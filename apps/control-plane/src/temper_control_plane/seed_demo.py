"""The zero-cost tier's demonstration content, seeded once at startup.

Spec 012's second front door is the existing double promoted: selecting the
zero-cost tier must *show* something the moment the stack is up, not ask the
reviewer to go and invent a dataset. So when the zero-cost tier boots against
an empty database, this module plants a sample dataset and one completed run.

The completed run is **not hand-written into the database.** It is a real
`orchestrator.run_job` against the in-package fake provider; the same call
the worker makes, the same state machine, the same events, the same artifact
verification and teardown. That is the only way the seeded record is
guaranteed to look exactly like a run a reviewer would drive themselves,
because it *is* one. ADR-0066 moved orchestration into a worker process; this
is the one deliberate exception, and it is scoped so it cannot reintroduce
what that record removed:

* it runs in the zero-cost tier only (`TEMPER_FAKE_PROVIDER`), where the
  double crosses no connection and bills nothing; there is no machine to
  orphan and no spend to watch, which is precisely what ADR-0066's worker
  exists to protect against;
* it runs at startup, synchronously, once, against an empty database, so the
  demonstration is complete before the first request is served;
* it is not the request path, and it starts no thread (ADR-0066's rule is
  about the request path starting threads, full stop).

The seeding is part of the mode, not a separate setting: flipping
`TEMPER_FAKE_PROVIDER` is what both enables the zero-cost tier and, on a fresh
database, brings its demonstration with it. Nothing else differs between the
modes (ADR-0024, ADR-0062, ADR-0071).
"""

from __future__ import annotations

import io
import time

from temper_control_plane import config, db, jobs, orchestrator, quote
from temper_control_plane.datasets import ingest
from temper_control_plane.logging import get_logger
from temper_core import catalog

logger = get_logger(__name__)

# The checked-in sample, read at seed time. One definition: the file is the
# same bytes a reviewer can open and read, so "the sample dataset included"
# and "the dataset the seeded run trained on" can never drift apart.
SAMPLE_DATASET_FILENAME = "sample-chat.jsonl"
SAMPLE_DATASET_PATH = config.REPO_ROOT / "samples" / SAMPLE_DATASET_FILENAME

# The delivery formats the seeded run asks for, so the finished record shows
# the full journey (issue #74): the canonical artifact plus the two optional
# formats, exactly as a user who ticked the boxes would get.
SAMPLE_DELIVERY = ("merged", "quantised")


def _database_is_empty() -> bool:
    """Whether a fresh start would collide with anything a human did.

    Seeding is for the reviewer's first boot. A database that already holds a
    dataset or a job belongs to someone who has been using the stack, and
    injecting a fake run into their history would be the dishonesty the whole
    mode exists to avoid.
    """
    return not db.list_datasets(limit=1) and not db.list_jobs(limit=1)


def _wait_for(
    fetch,
    done,
    what: str,
    timeout_s: float = 30.0,
) -> dict | None:
    """Poll `fetch` until `done`, or None when the wait expires."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        value = fetch()
        if done(value):
            return value
        time.sleep(0.05)
    logger.warning(
        "demo seed wait expired",
        what=what,
        timeout_s=timeout_s,
    )
    return None


def _seed_sample_dataset() -> str:
    """Store and validate the checked-in sample, returning its id."""
    if not SAMPLE_DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"sample dataset missing: {SAMPLE_DATASET_PATH}"
        )
    ds_id, _status = ingest(
        None,
        SAMPLE_DATASET_FILENAME,
        io.BytesIO(SAMPLE_DATASET_PATH.read_bytes()),
    )
    ds = _wait_for(
        lambda: db.get_dataset(ds_id),
        lambda d: d is not None and d.get("status") != "validating",
        "sample dataset validation",
    )
    if ds is None or ds.get("status") != "valid":
        problems = (ds or {}).get("report", {}).get("errors", [])
        raise RuntimeError(f"sample dataset failed validation: {problems!r}")
    # The token count is the expensive half of validation and lands a moment
    # after the report; the seeded run wants it present so the dataset's page
    # reads as complete. A count that never lands is an absent estimate, not a
    # broken seed, so this is a soft wait.
    _wait_for(
        lambda: db.get_dataset(ds_id),
        lambda d: (
            d is not None and d.get("token_count_status") in ("done", "failed")
        ),
        "sample dataset token count",
        timeout_s=15.0,
    )
    return ds_id


def _seed_completed_run(ds_id: str) -> str:
    """Create and drive one job to completion against the fake provider.

    The quote is computed through the same path the launch screen uses, the
    job is created through the one shared creation path, and the run is driven
    by the same `run_job` the worker calls, so the finished record carries
    every shape a real run carries (quote frozen, actuals measured, artifact
    verified, teardown confirmed).
    """
    quote_for_job = quote.quote_for_launch(ds_id, catalog.DEFAULT_MODEL, {})
    job_id = jobs.create(
        ds_id,
        catalog.DEFAULT_MODEL,
        {},
        quote=quote_for_job,
        delivery_request=list(SAMPLE_DELIVERY),
    )
    orchestrator.run_job(job_id)
    record = db.get_job(job_id)
    if record is None or record.get("status") != "complete":
        raise RuntimeError(
            f"demo job did not complete; status="
            f"{(record or {}).get('status')!r}"
        )
    return job_id


def maybe_seed() -> None:
    """Seed the demonstration content, or do nothing.

    Does nothing unless the zero-cost tier is in force and the database is
    empty. Failures are logged and swallowed: a broken seed must not take the
    stack down with it; the zero-cost tier still works without its demo, it
    just starts empty.
    """
    if not config.FAKE_PROVIDER:
        return
    if not _database_is_empty():
        return
    try:
        ds_id = _seed_sample_dataset()
        job_id = _seed_completed_run(ds_id)
        logger.info(
            "demo content seeded",
            dataset_id=ds_id,
            dataset=SAMPLE_DATASET_FILENAME,
            job_id=job_id,
        )
    except Exception:  # noqa: BLE001 - a broken seed is not a broken stack
        logger.exception("demo content seeding failed")
