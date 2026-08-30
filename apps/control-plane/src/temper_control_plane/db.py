"""PostgreSQL persistence behind the same functions Phase A's SQLite did.

Issue #43: the local file cannot express a claim on a row surviving
contention, so there is no path to running orchestration anywhere but inside
the request-serving process. A relational database with real migrations
replaces it. **Function names and return shapes stay; only their bodies
change** -- the measure of success is that nothing outside this file (and
`migrations.py`, its migration machinery) noticed.

What is kept from Phase A, because these are the parts that matter and they
travel with the seam rather than with SQLite:

* **The run spec is immutable after launch.** Hyperparameters are frozen into
  the job row at creation. `create_job` is the only function that writes
  `hyperparams_json`, `base_model` or `base_revision`; no function here
  updates them, so a later edit to a dataset or a default cannot
  retroactively change what a completed run claims to have done.
* **Every state transition appends an event, in the same transaction.** A job
  whose status moved with no event recorded is a job you cannot explain, and
  explaining runs is the product. `set_state` writes both inside one
  connection's transaction (psycopg's default: commit on success, rollback on
  any exception before it), so a failure between the two leaves neither.

See ADR-0064 for what the migration found: the seam bounded `db.py`'s
functions as claimed, but not the tests that reached past them at `DB_PATH`
directly, nor the health check's way of *causing* a database failure.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from temper_core import artifacts, delivery
from temper_core.errors import OrchestratorError

from . import config, migrations, storage

# Via config, not a literal: the value two processes (and every test that
# isolates itself) must agree on is the connection string, defined once here
# and read everywhere else through this module -- never retyped (ADR-0010).
DATABASE_URL = config.DATABASE_URL

# The full lifecycle. `preparing` covers image build and model download --
# separated from `training` because they fail for completely different reasons
# and a user staring at a progress bar deserves to know which one they are in.
JOB_STATES = (
    "queued",
    "provisioning",
    "preparing",
    "training",
    "packaging",
    "complete",
    "failed",
    "cancelled",
)
TERMINAL_STATES = {"complete", "failed", "cancelled"}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# One pool per `DATABASE_URL`, not one connection per call. Measured, not
# guessed: a single orchestrator test (`test_a_job_runs_to_completion_
# against_a_fake_provider`) opened 63 separate connections before pooling
# existed here -- one per state transition, one per promoted progress line,
# one per event -- and at roughly 17ms to open and close a connection on this
# machine, that is over a second of pure connection overhead inside one
# test's `call` phase. Across the suite it was the dominant cost behind the
# 2306-second run the PR for issue #43 records finding this in (see
# ADR-0064): per-test database cloning was real but secondary, on the order
# of 100ms a test, against a connection overhead that ran into the seconds
# for any test that drives the orchestrator through more than a handful of
# transitions.
#
# Keyed by URL rather than a single pool built at import, because tests
# monkeypatch `db.DATABASE_URL` to a fresh per-test database
# (`conftest.isolated`) and a pool bound to the wrong database would serve
# connections to a database that either is not this test's or, once its
# `isolated` fixture drops it, no longer exists. Only ever one pool is kept
# open: the moment a call is made against a new URL, every pool for a
# different URL is closed first. In production `DATABASE_URL` never changes,
# so this holds exactly one pool for the life of the process, opened once --
# and the lock below only ever guards the fast "it already exists" path there,
# since eviction never fires without a URL change.
#
# `_pools_lock` makes the read-check-create sequence atomic: without it, two
# threads racing on `pool is None` (the FastAPI request thread and a launched
# job's orchestrator thread both call `db.connect()`, per
# apps/control-plane/AGENTS.md) could each construct a `ConnectionPool` for
# the same URL, silently orphaning whichever loses the assignment -- a pool
# that is never closed, holding connections open for the life of the process.
_pools: dict[str, ConnectionPool] = {}
_pools_lock = threading.Lock()


def _pool() -> ConnectionPool:
    with _pools_lock:
        pool = _pools.get(DATABASE_URL)
        if pool is None:
            for stale_url, stale_pool in list(_pools.items()):
                if stale_url != DATABASE_URL:
                    stale_pool.close()
                    del _pools[stale_url]
            pool = _pools[DATABASE_URL] = ConnectionPool(
                DATABASE_URL,
                min_size=1,
                max_size=8,
                kwargs={"row_factory": dict_row, "autocommit": False},
                open=True,
            )
        return pool


@contextmanager
def connect():
    """One connection from the pool, one transaction: `pool.connection()`
    applies psycopg's normal connection context behaviour -- commits on
    success, rolls back on any exception raised inside the `with` block --
    and returns the connection to the pool either way rather than closing
    it."""
    with _pool().connection() as conn:
        yield conn


def init() -> None:
    with connect() as c:
        if config.DB_RESET:
            # A clean slate, once, at startup: roll every migration back and
            # forward again so the process boots against a fresh schema. The
            # e2e journeys ask for this so an interrupted run cannot leave an
            # orphaned non-terminal job behind that crashes the next run's
            # startup. The Phase A equivalent deleted the SQLite file; a
            # shared server has no file to delete, so this walks the same
            # migrations everything else runs through instead of a second,
            # untested way to build the schema.
            migrations.reset(c)
        else:
            migrations.migrate_up(c)


def ping() -> None:
    """One trivial read, proving the database opens and answers.

    The readiness probe the health endpoint reports per dependency (issue
    #29): a database that cannot be reached or whose connection fails is a
    broken dependency, and `/health` must be able to say so separately from a
    broken application. Raises with the driver's own error when the database
    cannot answer.
    """
    with connect() as c:
        c.execute("SELECT 1")


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


def create_dataset(
    filename: str, object_key: str, ds_id: str | None = None
) -> str:
    """Insert a dataset row. `ds_id` lets the caller own the id -- the upload
    path derives the key from it, so the id must exist before the row."""
    ds_id = ds_id or new_id("ds")
    with connect() as c:
        c.execute(
            "INSERT INTO datasets (id, filename, object_key, created_at, status)"
            " VALUES (%s,%s,%s,%s,%s)",
            (ds_id, filename, object_key, time.time(), "validating"),
        )
    return ds_id


def delete_dataset(ds_id: str) -> None:
    """Remove a dataset row whose object never made it into storage.

    Used only by the ingest path to clean up after a mid-stream refusal --
    an upload refused once its true size is known has no object behind it, so
    nothing but the row is left to remove. Deleting a dataset that reached a
    report is not this function's contract.
    """
    with connect() as c:
        c.execute("DELETE FROM datasets WHERE id=%s", (ds_id,))


def finish_dataset(ds_id: str, report: dict) -> None:
    with connect() as c:
        c.execute(
            "UPDATE datasets SET status=%s, row_count=%s, schema_type=%s, "
            "enable_thinking=%s, report_json=%s, progress_json=NULL WHERE id=%s",
            (
                "valid" if report.get("valid") else "invalid",
                report.get("row_count"),
                report.get("schema_type"),
                None
                if report.get("enable_thinking") is None
                else int(report["enable_thinking"]),
                json.dumps(report),
                ds_id,
            ),
        )


def set_dataset_progress(ds_id: str, progress: dict) -> None:
    """Record where validation has got to, for the page watching it run."""
    with connect() as c:
        c.execute(
            "UPDATE datasets SET progress_json=%s WHERE id=%s",
            (json.dumps(progress), ds_id),
        )


# --------------------------------------------------------------------------
# the counting phase (issue #42): its own state, recorded with the version
# --------------------------------------------------------------------------
# Counting runs as a separate phase after validation -- the measurement put the
# ticket on that branch -- so the count has its own state on the row rather
# than living inside the validation report. The report stands the moment
# validation finishes; the count lands later and is merged into the report the
# API publishes, which is the seam the quote already reads.


# The counting phase's own states (issue #42). Defined once here and read
# everywhere -- the phase's transitions write them, the tests assert against
# them -- so the vocabulary cannot drift between the two halves.
COUNT_PHASE_COUNTING = "counting"
COUNT_PHASE_DONE = "done"
COUNT_PHASE_FAILED = "failed"


def begin_token_count(ds_id: str) -> None:
    """Mark the counting phase as in progress, before it starts.

    The dataset stays `valid` -- it is launchable the whole time, and the
    quote renders the count as absent until it lands (ADR-0031's documented
    posture for a count that does not exist yet)."""
    with connect() as c:
        c.execute(
            "UPDATE datasets SET token_count_status=%s, "
            "token_count=NULL, token_stats_json=NULL, "
            "counting_progress_json=NULL WHERE id=%s",
            (COUNT_PHASE_COUNTING, ds_id),
        )


def set_counting_progress(ds_id: str, progress: dict) -> None:
    """Where the counting pass has got to, for the report page watching it."""
    with connect() as c:
        c.execute(
            "UPDATE datasets SET counting_progress_json=%s WHERE id=%s",
            (json.dumps(progress), ds_id),
        )


def finish_token_count(ds_id: str, counts: dict) -> None:
    """Record the completed count with the dataset version: the total, and the
    bounded distribution. A finished phase has no progress to show."""
    with connect() as c:
        c.execute(
            "UPDATE datasets SET token_count_status=%s, token_count=%s, "
            "token_stats_json=%s, counting_progress_json=NULL WHERE id=%s",
            (
                COUNT_PHASE_DONE,
                counts.get("total_tokens"),
                json.dumps(counts),
                ds_id,
            ),
        )


def fail_token_count(ds_id: str) -> None:
    """The count could not be produced. The dataset stays valid and launchable
    -- the quote renders the count absent -- and the phase's state records why
    rather than leaving the report page to wonder."""
    with connect() as c:
        c.execute(
            "UPDATE datasets SET token_count_status=%s, "
            "counting_progress_json=NULL WHERE id=%s",
            (COUNT_PHASE_FAILED, ds_id),
        )


def get_dataset(ds_id: str) -> dict | None:
    with connect() as c:
        r = c.execute(
            "SELECT * FROM datasets WHERE id=%s", (ds_id,)
        ).fetchone()
    return _dataset_row(r)


def require_dataset(ds_id: str) -> dict:
    """The dataset row, or a coded error -- never None.

    For callers that cannot proceed without the row: the orchestrator reads
    the dataset on the path that spends money, and indexing a None there
    must not be the way a missing row is discovered.
    """
    ds = get_dataset(ds_id)
    if ds is None:
        raise OrchestratorError(
            "dataset_not_found", f"No dataset with id '{ds_id}'."
        )
    return ds


def list_datasets(limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM datasets ORDER BY created_at DESC LIMIT %s",
            (limit,),
        ).fetchall()
    return [_present(_dataset_row(r)) for r in rows]


# --------------------------------------------------------------------------
# admitted models (issue #58): the probe result is persisted and shown, and a
# job can be created against a model only after its probe is on record
# --------------------------------------------------------------------------


def create_admitted_model(repo: str, revision: str, probe: dict) -> str:
    """Persist one admission. The reference is unique, so re-probing the same
    pinned model returns its existing row rather than a second one."""
    model_id = new_id("m")
    with connect() as c:
        c.execute(
            "INSERT INTO admitted_models "
            "(id, repo, revision, probe_json, created_at) VALUES (%s,%s,%s,%s,%s) "
            "ON CONFLICT (repo, revision) DO NOTHING",
            (model_id, repo, revision, json.dumps(probe), time.time()),
        )
        row = c.execute(
            "SELECT id FROM admitted_models WHERE repo=%s AND revision=%s",
            (repo, revision),
        ).fetchone()
    return row["id"]


def get_admitted_model(model_id: str) -> dict | None:
    """One admitted model as the API publishes it: its probe result parsed,
    never the raw JSON string."""
    with connect() as c:
        r = c.execute(
            "SELECT * FROM admitted_models WHERE id=%s", (model_id,)
        ).fetchone()
    return _admitted_row(r)


def list_admitted_models(limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM admitted_models ORDER BY created_at DESC LIMIT %s",
            (limit,),
        ).fetchall()
    return [_present(_admitted_row(r)) for r in rows]


def _admitted_row(r) -> dict | None:
    row = _row(r, {"probe_json": "probe"})
    return row


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


def create_job(
    dataset_id: str,
    base_model: str,
    hyperparams: dict,
    warnings: list | None = None,
    base_revision: str | None = None,
    quote: dict | None = None,
    overrides: list | None = None,
    retry_from: str | None = None,
    is_moe: bool | None = None,
    delivery_request: list | None = None,
    correlation_id: str | None = None,
) -> str:
    """Insert a job row. ``correlation_id`` is the request identifier the job
    inherits so every log line the job emits carries the same story (issue
    #52). ``None`` means a job created before that migration or outside a
    request context -- a missing correlation is an honest absence, not a
    redaction."""
    # Fall back to the current context's identifier when the caller did not
    # name one explicitly -- the request path's middleware already bound it,
    # so a plain ``jobs.create`` inside a request inherits it without threading
    # a parameter through every helper.
    if correlation_id is None:
        try:
            from temper_control_plane.correlation import get_correlation_id

            correlation_id = get_correlation_id()
        except Exception:
            correlation_id = None
    job_id = new_id("job")
    with connect() as c:
        c.execute(
            "INSERT INTO jobs (id, dataset_id, base_model, base_revision, "
            "hyperparams_json, status, warnings_json, quote_json, "
            "overrides_json, retry_from, is_moe, delivery_request_json, "
            "correlation_id, created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                job_id,
                dataset_id,
                base_model,
                base_revision,
                json.dumps(hyperparams),
                "queued",
                json.dumps(warnings) if warnings else None,
                json.dumps(quote) if quote else None,
                json.dumps(overrides) if overrides else None,
                retry_from,
                int(is_moe) if is_moe is not None else None,
                json.dumps(delivery_request) if delivery_request else None,
                correlation_id,
                time.time(),
            ),
        )
        _append_event(c, job_id, "state", "queued")
    return job_id


def has_retry(job_id: str) -> bool:
    """Whether a job already has a retry child (issue #36 single-retry offer)."""
    with connect() as c:
        row = c.execute(
            "SELECT id FROM jobs WHERE retry_from=%s LIMIT 1", (job_id,)
        ).fetchone()
    return row is not None


def set_state(
    job_id: str, state: str, message: str | None = None, **fields
) -> None:
    """Move a job to a new state and record why, atomically.

    The event is written in the same transaction as the status change on
    purpose: a status that moved with no event is a run you cannot account
    for. `connect()` commits both together on success and rolls both back on
    any exception raised before the block exits -- forcing the event write to
    fail (e.g. a job id the events table's foreign key rejects) leaves the
    status update rolled back too, which is what `test_transactions.py`
    proves rather than asserts.
    """
    if state not in JOB_STATES:
        raise ValueError(f"unknown state {state!r}")
    cols, vals = ["status=%s"], [state]
    if state == "training" and "started_at" not in fields:
        fields["started_at"] = time.time()
    if state in TERMINAL_STATES:
        fields.setdefault("finished_at", time.time())
    for k, v in fields.items():
        cols.append(f"{k}=%s")
        vals.append(json.dumps(v) if k.endswith("_json") else v)
    vals.append(job_id)
    with connect() as c:
        c.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id=%s", vals)
        # The event carries the state it entered as data, not only as a
        # message that may be the human narration ("Selecting hardware")
        # rather than the state's name: issue #77's stage durations are
        # measured from these events, and a state event that does not say
        # which state it entered cannot be measured.
        _append_event(c, job_id, "state", message or state, {"state": state})


def set_checkpoints(job_id: str, checkpoints: list[dict]) -> None:
    """Record the checkpoints the control plane has verified in storage.

    Written without an event: recording a verified checkpoint is bookkeeping,
    and the verification itself already appends the event a reader would want
    to see. The column is replaced whole each time -- the machine reports the
    run's checkpoints once, in result.json, and the control plane records its
    verdict on that set.
    """
    with connect() as c:
        c.execute(
            "UPDATE jobs SET checkpoints_json=%s WHERE id=%s",
            (json.dumps(checkpoints), job_id),
        )


def set_delivery(job_id: str, records: list[dict]) -> None:
    """Record the verified per-format delivery records (issue #74).

    Written without an event, like `set_checkpoints`: recording what the
    machine produced and the control plane verified is bookkeeping, and each
    format's verification already appends the event a reader would want to
    see. `records` is the trainer's `result["delivery"]` list, enriched by the
    orchestrator with each format's storage key and checksum after it verifies
    what landed -- the same shape the download path reads to serve each format.
    """
    with connect() as c:
        c.execute(
            "UPDATE jobs SET delivery_json=%s WHERE id=%s",
            (json.dumps(records), job_id),
        )


def set_best_checkpoint(job_id: str, selection: dict) -> None:
    """Record the run's chosen result checkpoint (issue #62).

    Written once, at the same moment the verified checkpoints are recorded,
    and never recomputed: the whole point of recording the choice is that a
    later reader sees the same answer even if retention evicts a checkpoint
    or the selection rule is edited. `selection` is the stored shape of
    `temper_core.checkpoint.CheckpointSelection` -- the step, the basis and
    the reason -- so the record carries its own why.
    """
    with connect() as c:
        c.execute(
            "UPDATE jobs SET best_checkpoint_json=%s WHERE id=%s",
            (json.dumps(selection), job_id),
        )


def set_attempts(job_id: str, attempts: list[dict]) -> None:
    """Record the job's executions (issue #35): one record per attempt, each
    carrying its own machine, spec and outcome.

    Written whole each time because the orchestrator appends as each attempt
    ends and there is one writer per job; a memory failure's automatic retry
    makes attempts plural, and the record is what "the attempts recorded"
    means in the retry-cap criterion.
    """
    with connect() as c:
        c.execute(
            "UPDATE jobs SET attempts_json=%s WHERE id=%s",
            (json.dumps(attempts), job_id),
        )


def request_cancel(job_id: str, note: str = "Cancellation requested") -> str:
    """Ask a job to stop. Says what it found, and never raises for it.

    Returns one of `accepted`, `already_cancelling`, `terminal` or `missing`.
    Strings rather than exceptions because none of the four is exceptional —
    they are the four honest answers to the request, and each maps to a
    different thing to tell the user.

    The read and the write are one transaction, so a job that reaches a
    terminal state between them cannot be flagged after the fact: the
    orchestrator would have stopped looking at the flag by then, and the row
    would claim a cancellation that will never happen.

    `note` is the event recorded against the job. It is the caller's words on
    purpose: the same sentence is what the request answers with, and one string
    said in both places is what stops the button and the run's own history from
    drifting apart.

    The event is written only on the transition. Cancelling is idempotent, and
    a second request that appended a second "cancellation requested" would make
    a double-clicked button look like two decisions in the job's own history.
    """
    with connect() as c:
        row = c.execute(
            "SELECT status, cancel_requested FROM jobs WHERE id=%s", (job_id,)
        ).fetchone()
        if row is None:
            return "missing"
        if row["status"] in TERMINAL_STATES:
            return "terminal"
        if row["cancel_requested"]:
            return "already_cancelling"
        c.execute("UPDATE jobs SET cancel_requested=1 WHERE id=%s", (job_id,))
        _append_event(c, job_id, "log", note)
    return "accepted"


def cancel_requested(job_id: str) -> bool:
    """Whether the user has asked this job to stop.

    Read from the row on every check rather than cached: the request arrives on
    a different thread than the one running the job, and a value read once at
    the start is a value that can never say yes.
    """
    with connect() as c:
        row = c.execute(
            "SELECT cancel_requested FROM jobs WHERE id=%s", (job_id,)
        ).fetchone()
    return bool(row and row["cancel_requested"])


def record_actuals(job_id: str, actuals: dict) -> None:
    """Freeze the measured figures onto a terminal job (issue #77).

    Written once, like the quote -- the mirror image of it: the quote says
    what was predicted before launch, the actuals say what was measured
    after. Called by the orchestrator the moment a run reaches a terminal
    state, so recording starts with the first run rather than the last.
    """
    with connect() as c:
        c.execute(
            "UPDATE jobs SET actuals_json=%s WHERE id=%s",
            (json.dumps(actuals), job_id),
        )


def add_event(
    job_id: str, kind: str, message: str, data: dict | None = None
) -> None:
    with connect() as c:
        _append_event(c, job_id, kind, message, data)


def _append_event(conn, job_id, kind, message, data=None) -> None:
    conn.execute(
        "INSERT INTO events (job_id, ts, kind, message, data_json) VALUES (%s,%s,%s,%s,%s)",
        (
            job_id,
            time.time(),
            kind,
            message,
            json.dumps(data) if data is not None else None,
        ),
    )


# --------------------------------------------------------------------------
# progress (issue #49): one superseding row per phase, and the raw lines that
# were promoted, retained whole
# --------------------------------------------------------------------------


def upsert_progress(
    job_id: str,
    phase: str,
    done: float | None,
    total: float | None,
    rate: float | None,
    eta_s: float | None,
    ts: float,
    message: str | None,
) -> None:
    """Record the phase's current progress, replacing any previous record.

    Progress supersedes rather than accumulates: this is the one row per phase
    the interface renders, so the hundreds of lines a pull produces never
    become hundreds of rows. The raw lines themselves are retained separately
    by `append_output`; this row is the summary that replaces itself.
    """
    with connect() as c:
        c.execute(
            "INSERT INTO job_progress (job_id, phase, done, total, rate, "
            "eta_s, ts, message) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT(job_id, phase) DO UPDATE SET "
            "done=excluded.done, total=excluded.total, rate=excluded.rate, "
            "eta_s=excluded.eta_s, ts=excluded.ts, message=excluded.message",
            (job_id, phase, done, total, rate, eta_s, ts, message),
        )


def append_output(job_id: str, phase: str, line: str) -> None:
    """Retain one promoted raw line on the job's output record.

    The line was promoted into progress and deliberately not emitted as an
    event; keeping it here is what makes "nothing is discarded" true. The
    interface offers these as collapsed detail per phase.
    """
    with connect() as c:
        c.execute(
            "INSERT INTO job_output (job_id, phase, line) VALUES (%s,%s,%s)",
            (job_id, phase, line),
        )


def get_progress(job_id: str) -> list[dict]:
    """The job's per-phase progress records, superseded latest."""
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM job_progress WHERE job_id=%s ORDER BY phase",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_output(job_id: str) -> list[dict]:
    """The job's retained promoted lines, in the order they were written."""
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM job_output WHERE job_id=%s ORDER BY id",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _with_warnings(job: dict | None) -> dict | None:
    if job is not None and job["warnings"] is None:
        # An absent list reads as empty, so no client special-cases null.
        job["warnings"] = []
    return job


def _with_is_moe(job: dict | None) -> dict | None:
    """Normalise the `is_moe` column for API publication.

    The store keeps booleans as 0/1 integers and legacy rows carry NULL.
    The API publishes `is_moe` as a boolean when known and null otherwise,
    so the stored integer is converted and an absent value stays absent.
    Frozen at creation (issue #65): the label travels with the job so a
    finished run says what it was trained on.
    """
    if job is not None and "is_moe" in job:
        val = job.get("is_moe")
        if val is None:
            job["is_moe"] = None
        else:
            job["is_moe"] = bool(val)
    return job


def artifact_members(job: dict) -> list[tuple[str, str]]:
    """The `(arcname, storage key)` pairs one job's artifact consists of.

    The one place a job's artifact is resolved to its stored objects (issue
    #32). The members come from the record the orchestrator assembled at
    packaging time -- whatever kind it is -- and, for a row written before the
    record existed, from the canonical adapter pair. The download path and the
    teardown path both read this, so the two cannot disagree about what the
    artifact is or drift apart in spelling (a value two components must agree
    on is defined once and read, never retyped).

    Issue #74 extends the resolution to the delivery formats: each produced
    format's verified object is a member too, recorded at packaging time, so
    the download path can serve it and teardown deletes it -- one resolution,
    every object a job owns.
    """
    members = _canonical_members(job)
    for record in job.get("delivery") or []:
        if not isinstance(record, dict) or not record.get("verified"):
            continue
        for m in record.get("members") or []:
            if isinstance(m, dict) and m.get("name") and m.get("key"):
                members.append((m["name"], m["key"]))
    return members


def _canonical_members(job: dict) -> list[tuple[str, str]]:
    """The canonical artifact's members: the record's, or the legacy pair."""
    record = job.get("artifact_record")
    if record and record.get("members"):
        return [
            (m["name"], m["key"])
            for m in record["members"]
            if m.get("name") and m.get("key")
        ]
    artifact_key = job.get("artifact_key")
    if artifact_key:
        # The legacy shape: a row written before the record existed named one
        # weights key, and the adapter's member names are the canonical pair,
        # each addressed by the seam's key helper.
        return [
            (name, storage.artifact_key(job["id"], name))
            for name in artifacts.ADAPTER_MEMBER_NAMES
        ]
    return []


def canonical_artifact_members(job: dict) -> list[tuple[str, str]]:
    """The canonical artifact's members only -- what the default download serves.

    The download endpoint offers the canonical artifact by default and each
    delivery format by a `format` query parameter, so it needs the canonical
    set separate from everything a job owns (`artifact_members`, which
    teardown deletes). Named for what it resolves: the one artifact a method
    produced, without the delivery formats.
    """
    return _canonical_members(job)


def delivery_format_members(
    job: dict, format_id: str
) -> list[tuple[str, str]]:
    """One produced delivery format's members, or an empty list.

    Issue #74. Reads the stored, verified delivery record for `format_id` and
    returns its member pairs -- the archive entry name and the storage key it
    was verified at. An unproduced or unverified format returns empty, which
    the download path turns into a coded refusal rather than an empty archive.
    """
    for record in job.get("delivery") or []:
        if (
            isinstance(record, dict)
            and record.get("format") == format_id
            and record.get("verified")
        ):
            return [
                (m["name"], m["key"])
                for m in (record.get("members") or [])
                if isinstance(m, dict) and m.get("name") and m.get("key")
            ]
    return []


def _with_artifact(job: dict | None) -> dict | None:
    """Publish the artifact record on a job row (issue #32).

    The artifact's kind is **derived from the job's method, never stored**: a
    QLoRA or LoRA run produces an adapter, a full fine-tune produces a fully
    trained model, and the derivation is what lets pre-existing rows read
    correctly with no migration. The members come from the same resolution the
    download path and teardown use (`artifact_members`).

    An artifact is available exactly when the job is complete: a failed or
    cancelled job that retains keys has had its objects deleted, so publishing
    one there would offer a download that cannot succeed. The published record
    is the safe subset: member *names* (arcnames) and the load path, never the
    storage keys -- where an object lives is the seam's business, not the
    browser's.
    """
    if job is None:
        return None
    members = _canonical_members(job)
    if not members or job.get("status") != "complete":
        job["artifact"] = None
        return job
    kind = artifacts.kind_for(job.get("method"))
    record = job.get("artifact_record")
    job["artifact"] = {
        "kind": kind,
        "members": [name for name, _ in members],
        "bytes": record.get("bytes") if record else None,
        "loading": artifacts.loading_instructions(kind),
    }
    # Issue #74: publish the delivery formats the machine produced, each with
    # its plain-language purpose and its member names -- the same safe subset
    # rule as the canonical artifact (names and load path, never storage keys).
    # The published record is built from the stored `delivery` list (which
    # keeps its keys for the download and teardown paths), so the two cannot
    # disagree about what a job produced. Published under `delivery_formats`
    # so the stored records (keys included) stay intact for those paths.
    job["delivery_formats"] = _published_delivery(job)
    return job


def _published_delivery(job: dict) -> list[dict]:
    """The delivery formats a complete job produced, for the interface.

    Reads the stored `delivery` records (written by the orchestrator at
    packaging time) and decorates each with the purpose and loading defined in
    the domain -- the same single definitions the download manifest reads, so
    a format is described once. A job that produced no extra formats (or a
    pre-delivery row) publishes an empty list.
    """
    records = job.get("delivery") or []
    out: list[dict] = []
    for record in records:
        if not isinstance(record, dict) or not record.get("format"):
            continue
        fmt_id = record["format"]
        try:
            kind = delivery.kind_for(fmt_id)
            what_for = delivery.what_for(fmt_id)
        except delivery.UnknownDeliveryFormat:
            continue
        member_names = [
            str(m.get("name"))
            for m in (record.get("members") or [])
            if isinstance(m, dict) and m.get("name")
        ]
        out.append(
            {
                "format": fmt_id,
                "kind": kind,
                "what_for": what_for,
                "members": member_names,
                "bytes": record.get("bytes"),
                "loading": artifacts.loading_instructions(kind),
            }
        )
    return out


def _with_best_checkpoint(job: dict | None) -> dict | None:
    """Publish the run's recorded choice, flagging the chosen checkpoint.

    The choice itself is stored once at terminal time (issue #62); this only
    decorates the published record so the interface can highlight which
    checkpoint the run stands by without re-selecting. Marking `selected`
    from the stored `best_checkpoint.step` is presentation, never a
    re-derivation of the choice: if the two ever disagreed, the stored choice
    is the one the run answers with, and the flag follows it.
    """
    if job is None:
        return None
    if job.get("checkpoints") is None:
        job["checkpoints"] = []
    if job.get("best_checkpoint") is None:
        return job
    best = job["best_checkpoint"]
    step = best.get("step") if isinstance(best, dict) else None
    if step is not None:
        for ckpt in job["checkpoints"]:
            if isinstance(ckpt, dict) and ckpt.get("step") == step:
                ckpt["selected"] = True
    return job


def _with_comparison(job: dict | None) -> dict | None:
    """Publish the run's recorded comparison from its result document.

    The trainer records the side-by-side comparison (issue #69) in
    result.json; this presents it typed, without recomputing or reshaping
    anything -- a run's answer is what the machine recorded, and a
    pre-existing row whose result carries no comparison simply has none
    published. On a failed comparison the trainer's `ok: false` and `reason`
    travel as recorded, so the interface can say why there is no side-by-side
    rather than pretending one exists.
    """
    if job is None:
        return None
    result = job.get("result")
    recorded = result.get("comparison") if isinstance(result, dict) else None
    job["comparison"] = recorded if isinstance(recorded, dict) else None
    return job


def get_job(job_id: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
    return _with_best_checkpoint(
        _with_artifact(
            _with_is_moe(
                _with_warnings(
                    _with_comparison(
                        _row(
                            r,
                            {
                                "hyperparams_json": "hyperparameters",
                                "result_json": "result",
                                "warnings_json": "warnings",
                                "quote_json": "quote",
                                "overrides_json": "overrides",
                                "actuals_json": "actuals",
                                "checkpoints_json": "checkpoints",
                                "best_checkpoint_json": "best_checkpoint",
                                "artifact_json": "artifact_record",
                                "delivery_request_json": "delivery_request",
                                "delivery_json": "delivery",
                                "attempts_json": "attempts",
                            },
                            defaults={
                                "overrides": [],
                                "checkpoints": [],
                                "best_checkpoint": None,
                                "delivery_request": [],
                                "delivery": [],
                                "attempts": [],
                            },
                        )
                    )
                )
            )
        )
    )


def require_job(job_id: str) -> dict:
    """The job row, or a coded error -- never None.

    The raising companion to `get_job`, for the orchestration path: a job
    row that goes missing between launch and a re-read must surface as a
    named failure on the record, not as a TypeError mid-provisioning.
    """
    job = get_job(job_id)
    if job is None:
        raise OrchestratorError("job_not_found", f"No job with id '{job_id}'.")
    return job


def list_jobs(limit: int | None = 50) -> list[dict]:
    """The job list, newest first. `limit=None` reads every row: the job list
    endpoint caps at 50, but an aggregate that says "across N runs" must not
    silently drop the runs older than its reader's page."""
    with connect() as c:
        if limit is None:
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
    return [
        _present(
            _with_best_checkpoint(
                _with_artifact(
                    _with_is_moe(
                        _with_warnings(
                            _with_comparison(
                                _row(
                                    r,
                                    {
                                        "hyperparams_json": "hyperparameters",
                                        "result_json": "result",
                                        "warnings_json": "warnings",
                                        "quote_json": "quote",
                                        "overrides_json": "overrides",
                                        "actuals_json": "actuals",
                                        "checkpoints_json": "checkpoints",
                                        "best_checkpoint_json": "best_checkpoint",
                                        "artifact_json": "artifact_record",
                                        "delivery_request_json": "delivery_request",
                                        "delivery_json": "delivery",
                                    },
                                    defaults={
                                        "overrides": [],
                                        "checkpoints": [],
                                        "best_checkpoint": None,
                                        "delivery_request": [],
                                        "delivery": [],
                                    },
                                )
                            )
                        )
                    )
                )
            )
        )
        for r in rows
    ]


def count_events(job_id: str) -> int:
    """How many events a job has in total, regardless of any paging window.

    The page returns a window (`after`/`limit`) but the interface must say
    what it is showing and of how many (issue #56): silently cutting after
    500 oldest-first hid the artifact verification, teardown and completion.
    The total is what makes a cut stated rather than silent, and a stated
    one is a feature.
    """
    with connect() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM events WHERE job_id=%s", (job_id,)
        ).fetchone()
    return int(row["n"]) if row else 0


def get_events(job_id: str, after_id: int = 0, limit: int = 500) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM events WHERE job_id=%s AND id>%s ORDER BY id LIMIT %s",
            (job_id, after_id, limit),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["data"] = (
            json.loads(d.pop("data_json")) if d.get("data_json") else None
        )
        out.append(d)
    return out


def claim_next_job() -> dict | None:
    """Atomically claim one queued job for a worker.

    The local file that preceded PostgreSQL could not express a claim on a
    row surviving contention, so there was no path to running orchestration
    outside the request-serving process. This is the mechanism that issue
    points at: ``SELECT ... FOR UPDATE SKIP LOCKED``.

    * ``FOR UPDATE`` takes a row-level lock that survives contention: a
      second worker trying to claim the same row blocks on the first's
      transaction.
    * ``SKIP LOCKED`` is what makes the second half of the criterion true:
      workers that would otherwise block *skip* the locked row instead.
      Without it two workers would serialize on the lock; with it they
      never wait and never claim the same job.

    The claim and its event are one transaction: a status that moved with
    no event is a run you cannot explain.
    """

    with connect() as c:
        row = c.execute(
            """
            WITH claimed AS (
                SELECT id FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE jobs
            SET status = 'provisioning'
            WHERE id IN (SELECT id FROM claimed)
            RETURNING *
            """
        ).fetchone()
        if row is None:
            return None
        _append_event(
            c, row["id"], "state", "provisioning", {"state": "provisioning"}
        )
        job = _row(
            row,
            {
                "hyperparams_json": "hyperparameters",
                "result_json": "result",
                "warnings_json": "warnings",
                "quote_json": "quote",
                "overrides_json": "overrides",
                "actuals_json": "actuals",
                "checkpoints_json": "checkpoints",
                "best_checkpoint_json": "best_checkpoint",
                "artifact_json": "artifact_record",
                "delivery_request_json": "delivery_request",
                "delivery_json": "delivery",
                "attempts_json": "attempts",
            },
            defaults={
                "overrides": [],
                "checkpoints": [],
                "best_checkpoint": None,
                "delivery_request": [],
                "delivery": [],
                "attempts": [],
            },
        )
        return _with_best_checkpoint(
            _with_artifact(_with_is_moe(_with_warnings(_with_comparison(job))))
        )


def _row(
    r, json_fields: dict[str, str], defaults: dict[str, Any] | None = None
) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for col, name in json_fields.items():
        raw = d.pop(col, None)
        if raw:
            d[name] = json.loads(raw)
        elif defaults and name in defaults:
            d[name] = defaults[name]
        else:
            d[name] = None
    return d


def _dataset_row(r) -> dict | None:
    """A dataset row as the API publishes it.

    The token fields (issue #42) are produced by the phase that runs after
    validation, so they are stored separately and merged into the report here:
    `report.token_count` is the seam the quote reads, and the distribution
    travels beside it for the report page. The counting phase's own state and
    progress ride on the record too.
    """
    ds = _row(
        r,
        {
            "report_json": "report",
            "progress_json": "progress",
            "token_stats_json": "token_distribution",
            "counting_progress_json": "counting_progress",
        },
    )
    if ds is None:
        return None
    token_count = ds.pop("token_count", None)
    token_distribution = ds.pop("token_distribution", None)
    if ds.get("report") is not None:
        ds["report"]["token_count"] = token_count
        ds["report"]["token_distribution"] = token_distribution
    return ds


def _present(row: dict | None) -> dict:
    """Narrow a listed row that cannot actually be absent.

    `fetchall()` returns one row object per matched record, so the None
    case of `_row` is unreachable in a list comprehension over its output
    -- but the Optional still fails the `list[dict]` return type, and
    filtering the Nones out would hide a defect rather than surface it.
    """
    if row is None:
        raise ValueError("a listed row came back missing")
    return row


# --------------------------------------------------------------------------
# serving endpoints (issue #78): temporary authenticated endpoints
# --------------------------------------------------------------------------

ENDPOINT_STATUSES = ("running", "stopped", "expired")
ENDPOINT_ACTIVE = "running"


def create_endpoint(
    endpoint_id: str,
    job_id: str,
    api_key_hash: str,
    api_key_prefix: str,
    created_at: float,
    expires_at: float,
    max_expires_at: float,
    machine_id: int | None = None,
    price_per_hour: float | None = None,
    currency: str | None = None,
) -> None:
    """Insert one endpoint row. The key is stored hashed, never plaintext."""
    with connect() as c:
        c.execute(
            "INSERT INTO endpoints (id, job_id, status, api_key_hash, api_key_prefix, "
            "created_at, expires_at, last_used_at, max_expires_at, machine_id, "
            "price_per_hour, currency) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                endpoint_id,
                job_id,
                ENDPOINT_ACTIVE,
                api_key_hash,
                api_key_prefix,
                created_at,
                expires_at,
                created_at,
                max_expires_at,
                machine_id,
                price_per_hour,
                currency,
            ),
        )


def get_endpoint(endpoint_id: str) -> dict | None:
    with connect() as c:
        r = c.execute(
            "SELECT * FROM endpoints WHERE id=%s", (endpoint_id,)
        ).fetchone()
    if r is None:
        return None
    return dict(r)


def get_endpoint_by_job(
    job_id: str, status: str | None = ENDPOINT_ACTIVE
) -> dict | None:
    """The job's active endpoint, if any. One active endpoint per job."""
    with connect() as c:
        if status is None:
            r = c.execute(
                "SELECT * FROM endpoints WHERE job_id=%s ORDER BY created_at DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        else:
            r = c.execute(
                "SELECT * FROM endpoints WHERE job_id=%s AND status=%s "
                "ORDER BY created_at DESC LIMIT 1",
                (job_id, status),
            ).fetchone()
    if r is None:
        return None
    return dict(r)


def list_endpoints(
    job_id: str | None = None, status: str | None = None
) -> list[dict]:
    with connect() as c:
        q = "SELECT * FROM endpoints WHERE 1=1"
        params: list = []
        if job_id is not None:
            q += " AND job_id=%s"
            params.append(job_id)
        if status is not None:
            q += " AND status=%s"
            params.append(status)
        q += " ORDER BY created_at DESC"
        rows = c.execute(q, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def active_endpoints() -> list[dict]:
    """All running endpoints. Used at startup to re-arm timers and at sweep."""
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM endpoints WHERE status=%s", (ENDPOINT_ACTIVE,)
        ).fetchall()
    return [dict(r) for r in rows]


def touch_endpoint(
    endpoint_id: str, now: float, new_expires_at: float
) -> None:
    """Extend an endpoint's idle expiry after a successful use."""
    with connect() as c:
        c.execute(
            "UPDATE endpoints SET last_used_at=%s, expires_at=%s WHERE id=%s",
            (now, new_expires_at, endpoint_id),
        )


def set_endpoint_status(
    endpoint_id: str,
    status: str,
    stopped_at: float | None = None,
    stop_reason: str | None = None,
) -> None:
    if status not in ENDPOINT_STATUSES:
        raise ValueError(f"unknown endpoint status {status!r}")
    with connect() as c:
        c.execute(
            "UPDATE endpoints SET status=%s, stopped_at=%s, stop_reason=%s WHERE id=%s",
            (status, stopped_at, stop_reason, endpoint_id),
        )


def delete_endpoint(endpoint_id: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM endpoints WHERE id=%s", (endpoint_id,))
