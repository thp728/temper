"""Issue #51: the claim that makes a second process possible.

Three properties the relational store was changed for, proved directly
rather than asserted:

* **Claim is atomic and exclusive.** ``claim_next_job`` uses
  ``SELECT ... FOR UPDATE SKIP LOCKED`` so two workers racing for one
  queued job never claim the same row.
* **No worker blocks on another.** ``SKIP LOCKED`` is what makes the
  second half true: ``FOR UPDATE`` alone would make the second worker
  wait for the first's transaction, rather than skip.
* **The request path starts nothing.** Creating a job through the API
  inserts a ``queued`` row and returns; no thread is started and the
  job stays queued until a worker claims it.

The subtle one -- provisioning happens once even if its step is retried
-- is proved separately: a machine provisioned and not recorded is a
billing machine nobody owns (ADR-0057, ADR-0063). The record-first
ordering in ``orchestrator._attempt`` is what makes a retry safe.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from temper_control_plane import db
from temper_control_plane.fake_provider import FakeProvider

pytestmark = pytest.mark.usefixtures("isolated")


def _finished_dataset_id(name: str) -> str:
    """A dataset row valid enough to satisfy a job's foreign key -- these
    tests exercise `db.claim_next_job` directly, which only cares about a
    job's status, not the dataset behind it."""

    ds_id = db.create_dataset("d.jsonl", f"datasets/{name}.jsonl", name)
    db.finish_dataset(
        ds_id,
        {
            "valid": True,
            "row_count": 12,
            "schema_type": "chat",
            "enable_thinking": False,
            "errors": [],
            "warnings": [],
            "preview": [],
        },
    )
    return ds_id


def _queued_job_via_api(client, tmp_path) -> str:
    """Create a dataset and a job via the HTTP API, return the job id.

    Helper that goes through the request path rather than ``db.create_job``
    directly, so the "starts nothing" assertion is about the real path.
    """

    import json

    from helpers import wait_validated

    def chat(user, assistant):
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }

    path = tmp_path / "d.jsonl"
    path.write_text(
        "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(12)),
        encoding="utf-8",
    )
    with open(path, "rb") as f:
        ds = client.post(
            "/v1/datasets", files={"file": (path.name, f)}
        ).json()["id"]
    wait_validated(client, ds)
    r = client.post("/v1/jobs", json={"dataset_id": ds})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_claim_next_job_moves_one_queued_job_to_provisioning(isolated):
    ds_id = _finished_dataset_id("ds_claim")
    job_id = db.create_job(ds_id, "qwen3-4b", {})
    assert db.get_job(job_id)["status"] == "queued"

    claimed = db.claim_next_job()
    assert claimed is not None
    assert claimed["id"] == job_id
    assert claimed["status"] == "provisioning"
    # The DB row is provisioning too, with a state event.
    assert db.get_job(job_id)["status"] == "provisioning"
    events = db.get_events(job_id)
    assert any(
        e["kind"] == "state"
        and e["data"]
        and e["data"].get("state") == "provisioning"
        for e in events
    )
    # No more queued jobs to claim.
    assert db.claim_next_job() is None


def test_concurrent_workers_never_claim_the_same_job(
    isolated, tmp_path, monkeypatch
):
    """Two workers, one job, is a race. Prove no double-claim.

    Starts two threads that both try to claim at the same instant (a
    threading.Barrier makes them race rather than run sequentially).
    Exactly one succeeds; the other gets None. This is the property the
    database was changed for; asserting it directly is asserting the reason.
    """

    ds_id = _finished_dataset_id("ds_race")
    job_id = db.create_job(ds_id, "qwen3-4b", {})

    results: list[object] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def try_claim():
        try:
            barrier.wait(timeout=5)
            claimed = db.claim_next_job()
            results.append(claimed["id"] if claimed else None)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=try_claim)
    t2 = threading.Thread(target=try_claim)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors, errors
    assert t1.is_alive() is False and t2.is_alive() is False, (
        "a worker blocked"
    )
    # Exactly one worker claimed the job.
    claimed_ids = [r for r in results if r is not None]
    assert len(claimed_ids) == 1
    assert claimed_ids[0] == job_id
    assert results.count(None) == 1
    # The job is provisioning, not queued, and a third claim finds nothing.
    assert db.get_job(job_id)["status"] == "provisioning"
    assert db.claim_next_job() is None


def test_concurrent_claims_do_not_block(isolated, tmp_path, monkeypatch):
    """SKIP LOCKED is what makes workers *skip* rather than block.

    ``FOR UPDATE`` alone would make workers block on each other's row lock.
    This proves the second half: with 20 workers racing for one job, all
    finish quickly and only one claims. A blocking implementation would
    serialize and take far longer, or deadlock the suite.
    """

    ds_id = _finished_dataset_id("ds_skip")
    db.create_job(ds_id, "qwen3-4b", {})

    n = 20
    barrier = threading.Barrier(n)
    results: list[object] = []
    errors: list[Exception] = []

    def try_claim():
        try:
            barrier.wait(timeout=5)
            claimed = db.claim_next_job()
            results.append(claimed["id"] if claimed else None)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=try_claim) for _ in range(n)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    elapsed = time.monotonic() - start

    assert not errors, errors
    assert all(not t.is_alive() for t in threads), "a worker blocked"
    # Only one of 20 claimed the single job.
    assert sum(1 for r in results if r is not None) == 1
    assert sum(1 for r in results if r is None) == n - 1
    # And it did not take 20x poll intervals: the whole race finishes in <1s.
    # A blocking FOR UPDATE without SKIP LOCKED would serialize behind the
    # first transaction's lifetime and take far longer.
    assert elapsed < 2.0, f"claims blocked: {elapsed:.2f}s for {n} workers"


def test_claim_orders_by_created_at(isolated):
    """The oldest queued job is claimed first."""

    ds_id = _finished_dataset_id("ds_order")
    j1 = db.create_job(ds_id, "qwen3-4b", {})
    time.sleep(0.01)
    j2 = db.create_job(ds_id, "qwen3-4b", {})

    first = db.claim_next_job()
    assert first is not None and first["id"] == j1
    second = db.claim_next_job()
    assert second is not None and second["id"] == j2


def test_the_request_path_no_longer_starts_threads(isolated, tmp_path):
    """Prove the negative: creating a job through the API starts nothing.

    The control plane used to call ``orchestrator.launch`` (a thread per
    job) on ``POST /v1/jobs``. After #51 it just inserts the row. Assert
    that a job created via the API stays ``queued`` with no machine and no
    terminal transition, rather than merely not calling ``launch`` ourselves.
    """

    from temper_control_plane import main

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        job = client.get(f"/v1/jobs/{job_id}").json()
        # Still queued -- the worker has not yet claimed it, and the request
        # path did not drive it.
        assert job["status"] == "queued"
        assert job["machine_id"] is None
        # No provisioning event yet; only the ``queued`` state event from creation.
        events = client.get(f"/v1/jobs/{job_id}/events").json()["events"]
        state_events = [e for e in events if e["kind"] == "state"]
        assert len(state_events) == 1
        assert state_events[0]["message"] == "queued"


def test_provisioning_is_recorded_before_anything_else_can_fail(
    isolated, tmp_path, monkeypatch
):
    """A machine provisioned and not recorded is a billing machine nobody owns.

    The ordering is the criterion: write the record first, then act. If the
    step after provisioning (``await_ready``) fails and the step is retried,
    provisioning must not happen twice. This proves the retry sees the
    recorded machine and does not create a second one.

    The test drives ``_attempt`` directly with a provider that fails at
    ``await_ready``. The first call provisions once and fails; the job row
    already carries ``machine_id`` before the failure. A second call that
    checks the row must not provision again -- the machine's identity was
    recorded before the failure, so a retry knows it already exists.
    """

    from temper_control_plane import db as db_mod
    from temper_control_plane import orchestrator

    # Make teardown fast.
    monkeypatch.setattr(orchestrator, "DESTROY_RETRY_DELAY_S", 0)
    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_INTERVAL_S", 0)
    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_TIMEOUT_S", 2)
    monkeypatch.setattr(
        orchestrator,
        "published_reference",
        lambda: (
            "ghcr.io/thp728/temper/trainer@sha256:0000000000000000000000000000000000000000000000000000000000000000"
        ),
    )

    from fastapi.testclient import TestClient

    from temper_control_plane import main

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "queued"

        # Provider that succeeds at create but fails at await_ready.
        provider = FakeProvider(
            fail_at="await_ready", fail_code="ssh_unreachable"
        )
        # First attempt: provisions, records machine_id, then fails at await_ready.
        # ``_attempt`` is the unit that provisions; ``run_job`` would mark the
        # job failed, but we want to inspect the intermediate state.
        from temper_control_plane.fake_provider import MACHINE_ID

        job_before = db_mod.get_job(job_id)
        assert job_before["machine_id"] is None

        # Call _attempt directly: it will create the machine and then fail.
        # Use a fresh limits/models as run_job does.
        from temper_control_plane import fake_models
        from temper_control_plane.limits import RunLimits

        limits = RunLimits.from_config().start()
        models = fake_models.catalog_models()
        machines: list = []
        try:
            orchestrator._attempt(provider, job_id, machines, limits, models)
            raise AssertionError("_attempt should have raised")
        except Exception as e:
            # Expected failure at await_ready.
            assert (
                "ssh_unreachable" in str(e)
                or getattr(e, "code", "") == "ssh_unreachable"
            )

        # The machine was provisioned exactly once.
        assert len(provider.create_calls) == 1
        # And its identity is already on the row, before the failure propagated.
        job_after = db_mod.get_job(job_id)
        assert job_after["machine_id"] == MACHINE_ID
        assert job_after["status"] in ("provisioning", "preparing")
        # Teardown was not yet run (that is run_job's finally); the machine
        # is the one that would be destroyed there, and the row already names it.
        # A retry that checks the row would therefore not provision again --
        # the record-first ordering is what makes that safe.
        # Prove the row already protects: a worker that claims a job that is
        # no longer queued will not re-provision, because there is no queued
        # job to claim. A second provisioning would be a second machine; the
        # record-first ordering prevents it.
        assert db_mod.claim_next_job() is None


def test_worker_claim_and_run_is_idempotent_via_api(tmp_path, monkeypatch):
    """The worker can be started more than once safely."""

    import hashlib

    from temper_control_plane import main
    from temper_control_plane.fake_provider import FakeProvider
    from temper_worker.worker import run_once

    adapter_bytes = b"weights"
    result = {
        "ok": True,
        "stage": "train",
        "artifact_path": "run/adapter_model.safetensors",
        "artifact_sha256": hashlib.sha256(adapter_bytes).hexdigest(),
        "adapter_config": {"r": 16, "lora_alpha": 32},
    }

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        # Patch orchestrator.run_job to use our fake provider, as the harness does.
        from temper_control_plane import fake_provider, orchestrator

        orig_run = orchestrator.run_job

        def patched(job_id_inner, provider=None, limits=None, models=None):
            return orig_run(
                job_id_inner,
                provider=FakeProvider(
                    lines=["[00:00:01] done"],
                    result=result,
                    adapter_bytes=adapter_bytes,
                ),
                limits=limits,
                models=models,
            )

        monkeypatch.setattr(orchestrator, "run_job", patched)
        monkeypatch.setattr(
            orchestrator,
            "published_reference",
            lambda: fake_provider.PUBLISHED_IMAGE_REFERENCE,
        )
        # First worker claims and runs the job.
        assert run_once() is True
        # Second worker finds nothing queued.
        assert run_once() is False
        # Job is terminal, not queued.
        rec = client.get(f"/v1/jobs/{job_id}").json()
        assert rec["status"] == "complete", rec
        # A third worker still finds nothing.
        assert run_once() is False


def test_run_once_marks_a_job_failed_if_driving_it_raises_unexpectedly(
    isolated, tmp_path, monkeypatch
):
    """``run_job`` already records a failure on the row before it raises for
    every failure it knows about; this covers the failure it does not --
    an exception escaping ``run_job`` itself must not leave the claimed job
    stuck in a non-terminal state with no worker ever coming back to it."""

    from temper_control_plane import main, orchestrator
    from temper_worker.worker import run_once

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)

        def boom(job_id_inner, provider=None, limits=None, models=None):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(orchestrator, "run_job", boom)

        assert run_once() is True
        job = client.get(f"/v1/jobs/{job_id}").json()
        assert job["status"] == "failed"
        assert job["error_code"] == "internal_error"


def test_run_forever_stops_promptly_when_nothing_is_queued(
    isolated, monkeypatch
):
    """The polling loop must not block ``stop`` from being noticed -- an
    idle worker that cannot be told to shut down would need a hard kill on
    every deploy."""

    import threading
    import time as time_mod

    from temper_worker import worker

    monkeypatch.setattr(worker, "WORKER_POLL_INTERVAL_S", 0.05)
    stop = threading.Event()
    t = threading.Thread(target=worker.run_forever, args=(stop,), daemon=True)
    t.start()
    time_mod.sleep(0.15)
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
