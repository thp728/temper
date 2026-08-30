"""The machine-lifetime reconciler (issue #61 / Spec 010).

The reconciler lists machines, matches them against jobs that are not in a
terminal state and against served endpoints, destroys anything it cannot
account for, records what it did, and marks a job whose machine was destroyed
as unowned failed with a reason. These tests prove the behaviours the issue
names on the zero-cost provider:

* an unowned machine is destroyed, and the destruction is recorded;
* a machine owned by a live (non-terminal) job is left alone -- including a
  machine whose owner is only `provisioning`/`preparing`, which
  `claim_next_job` would never match (the "must be matched by you" gap);
* a served endpoint's machine (ADR-0065) is left alone, because an endpoint
  is not a job;
* a job that still claims a machine the reconciler destroyed as unowned is
  marked failed with a reason rather than left running forever;
* a machine reported as `destroying` is not issued a second destroy
  (ADR-0057);
* the `orphan` fault leaves a machine with no owner listed, and the
  reconciler finds and destroys it.

What the double proves is the orchestration logic: that an unowned machine is
matched, destroyed through the same confirmed path, and recorded. What it
cannot prove is the recovery on real hardware -- a machine deliberately
orphaned on a live provider and destroyed there -- which is the adversarial
tier's fixture and is stated as such in the PR.
"""

from __future__ import annotations

import time

import pytest

from temper_control_plane import db
from temper_control_plane.fake_provider import (
    MACHINE_ID,
    ORPHAN_MACHINE_ID,
    PUBLISHED_IMAGE_REFERENCE,
    FakeProvider,
    completed_run,
)
from temper_worker.reconciler import reconcile_once

pytestmark = pytest.mark.usefixtures("isolated")


@pytest.fixture()
def fast_teardown(monkeypatch):
    """Make the confirmed-destroy loop fast, as the teardown tests do."""
    from temper_control_plane import orchestrator

    monkeypatch.setattr(orchestrator, "DESTROY_RETRY_DELAY_S", 0)
    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_INTERVAL_S", 0)
    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_TIMEOUT_S", 2)


def _finished_dataset(name: str) -> str:
    """A dataset row and object valid enough to satisfy a job's foreign key
    and the orchestrator's dataset-streaming leg (which reads the object
    from storage by key)."""
    from temper_control_plane import storage

    object_key = f"datasets/{name}.jsonl"
    storage.STORE.put(
        object_key,
        "\n".join('{"messages":[]}' for _ in range(12)).encode("utf-8"),
    )
    ds_id = db.create_dataset("d.jsonl", object_key, name)
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


def _non_terminal_job(machine_id: int, state: str = "training") -> str:
    """A job in a non-terminal state that claims `machine_id`, with no driver."""
    ds_id = _finished_dataset("ds_recon")
    job_id = db.create_job(ds_id, "qwen3-4b", {})
    db.set_state(job_id, state, f"in {state}", machine_id=machine_id)
    return job_id


def _complete_job_with_endpoint(machine_id: int) -> tuple[str, str]:
    """A complete job with an active endpoint serving `machine_id`."""
    ds_id = _finished_dataset("ds_endpoint")
    job_id = db.create_job(ds_id, "qwen3-4b", {})
    db.set_state(
        job_id,
        "complete",
        "done",
        method="qlora",
        gpu_type="L4",
        price_per_hour=41.31,
        currency="INR",
    )
    ep_id = "ep_recon_1"
    now = time.time()
    db.create_endpoint(
        ep_id,
        job_id,
        "some_hash",
        "pref",
        now,
        now + 100,
        now + 1000,
        machine_id=machine_id,
        price_per_hour=41.31,
        currency="INR",
    )
    return job_id, ep_id


def test_a_machine_with_no_job_that_owns_it_is_destroyed(fast_teardown):
    """The primary criterion: a listed machine no live job or endpoint owns
    is destroyed, and the destruction is recorded so it is visible rather
    than silently cleaned."""
    from temper_worker.reconciler import reconcile_once

    # The listing goes absent after the destroy so confirmation succeeds.
    provider = FakeProvider(list_sequence=[[7777], [], [], []])
    report = reconcile_once(provider)

    assert report["destroyed"] == [7777]
    assert provider.destroy_attempts == 1
    # Recorded: one row naming the machine, the action and why.
    recs = db.list_reconciliations()
    assert any(
        r["machine_id"] == 7777
        and r["action"] == "destroyed"
        and r["status"] == "running"
        for r in recs
    )


def test_a_machine_owned_by_a_live_job_is_left_alone(fast_teardown):
    job_id = _non_terminal_job(MACHINE_ID, state="training")
    provider = FakeProvider()  # lists [MACHINE_ID]
    report = reconcile_once(provider)

    assert report["owned"] == 1
    assert report["destroyed"] == []
    assert provider.destroy_attempts == 0
    assert db.get_job(job_id)["status"] == "training"
    # An owned machine is the normal case, not an event: nothing recorded.
    assert db.list_reconciliations() == []


def test_a_machine_owned_by_a_provisioning_or_preparing_job_is_matched(
    fast_teardown,
):
    """`claim_next_job` only looks at queued rows; a machine owned by a
    provisioning/preparing job must be matched here or this pass would
    destroy a machine a worker is actively driving."""
    job_id = _non_terminal_job(MACHINE_ID, state="preparing")
    provider = FakeProvider()
    report = reconcile_once(provider)

    assert report["owned"] == 1
    assert report["destroyed"] == []
    assert provider.destroy_attempts == 0
    assert db.get_job(job_id)["status"] == "preparing"


def test_a_served_endpoints_machine_is_left_alone(fast_teardown):
    """ADR-0065: a served endpoint is a billed, warm machine, not a job; the
    reconciler must not destroy it while a user is serving."""
    _job_id, _ep_id = _complete_job_with_endpoint(machine_id=9001)
    provider = FakeProvider(list_sequence=[[9001]])
    report = reconcile_once(provider)

    assert report["owned"] == 1
    assert report["destroyed"] == []
    assert provider.destroy_attempts == 0


def test_a_job_that_still_claims_a_destroyed_machine_is_marked_failed(
    fast_teardown, monkeypatch
):
    """The "rather than left running forever" clause: a non-terminal job whose
    machine the reconciler destroys as unowned is marked failed with a reason.

    The branch defends the record-first race -- a worker records its
    machine_id onto a live row in the same instant the reconciler is acting on
    the unowned machine. The test simulates the stale ownership snapshot: the
    reconciler's read of non-terminal machines predates the machine_id write,
    so it sees no owner and acts, and the owner lookup afterwards finds the
    live row and fails it rather than leaving it non-terminal forever.
    """
    from temper_worker.reconciler import (
        ORPHANED_MACHINE_CODE,
        reconcile_once,
    )

    job_id = _non_terminal_job(MACHINE_ID, state="training")
    # The reconciler's ownership snapshot is stale (the record-first race).
    monkeypatch.setattr(db, "list_non_terminal_machine_ids", lambda: [])
    provider = FakeProvider()  # lists [MACHINE_ID]
    report = reconcile_once(provider)

    assert report["destroyed"] == [MACHINE_ID]
    assert job_id in report["marked_failed"]
    job = db.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == ORPHANED_MACHINE_CODE
    assert "destroyed by the reconciler" in str(job["error_message"])
    # The failure is named in the job's own history.
    msgs = [e["message"] for e in db.get_events(job_id)]
    assert any(
        "destroyed by the reconciler" in m and "cannot continue" in m
        for m in msgs
    )


def test_a_terminal_jobs_leaked_machine_is_destroyed_and_recorded(
    fast_teardown,
):
    """A machine that outlives its terminal job (a teardown that failed) is an
    orphan -- no live job owns it -- and is destroyed; the destruction is
    recorded on the job's history and the job is not re-failed."""
    ds_id = _finished_dataset("ds_leak")
    job_id = db.create_job(ds_id, "qwen3-4b", {})
    db.set_state(job_id, "complete", "done", machine_id=MACHINE_ID)
    provider = FakeProvider(list_sequence=[[MACHINE_ID], [], [], []])
    report = reconcile_once(provider)

    assert report["destroyed"] == [MACHINE_ID]
    # A complete job is not re-failed -- the row is already terminal.
    assert db.get_job(job_id)["status"] == "complete"
    msgs = [e["message"] for e in db.get_events(job_id)]
    assert any("Machine" in m and "destroyed" in m for m in msgs)


def test_a_machine_reported_as_destroying_is_not_double_destroyed(
    fast_teardown,
):
    """ADR-0057: destroying is present, not yet confirmed. A second destroy
    against a machine already being destroyed is the mistake the rule exists
    to stop, so the reconciler skips it -- and records the skip when it has no
    owner, so it is visible rather than silently dropped."""
    provider = FakeProvider(list_sequence=[[(8888, "destroying")]])
    report = reconcile_once(provider)

    assert report["destroying"] == 1
    assert report["destroyed"] == []
    assert provider.destroy_attempts == 0
    recs = db.list_reconciliations()
    assert any(
        r["machine_id"] == 8888 and r["action"] == "skipped_destroying"
        for r in recs
    )


def test_reconcile_once_destroys_the_machine_the_orphan_fault_left(
    fast_teardown, monkeypatch
):
    """Issue #61's story on the double: the `orphan` fault leaves a machine
    with no job that owns it listed; the reconciler finds and destroys it,
    and the job's own machine is untouched."""
    from temper_control_plane import config, fake_models, orchestrator
    from temper_worker.reconciler import reconcile_once

    monkeypatch.setattr(config, "FAKE_PROVIDER", True)
    monkeypatch.setattr(
        orchestrator,
        "published_reference",
        lambda: PUBLISHED_IMAGE_REFERENCE,
    )
    ds_id = _finished_dataset("ds_orphan_fault")
    job_id = db.create_job(
        ds_id,
        "qwen3-4b",
        {"simulated_failure_code": {"name": "orphan"}},
    )
    provider = completed_run()
    orchestrator.run_job(
        job_id, provider=provider, models=fake_models.catalog_models()
    )
    assert db.get_job(job_id)["status"] == "complete"
    assert provider.fault_applied == "orphan"

    # Before reconciliation, the orphan is listed and has no owner; the job's
    # own machine was torn down normally.
    listed = provider.list_machines()
    assert ORPHAN_MACHINE_ID in [m.machine_id for m in listed]
    assert db.job_rows_for_machine(ORPHAN_MACHINE_ID) == []

    report = reconcile_once(provider)
    assert report["destroyed"] == [ORPHAN_MACHINE_ID]
    # The job's own machine was not touched again -- the fault's whole point
    # is that it is a distinct id from the job's, which was already torn down.
    recs = db.list_reconciliations()
    assert any(
        r["machine_id"] == ORPHAN_MACHINE_ID and r["action"] == "destroyed"
        for r in recs
    )
    assert db.get_job(job_id)["status"] == "complete"


def test_start_and_stop_reconciler_thread(monkeypatch):
    """The worker's scheduled pass starts on demand and stops promptly."""
    import threading
    import time as time_mod

    from temper_worker import reconciler

    monkeypatch.setattr(reconciler, "RECONCILE_INTERVAL_S", 0.05)
    reconciler.start_reconciler_thread()
    threads = [
        t
        for t in threading.enumerate()
        if t.name == "machine-reconciler" and t.is_alive()
    ]
    try:
        assert threads, "reconciler thread did not start"
        time_mod.sleep(0.2)
        assert threads[0].is_alive()
    finally:
        reconciler.stop_reconciler_thread()
    # The stop is observed promptly (Event.wait, not sleep); join so the
    # assertion is about the thread having really finished, not merely about
    # having been asked to stop.
    threads[0].join(timeout=2.0)
    assert not threads[0].is_alive()
