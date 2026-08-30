"""Durable execution: a job survives the process driving it (issue #68).

Spec 010's answer to "what happens when your control plane dies while my
machine is billing": the job sequence is driven durably. The worker claims a
job under a lease (`claimed_at`); while it drives, a heartbeat refreshes the
lease; if the worker dies, the lease goes stale and a fresh worker reclaims
the job and resumes it from its recorded step rather than from the beginning.

The recoveries are proven on the double, the way the other fault-surface
issues proved theirs. An abandoned job in `training` has its recorded machine
torn down before anything new provisions -- never a second machine -- and
resumes from the checkpoint that had already left the machine (issue #60,
reused rather than duplicated). An abandoned job in `packaging` is
re-collected from storage and finishes without re-running anything. An
abandoned job in `provisioning` re-runs the attempt. What the double cannot
prove -- a worker killed on real hardware mid-training, the job completing
anyway -- is stated in the ADR and the PR body as unexercised, in those
words, rather than claimed.

The reconciler relationship is asserted rather than argued: a job this path
is actively recovering is still non-terminal, so the reconciler's ownership
query (every non-terminal state) leaves its machine alone, and this path is
what carries the job to terminal. The two address different failures.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from temper_control_plane import db
from temper_control_plane.fake_provider import (
    MACHINE_ID,
    FakeProvider,
)
from temper_core import resume as resume_logic

pytestmark = pytest.mark.usefixtures("isolated")

ADAPTER_BYTES = b"weights"
RESULT = {
    "ok": True,
    "stage": "train",
    "artifact_path": "run/adapter_model.safetensors",
    "artifact_sha256": hashlib.sha256(ADAPTER_BYTES).hexdigest(),
    "adapter_config": {"r": 16, "lora_alpha": 32},
}


def chat(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _queued_job_via_api(client, tmp_path) -> str:
    """Create a dataset and a queued job through the real request path."""
    from helpers import wait_validated

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


def _age_claim(job_id: str, seconds: float | None = None) -> None:
    """Push a job's claim lease into the past so it looks abandoned."""
    if seconds is None:
        seconds = db.CLAIM_STALE_AFTER_S + 1
    with db.connect() as c:
        c.execute(
            "UPDATE jobs SET claimed_at=%s WHERE id=%s",
            (time.time() - seconds, job_id),
        )


def _claim_and_set_training(job_id: str) -> None:
    """Claim a queued job and move it to `training` with a recorded machine,
    exactly as a live worker would have by that point in `_attempt`."""
    claimed = db.claim_next_job()
    assert claimed is not None and claimed["id"] == job_id
    db.set_state(
        job_id,
        "preparing",
        f"Machine {MACHINE_ID} running; waiting for SSH",
        machine_id=MACHINE_ID,
    )
    db.set_state(job_id, "training", "Pulling image and training")


def _set_training_without_claim(job_id: str) -> None:
    """Move a queued job to `training` with a recorded machine without going
    through the worker's claim -- the shape a direct driver (tests, the
    boot-time seed) leaves: non-terminal, machine recorded, lease never
    taken."""
    db.set_state(
        job_id,
        "preparing",
        f"Machine {MACHINE_ID} running; waiting for SSH",
        machine_id=MACHINE_ID,
    )
    db.set_state(job_id, "training", "Pulling image and training")


def _abandon_after_training_checkpoint(job_id: str) -> None:
    """Leave a job in the state a worker killed mid-training leaves behind:
    claimed (lease taken), machine recorded, status `training`, and one
    checkpoint already off the machine in storage (issue #37 writes them off
    as training produces them). The machine's own write is reproduced through
    the same scoped-grant redeem the fake machine uses, so discovery reads a
    real stored checkpoint."""
    from temper_control_plane import storage
    from temper_control_plane.fake_provider import fake_checkpoint_tar

    _claim_and_set_training(job_id)
    key = storage.checkpoint_key(job_id, 0)
    grant = storage.STORE.mint_write_grant(key, 100.0)
    storage.STORE.redeem(
        grant,
        fake_checkpoint_tar(30, b"ckpt-30", loss=0.31, held_out_loss=0.52),
    )


@pytest.fixture()
def fast_teardown(monkeypatch):
    from temper_control_plane import orchestrator

    monkeypatch.setattr(orchestrator, "DESTROY_RETRY_DELAY_S", 0)
    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_INTERVAL_S", 0)
    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_TIMEOUT_S", 2)
    monkeypatch.setattr(
        orchestrator,
        "published_reference",
        lambda: (
            "ghcr.io/thp728/temper/trainer@sha256:"
            "0000000000000000000000000000000000000000000000000000000000000000"
        ),
    )


def _patched_run_with(orig_run, provider):
    """A wrapper around `orchestrator.run_job` that injects a fake provider
    and fake model facts -- the same shape the worker's real call would take
    against the real account, with the money-spending half replaced."""
    from temper_control_plane import fake_models

    def patched(job_id_inner, _provider=None, limits=None, models=None):
        return orig_run(
            job_id_inner,
            provider=provider,
            limits=limits,
            models=models or fake_models.catalog_models(),
        )

    return patched


# --- the claim lease: what is reclaimable and what is not -------------------


def test_a_stale_job_is_reclaimed_from_its_recorded_step(
    tmp_path, monkeypatch
):
    """A non-terminal job whose lease has gone stale -- the worker driving it
    died -- is claimed again by a fresh worker, with its status preserved (the
    status is the step cursor) and a reclaim recorded in its own history."""
    from temper_control_plane import main

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        assert (
            db.claim_next_job() is not None
        )  # claimed: queued -> provisioning
        _age_claim(job_id)

        reclaimed = db.claim_next_job()
        assert reclaimed is not None and reclaimed["id"] == job_id
        # The step cursor is preserved, not reset to queued.
        assert reclaimed["status"] == "provisioning"
        # A second worker finds nothing to claim.
        assert db.claim_next_job() is None
        # The lease is fresh again, so the reclaimed job is not instantly
        # reclaimable a second time.
        _age_claim(job_id)
        assert db.claim_next_job() is not None
        # The reclaim is visible in the job's own history.
        events = db.get_events(job_id)
        assert any(
            e["kind"] == "log"
            and e.get("data")
            and e["data"].get("code") == db.RECLAIMED_CODE
            for e in events
        )


def test_a_live_job_is_never_reclaimed(tmp_path, monkeypatch):
    """The whole point of the lease: a job a live worker is driving (fresh
    lease) is not reclaimable, and neither is a non-terminal job nobody ever
    claimed (NULL lease). Reclaiming either would double-drive it."""
    from temper_control_plane import main

    with TestClient(main.app) as client:
        _queued_job_via_api(client, tmp_path)  # the live job, claimed below
        direct_id = _queued_job_via_api(client, tmp_path)
        # A non-terminal job whose lease was never taken (a direct driver) is
        # not treated as abandoned.
        _set_training_without_claim(direct_id)
        assert db.get_job(direct_id)["claimed_at"] is None
        # A freshly claimed job: the worker is driving it.
        assert db.claim_next_job() is not None  # claims the live job
        assert db.claim_next_job() is None, "a live job was reclaimed"


def test_reclaiming_is_as_exclusive_as_claiming(tmp_path, monkeypatch):
    """Two workers racing to reclaim the same stale job: exactly one wins.
    The reclaim uses the same FOR UPDATE SKIP LOCKED as the fresh claim, so a
    stale job is never handed to two drivers."""
    from temper_control_plane import main

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        assert db.claim_next_job() is not None
        _age_claim(job_id)

        results: list[object] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def try_reclaim():
            try:
                barrier.wait(timeout=5)
                claimed = db.claim_next_job()
                results.append(claimed["id"] if claimed else None)
            except Exception as e:  # pragma: no cover - a race failure is loud
                errors.append(e)

        threads = [threading.Thread(target=try_reclaim) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, errors
        assert sum(1 for r in results if r == job_id) == 1
        assert results.count(None) == 1


# --- a worker killed during a live run resumes and completes -----------------


def test_a_worker_killed_during_training_resumes_and_completes(
    tmp_path, fast_teardown, monkeypatch
):
    """The headline: the worker is killed while a live job is mid-training.
    The job is reclaimed, its recorded machine destroyed before anything new
    provisions (no second machine), and it resumes from the checkpoint that
    had already left the machine -- reusing the issue #60 resumption rather
    than duplicating it -- and completes with its record and artifact intact."""
    from temper_control_plane import main, orchestrator
    from temper_worker.worker import run_once

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        # The worker drove the job to mid-training, wrote its checkpoint off
        # the machine, and then died; only the lease and the step cursor tell
        # a fresh worker it was abandoned.
        _abandon_after_training_checkpoint(job_id)
        mid = db.get_job(job_id)
        assert mid["status"] == "training"
        assert mid["machine_id"] == MACHINE_ID
        assert mid["claimed_at"] is not None
        _age_claim(job_id)

    # A fresh worker reclaims and drives the job to completion.
    completion = FakeProvider(
        lines=[
            "[00:00:01] pulling trainer image",
            "[00:00:02] running training",
            "{'loss': 0.31, 'epoch': 0.5}",
        ],
        result=RESULT,
        adapter_bytes=ADAPTER_BYTES,
        checkpoints=[
            {
                "step": 30,
                "loss": 0.31,
                "held_out_loss": 0.52,
                "bytes": b"ckpt-30",
            }
        ],
    )
    orig_run = orchestrator.run_job
    monkeypatch.setattr(
        orchestrator, "run_job", _patched_run_with(orig_run, completion)
    )
    assert run_once() is True
    assert run_once() is False, "nothing left to claim"

    with TestClient(main.app) as client:
        job = client.get(f"/v1/jobs/{job_id}").json()
        assert job["status"] == "complete", job

    # The history says what actually happened: the first attempt interrupted,
    # the second a resumption that completed from step 30.
    record = db.get_job(job_id)
    attempts = record["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["outcome"] == "interrupted"
    assert attempts[0]["error_code"] == resume_logic.INTERRUPTED_CODE
    assert attempts[1]["outcome"] == "complete"
    assert attempts[1]["resumed_from"] == 30
    # The job's record and artifact are intact.
    assert record["result"] is not None
    assert record["artifact_key"] is not None
    assert record["artifact_record"] is not None
    # The recovery itself is named in the history: the reclaim event (from the
    # claim) and the orchestrator's own recovery event both carry their codes.
    events = db.get_events(job_id)
    assert any(
        e.get("data", {}) or {} and e["data"].get("code") == db.RECLAIMED_CODE
        for e in events
    ), "the claim's reclaim event is missing"
    from temper_control_plane import orchestrator as orchestrator_mod

    assert any(
        (e.get("data") or {}).get("code")
        == orchestrator_mod.RECLAIM_RECOVERY_CODE
        for e in events
    ), "the orchestrator's recovery event is missing"


def test_the_recorded_machine_is_destroyed_before_anything_new_provisions(
    tmp_path, fast_teardown, monkeypatch
):
    """Recovery never stacks machines: the abandoned job's recorded machine is
    destroyed -- through the same confirmed teardown a normal attempt uses --
    before the resumption provisions a fresh one. The completion worker's fake
    listing still shows the recorded machine (it is the same account), and the
    teardown is ordered before the fresh provision."""
    from temper_control_plane import main, orchestrator
    from temper_worker.worker import run_once

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        _abandon_after_training_checkpoint(job_id)
        _age_claim(job_id)

    completion = FakeProvider(
        lines=["[00:00:01] running training"],
        result=RESULT,
        adapter_bytes=ADAPTER_BYTES,
        checkpoints=[{"step": 30, "loss": 0.31, "bytes": b"ckpt-30"}],
    )
    orig_run = orchestrator.run_job
    monkeypatch.setattr(
        orchestrator, "run_job", _patched_run_with(orig_run, completion)
    )
    assert run_once() is True

    # The recorded machine was destroyed before the resumption provisioned:
    # the destroy call precedes the create call in the fresh worker's drive,
    # and only one machine was ever provisioned by it.
    assert completion.destroyed, (
        "the abandoned job's machine was not destroyed"
    )
    assert completion.calls.index("destroy") < completion.calls.index("create")
    assert len(completion.created) == 1, (
        "recovery provisioned a second machine"
    )
    assert db.get_job(job_id)["status"] == "complete"


def test_a_worker_killed_during_preparing_restarts_and_completes(
    tmp_path, fast_teardown, monkeypatch
):
    """A job killed while its machine was still being set up (status
    `preparing`) has nothing to resume from -- training never started -- so the
    recovery restarts the attempt rather than failing a run that never began.
    The interrupted preparing attempt is still recorded with its machine."""
    from temper_control_plane import main, orchestrator
    from temper_worker.worker import run_once

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        _claim_and_set_training(job_id)
        # Back up: the worker died during preparing, before training started.
        db.set_state(job_id, "preparing", "Waiting for SSH")
        assert db.get_job(job_id)["machine_id"] == MACHINE_ID
        _age_claim(job_id)

    completion = FakeProvider(
        lines=["[00:00:01] running training"],
        result=RESULT,
        adapter_bytes=ADAPTER_BYTES,
    )
    orig_run = orchestrator.run_job
    monkeypatch.setattr(
        orchestrator, "run_job", _patched_run_with(orig_run, completion)
    )
    assert run_once() is True

    record = db.get_job(job_id)
    assert record["status"] == "complete", record
    attempts = record["attempts"]
    # Attempt 1: interrupted before training, on the recorded machine.
    # Attempt 2: a fresh attempt that completed (no `resumed_from` -- nothing
    # was trained to resume from, so this is a restart, not a resumption).
    assert len(attempts) == 2
    assert attempts[0]["outcome"] == "interrupted"
    assert attempts[0]["machine_id"] == MACHINE_ID
    assert attempts[1]["outcome"] == "complete"
    assert attempts[1].get("resumed_from") is None
    # The recorded machine was destroyed before the fresh attempt provisioned.
    assert completion.destroyed


def test_a_reclaimed_provisioning_job_reprovisions_and_completes(
    tmp_path, fast_teardown, monkeypatch
):
    """A job killed mid-provisioning (status `provisioning`, no machine
    recorded) resumes by running the attempt again: nothing was created yet,
    so the recovery provisions exactly one fresh machine and completes."""
    from temper_control_plane import main, orchestrator
    from temper_worker.worker import run_once

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        # The worker died right after claiming: status `provisioning`, no
        # machine recorded, lease taken.
        assert db.claim_next_job() is not None
        assert db.get_job(job_id)["machine_id"] is None
        _age_claim(job_id)

    completion = FakeProvider(
        lines=["[00:00:01] running training"],
        result=RESULT,
        adapter_bytes=ADAPTER_BYTES,
    )
    orig_run = orchestrator.run_job
    monkeypatch.setattr(
        orchestrator, "run_job", _patched_run_with(orig_run, completion)
    )
    assert run_once() is True

    record = db.get_job(job_id)
    assert record["status"] == "complete", record
    # Exactly one machine was provisioned by the recovery: the provisioning
    # step ran once. The invariant that makes this safe -- a `provisioning`
    # status never carries a recorded machine -- is what the reclaim relies
    # on, and is asserted here directly.
    assert len(completion.created) == 1
    assert record["attempts"][-1]["outcome"] == "complete"


# --- a worker killed during packaging is re-collected, not re-run --------------


def test_a_worker_killed_during_packaging_is_recollected_not_rerun(
    tmp_path, fast_teardown, monkeypatch
):
    """A job whose run finished training but whose worker died during
    collection resumes at packaging: the stored artifact is re-verified and
    the job completes without provisioning a single machine -- nothing is
    re-run, only re-collected."""
    from temper_control_plane import main, orchestrator, storage
    from temper_control_plane.fake_provider import MACHINE_ID as MID
    from temper_worker.worker import run_once

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        _claim_and_set_training(job_id)
        # The machine wrote its artifact to storage before reporting, as the
        # trainer does (ADR-0009); only the collection remains.
        weights_key = storage.artifact_key(
            job_id, storage.ADAPTER_WEIGHTS_NAME
        )
        grant = storage.STORE.mint_write_grant(weights_key, 100.0)
        storage.STORE.redeem(grant, ADAPTER_BYTES)
        # The result manifest was persisted when packaging began (issue #68),
        # exactly as the orchestrator now writes it.
        db.set_state(
            job_id, "packaging", "Verifying artifact", result_json=RESULT
        )
        _age_claim(job_id)

    completion = FakeProvider(result=RESULT, adapter_bytes=ADAPTER_BYTES)
    orig_run = orchestrator.run_job
    monkeypatch.setattr(
        orchestrator, "run_job", _patched_run_with(orig_run, completion)
    )
    assert run_once() is True

    record = db.get_job(job_id)
    assert record["status"] == "complete"
    # No machine was provisioned: the finished run was re-collected, not re-run.
    assert completion.created == []
    assert record["artifact_key"] is not None
    # The single attempt that did the training is recorded as complete.
    assert len(record["attempts"]) == 1
    assert record["attempts"][0]["outcome"] == "complete"
    assert record["attempts"][0]["machine_id"] == MID


# --- the worker is a separate process: the control plane can restart ----------


def test_a_worker_survives_the_control_plane_restarting(
    tmp_path, fast_teardown, monkeypatch
):
    """The worker is a separate process that polls the database; a control
    plane restart must not stop it. Create a job, take the control plane down,
    let the worker drive it to completion, and restart the control plane to
    read the finished record."""
    from temper_control_plane import main, orchestrator
    from temper_worker.worker import run_once

    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
    # The control plane process is gone. The worker, untouched, drives the job.
    completion = FakeProvider(
        lines=["[00:00:01] running training"],
        result=RESULT,
        adapter_bytes=ADAPTER_BYTES,
    )
    orig_run = orchestrator.run_job
    monkeypatch.setattr(
        orchestrator, "run_job", _patched_run_with(orig_run, completion)
    )
    assert run_once() is True
    # The control plane comes back and the finished job is there, intact.
    with TestClient(main.app) as client:
        job = client.get(f"/v1/jobs/{job_id}").json()
        assert job["status"] == "complete", job


# --- the heartbeat: a driving worker never looks abandoned -------------------


def test_the_heartbeat_refreshes_the_lease_while_a_worker_drives(
    tmp_path, fast_teardown, monkeypatch
):
    """The lease that makes reclaim safe is refreshed by the worker's
    heartbeat while it drives, so a live driver -- even one quiet for longer
    than the staleness window -- is never mistaken for a dead one."""
    from temper_control_plane import main
    from temper_worker import worker as worker_mod

    monkeypatch.setattr(worker_mod, "HEARTBEAT_INTERVAL_S", 0.1)
    with TestClient(main.app) as client:
        job_id = _queued_job_via_api(client, tmp_path)
        assert db.claim_next_job() is not None
        first = db.get_job(job_id)["claimed_at"]
        assert first is not None

        stop = threading.Event()
        t = threading.Thread(
            target=worker_mod._heartbeat_loop,
            args=(stop, job_id),
            daemon=True,
            name="heartbeat-test",
        )
        t.start()
        try:
            # The heartbeat touches well within the staleness window.
            deadline = time.time() + 5
            while time.time() < deadline:
                if db.get_job(job_id)["claimed_at"] != first:
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("heartbeat never refreshed the lease")
            assert db.get_job(job_id)["claimed_at"] > first
        finally:
            stop.set()
            t.join(timeout=1.0)


def test_heartbeat_interval_is_safely_below_the_staleness_window():
    """The invariant that keeps live jobs from being stolen: the heartbeat
    refreshes the lease far more often than the reclaim window. A future edit
    that inverts this would let a live driver look abandoned and be stolen."""
    from temper_worker import worker as worker_mod

    assert worker_mod.HEARTBEAT_INTERVAL_S < db.CLAIM_STALE_AFTER_S
