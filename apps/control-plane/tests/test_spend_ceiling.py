"""The spend ceiling: enforced from outside the training process (issue #46).

Spec 010's implementation decision is the load-bearing one here: a spend
ceiling must be enforced by something other than the process that is spending
-- a process that has stopped responding cannot enforce its own limit. The
ceiling in this ticket is measured by the control plane from wall clock and
the job's frozen price (`price_per_hour` x elapsed, the same derivation the
actuals use), never from anything the trainer reports, so it fires even
against a machine that produces no output at all. On reaching it the order is
checkpoint, terminate, destroy, and the job fails with the specific reason
`budget_exhausted`, distinct from a stall and from the duration ceiling.

The checkpoint half is issue #37's point: the machine writes checkpoints off
itself as it trains, so the control plane asks it to report them
(`request_checkpoint`), records what it can verify, and the recorded set feeds
the held-out-loss selection (ADR-0049) -- a checkpoint written at the ceiling
must be visible to that selection, and retrievable by the user, rather than
stored somewhere the selection cannot see.
"""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from helpers import wait_validated

from temper_control_plane.fake_provider import (
    MACHINE_ID,
    PUBLISHED_IMAGE_REFERENCE,
    FakeProvider,
    completed_run,
    fake_checkpoint_tar,
    simulated_limits,
)
from temper_control_plane.limits import RunLimits
from temper_core.selection import GpuAvailability


def chat(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


TRAINING_LINES = [
    "[10:00:01] pulling trainer image",
    "[10:03:04] image pulled in 183s",
    "[10:03:04] running training",
    "{'loss': 1.9042, 'grad_norm': 1.5, 'epoch': 0.5}",
]

RESULT = {
    "ok": True,
    "stage": "train",
    "artifact_path": "run/adapter_model.safetensors",
    "artifact_sha256": hashlib.sha256(b"weights").hexdigest(),
    "adapter_config": {"r": 16, "lora_alpha": 32},
}


class Harness:
    """Upload a dataset, launch a job on a given provider, read it back."""

    def __init__(self, client, monkeypatch, tmp_path):
        self._client = client
        self._monkeypatch = monkeypatch
        self._tmp_path = tmp_path

    def enable_fake_tier(self):
        from temper_control_plane import config

        self._monkeypatch.setattr(config, "FAKE_PROVIDER", True)

    def run(self, provider, hyperparameters=None, limits=None) -> str:
        from temper_control_plane import fake_models, orchestrator

        models = fake_models.catalog_models()
        # Issue #51: the request path no longer starts threads. Drive
        # the job directly as the worker would.
        job_id = self._create(hyperparameters)
        orchestrator.run_job(
            job_id, provider=provider, limits=limits, models=models
        )
        return job_id

    def _create(self, hyperparameters=None) -> str:
        path = self._tmp_path / "d.jsonl"
        path.write_text(
            "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(12)),
            encoding="utf-8",
        )
        with open(path, "rb") as f:
            ds = self._client.post(
                "/v1/datasets", files={"file": (path.name, f)}
            ).json()["id"]
        wait_validated(self._client, ds)
        r = self._client.post(
            "/v1/jobs",
            json={"dataset_id": ds, "hyperparameters": hyperparameters or {}},
        )
        assert r.status_code == 201, r.text
        return r.json()["id"]

    def job(self, job_id) -> dict:
        return self._client.get(f"/v1/jobs/{job_id}").json()

    def stored(self, job_id) -> dict:
        from temper_control_plane import db

        return db.get_job(job_id)

    def events(self, job_id) -> list[dict]:
        return self._client.get(f"/v1/jobs/{job_id}/events").json()["events"]

    def messages(self, job_id) -> list[str]:
        return [e["message"] for e in self.events(job_id)]


@pytest.fixture()
def harness(isolated, tmp_path, monkeypatch):
    from temper_control_plane import main, orchestrator

    monkeypatch.setattr(orchestrator, "DESTROY_RETRY_DELAY_S", 0)
    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_INTERVAL_S", 0)
    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_TIMEOUT_S", 2)
    monkeypatch.setattr(
        orchestrator,
        "published_reference",
        lambda: PUBLISHED_IMAGE_REFERENCE,
    )
    with TestClient(main.app) as c:
        yield Harness(c, monkeypatch, tmp_path)


def spend_limits(**kwargs) -> RunLimits:
    """Limits whose minutes pass in milliseconds, plus a spend ceiling.

    The price is the fake provider's L4 rate unless the test overrides it,
    matching what `_attempt` will freeze at provisioning."""
    kwargs.setdefault("stall", 900.0)
    kwargs.setdefault("maximum", 86400.0)
    return simulated_limits(**kwargs)


# --- the ceiling is a safety control, not a gate ---------------------------------


def test_a_healthy_run_is_untouched_by_the_default_ceiling(harness):
    """The default ceiling is far above any legitimate run on the catalog, so
    a normal job completes exactly as it did before the ceiling existed. A
    spend ceiling that fires on legitimate runs is a bug, not a safety net."""
    provider = FakeProvider(
        lines=TRAINING_LINES,
        result={
            **RESULT,
            "artifact_sha256": hashlib.sha256(b"weights").hexdigest(),
        },
        adapter_bytes=b"weights",
    )
    job_id = harness.run(provider, limits=spend_limits(step=60.0))

    job = harness.job(job_id)
    assert job["status"] == "complete"
    assert job["error_code"] is None
    assert not any("spend ceiling" in m for m in harness.messages(job_id))


# --- reaching the ceiling: checkpoint, terminate, destroy -------------------------


def test_a_run_that_exceeds_the_ceiling_fails_with_budget_exhausted(
    harness, monkeypatch
):
    """The ceiling is a hard cap on what one job can spend: a run that passes
    it is stopped, its machine destroyed through the confirmed teardown path,
    and the failure is named budget_exhausted -- distinguishable from a stall
    and from the duration ceiling, which is the point of user story 13."""
    from temper_control_plane import orchestrator

    monkeypatch.setattr(orchestrator, "SPEND_CEILING_MINOR", 100)  # Rs 1
    provider = FakeProvider(
        lines=TRAINING_LINES,
        result={
            **RESULT,
            "artifact_sha256": hashlib.sha256(b"weights").hexdigest(),
        },
        adapter_bytes=b"weights",
    )
    job_id = harness.run(provider, limits=spend_limits(step=300.0))

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "budget_exhausted"
    assert "spend ceiling" in job["error_message"]
    assert job["machine_id"] == MACHINE_ID
    assert provider.destroyed, (
        "a job stopped by the ceiling must not leave a machine billing"
    )
    assert provider.calls.count("list_machines") >= 3, (
        "teardown must go through the confirmed path, not a bare destroy"
    )


def test_the_ordered_shutdown_at_the_ceiling_is_checkpoint_then_destroy(
    harness, monkeypatch
):
    """The emergency checkpoint is requested before the machine is destroyed,
    and the teardown confirmation still precedes the terminal state -- so the
    saved progress is on record before anything is torn down, and a client
    that stops polling at terminal still sees the confirmation."""
    from temper_control_plane import orchestrator

    monkeypatch.setattr(orchestrator, "SPEND_CEILING_MINOR", 100)
    provider = FakeProvider(
        lines=TRAINING_LINES,
        result={
            **RESULT,
            "artifact_sha256": hashlib.sha256(b"weights").hexdigest(),
        },
        adapter_bytes=b"weights",
        checkpoints=[
            {"step": 10, "loss": 1.0, "bytes": b"ckpt-10"},
            {"step": 20, "loss": 0.5, "bytes": b"ckpt-20"},
        ],
    )
    job_id = harness.run(provider, limits=spend_limits(step=300.0))

    events = harness.events(job_id)
    messages = [e["message"] for e in events]
    asked = next(
        i
        for i, m in enumerate(messages)
        if "requesting an emergency checkpoint" in m
    )
    destroyed = next(i for i, m in enumerate(messages) if "destroyed" in m)
    assert asked < destroyed, (
        "checkpoint must be asked for before the machine is destroyed"
    )
    states = [i for i, e in enumerate(events) if e["kind"] == "state"]
    assert states[-1] == len(events) - 1, (
        "terminal state must be the last word"
    )


# --- the checkpoint written at the ceiling is retrievable --------------------------


def test_the_checkpoint_saved_at_the_ceiling_is_retrievable_and_selected(
    harness, monkeypatch
):
    """The checkpoint written at the ceiling is not asserted-to-exist: it is
    fetched back through the download route and its bytes come out, and the
    held-out-loss selection (ADR-0049) can see it -- the chosen result
    checkpoint is the ceiling checkpoint, not nothing."""
    from temper_control_plane import orchestrator

    # The ceiling trips a few simulated minutes into the run, once the
    # machine's stream is genuinely underway (and its script -- which carries
    # the checkpoint grants -- is on record), not on the first clock read.
    monkeypatch.setattr(orchestrator, "SPEND_CEILING_MINOR", 600)  # Rs 6
    provider = FakeProvider(
        lines=TRAINING_LINES,
        result={
            **RESULT,
            "artifact_sha256": hashlib.sha256(b"weights").hexdigest(),
        },
        adapter_bytes=b"weights",
        checkpoints=[
            {
                "step": 10,
                "loss": 1.0,
                "held_out_loss": 0.44,
                "bytes": b"ckpt-10",
            },
            {
                "step": 20,
                "loss": 0.5,
                "held_out_loss": 0.39,
                "bytes": b"ckpt-20",
            },
        ],
    )
    job_id = harness.run(provider, limits=spend_limits(step=60.0))

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "budget_exhausted"

    # Recorded, verified, and the best by held-out loss -- the ceiling
    # checkpoint is visible to ADR-0049's selection, not parked somewhere it
    # cannot see.
    records = harness.stored(job_id)["checkpoints"]
    assert {r["step"] for r in records} == {10, 20}
    assert all(r["verified"] is True for r in records)
    assert harness.stored(job_id)["best_checkpoint"]["step"] == 20

    # Retrieved, not merely written: the bytes come back out of the seam, as
    # the checkpoint's own tar (the shape a resumption parses back).
    r = harness._client.get(f"/v1/jobs/{job_id}/checkpoints/20")
    assert r.status_code == 200, r.text
    assert r.content == fake_checkpoint_tar(
        20, b"ckpt-20", loss=0.5, held_out_loss=0.39
    )


def test_a_ceiling_stop_with_no_checkpoints_records_none_honestly(
    harness, monkeypatch
):
    """A machine that has nothing to save at the ceiling records nothing, and
    the record says so plainly rather than claiming a checkpoint it does not
    have. The shutdown still completes: the reason is budget_exhausted and
    the machine is destroyed."""
    from temper_control_plane import orchestrator

    monkeypatch.setattr(orchestrator, "SPEND_CEILING_MINOR", 100)
    provider = FakeProvider(
        lines=TRAINING_LINES,
        result={
            **RESULT,
            "artifact_sha256": hashlib.sha256(b"weights").hexdigest(),
        },
        adapter_bytes=b"weights",
    )
    job_id = harness.run(provider, limits=spend_limits(step=300.0))

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "budget_exhausted"
    assert provider.destroyed
    assert harness.stored(job_id)["checkpoints"] == []
    assert any(
        "No checkpoints" in m or "reported none" in m
        for m in harness.messages(job_id)
    )


# --- enforced from outside the training process ----------------------------------


def test_the_ceiling_fires_when_the_machine_stops_responding(
    harness, monkeypatch
):
    """The criterion the issue states in one line: a process that has stopped
    responding cannot enforce its own limit. The ceiling is measured by the
    control plane from wall clock and the frozen price -- never from anything
    the trainer reports -- so a machine that goes silent (the `machine_silent`
    fault, exercised through the fault surface) still trips it, before the
    stall detector has any reason to fire."""
    from temper_control_plane import orchestrator

    harness.enable_fake_tier()
    # Enough simulated time that the machine's narration lines have been
    # emitted and the silence has started, but far short of the stall budget
    # (900s): the only thing that can have stopped the job is the ceiling.
    monkeypatch.setattr(orchestrator, "SPEND_CEILING_MINOR", 600)
    provider = completed_run()
    job_id = harness.run(
        provider,
        hyperparameters={
            "simulated_failure_code": {
                "name": "machine_silent",
                "after_line": 1,
            }
        },
        limits=spend_limits(step=60.0),
    )

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "budget_exhausted"
    assert provider.fault_applied == "machine_silent"
    assert provider.destroyed
    # The stall detector did not win the race: the reason is the ceiling.
    assert "gpu_stalled" not in (job.get("error_code") or "")
    assert any(
        "Simulated fault injected: machine_silent" in m
        for m in harness.messages(job_id)
    )


def test_the_ceiling_is_derived_from_the_machines_price(harness, monkeypatch):
    """The ceiling is a cost, and the same ceiling is reached faster on an
    expensive machine than a cheap one -- the ceiling is derived from the
    job's own frozen price, not from a fixed number of minutes. With the same
    simulated clock, a job on an H100 (250 INR/hr) exceeds the ceiling while
    the identical job on an L4 (41.31 INR/hr) completes."""
    from temper_control_plane import orchestrator

    monkeypatch.setattr(orchestrator, "SPEND_CEILING_MINOR", 1000)  # Rs 10

    cheap = FakeProvider(
        lines=TRAINING_LINES,
        result={
            **RESULT,
            "artifact_sha256": hashlib.sha256(b"weights").hexdigest(),
        },
        adapter_bytes=b"weights",
    )
    cheap_job = harness.run(cheap, limits=spend_limits(step=60.0))
    assert harness.job(cheap_job)["status"] == "complete"

    dear = FakeProvider(
        lines=TRAINING_LINES,
        result={
            **RESULT,
            "artifact_sha256": hashlib.sha256(b"weights").hexdigest(),
        },
        adapter_bytes=b"weights",
        availability=[GpuAvailability("H100", 250.0, 2)],
    )
    dear_job = harness.run(dear, limits=spend_limits(step=60.0))
    assert harness.job(dear_job)["status"] == "failed"
    assert harness.job(dear_job)["error_code"] == "budget_exhausted"
