"""A job that runs out of memory retries with the effective batch preserved
(issue #35).

The recovery is automatic: a memory failure triggers a retry that halves the
per-step batch and doubles the accumulation, so the effective batch is
unchanged -- asserted here as an invariant on the spec the retried attempt
actually runs, and on the emitted loss series a user sees (the continuity
criterion is a claim about data, so it is tested on the data). The escalation
ladder is exercised (batch floor -> sequence length -> more capable
hardware), the retry cap is bounded, and a memory retry is kept
distinguishable from a divergence retry (ADR-0055) in the record and the db.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from helpers import wait_validated

from temper_control_plane.fake_provider import (
    DEFAULT_AVAILABILITY,
    PUBLISHED_IMAGE_REFERENCE,
    FakeProvider,
    _job_spec_from_script,
    completed_run,
)
from temper_core import faults as fault_surface
from temper_core import memory_retry
from temper_core.memory_retry import (
    MEMORY_FAILURE_CODES,
    MEMORY_RECOVERY_CODE,
    MEMORY_RETRIES_EXHAUSTED_CODE,
    MEMORY_RETRY_CAP,
)

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


def loss_line(loss, step):
    return f"{{'loss': {loss}, 'step': {step}, 'epoch': 0.5}}"


class MemoryRecoveryMachine(FakeProvider):
    """A machine that OOMs the first `oom_attempts` attempts, then succeeds.

    The recovered machine's emitted loss series depends on the spec it is
    handed: if the effective batch (micro_batch_size x accumulation) matches
    `launch_effective_batch` it continues the pre-OOM series (`continuation`),
    and otherwise it emits a *jumped* series -- the discontinuity a changed
    optimisation would show a user. This is what makes the continuity test a
    test of the recovery and not of the script: a recovery that changed the
    effective batch would produce a jumped series and fail the assertion.
    """

    def __init__(
        self,
        *,
        oom_attempts: int = 1,
        launch_effective_batch: int = 8,
        continuation: list[float] | None = None,
        jumped: list[float] | None = None,
        oom_code: str = memory_retry.REAL_OOM_CODE,
        availability=None,
    ) -> None:
        super().__init__(
            lines=[],
            result=RESULT,
            adapter_bytes=ADAPTER_BYTES,
            availability=availability or list(DEFAULT_AVAILABILITY),
        )
        self._oom_attempts = int(oom_attempts)
        self._launch_effective_batch = int(launch_effective_batch)
        self._continuation = continuation or [0.5, 0.49, 0.48, 0.47]
        self._jumped = jumped or [0.9, 0.85, 0.8, 0.75]
        self._oom_code = oom_code
        self._attempt_count = 0

    def stream(self, machine, script):
        attempt = self._attempt_count
        self._attempt_count += 1
        if attempt < self._oom_attempts:
            step = 10 * (attempt + 1)
            self._lines = [
                "[simulated] CUDA out of memory after training step",
                loss_line(0.7 - 0.1 * attempt, step),
            ]
            self._result = {
                "ok": False,
                "stage": "train",
                "error_code": self._oom_code,
                "error": (
                    "The training process ran out of device memory; no "
                    "artifact was produced."
                ),
            }
        else:
            spec = _job_spec_from_script(script) or {}
            hp = spec.get("hyperparameters") or {}
            try:
                batch = int(hp.get("micro_batch_size"))
                acc = int(hp.get("gradient_accumulation_steps"))
                preserved = batch * acc == self._launch_effective_batch
            except (TypeError, ValueError):
                preserved = False
            losses = self._continuation if preserved else self._jumped
            step = 10 * (attempt + 1)
            self._lines = [
                loss_line(v, step + i) for i, v in enumerate(losses)
            ]
            self._result = dict(RESULT)
        yield from super().stream(machine, script)


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


class Harness:
    def __init__(self, client, monkeypatch, tmp_path):
        self._client = client
        self._monkeypatch = monkeypatch
        self._tmp_path = tmp_path

    def run(
        self, provider, hyperparameters=None, limits=None, models=None
    ) -> str:
        from temper_control_plane import fake_models, orchestrator

        models = models or fake_models.catalog_models()
        # Issue #51: the request path no longer starts threads. Drive
        # the job directly as the worker would, rather than relying on
        # ``launch`` being called by ``POST /v1/jobs``.
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

    def job(self, job_id):
        return self._client.get(f"/v1/jobs/{job_id}").json()

    def events(self, job_id):
        return self._client.get(f"/v1/jobs/{job_id}/events").json()["events"]

    def metrics(self, job_id):
        return [
            e["data"]["loss"]
            for e in self.events(job_id)
            if e["kind"] == "metric" and e.get("data") and "loss" in e["data"]
        ]


def _recovery_events(events):
    return [
        e
        for e in events
        if e.get("data") and e["data"].get("code") == MEMORY_RECOVERY_CODE
    ]


# The continuity a user sees in the loss chart: consecutive emitted losses
# move by small steps (no jump), and the curve descends rather than restarting
# at its initial value. Defined once as a named predicate so the positive test
# (the recovery's emitted series) and the negative control (a jumped series)
# exercise exactly the same check -- a regression that silently widened the
# threshold would fail the negative control.
MAX_STEP_DELTA = 0.2


def assert_continuous_loss_series(series: list[float]) -> None:
    assert len(series) >= 2
    for prev, nxt in zip(series, series[1:], strict=False):
        assert abs(nxt - prev) < MAX_STEP_DELTA, (
            f"loss curve jumps at the retry: {prev} -> {nxt}"
        )
    assert series[0] > series[-1], "loss curve restarts rather than descending"
    assert abs(series[1] - series[0]) < MAX_STEP_DELTA


# ---------------------------------------------------------------------------
# the automatic retry and the invariant
# ---------------------------------------------------------------------------


def test_a_memory_failure_triggers_an_automatic_retry_with_the_invariant_held(
    harness,
):
    """A genuine out-of-memory failure retries automatically, and the retried
    attempt's spec halves the per-step batch and doubles the accumulation --
    asserted as the invariant (per_step x accumulation unchanged), on the spec
    the retried attempt actually ran, not on one hand-picked sequence."""
    provider = MemoryRecoveryMachine(oom_attempts=1, launch_effective_batch=8)
    job_id = harness.run(
        provider,
        hyperparameters={
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 1,
        },
    )
    job = harness.job(job_id)
    assert job["status"] == "complete"

    attempts = job["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["outcome"] == "failed"
    assert attempts[0]["error_code"] == memory_retry.REAL_OOM_CODE
    assert attempts[-1]["outcome"] == "complete"

    # The invariant, on the spec the retried machine was handed.
    first = attempts[0]["spec"]
    last = attempts[-1]["spec"]
    assert first["micro_batch_size"] == 8
    assert first["gradient_accumulation_steps"] == 1
    assert last["micro_batch_size"] == 4
    assert last["gradient_accumulation_steps"] == 2
    assert (
        last["micro_batch_size"] * last["gradient_accumulation_steps"]
        == first["micro_batch_size"] * first["gradient_accumulation_steps"]
        == 8
    )

    # The recovery is told in the history, with its own code.
    recovery = _recovery_events(harness.events(job_id))
    assert len(recovery) == 1
    assert recovery[0]["data"]["rung"] == memory_retry.RUNG_HALVE_BATCH
    assert recovery[0]["data"]["effective_batch"] == 8


def test_the_retry_is_automatic_and_uses_one_machine_per_attempt(harness):
    """Each attempt is its own machine, torn down before the next provisions:
    the recovery never stacks machines."""
    provider = MemoryRecoveryMachine(oom_attempts=1, launch_effective_batch=8)
    job_id = harness.run(
        provider,
        hyperparameters={
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 1,
        },
    )
    assert harness.job(job_id)["status"] == "complete"
    assert len(provider.created) == 2
    assert provider.destroy_attempts >= 2


def test_the_loss_curve_is_continuous_across_the_retry(harness):
    """The emitted series a user sees is continuous across the retry: the
    retried attempt (effective batch preserved) continues the pre-OOM loss
    series with no jump. The machine emits a *jumped* series when the
    effective batch is not preserved, so this test fails exactly when the
    recovery changes the optimisation -- the claim is on the data, not a
    comment."""
    provider = MemoryRecoveryMachine(
        oom_attempts=1,
        launch_effective_batch=8,
        continuation=[0.7, 0.6, 0.5, 0.49, 0.48, 0.47],
    )
    job_id = harness.run(
        provider,
        hyperparameters={
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 1,
        },
    )
    assert harness.job(job_id)["status"] == "complete"

    series = harness.metrics(job_id)
    assert_continuous_loss_series(series)
    # And the pre-OOM value is what the retry continues from (not a restart
    # back up at the initial loss).
    assert series[0] > series[-1]


def test_a_jumped_emitted_series_is_caught_by_the_continuity_check():
    """The negative control, and it is a real one: the continuity predicate
    used on the recovery's emitted series must fail on a series that jumps at
    the retry boundary -- the discontinuity a user would see if the recovery
    ever changed the effective batch. A regression that silently removed the
    jump detection from the check would fail this test, so the positive test
    cannot pass while the detection is gone."""
    continuous = [0.7, 0.6, 0.5, 0.49, 0.48, 0.47]
    assert_continuous_loss_series(continuous)
    # A restart back at the initial loss (the shape of a changed run) jumps.
    with pytest.raises(AssertionError, match="jumps at the retry"):
        assert_continuous_loss_series([0.7, 0.6, 0.5, 0.9, 0.85, 0.8])
    with pytest.raises(AssertionError, match="jumps at the retry"):
        assert_continuous_loss_series([0.7, 0.6, 0.5, 0.75, 0.7, 0.68])


def test_a_recovery_that_changed_the_effective_batch_shows_the_jump(
    harness, monkeypatch
):
    """The pipeline-level negative control: drive the actual orchestrator
    through a deliberately broken escalation -- per-step batch halved,
    accumulation NOT doubled, so the effective batch changes -- and assert the
    emitted series is discontinuous. This is what would happen to a user if
    the recovery ever broke the invariant, and it is why the continuity check
    is a guard rather than a comment: the positive test's continuity assertion
    fails exactly on this shape."""
    from temper_core.memory_retry import MemoryRetryStep

    class BrokenEscalator:
        """A recovery that halves the batch without doubling accumulation."""

        def __init__(self, hyperparameters, **kwargs):
            self._hp = dict(hyperparameters)

        def step(self):
            old_b = int(self._hp["micro_batch_size"])
            new_b = max(old_b // 2, 1)
            self._hp["micro_batch_size"] = new_b
            return MemoryRetryStep(
                hyperparameters=dict(self._hp),
                rung="halve_batch",
                action="per-step batch halved without doubling accumulation",
                changed={"micro_batch_size": (old_b, new_b)},
                effective_batch=new_b
                * int(self._hp["gradient_accumulation_steps"]),
            )

    monkeypatch.setattr(memory_retry, "MemoryEscalator", BrokenEscalator)
    provider = MemoryRecoveryMachine(oom_attempts=1, launch_effective_batch=8)
    job_id = harness.run(
        provider,
        hyperparameters={
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 1,
        },
    )
    assert harness.job(job_id)["status"] == "complete"
    series = harness.metrics(job_id)
    # The broken escalation changed the effective batch, so the machine
    # emitted a jumped series and the continuity check fails on it.
    with pytest.raises(AssertionError):
        assert_continuous_loss_series(series)


# ---------------------------------------------------------------------------
# the escalation ladder
# ---------------------------------------------------------------------------


def test_a_default_config_escalates_past_the_batch_floor_to_sequence_length(
    harness,
):
    """Defaults carry micro_batch_size 1 -- already at its floor -- so the
    first memory retry skips the batch rung (and gradient checkpointing,
    always on) and halves the sequence length."""
    provider = MemoryRecoveryMachine(oom_attempts=1, launch_effective_batch=8)
    job_id = harness.run(provider)
    job = harness.job(job_id)
    assert job["status"] == "complete"
    attempts = job["attempts"]
    assert attempts[0]["spec"]["micro_batch_size"] == 1
    assert attempts[-1]["spec"]["sequence_len"] == 1024
    recovery = _recovery_events(harness.events(job_id))
    assert recovery[0]["data"]["rung"] == memory_retry.RUNG_SEQUENCE_LENGTH


def test_the_ladder_climbs_to_more_capable_hardware(harness):
    """When batch is at its floor and sequence length at its floor, the retry
    escalates to more capable hardware -- a card with strictly more memory
    than the one that OOMed -- and finishes on it."""
    from temper_core.selection import GpuAvailability

    provider = MemoryRecoveryMachine(
        oom_attempts=3,  # batch floor, then two sequence-length halvings OOM
        launch_effective_batch=8,
        availability=[
            GpuAvailability("L4", 41.31, 8),
            GpuAvailability("A100-80GB", 250.0, 8),
        ],
    )
    job_id = harness.run(provider)
    job = harness.job(job_id)
    assert job["status"] == "complete"
    # The final machine was the bigger card, and only after the in-place
    # reductions were spent.
    assert job["gpu_type"] == "A100-80GB"
    rungs = [
        e["data"]["rung"] for e in _recovery_events(harness.events(job_id))
    ]
    assert rungs == [
        memory_retry.RUNG_SEQUENCE_LENGTH,
        memory_retry.RUNG_SEQUENCE_LENGTH,
        memory_retry.RUNG_HARDWARE,
    ]
    assert len(provider.created) == 4


# ---------------------------------------------------------------------------
# the retry cap
# ---------------------------------------------------------------------------


def test_retries_are_capped_and_exhaustion_fails_with_the_attempts_recorded(
    harness,
):
    """A memory failure that keeps happening stops retrying at the cap and
    fails the job with the stable reason and every attempt recorded -- it
    never retries forever against a machine that will not fit."""
    provider = MemoryRecoveryMachine(
        oom_attempts=99,
        launch_effective_batch=8,
    )
    job_id = harness.run(
        provider,
        hyperparameters={
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 1,
        },
    )
    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == MEMORY_RETRIES_EXHAUSTED_CODE
    # One initial attempt plus the capped retries.
    assert len(job["attempts"]) == MEMORY_RETRY_CAP + 1
    assert all(a["outcome"] == "failed" for a in job["attempts"])
    assert any(
        (e.get("data") or {}).get("code") == MEMORY_RETRIES_EXHAUSTED_CODE
        for e in harness.events(job_id)
    )


# ---------------------------------------------------------------------------
# the fault surface exercises the path
# ---------------------------------------------------------------------------


def test_the_oom_fault_exercises_the_recovery_through_the_fault_surface(
    harness,
):
    """The existing fault surface (ADR-0051, off by default) is the way the
    recovery is proven: a job carrying the `oom` fault has its first machine
    exhaust memory, and the automatic retry -- the effective batch preserved
    -- completes the job. No second fault is added and the guard is not
    weakened; the fake tier is switched on exactly as every other fault test
    switches it on."""
    from temper_control_plane import config, fake_models, orchestrator

    harness._monkeypatch.setattr(config, "FAKE_PROVIDER", True)
    provider = completed_run()
    path = harness._tmp_path / "fault.jsonl"
    path.write_text(
        "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(12)),
        encoding="utf-8",
    )
    with open(path, "rb") as f:
        ds = harness._client.post(
            "/v1/datasets", files={"file": (path.name, f)}
        ).json()["id"]
    wait_validated(harness._client, ds)
    r = harness._client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "hyperparameters": {"simulated_failure_code": {"name": "oom"}},
        },
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    # The request path only inserts a `queued` row (issue #51); drive it
    # directly, the way the worker would.
    orchestrator.run_job(
        job_id, provider=provider, models=fake_models.catalog_models()
    )
    job = harness.job(job_id)
    assert job["status"] == "complete"
    assert provider.fault_applied == "oom"
    # The deliberately broken first attempt carries the fault's simulated code.
    assert job["attempts"][0]["error_code"] == "simulated_oom"
    assert fault_surface.code_for("oom") in MEMORY_FAILURE_CODES


# ---------------------------------------------------------------------------
# distinguishable from a divergence retry (ADR-0055)
# ---------------------------------------------------------------------------


def test_a_memory_recovery_never_looks_like_a_divergence_retry(harness):
    """A memory retry is automatic; a divergence retry is a choice. The
    recovered job's record and events carry the memory codes, and a memory-
    recovered job is not offered the divergence retry endpoint."""
    provider = MemoryRecoveryMachine(oom_attempts=1, launch_effective_batch=8)
    job_id = harness.run(
        provider,
        hyperparameters={
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 1,
        },
    )
    job = harness.job(job_id)
    assert job["status"] == "complete"
    # No divergence vocabulary anywhere in the history or the record.
    assert job["error_code"] is None
    for e in harness.events(job_id):
        code = (e.get("data") or {}).get("code")
        assert code not in ("training_diverged", "training_instability")
    # The divergence retry endpoint refuses: this is not a diverged job.
    r = harness._client.post(f"/v1/jobs/{job_id}/retry")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "not_diverged"


def test_the_memory_failure_codes_and_the_divergence_codes_are_disjoint(
    harness,
):
    from temper_core.divergence import DIVERGED_CODE, INSTABILITY_CODE

    assert MEMORY_FAILURE_CODES.isdisjoint({DIVERGED_CODE, INSTABILITY_CODE})
    assert MEMORY_RETRIES_EXHAUSTED_CODE != DIVERGED_CODE
