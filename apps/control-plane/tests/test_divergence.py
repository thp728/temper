"""A diverging job stops early with a plain cause (issue #36).

The detector reads the measurements the platform already streams (training loss
as metric events and the non-finite loss log line) against thresholds published
in the research rather than invented here. The numbers are report-b Section
5.7: NaN immediate, >2x trailing 50 average for >20 consecutive steps.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from helpers import wait_validated

from temper_control_plane.fake_provider import FakeProvider
from temper_core.divergence import DIVERGED_CODE, INSTABILITY_CODE

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


@pytest.fixture()
def harness(isolated, tmp_path, monkeypatch):
    from temper_control_plane import fake_provider, main, orchestrator

    monkeypatch.setattr(orchestrator, "DESTROY_RETRY_DELAY_S", 0)
    monkeypatch.setattr(
        orchestrator,
        "published_reference",
        lambda: fake_provider.PUBLISHED_IMAGE_REFERENCE,
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
        self._monkeypatch.setattr(
            orchestrator,
            "launch",
            lambda job_id: orchestrator.run_job(
                job_id, provider=provider, limits=limits, models=models
            ),
        )
        return self._create(hyperparameters)

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


def test_a_nan_loss_aborts_with_training_diverged(harness):
    # A single nan loss line is immediate divergence -- unrecoverable without
    # rollback, so v1 aborts rather than rolling back 100 steps.
    lines = [
        "[00:00:00] running training",
        "{'loss': 0.5, 'step': 10, 'epoch': 0.2}",
        "{'loss': nan, 'step': 20, 'epoch': 0.4}",
    ]
    provider = FakeProvider(
        lines=lines, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider)
    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == DIVERGED_CODE
    assert "diverged" in job["error_message"].lower()
    assert "lower learning rate" in job["error_message"].lower()


def test_instability_is_a_warning_not_an_abort(harness, monkeypatch):
    """Loss spiking for WARNING_CONSECUTIVE but not DIVERGENCE_CONSECUTIVE
    surfaces as a warning event and the job continues to completion.
    """
    from temper_control_plane import config

    # Shrink thresholds so the test is fast but preserves the published ratio:
    # warning at 2 steps, divergence at 4, window 3, multiplier 2.0.
    monkeypatch.setattr(config, "DIVERGENCE_WINDOW", 3)
    monkeypatch.setattr(config, "DIVERGENCE_CONSECUTIVE", 4)
    monkeypatch.setattr(config, "WARNING_CONSECUTIVE", 2)
    monkeypatch.setattr(config, "DIVERGENCE_MULTIPLIER", 2.0)

    # Fill window with 0.5, then spike for 2 steps -> warning, then normal -> no abort
    # Use increasing spikes so sliding window does not hide the exceedance.
    lines = [
        "[00:00:00] running training",
        "{'loss': 0.5, 'step': 1, 'epoch': 0.1}",
        "{'loss': 0.5, 'step': 2, 'epoch': 0.2}",
        "{'loss': 0.5, 'step': 3, 'epoch': 0.3}",
        "{'loss': 1.5, 'step': 4, 'epoch': 0.4}",  # exceed 1
        "{'loss': 3.0, 'step': 5, 'epoch': 0.5}",  # 2nd exceed -> warning
        "{'loss': 0.5, 'step': 6, 'epoch': 0.6}",  # reset with normals to flush window
        "{'loss': 0.5, 'step': 7, 'epoch': 0.7}",
        "{'loss': 0.5, 'step': 8, 'epoch': 0.8}",
        "{'loss': 0.5, 'step': 9, 'epoch': 0.9}",
    ]
    provider = FakeProvider(
        lines=lines, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider)
    job = harness.job(job_id)
    assert job["status"] == "complete"
    warnings = [
        e
        for e in harness.events(job_id)
        if e.get("data") and e["data"].get("code") == INSTABILITY_CODE
    ]
    assert len(warnings) == 1
    assert "instability" in warnings[0]["message"].lower()


def test_sustained_exceedance_diverges_and_aborts(harness, monkeypatch):
    from temper_control_plane import config

    monkeypatch.setattr(config, "DIVERGENCE_WINDOW", 3)
    monkeypatch.setattr(config, "DIVERGENCE_CONSECUTIVE", 4)
    monkeypatch.setattr(config, "WARNING_CONSECUTIVE", 2)
    monkeypatch.setattr(config, "DIVERGENCE_MULTIPLIER", 2.0)

    lines = [
        "[00:00:00] running training",
        "{'loss': 0.5, 'step': 1, 'epoch': 0.1}",
        "{'loss': 0.5, 'step': 2, 'epoch': 0.2}",
        "{'loss': 0.5, 'step': 3, 'epoch': 0.3}",
        "{'loss': 1.5, 'step': 4, 'epoch': 0.4}",
        "{'loss': 3.0, 'step': 5, 'epoch': 0.5}",
        "{'loss': 6.0, 'step': 6, 'epoch': 0.6}",
        "{'loss': 12.0, 'step': 7, 'epoch': 0.7}",
    ]
    provider = FakeProvider(
        lines=lines, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider)
    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == DIVERGED_CODE


def test_the_fault_surface_divergence_trips_the_detector(harness, monkeypatch):
    """The divergence fault is the proof: a job whose spec names the fault must
    be observed to abort with training_diverged. Uses SimulatedMachine so the
    fault is applied exactly as the orchestrator and fake provider agree.
    """
    from temper_control_plane import config, fake_provider

    monkeypatch.setattr(config, "FAKE_PROVIDER", True)
    # Build a harness that drives through the fake's SimulatedMachine path
    # rather than a plain FakeProvider: we need the dict form of the fault
    # to be read from the job spec inside SimulatedMachine.stream.
    from temper_control_plane import fake_models, orchestrator

    # Monkeypatch launch to use SimulatedMachine with no custom lines -- the
    # fault configures the lines to include the nan.
    provider = fake_provider.SimulatedMachine(
        lines=fake_provider.DEMO_LINES,
        result=fake_provider.DEMO_RESULT,
        adapter_bytes=fake_provider.DEMO_ADAPTER_BYTES,
    )

    # We need to create the job with the fault spec while FAKE_PROVIDER is on.
    # Harness.run replaces launch; we do it manually.
    harness._monkeypatch.setattr(
        orchestrator,
        "launch",
        lambda job_id: orchestrator.run_job(
            job_id, provider=provider, models=fake_models.catalog_models()
        ),
    )
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
            "hyperparameters": {
                "simulated_failure_code": {"name": "divergence"}
            },
        },
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    # Run is synchronous in test harness via run_job, so job is terminal now.
    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == DIVERGED_CODE
    assert provider.fault_applied == "divergence"


def test_a_diverged_job_offers_a_single_half_lr_retry_as_a_choice(
    harness, monkeypatch
):
    from temper_control_plane import config

    monkeypatch.setattr(config, "DIVERGENCE_WINDOW", 3)
    monkeypatch.setattr(config, "DIVERGENCE_CONSECUTIVE", 3)
    monkeypatch.setattr(config, "WARNING_CONSECUTIVE", 2)

    lines = [
        "[00:00:00] running training",
        "{'loss': 0.5, 'step': 1, 'epoch': 0.1}",
        "{'loss': 0.5, 'step': 2, 'epoch': 0.2}",
        "{'loss': 0.5, 'step': 3, 'epoch': 0.3}",
        "{'loss': nan, 'step': 4, 'epoch': 0.4}",
    ]
    provider = FakeProvider(
        lines=lines, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    # Use a known learning_rate so halving is observable
    job_id = harness.run(provider, hyperparameters={"learning_rate": 0.002})
    job = harness.job(job_id)
    assert job["error_code"] == DIVERGED_CODE
    old_lr = job["hyperparameters"]["learning_rate"]
    assert old_lr == 0.002

    # Retry is a choice, not automatic: no new job exists until the endpoint is called.
    assert not job.get("retry_from")
    # Offer the retry
    r = harness._client.post(f"/v1/jobs/{job_id}/retry")
    assert r.status_code == 201, r.text
    new_job = r.json()
    assert new_job["hyperparameters"]["learning_rate"] == pytest.approx(
        old_lr * 0.5
    )
    assert new_job["retry_from"] == job_id

    # Single retry: a second attempt is refused with already_retried
    r2 = harness._client.post(f"/v1/jobs/{job_id}/retry")
    assert r2.status_code == 409
    assert r2.json()["detail"]["code"] == "already_retried"

    # A non-diverged job cannot be retried this way
    ok_provider = FakeProvider(
        lines=["{'loss': 0.5, 'step': 1, 'epoch': 0.1}"],
        result=RESULT,
        adapter_bytes=ADAPTER_BYTES,
    )
    ok_id = harness.run(ok_provider)
    assert harness.job(ok_id)["status"] == "complete"
    r3 = harness._client.post(f"/v1/jobs/{ok_id}/retry")
    assert r3.status_code == 409
    assert r3.json()["detail"]["code"] == "not_diverged"


def test_retry_is_not_automatic(harness, monkeypatch):
    """A diverging run does not automatically launch a second machine; the
    retry must be asked for."""
    from temper_control_plane import config

    monkeypatch.setattr(config, "DIVERGENCE_WINDOW", 3)
    monkeypatch.setattr(config, "DIVERGENCE_CONSECUTIVE", 2)
    monkeypatch.setattr(config, "WARNING_CONSECUTIVE", 1)

    lines = [
        "[00:00:00] running training",
        "{'loss': 0.5, 'step': 1, 'epoch': 0.1}",
        "{'loss': 0.5, 'step': 2, 'epoch': 0.2}",
        "{'loss': 0.5, 'step': 3, 'epoch': 0.3}",
        "{'loss': nan, 'step': 4, 'epoch': 0.4}",
    ]
    provider = FakeProvider(
        lines=lines, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider)
    job = harness.job(job_id)
    assert job["error_code"] == DIVERGED_CODE
    # Only one machine was created; no automatic retry provisioned a second.
    assert len(provider.created) == 1
