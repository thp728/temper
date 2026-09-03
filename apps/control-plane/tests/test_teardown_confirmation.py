"""Teardown is confirmed across consecutive observations (issue #34).

Spec 010 / spike/teardown.py C17: after a destroy the provider's listing is
eventually consistent; the machine reads absent, then reappears as
`destroying`, then goes absent for good. A single absent observation is not
proof; confirmation requires consecutive absences, and `destroying` is
treated as not yet confirmed rather than as a stray. A destroy the provider
refuses is retried and, when exhausted, recorded loudly. The path is
exercised against the `destroy_refused` fault.
"""

import json

import pytest
from fastapi.testclient import TestClient
from helpers import wait_validated

from temper_control_plane.fake_provider import (
    MACHINE_ID,
    PUBLISHED_IMAGE_REFERENCE,
    FakeProvider,
    completed_run,
)


def chat(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


class Harness:
    def __init__(self, client, monkeypatch, tmp_path):
        self._client = client
        self._monkeypatch = monkeypatch
        self._tmp_path = tmp_path
        self.provider = None

    def enable_fake_tier(self):
        from temper_control_plane import config

        self._monkeypatch.setattr(config, "FAKE_PROVIDER", True)

    def create_and_run(self, provider, hyperparameters=None):
        from temper_control_plane import fake_models, orchestrator

        self.provider = provider
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
        job_id = r.json()["id"]
        # The request path only inserts a `queued` row (issue #51); drive it
        # directly, the way the worker would.
        orchestrator.run_job(
            job_id,
            provider=provider,
            models=fake_models.catalog_models(),
        )
        return job_id

    def job(self, job_id):
        return self._client.get(f"/v1/jobs/{job_id}").json()

    def events(self, job_id):
        return self._client.get(f"/v1/jobs/{job_id}/events").json()["events"]

    def messages(self, job_id):
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


def test_confirmation_requires_absence_across_consecutive_observations(
    harness, monkeypatch
):
    """A single absent listing is not proof: the provider can read absent
    then reappear as destroying. Confirmation requires consecutive absences."""
    from temper_control_plane import orchestrator

    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_SAMPLES", 3)

    # After destroy, script: absent (gap), then present (destroying), then
    # three consecutively absent. A confirmation that stopped at the first
    # absent would have succeeded immediately; the correct one needs the run
    # of three.
    seq = [
        [],  # gap: would fool a single-sample check
        [MACHINE_ID],  # reappears (destroying in real provider)
        [],
        [],
        [],
    ]
    provider = FakeProvider(
        lines=["[00:00:00] running"],
        result={
            "ok": True,
            "stage": "train",
            "artifact_path": "run/adapter_model.safetensors",
            "artifact_sha256": "abc",
            "adapter_config": {"r": 16},
        },
        adapter_bytes=b"weights",
        list_sequence=seq,
    )
    # Need a valid artifact checksum handling; bypass artifact verification
    # by making result have no artifact_path so teardown is the only focus
    provider._result = {
        "ok": True,
        "stage": "train",
        "artifact_path": None,
    }
    # Directly exercise _teardown via run_job to keep orchestration path

    job_id = harness.create_and_run(provider)
    events = harness.events(job_id)
    # Teardown must have required multiple listings, not just one
    assert provider.calls.count("list_machines") >= 3
    # Must have been confirmed (no STRAY) because the run eventually
    # reached three consecutive absences
    assert not any("STRAY" in e["message"] for e in events)
    # And must have logged the confirmation
    assert any("Teardown confirmed" in m for m in harness.messages(job_id))


def test_a_machine_reported_as_destroying_is_not_yet_confirmed(
    harness, monkeypatch
):
    """Destroying is treated as present, not as confirmed and not as stray."""
    from temper_control_plane import orchestrator

    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_SAMPLES", 2)
    # Script destroying status explicitly
    seq = [
        [(MACHINE_ID, "destroying")],
        [(MACHINE_ID, "destroying")],
        [],
        [],
    ]
    provider = FakeProvider(
        lines=["[00:00:00] running"],
        result={"ok": True, "stage": "train"},
        adapter_bytes=b"x",
        list_sequence=seq,
    )
    provider._result = {"ok": True, "stage": "train", "artifact_path": None}
    job_id = harness.create_and_run(provider)
    messages = harness.messages(job_id)
    # Destroying must have been logged as not yet confirmed, not as STRAY
    assert any("still destroying" in m for m in messages)
    # No STRAY because the machine eventually went absent for two consecutive
    assert not any("STRAY" in m for m in messages)
    # Confirmation eventually succeeded
    assert any("Teardown confirmed" in m for m in messages)


def test_a_destroy_the_provider_refuses_is_retried(harness):
    """A transient destroy failure is retried rather than forgotten."""
    provider = FakeProvider(
        lines=["[00:00:00] running"],
        result={"ok": True, "stage": "train", "artifact_path": None},
        adapter_bytes=b"weights",
        destroy_failures=1,
    )
    provider._result = {"ok": True, "stage": "train", "artifact_path": None}
    job_id = harness.create_and_run(provider)
    assert provider.destroy_attempts == 2
    assert provider.destroyed
    errors = [
        e["message"] for e in harness.events(job_id) if e["kind"] == "error"
    ]
    assert any("Destroy attempt failed" in m for m in errors)
    assert not any("STRAY" in m for m in errors)


def test_a_machine_that_cannot_be_destroyed_is_recorded_loudly(harness):
    """When retries are exhausted the machine is recorded loudly rather than
    silently forgotten; an orphaned GPU bills until someone notices."""
    provider = FakeProvider(
        lines=["[00:00:00] running"],
        result={"ok": True, "stage": "train", "artifact_path": None},
        adapter_bytes=b"weights",
        destroy_failures=99,
        stays_listed=True,
    )
    provider._result = {"ok": True, "stage": "train", "artifact_path": None}
    job_id = harness.create_and_run(provider)
    errors = [
        e["message"] for e in harness.events(job_id) if e["kind"] == "error"
    ]
    # Retried
    assert provider.destroy_attempts == 3
    assert any("Destroy attempt" in m for m in errors)
    # Escalated: destroy refused loudly
    assert any("destroy refused" in m.lower() for m in errors)
    # And STRAY with billing warning
    assert any(
        "STRAY" in m and str(MACHINE_ID) in m and "billing" in m.lower()
        for m in errors
    )


def test_the_path_is_exercised_against_the_destroy_refused_fault(harness):
    """The fault surface's `destroy_refused` makes the provider refuse a
    destroy, and the orchestrator's path is exercised against it."""
    harness.enable_fake_tier()
    provider = completed_run()
    # Use SimulatedMachine via completed_run but inject fault through job spec
    job_id = harness.create_and_run(
        provider,
        hyperparameters={
            "simulated_failure_code": {"name": "destroy_refused", "times": 99}
        },
    )
    job = harness.job(job_id)
    # Training itself completes; teardown is what is refused
    assert job["status"] == "complete"
    errors = [
        e["message"] for e in harness.events(job_id) if e["kind"] == "error"
    ]
    assert any("Destroy attempt failed" in m for m in errors)
    assert any("STRAY" in m and str(MACHINE_ID) in m for m in errors)
    assert provider.fault_applied == "destroy_refused"
    assert any(
        "Simulated fault injected: destroy_refused" in m
        for m in harness.messages(job_id)
    )
