"""The money-spending path, exercised without spending money.

Every test here drives a job through the HTTP seam with a fake provider
injected. What is asserted is what a user or an operator can observe: the job
record, and the ordered event log. How many times an internal helper was called
is not behaviour. The one exception is teardown, where the observable outcome
*is* an event and the fact that destroy was attempted is the thing under test.
"""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.fake_provider import MACHINE_ID, FakeProvider

ADAPTER_BYTES = b"weights"
RESULT = {
    "ok": True,
    "stage": "train",
    "adapter_path": "run/adapter_model.safetensors",
    "adapter_sha256": hashlib.sha256(ADAPTER_BYTES).hexdigest(),
    "adapter_config": {"r": 16, "lora_alpha": 32},
}
TRAINING_LINES = [
    "[10:00:01] building trainer image",
    "[10:03:04] image built in 183s",
    "[10:03:04] running training",
    "{'loss': 1.9042, 'epoch': 0.5}",
]


def chat(user, assistant):
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


class Harness:
    """Upload a dataset, launch a job on a given provider, read it back."""

    def __init__(self, client, monkeypatch, tmp_path):
        self._client = client
        self._monkeypatch = monkeypatch
        self._tmp_path = tmp_path

    def run(self, provider, hyperparameters=None) -> str:
        from api import orchestrator

        # Run inline rather than on a thread: the job is the system under test,
        # so the test should observe its finished state rather than race it.
        self._monkeypatch.setattr(
            orchestrator, "launch",
            lambda job_id: orchestrator.run_job(job_id, provider=provider))

        path = self._tmp_path / "d.jsonl"
        path.write_text(
            "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(12)),
            encoding="utf-8")
        with open(path, "rb") as f:
            ds = self._client.post(
                "/v1/datasets", files={"file": (path.name, f)}).json()["id"]

        r = self._client.post("/v1/jobs", json={
            "dataset_id": ds, "hyperparameters": hyperparameters or {}})
        assert r.status_code == 201
        return r.json()["id"]

    def job(self, job_id) -> dict:
        return self._client.get(f"/v1/jobs/{job_id}").json()

    def events(self, job_id) -> list[dict]:
        return self._client.get(f"/v1/jobs/{job_id}/events").json()["events"]

    def messages(self, job_id) -> list[str]:
        return [e["message"] for e in self.events(job_id)]


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    from api import db, main, orchestrator

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(main, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(orchestrator, "ARTIFACTS", tmp_path / "artifacts")
    # Teardown retries sleep between attempts. Tests do not need to.
    monkeypatch.setattr(orchestrator, "DESTROY_RETRY_DELAY_S", 0)
    with TestClient(main.app) as c:
        yield Harness(c, monkeypatch, tmp_path)


def index_of(events, predicate) -> int:
    for i, e in enumerate(events):
        if predicate(e):
            return i
    raise AssertionError("no event matched")


def terminal_index(events) -> int:
    """Where the job reported that it was over."""
    states = [i for i, e in enumerate(events) if e["kind"] == "state"]
    return states[-1]


# --- the whole loop, no GPU ------------------------------------------------

def test_a_job_runs_to_completion_against_a_fake_provider(harness):
    provider = FakeProvider(lines=TRAINING_LINES, result=RESULT,
                            adapter_bytes=ADAPTER_BYTES)
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "complete"
    assert job["gpu_type"] == "L4"
    assert job["currency"] == "INR"
    assert job["machine_id"] == MACHINE_ID
    assert job["result"]["adapter_path"] == RESULT["adapter_path"]

    # The artifact is the adapter *and* the config that makes it loadable.
    directory = Path(job["adapter_path"]).parent
    assert (directory / "adapter_model.safetensors").read_bytes() == ADAPTER_BYTES
    assert json.loads((directory / "adapter_config.json").read_text()) == \
        RESULT["adapter_config"]

    assert provider.destroyed
    # An injected provider belongs to whoever injected it; the job does not
    # close a resource it did not open.
    assert not provider.closed


def test_scripted_output_lines_reach_the_event_log(harness):
    provider = FakeProvider(lines=TRAINING_LINES, result=RESULT,
                            adapter_bytes=ADAPTER_BYTES)
    job_id = harness.run(provider)
    messages = harness.messages(job_id)
    for line in TRAINING_LINES:
        assert line in messages
    # The result marker and the JSON behind it are machinery, not output.
    assert "---RESULT---" not in messages


def test_the_job_spec_reaches_the_machine(harness):
    provider = FakeProvider(lines=TRAINING_LINES, result=RESULT,
                            adapter_bytes=ADAPTER_BYTES)
    job_id = harness.run(provider, hyperparameters={"lora_r": 32})
    script = provider.script.decode("utf-8")
    assert job_id in script
    assert '"lora_r": 32' in script
    # Trainer sources travel as one payload rather than one round trip per file.
    assert len(provider.pushed) == 1


# --- teardown ordering: the bug this ticket fixes ---------------------------

def test_destroy_confirmation_precedes_the_terminal_state(harness):
    """A client that stops polling on a terminal status still sees teardown.

    Asserted explicitly because it is the regression: teardown used to run in a
    `finally` that executed after the terminal transition, so the one
    confirmation an operator most wants arrived after everyone stopped reading.
    """
    provider = FakeProvider(lines=TRAINING_LINES, result=RESULT,
                            adapter_bytes=ADAPTER_BYTES)
    job_id = harness.run(provider)

    events = harness.events(job_id)
    destroyed = index_of(events, lambda e: "destroyed" in (e["message"] or ""))
    assert destroyed < terminal_index(events)
    assert events[terminal_index(events)] is events[-1], \
        "the terminal state must be the last word on a job"


def test_teardown_confirmation_precedes_the_terminal_state_on_failure(harness):
    provider = FakeProvider(fail_at="push", fail_code="source_upload_failed")
    job_id = harness.run(provider)

    events = harness.events(job_id)
    assert harness.job(job_id)["status"] == "failed"
    destroyed = index_of(events, lambda e: "destroyed" in (e["message"] or ""))
    assert destroyed < terminal_index(events)


def test_teardown_runs_when_the_run_raises_unexpectedly(harness):
    """The path nobody anticipated still destroys the machine."""
    provider = FakeProvider(fail_at="await_ready", fail_unexpectedly=True)
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "internal_error"
    assert "RuntimeError" in job["error_message"] or \
        "exploded" in job["error_message"]
    assert provider.destroyed


def test_teardown_is_confirmed_by_listing_not_by_the_destroy_call(harness):
    provider = FakeProvider(lines=TRAINING_LINES, result=RESULT,
                            adapter_bytes=ADAPTER_BYTES)
    harness.run(provider)
    assert "list_machine_ids" in provider.calls, \
        "a destroy call's return value is a claim, not evidence"


def test_a_machine_that_survives_teardown_is_reported_loudly(harness):
    provider = FakeProvider(fail_at="push", destroy_failures=99,
                            stays_listed=True)
    job_id = harness.run(provider)

    errors = [e["message"] for e in harness.events(job_id)
              if e["kind"] == "error"]
    assert any("STRAY" in m and str(MACHINE_ID) in m for m in errors)
    assert provider.destroy_attempts > 1, "one transient error is not give-up"


def test_a_transient_destroy_failure_is_retried_rather_than_given_up_on(harness):
    """One flaky provider call must not be what leaves a machine billing."""
    provider = FakeProvider(fail_at="push", destroy_failures=1)
    job_id = harness.run(provider)

    assert provider.destroy_attempts == 2
    assert provider.destroyed
    errors = [e["message"] for e in harness.events(job_id)
              if e["kind"] == "error"]
    assert any("Destroy attempt failed" in m for m in errors)
    assert not any("STRAY" in m for m in errors)


def test_nothing_is_destroyed_when_no_machine_was_created(harness):
    provider = FakeProvider(fail_at="select_gpu",
                            fail_code="provider_capacity_unavailable")
    job_id = harness.run(provider)

    assert harness.job(job_id)["error_code"] == "provider_capacity_unavailable"
    assert "destroy" not in provider.calls


# --- failure at each stage --------------------------------------------------

@pytest.mark.parametrize("stage,code", [
    ("select_gpu", "provider_capacity_unavailable"),
    ("create", "provider_capacity_unavailable"),
    ("await_ready", "ssh_unreachable"),
    ("await_ready", "ssh_auth_failed"),
    ("push", "source_upload_failed"),
    ("stream", "training_failed"),
])
def test_failure_at_a_stage_fails_the_job_with_its_code(harness, stage, code):
    provider = FakeProvider(fail_at=stage, fail_code=code)
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == code
    assert job["error_message"]
    assert job["adapter_path"] is None
    # A machine only exists from `create` onwards; before that there is
    # nothing to tear down, and after it there always is.
    if stage in ("select_gpu", "create"):
        assert "destroy" not in provider.calls
    else:
        assert provider.destroyed, "a created machine is always destroyed"


def test_a_provider_that_stops_producing_output_fails_the_job(harness):
    """No result marker means no result, whatever the output said."""
    provider = FakeProvider(lines=TRAINING_LINES, result=RESULT, stop_after=2)
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "training_failed"
    assert provider.destroyed


def test_a_trainer_that_reports_failure_fails_the_job(harness):
    provider = FakeProvider(lines=["[10:00:01] BUILD FAILED"],
                            result={"ok": False, "stage": "build",
                                    "error": "no space left on device"})
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "training_failed"
    assert "no space left on device" in job["error_message"]


def test_a_failure_before_training_keeps_its_own_error_code(harness):
    """An unpacking failure is not a training failure.

    The machine reports the stage that failed and the code that names it;
    telling a user their training failed sends them to read the wrong logs.
    """
    provider = FakeProvider(
        lines=["[10:00:01] SOURCE UNPACK FAILED"],
        result={"stage": "source", "ok": False,
                "error_code": "source_upload_failed",
                "error": "The trainer sources did not unpack."})
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "source_upload_failed"


def test_a_corrupt_adapter_download_is_refused(harness):
    """The container reported a hash; a truncated transfer must not pass."""
    provider = FakeProvider(lines=TRAINING_LINES, result=RESULT,
                            adapter_bytes=b"truncated")
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "artifact_corrupt"


def test_an_unreadable_adapter_is_reported_without_failing_the_job(harness):
    provider = FakeProvider(lines=TRAINING_LINES, result=RESULT,
                            adapter_bytes=b"")
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "complete"
    assert job["adapter_path"] is None
    assert any("fetch failed" in (e["message"] or "").lower()
               for e in harness.events(job_id) if e["kind"] == "error")


# --- the default provider ---------------------------------------------------

def test_missing_credentials_fail_the_job_before_anything_is_provisioned(
        harness, monkeypatch):
    """No provider passed means the real one — which refuses without a key."""
    from api import config, orchestrator

    monkeypatch.setattr(config, "provider_credentials_present", lambda: False)
    monkeypatch.setattr(orchestrator, "launch", orchestrator.run_job)

    path = harness._tmp_path / "creds.jsonl"
    path.write_text("\n".join(json.dumps(chat(f"q{i}", f"a{i}"))
                              for i in range(12)), encoding="utf-8")
    with open(path, "rb") as f:
        ds = harness._client.post(
            "/v1/datasets", files={"file": (path.name, f)}).json()["id"]
    job_id = harness._client.post(
        "/v1/jobs", json={"dataset_id": ds}).json()["id"]

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "provider_unauthenticated"
    assert job["machine_id"] is None


def test_the_suite_refuses_to_build_a_real_provider_client():
    """The guard itself is tested: a test that escapes its fake must fail."""
    from api import config, provider

    original = config.provider_credentials_present
    config.provider_credentials_present = lambda: True
    try:
        with pytest.raises(AssertionError, match="real provider client"):
            provider.new_provider()
    finally:
        config.provider_credentials_present = original
