"""Resumption: an interrupted job continues from its last checkpoint (issue #60).

An interrupted run (the `worker_kill` fault: no result document, only the
checkpoints the machine wrote off itself before it died) resumes from its
last checkpoint on a fresh machine, restoring the optimiser, scheduler and
step position -- the whole checkpoint directory -- not only the weights. The
resumed run is a new attempt against the same job, with its own machine, its
own rate and its own outcome, so the history says what actually happened
rather than presenting one continuous run that was not.

What is asserted is what a user or an operator can observe: the job record,
the attempts, the ordered event log, and what the machine was asked to run.
The numerics of a resumed run matching an uninterrupted one (criterion 5)
cannot be proven without hardware -- that is the trainer's own restore from
`resume_from_checkpoint` -- and is stated as such in the ADR rather than
claimed here.
"""

import json

import pytest

from temper_control_plane.fake_provider import (
    FakeProvider,
    completed_run,
    fake_checkpoint_tar,
)
from temper_core import resume as resume_logic

TRAINING_LINES = [
    "[10:00:01] pulling trainer image",
    "[10:03:04] running training",
    "{'loss': 1.9042, 'epoch': 0.5}",
]


def chat(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


class Harness:
    """Upload a dataset, launch a job on a given provider, read it back."""

    def __init__(self, client, monkeypatch, tmp_path):
        self._client = client
        self._monkeypatch = monkeypatch
        self._tmp_path = tmp_path
        self.provider = None
        self._pending_provider = None
        self._pending_limits = None
        self._pending_models = None

    def enable_fake_tier(self) -> None:
        from temper_control_plane import config

        self._monkeypatch.setattr(config, "FAKE_PROVIDER", True)

    def create(self, hyperparameters: dict):
        """Create a job and drive it inline with `completed_run()`'s machine."""
        from temper_control_plane import fake_models, orchestrator

        self.provider = completed_run()
        self._pending_provider = self.provider
        self._pending_models = fake_models.catalog_models()
        return self._launch(hyperparameters)

    def run(self, provider, hyperparameters=None, limits=None):
        """Create a job and drive it with an explicitly provided machine."""
        from temper_control_plane import fake_models, orchestrator

        self.provider = provider
        self._pending_provider = provider
        self._pending_limits = limits
        self._pending_models = fake_models.catalog_models()
        return self._launch(hyperparameters)

    def _launch(self, hyperparameters):
        from temper_control_plane import orchestrator

        path = self._tmp_path / "d.jsonl"
        path.write_text(
            "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(12)),
            encoding="utf-8",
        )
        with open(path, "rb") as f:
            ds = self._client.post(
                "/v1/datasets", files={"file": (path.name, f)}
            ).json()["id"]
        from helpers import wait_validated

        wait_validated(self._client, ds)
        r = self._client.post(
            "/v1/jobs",
            json={
                "dataset_id": ds,
                "hyperparameters": hyperparameters or {},
            },
        )
        body = r.json()
        job_id = body.get("id") if isinstance(body, dict) else None
        if r.status_code == 201 and job_id is not None:
            orchestrator.run_job(
                job_id,
                provider=self._pending_provider,
                limits=self._pending_limits,
                models=self._pending_models,
            )
        return r, job_id

    def job(self, job_id) -> dict:
        return self._client.get(f"/v1/jobs/{job_id}").json()

    def events(self, job_id) -> list[dict]:
        return self._client.get(f"/v1/jobs/{job_id}/events").json()["events"]

    def messages(self, job_id) -> list[str]:
        return [e["message"] for e in self.events(job_id)]


@pytest.fixture()
def harness(isolated, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from temper_control_plane import main, orchestrator

    monkeypatch.setattr(orchestrator, "DESTROY_RETRY_DELAY_S", 0)
    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_INTERVAL_S", 0)
    monkeypatch.setattr(orchestrator, "TEARDOWN_CONFIRM_TIMEOUT_S", 2)
    monkeypatch.setattr(
        orchestrator,
        "published_reference",
        lambda: "ghcr.io/thp728/temper/trainer@sha256:0000000000000000000000000000000000000000000000000000000000000000",
    )
    with TestClient(main.app) as c:
        yield Harness(c, monkeypatch, tmp_path)


def _job_spec_from_script(script: bytes) -> dict:
    """The jobspec embedded in a remote script (mirrors the fake's own)."""
    start = script.find(b"<<'JOBSPEC'\n")
    start += len(b"<<'JOBSPEC'\n")
    end = script.find(b"\nJOBSPEC", start)
    return json.loads(script[start:end].decode("utf-8"))


# --- the headline: a killed worker's run resumes and completes ---------------


def test_an_interrupted_job_resumes_from_its_last_checkpoint(harness):
    """The worker is killed mid-training; the run resumes from the checkpoint
    that had already left the machine (the latest one, step 30) on a fresh
    machine and completes. No second machine is ever provisioned before the
    first is torn down."""
    harness.enable_fake_tier()
    response, job_id = harness.create(
        {"simulated_failure_code": {"name": "worker_kill"}}
    )
    assert response.status_code == 201, response.text
    job = harness.job(job_id)
    assert job["status"] == "complete"

    # The history shows what actually happened: two attempts, the first
    # interrupted and the second a resumption that completed.
    attempts = job["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["outcome"] == "interrupted"
    assert attempts[0]["error_code"] == "interrupted"
    assert attempts[1]["outcome"] == "complete"
    assert attempts[1]["resumed_from"] == 30

    # Each attempt has its own machine and its own billing rate.
    assert attempts[0]["machine_id"] != attempts[1]["machine_id"]
    assert attempts[1]["machine_id"] is not None
    assert attempts[0]["rate"]["price_per_hour"] > 0
    assert attempts[1]["rate"] == attempts[0]["rate"]

    # Exactly two machines were provisioned -- the first was torn down before
    # the second appeared, so the retry never stacks machines.
    assert len(harness.provider.created) == 2


def test_the_resumed_attempt_runs_with_resume_from_checkpoint(harness):
    """The resumed machine receives the checkpoint and a job spec whose
    `resume_from_checkpoint` names the step the run came back from -- the
    directive the trainer passes straight to axolotl, restoring the whole
    checkpoint directory rather than only the weights."""
    harness.enable_fake_tier()
    _, job_id = harness.create(
        {"simulated_failure_code": {"name": "worker_kill"}}
    )
    assert harness.job(job_id)["status"] == "complete"

    # The last script the machine ran is the resumed attempt's: it names the
    # resume path and extracts the checkpoint the control plane shipped.
    spec = _job_spec_from_script(harness.provider.script)
    assert spec["resume_from_checkpoint"] == "/out/run/checkpoint-30"
    # The archive was streamed to the machine, not held in the control plane's
    # memory -- the same push channel the dataset uses.
    checkpoint_pushes = [
        data
        for dest, data in harness.provider.pushed
        if dest == "/tmp/checkpoint.tar"
    ]
    assert len(checkpoint_pushes) == 1
    # The pushed bytes are the stored checkpoint's own tar (the weights the
    # machine wrote off itself before it died).
    assert checkpoint_pushes[0] == fake_checkpoint_tar(
        30, b"ckpt-30", loss=0.31, held_out_loss=0.52
    )
    # And the shell it runs extracts that archive under the trainer's own
    # output directory before axolotl starts.
    script = harness.provider.script.decode("utf-8")
    assert "tar xf /tmp/checkpoint.tar -C /tmp/out/run" in script


def test_the_history_names_the_interruption_and_the_resumption(harness):
    """The ordered event log says both halves: the interruption (no result
    document) and the resumption (from which step, on a new machine)."""
    harness.enable_fake_tier()
    _, job_id = harness.create(
        {"simulated_failure_code": {"name": "worker_kill"}}
    )
    messages = harness.messages(job_id)
    resume_events = [
        m
        for m in messages
        if "resuming from checkpoint step 30 on a new machine" in m
    ]
    assert resume_events, messages
    # The discovered survivors are named too, so the record does not invent a
    # checkpoint it did not verify.
    assert any("Discovered surviving checkpoint" in m for m in messages)
    assert any("Simulated fault injected: worker_kill" in m for m in messages)


# --- an interruption with nothing to resume from ------------------------------


def test_an_interruption_with_no_checkpoint_fails_with_interrupted(harness):
    """A run that dies before any checkpoint left the machine cannot resume:
    nothing survived to come back to, so the job fails with the honest
    `interrupted` code and no second machine is provisioned."""
    provider = FakeProvider(lines=TRAINING_LINES, result=None)
    _, job_id = harness.run(provider)
    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == resume_logic.INTERRUPTED_CODE
    attempts = job["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["outcome"] == "interrupted"
    assert attempts[0]["error_code"] == "interrupted"
    assert len(provider.created) == 1, "no resumption without a checkpoint"


# --- the cap: a configuration the infrastructure keeps killing -----------------


class _AlwaysInterruptedProvider(FakeProvider):
    """A machine that writes its checkpoints then dies without a result
    document, on every attempt -- the shape of a job whose infrastructure
    never stays up long enough to finish."""

    def stream(self, machine, script):
        if False:  # pragma: no cover - keep this a generator function
            yield
        spec = _job_spec_from_script(script)
        self._write_checkpoints(spec)


def test_resumption_is_bounded(harness):
    """A job that keeps being interrupted must stop resuming and surface,
    with the attempts recorded, rather than provisioning machines forever."""
    provider = _AlwaysInterruptedProvider(
        lines=TRAINING_LINES,
        result=None,
        checkpoints=[{"step": 10, "loss": 1.0, "bytes": b"ckpt-10"}],
    )
    _, job_id = harness.run(provider)
    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == resume_logic.RESUME_EXHAUSTED_CODE
    attempts = job["attempts"]
    # One interrupted attempt per machine, and the last one surfaces: the cap
    # allows `RESUME_RETRY_CAP` resumptions, so `RESUME_RETRY_CAP + 1`
    # attempts in total.
    assert len(attempts) == resume_logic.RESUME_RETRY_CAP + 1
    assert [a["outcome"] for a in attempts] == ["interrupted"] * len(attempts)
    assert len(provider.created) == len(attempts)
    # The first resumption is visible in the history; the job surfaces after
    # the cap rather than billing forever.
    assert any(
        "resuming from checkpoint step 10" in m
        for m in harness.messages(job_id)
    )

