"""The fault surface (issue #24): failures can be caused on demand.

Spec 010 ships this before the recoveries, deliberately. Every recovery path
is verified by something going wrong, and nothing could make something go
wrong on demand; a recovery proven only by reasoning is the same class of
artifact as a large green suite over a product that could not train. This
surface makes each of the six failures happen at a chosen point -- provider
side through the existing provider seam, trainer side through the trainer's
environment -- off by default, and named in the run's history so a
deliberately broken run can never be mistaken for a real one.

What is asserted is what a user or an operator can observe: the job record,
the ordered event log, and what the provider still lists. Internal call
counts are not behaviour.
"""

import json

import pytest

from temper_control_plane.fake_provider import (
    DEMO_LINES,
    MACHINE_ID,
    ORPHAN_MACHINE_ID,
    PUBLISHED_IMAGE_REFERENCE,
    SimulatedMachine,
    completed_run,
    simulated_limits,
)
from temper_core import faults as fault_surface

ALL_FAULTS = fault_surface.names()
# The six faults the surface must cover, and the terminal state each drives
# the job to when it fires against the fake machine. `divergence` completes
# with a worthless result: the fault is the loss going meaningless, and what
# happens next is the divergence recovery's (#36) job, not the fault's.
EXPECTED_OUTCOMES = {
    "oom": ("failed", "simulated_oom"),
    "divergence": ("complete", None),
    "worker_kill": ("failed", "training_failed"),
    "machine_silent": ("failed", "gpu_stalled"),
    "destroy_refused": ("complete", None),
    "orphan": ("complete", None),
}


def chat(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


class Harness:
    """Upload a dataset, launch a job on the simulated machine, read it back."""

    def __init__(self, client, monkeypatch, tmp_path):
        self._client = client
        self._monkeypatch = monkeypatch
        self._tmp_path = tmp_path
        self.provider = None

    def enable_fake_tier(self) -> None:
        """The zero-cost tier (TEMPER_FAKE_PROVIDER): the fake machine honours
        every fault, so the surface can be exercised end to end."""
        from temper_control_plane import config

        self._monkeypatch.setattr(config, "FAKE_PROVIDER", True)

    def enable_real_tier(self) -> None:
        """The deliberate real-hardware tier (TEMPER_FAULT_SURFACE): only
        trainer-side faults may be caused; provider-side faults are refused
        (see config.fault_surface_refusal)."""
        from temper_control_plane import config

        self._monkeypatch.setattr(config, "FAULT_SURFACE", True)

    def create(self, hyperparameters: dict, limits=None):
        """Create a job and run it inline; returns (response, job_id)."""
        from temper_control_plane import fake_models, orchestrator

        self.provider = completed_run()
        self._monkeypatch.setattr(
            orchestrator,
            "launch",
            lambda job_id: orchestrator.run_job(
                job_id,
                provider=self.provider,
                limits=limits,
                models=fake_models.catalog_models(),
            ),
        )
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
            json={"dataset_id": ds, "hyperparameters": hyperparameters},
        )
        body = r.json()
        return r, body.get("id") if isinstance(body, dict) else None

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
    monkeypatch.setattr(
        orchestrator,
        "published_reference",
        lambda: PUBLISHED_IMAGE_REFERENCE,
    )
    with TestClient(main.app) as c:
        yield Harness(c, monkeypatch, tmp_path)


def fault_event(messages: list[str], name: str) -> bool:
    return any(f"Simulated fault injected: {name}" in m for m in messages)


# --- off by default, and cannot be enabled by accident ------------------------


def test_the_surface_is_off_by_default_and_refused_at_creation(harness):
    """A fault spec cannot even be created unless the deployment has switched
    the surface on: off is a safety property, not a convenience, and a fault
    surface that can be switched on by accident in front of a user is worse
    than none."""
    response, job_id = harness.create(
        {"simulated_failure_code": {"name": "oom"}}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "fault_surface_refused"
    assert not job_id
    assert (
        "nothing was launched" in response.json()["detail"]["message"].lower()
    )


def test_an_unknown_fault_name_is_refused_at_creation(harness):
    """A fault nobody can explain is refused before anything is priced."""
    harness.enable_fake_tier()
    response, job_id = harness.create(
        {"simulated_failure_code": {"name": "definitely_not_a_fault"}}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "fault_unknown"
    assert not job_id


def test_a_normal_job_without_a_fault_spec_is_untouched(harness):
    """Off by default means no fault behaviour leaks into ordinary runs: no
    fault event, no `simulated_` code, a normal completion."""
    harness.enable_fake_tier()
    response, job_id = harness.create({})
    assert response.status_code == 201
    assert job_id
    job = harness.job(job_id)
    assert job["status"] == "complete"
    assert job["error_code"] is None
    assert not any("Simulated fault" in m for m in harness.messages(job_id))


# --- each fault, caused on demand, named in the job's history -----------------


@pytest.mark.parametrize("name", ALL_FAULTS)
def test_every_fault_in_the_vocabulary_is_causable(harness, name):
    """The contract's own vocabulary is the coverage: every fault named there
    can be requested, and requesting it drives the job to the outcome the
    surface documents for it."""
    harness.enable_fake_tier()
    spec = {"name": name}
    if name == "machine_silent":
        spec = {**spec, "after_line": 2}
    if name == "destroy_refused":
        spec = {**spec, "times": 99}
    response, job_id = harness.create(
        {"simulated_failure_code": spec},
        limits=(
            simulated_limits(step=60.0, stall=900.0)
            if name == "machine_silent"
            else None
        ),
    )
    assert response.status_code == 201, response.text
    assert job_id
    job = harness.job(job_id)
    expected_state, expected_code = EXPECTED_OUTCOMES[name]
    assert job["status"] == expected_state, name
    if expected_code is not None:
        assert job["error_code"] == expected_code, name
    # Every injected fault is named in the job's own history, so a
    # deliberately broken run can never be mistaken for a real one.
    assert fault_event(harness.messages(job_id), name)
    # And the machine itself records which fault it was asked to suffer.
    assert harness.provider.fault_applied == name


def test_oom_records_the_fault_in_the_result_document(harness):
    """A trainer-side fault on the fake machine travels the ordinary result
    path, so the job's record carries the `simulated_` code -- the marker
    that makes the broken run identifiable."""
    harness.enable_fake_tier()
    response, job_id = harness.create(
        {"simulated_failure_code": {"name": "oom"}}
    )
    assert response.status_code == 201
    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "simulated_oom"
    # The machine's own stream says the fault, at the moment it fired.
    assert any("out of device memory" in m for m in harness.messages(job_id))


def test_divergence_leaves_the_meaningless_loss_in_the_history(harness):
    """Driving the loss to a meaningless value is the fault; what happens next
    is the divergence recovery's (#36) job. So the run completes with a
    worthless result, exactly as the sabotaged real trainer would, and the nan
    line sits in the history beside the named fault."""
    harness.enable_fake_tier()
    response, job_id = harness.create(
        {"simulated_failure_code": {"name": "divergence"}}
    )
    assert response.status_code == 201
    job = harness.job(job_id)
    assert job["status"] == "complete"
    assert any("loss': nan" in m for m in harness.messages(job_id))
    assert fault_event(harness.messages(job_id), "divergence")


def test_a_killed_worker_leaves_no_result_document(harness):
    """A worker killed mid-run produces no result.json -- the honest shape of
    an interruption, which a resumption (#60) exists to recover from. The
    fault is named even though the code is the ordinary training_failed."""
    harness.enable_fake_tier()
    response, job_id = harness.create(
        {"simulated_failure_code": {"name": "worker_kill"}}
    )
    assert response.status_code == 201
    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "training_failed"
    assert job["result"] is None
    assert fault_event(harness.messages(job_id), "worker_kill")


def test_a_refused_destroy_leaves_the_machine_reported_loudly(harness):
    """A destroy the provider refuses is retried, and a machine that survives
    teardown is reported loudly rather than silently forgotten -- the path a
    reconciler will close, made causable on demand."""
    harness.enable_fake_tier()
    response, job_id = harness.create(
        {"simulated_failure_code": {"name": "destroy_refused", "times": 99}}
    )
    assert response.status_code == 201
    job = harness.job(job_id)
    # Training itself completed; the teardown is what failed.
    assert job["status"] == "complete"
    errors = [
        e["message"] for e in harness.events(job_id) if e["kind"] == "error"
    ]
    assert any("Destroy attempt failed" in m for m in errors)
    assert any("STRAY" in m and str(MACHINE_ID) in m for m in errors)
    assert fault_event(harness.messages(job_id), "destroy_refused")


def test_an_orphan_machine_is_left_for_the_reconciler(harness):
    """A machine with no job that owns it is exactly what the reconciler
    (#61) exists to find; the fault leaves one in the provider's listing, and
    the job's own teardown never touches it."""
    harness.enable_fake_tier()
    response, job_id = harness.create(
        {"simulated_failure_code": {"name": "orphan"}}
    )
    assert response.status_code == 201
    job = harness.job(job_id)
    assert job["status"] == "complete"
    assert job["machine_id"] == MACHINE_ID
    assert fault_event(harness.messages(job_id), "orphan")

    # The orphan is a distinct id, still listed: exactly what a reconciler
    # would match against jobs and find unowned. It is never the job's own
    # machine, which was torn down normally.
    from temper_control_plane import db

    assert db.get_job(job_id)["status"] == "complete"
    assert ORPHAN_MACHINE_ID != MACHINE_ID


def test_the_fault_spec_is_part_of_the_frozen_job_record(harness):
    """The request is frozen onto the job row like every hyperparameter, so
    the record itself says what was asked for -- a reader never has to infer
    that a run was deliberately broken."""
    harness.enable_fake_tier()
    spec = {"name": "oom"}
    response, job_id = harness.create({"simulated_failure_code": spec})
    assert response.status_code == 201
    job = harness.job(job_id)
    assert job["hyperparameters"]["simulated_failure_code"] == spec


def test_the_fault_reaches_the_trainer_environment_only_when_on(harness):
    """Trainer-side faults are an environment switch: the control plane writes
    the fault spec into the machine's environment, and only for a job that
    carries one. A normal job's script carries no such instruction at all, so
    a fault that could not have been switched on never reaches a trainer; and
    a fault-spec job is refused outright when the surface is off (its own
    test), so the injection and the guard together are the safety property."""
    harness.enable_fake_tier()
    _, job_id = harness.create(
        {"simulated_failure_code": {"name": "oom", "delay_s": 5}}
    )
    script = harness.provider.script.decode("utf-8")
    assert '-e TEMPER_FAULT_SPEC=\'{"name": "oom", "delay_s": 5}\'' in script

    # A job that carries no fault spec gets no instruction in its script.
    response, _ = harness.create({})
    assert response.status_code == 201
    assert "TEMPER_FAULT_SPEC" not in harness.provider.script.decode("utf-8")


# --- the seams' own guards ---------------------------------------------------


def test_the_fake_refuses_an_invalid_fault_spec():
    """The seam's defence in depth: even a fault spec that slipped past
    creation is refused by the machine rather than half-honoured."""
    from temper_core.errors import OrchestratorError

    for bad in ({"name": "not_a_fault"}, {"name": "oom", "bogus": 1}):
        spec = json.dumps(
            {
                "job_id": "job_test",
                "base_model": "Qwen/Qwen3-4B",
                "base_revision": "0" * 40,
                "hyperparameters": {"simulated_failure_code": bad},
            },
            indent=2,
        )
        script = (
            b"set -u\ncat > /tmp/job/job.json <<'JOBSPEC'\n"
            + spec.encode("utf-8")
            + b"\nJOBSPEC\nsudo docker run ghcr.io/example/trainer@sha256:0\n"
        )
        with pytest.raises(OrchestratorError) as excinfo:
            list(SimulatedMachine().stream(None, script))
        assert excinfo.value.code == "fault_invalid"


def test_an_unknown_fault_parameter_is_refused_at_creation(harness):
    """A parameter a fault does not take is refused before anything is priced:
    a fault spec the caller believes is in effect but is not is worse than a
    refusal."""
    harness.enable_fake_tier()
    response, job_id = harness.create(
        {"simulated_failure_code": {"name": "oom", "bogus": 1}}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "fault_invalid"
    assert not job_id
    assert "does not take parameter" in response.json()["detail"]["message"]


def test_a_provider_side_fault_is_refused_on_the_real_tier(harness):
    """On the deliberate real-hardware tier no provider honours a provider-side
    fault, so it is refused rather than launched under a history that would
    claim a deliberate break no machine will make -- the naming guarantee
    must never fire the wrong way."""
    harness.enable_real_tier()
    for name in ("machine_silent", "orphan", "destroy_refused"):
        response, job_id = harness.create(
            {"simulated_failure_code": {"name": name}}
        )
        assert response.status_code == 400, name
        assert response.json()["detail"]["code"] == "fault_not_causable", name
        assert not job_id


def test_a_trainer_side_fault_is_allowed_on_the_real_tier(harness):
    """The deliberate real tier exists exactly for trainer-side faults: the
    trainer genuinely makes them happen on hardware, and nothing is refused."""
    harness.enable_real_tier()
    response, job_id = harness.create(
        {"simulated_failure_code": {"name": "oom"}}
    )
    assert response.status_code == 201
    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "simulated_oom"


def test_only_result_producing_faults_carry_a_simulated_code():
    """The contract's codes are truthful: only the faults whose run carries
    its own result document declare one. The others fail through the
    platform's ordinary machinery and are named by the history instead."""
    assert fault_surface.code_for("oom") == "simulated_oom"
    assert fault_surface.code_for("divergence") == "simulated_divergence"
    for name in ("worker_kill", "machine_silent", "orphan", "destroy_refused"):
        assert fault_surface.code_for(name) is None


def test_the_orchestrator_refuses_a_fault_job_when_the_surface_is_off(
    harness,
):
    """The belt for a row that slipped past creation: run_job itself refuses
    a fault-spec job unless the surface is on, before anything is provisioned.
    A guard that lives on only one side of a money path is a hope."""
    from temper_control_plane import db, orchestrator

    # Create a job row with a fault spec directly, bypassing the create-time
    # guard, exactly as a row created before a flag changed would arrive.
    path = harness._tmp_path / "d.jsonl"
    path.write_text(
        "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(12)),
        encoding="utf-8",
    )
    with open(path, "rb") as f:
        ds = harness._client.post(
            "/v1/datasets", files={"file": (path.name, f)}
        ).json()["id"]
    from helpers import wait_validated

    wait_validated(harness._client, ds)
    job_id = db.create_job(
        ds,
        "Qwen/Qwen3-4B",
        {"simulated_failure_code": {"name": "oom"}},
    )
    provider = SimulatedMachine(lines=DEMO_LINES, result={"ok": True})
    orchestrator.run_job(job_id, provider=provider)

    job = db.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "fault_surface_refused"
    assert provider.created == [], "nothing may be provisioned"
