"""A failure the journeys can cause on demand, without hardware.

Spec 007's testing decisions name "land on a failed job and read the reason"
as one of the journeys that must run on every push. Nothing in main could
produce a failed job: ADR-0024's journey fake (`completed_run`) only ever
succeeds, and issue #24 -- product-level fault injection -- is still open.
This fills exactly that gap for the browser tier, in the same spirit as the
canned success: the simulated machine reads the job spec it was handed, and a
reserved hyperparameter asks it to end with a named code.

The reserved key is honoured nowhere else and documented nowhere as product
surface; #24 will replace it with something a user can be offered honestly.
"""

import json

from temper_control_plane.fake_provider import (
    DEMO_LINES,
    SIMULATED_FAILURE_KEY,
    _job_spec_from_script,
    completed_run,
)


def script_with(hyperparameters: dict) -> bytes:
    """A remote script carrying its jobspec, shaped like the real one."""
    spec = json.dumps(
        {
            "job_id": "job_test",
            "base_model": "Qwen/Qwen3-4B",
            "base_revision": "0" * 40,
            "hyperparameters": hyperparameters,
        },
        indent=2,
    )
    return (
        b"set -u\ncat > /tmp/job/job.json <<'JOBSPEC'\n"
        + spec.encode("utf-8")
        + b"\nJOBSPEC\nsudo docker run ghcr.io/example/trainer@sha256:0\n"
    )


def consume(provider, script: bytes):
    """Split a simulated stream into its log lines and result document."""
    lines, result_text, seen_marker = [], [], False
    for line in provider.stream(None, script):
        if line == "---RESULT---":
            seen_marker = True
        elif seen_marker:
            result_text.append(line)
        else:
            lines.append(line)
    assert seen_marker, "the simulated machine always reports a result"
    return lines, json.loads("\n".join(result_text))


def test_the_jobspec_is_read_out_of_the_script():
    spec = _job_spec_from_script(script_with({"lora_r": 16}))
    assert spec is not None
    assert spec["hyperparameters"] == {"lora_r": 16}


def test_a_script_without_a_jobspec_parses_to_none():
    assert _job_spec_from_script(b"echo nothing here") is None


def test_by_default_the_simulated_machine_completes():
    _, result = consume(completed_run(), script_with({}))
    assert result["ok"] is True


def test_the_machine_reports_the_failure_it_was_asked_for():
    provider = completed_run()
    log, result = consume(
        provider, script_with({SIMULATED_FAILURE_KEY: "gpu_stalled"})
    )
    assert result["ok"] is False
    assert result["error_code"] == "gpu_stalled"
    # This sentence is what the user reads under the code on the finished-job
    # page, so it states what happened rather than what was simulated.
    assert len(result["error"]) > 20
    # Output from before the failure stays in the history: the reason a job
    # died is usually read beside what it was last doing.
    assert log == list(DEMO_LINES[:1])


def test_launching_with_the_key_yields_a_failed_job_record(
    client, monkeypatch
):
    """End to end through the API: create with the key, and the record comes
    back failed with its stable code -- the shape the finished-job view
    renders."""
    from temper_control_plane import orchestrator

    payload = "\n".join(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            }
        )
        for i in range(12)
    )
    r = client.post(
        "/v1/datasets",
        files={
            "file": (
                "d.jsonl",
                payload.encode("utf-8"),
                "application/octet-stream",
            )
        },
    )
    ds_id = r.json()["id"]

    from helpers import wait_validated

    wait_validated(client, ds_id)  # the job needs the finished report

    def start(job_id):
        from temper_control_plane import fake_models

        orchestrator.run_job(
            job_id,
            provider=completed_run(),
            models=fake_models.catalog_models(),
        )

    monkeypatch.setattr(orchestrator, "launch", start)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds_id,
            "hyperparameters": {SIMULATED_FAILURE_KEY: "gpu_stalled"},
        },
    )
    assert r.status_code == 201, r.text
    record = r.json()
    assert record["status"] == "failed"
    assert record["error_code"] == "gpu_stalled"
    assert record["error_message"]
    # The leak ADR-0023 removed stays removed on every path.
    assert "adapter_path" not in record
