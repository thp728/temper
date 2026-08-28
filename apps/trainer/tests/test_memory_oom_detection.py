"""The trainer's half of memory-failure detection (issue #35).

A job that runs out of device memory retries automatically with the effective
batch preserved. For the control plane to know a failure was a memory
exhaustion -- and not, say, a divergence -- the trainer names it in the result
document: a genuine exhaustion carries the platform's `training_oom`, and a
deliberately caused one (the `oom` fault) keeps the fault surface's own
`simulated_oom`, so a deliberately broken run is never mistaken for a real one
(ADR-0051).

The trainer image cannot import `temper_core` (ADR-0010), so the value two
components must agree on is pinned equal by a test, exactly as FAULT_ENV is.
"""

from __future__ import annotations

import json

import entrypoint

from temper_core import memory_retry


def test_the_oom_code_is_defined_once_and_pinned_to_the_domain():
    """The trainer cannot import `temper_core` in the image, so the shared
    value is pinned by a test -- the FAULT_ENV pattern for issue #35's real
    out-of-memory code."""
    assert entrypoint.TRAINER_OOM_CODE == memory_retry.REAL_OOM_CODE
    assert entrypoint.TRAINER_OOM_CODE == "training_oom"


def test_is_oom_tail_recognises_a_cuda_out_of_memory_line():
    assert entrypoint.is_oom_tail(
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    )
    assert entrypoint.is_oom_tail(
        "torch.OutOfMemoryError: CUDA out of memory."
    )
    assert entrypoint.is_oom_tail("cuda runtime error (2): out of memory")


def test_is_oom_tail_does_not_recognise_a_cpu_exhaustion():
    """A host that ran out of RAM is not a memory retry: halving the batch
    does not add RAM, so it is not named as one."""
    assert not entrypoint.is_oom_tail("MemoryError: unable to allocate 8 GiB")
    assert not entrypoint.is_oom_tail("ran out of memory on the CPU")
    assert not entrypoint.is_oom_tail("loss: 0.5, step: 10")
    assert not entrypoint.is_oom_tail("")


def test_a_real_oom_carries_the_real_code():
    tail = [
        "{'loss': 0.5, 'step': 10, 'epoch': 0.2}",
        "RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB",
    ]
    assert entrypoint.oom_error_code(tail, None) == entrypoint.TRAINER_OOM_CODE


def test_a_deliberate_oom_keeps_the_faults_simulated_code():
    """A run broken on purpose keeps the fault surface's own code -- the
    marker that makes the deliberate break identifiable (ADR-0051)."""
    tail = ["RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB"]
    assert (
        entrypoint.oom_error_code(tail, {"name": "oom", "delay_s": 5})
        == "simulated_oom"
    )
    # A different fault (or no fault) on the same tail is a real OOM.
    assert (
        entrypoint.oom_error_code(tail, {"name": "divergence"})
        == entrypoint.TRAINER_OOM_CODE
    )


def test_a_failure_without_an_oom_line_is_not_named_as_one():
    tail = ["ValueError: unexpected tensor shape"]
    assert entrypoint.oom_error_code(tail, None) is None


def _job_dir(tmp_path):
    """A minimal job directory: one valid multi-row dataset and a job spec
    carrying the resolved hyperparameters the trainer refuses to invent."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(12)
    ]
    (job_dir / "dataset.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "job_test",
                "base_model": "Qwen/Qwen3-4B",
                "base_revision": "0" * 40,
                "hyperparameters": {
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.0,
                    "learning_rate": 2e-4,
                    "num_epochs": 3,
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "sequence_len": 2048,
                    "warmup_ratio": 0.1,
                    "lr_scheduler": "cosine",
                    "val_set_size": 0.05,
                    "save_total_limit": 3,
                    "lora_use_rslora": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return job_dir


def _drive_failed_run(tmp_path, monkeypatch, tail, fault_spec=None):
    """Drive main() through a failed training run, returning result.json."""
    job_dir = _job_dir(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(entrypoint, "JOB_DIR", job_dir)
    monkeypatch.setattr(entrypoint, "OUT_DIR", out_dir)
    monkeypatch.setattr(entrypoint, "CONFIG", out_dir / "config.yaml")
    monkeypatch.setattr(entrypoint, "RESULT", out_dir / "result.json")
    monkeypatch.setattr(entrypoint, "LOG", out_dir / "train.log")
    monkeypatch.setattr(entrypoint, "prefetch_model", lambda job: None)
    monkeypatch.setattr(
        entrypoint, "run_streaming", lambda cmd: (1, list(tail))
    )
    # The runtime half of the oom fault schedules a timer that exhausts real
    # device memory -- torch, base-image only -- which no host test can run.
    # Detection is what is under test here, so the runtime half is stubbed.
    monkeypatch.setattr(entrypoint, "schedule_fault", lambda spec: None)
    if fault_spec is not None:
        monkeypatch.setenv(entrypoint.FAULT_ENV, json.dumps(fault_spec))
    else:
        monkeypatch.delenv(entrypoint.FAULT_ENV, raising=False)
    entrypoint.main()
    return json.loads((out_dir / "result.json").read_text())


def test_a_failed_run_that_ran_out_of_memory_names_training_oom(
    tmp_path, monkeypatch
):
    result = _drive_failed_run(
        tmp_path,
        monkeypatch,
        [
            "{'loss': 0.5, 'step': 10, 'epoch': 0.2}",
            "RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB",
        ],
    )
    assert result["ok"] is not True
    assert result["error_code"] == entrypoint.TRAINER_OOM_CODE
    assert "out of device memory" in result["error"]
    assert "CUDA out of memory" in "\n".join(result["log_tail"])


def test_a_failed_run_under_the_oom_fault_keeps_the_simulated_code(
    tmp_path, monkeypatch
):
    result = _drive_failed_run(
        tmp_path,
        monkeypatch,
        ["RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB"],
        fault_spec={"name": "oom", "delay_s": 5},
    )
    assert result["error_code"] == "simulated_oom"
    # The deliberately broken run is named as such in its own record.
    assert result["simulated_fault"]["name"] == "oom"


def test_a_failed_run_that_did_not_oom_is_not_named_as_one(
    tmp_path, monkeypatch
):
    result = _drive_failed_run(
        tmp_path,
        monkeypatch,
        ["ValueError: unexpected tensor shape"],
    )
    assert result["ok"] is not True
    assert result.get("error_code") is None
    assert "out of device memory" not in result["error"]
