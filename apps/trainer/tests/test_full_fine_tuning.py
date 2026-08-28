"""Full fine-tuning's trainer side (issue #66): config and artifact.

A full fine-tune trains every weight, so its Axolotl config configures no
PEFT adapter and no 4-bit quantisation, and its artifact is a whole model
directory rather than an adapter pair. Both are asserted here against canned
specs and directories, the same way the adapter paths are -- the machine is
not needed to pin the shape this entrypoint produces.
"""

from __future__ import annotations

import hashlib
import json
import tarfile

import entrypoint
import pytest

from temper_core import hyperparams


def full_job(**hp_changes) -> dict:
    """A full fine-tune job spec shaped like the one the control plane writes:
    the method-resolved hyperparameters (its own learning rate) beside the
    method the provisioning-time selection chose."""
    hp = hyperparams.effective({}, method="full")
    hp.update(hp_changes)
    return {
        "job_id": "example-full-0001",
        "base_model": "Qwen/Qwen3-4B",
        "method": "full",
        "hyperparameters": hp,
    }


def test_a_full_fine_tune_configures_no_adapter():
    """The config for method=full carries no adapter, no 4-bit quantisation
    and no rank/scale claims: it trains every weight in bf16."""
    cfg, rejected = entrypoint.build_config(full_job())
    assert rejected == {}
    assert "adapter" not in cfg
    assert cfg["load_in_4bit"] is False
    for key in entrypoint.ADAPTER_ONLY_HYPERPARAMETERS:
        assert key not in cfg
    assert cfg["bf16"] is True
    assert "lora_target_linear" not in cfg


def test_a_qlora_job_still_configures_the_adapter():
    """The adapter shape is untouched for qlora (and for a pre-method spec,
    which reads as qlora): the two methods are different configs, not a
    second resolver guessing."""
    hp = hyperparams.effective({}, method="qlora")
    job = {
        "job_id": "j",
        "base_model": "Qwen/Qwen3-4B",
        "method": "qlora",
        "hyperparameters": hp,
    }
    cfg, _rejected = entrypoint.build_config(job)
    assert cfg["adapter"] == "qlora"
    assert cfg["load_in_4bit"] is True
    assert cfg["lora_r"] == 16


def test_a_spec_without_a_method_reads_as_qlora():
    """A job spec written before the method key existed runs qlora -- the
    only method that ever ran before it, the same reading
    `temper_core.artifacts.kind_for` gives a missing method."""
    hp = hyperparams.effective({})
    job = {"job_id": "j", "base_model": "Qwen/Qwen3-4B", "hyperparameters": hp}
    cfg, _rejected = entrypoint.build_config(job)
    assert cfg["adapter"] == "qlora"


def test_an_unknown_method_is_refused_not_run_as_qlora():
    """A missing method reads as qlora, but a present-and-unknown method is
    refused loudly: running it as qlora would be a run that lies about what
    it did -- the same refusal `temper_core.artifacts.kind_for` gives an
    unknown method string."""
    hp = hyperparams.effective({})
    for bogus in ("ful", "lora", "garbage"):
        job = {
            "job_id": "j",
            "base_model": "Qwen/Qwen3-4B",
            "method": bogus,
            "hyperparameters": hp,
        }
        with pytest.raises(entrypoint.IncompleteJobSpec) as excinfo:
            entrypoint.build_config(job)
        assert bogus in str(excinfo.value)


def test_a_full_job_uses_the_specs_own_learning_rate():
    """The resolver's method-specific value (issue #66) reaches the config
    untouched: the trainer applies what it is given and derives nothing."""
    cfg, _rejected = entrypoint.build_config(full_job())
    assert (
        cfg["learning_rate"]
        == hyperparams.effective({}, method="full")["learning_rate"]
    )


def test_a_full_model_artifact_is_the_tarred_model_directory(
    tmp_path, monkeypatch
):
    """collect_artifacts(method="full") tars the trained model's directory --
    the final model at the top of the output directory -- into one archive
    with a `model/` top-level folder, and records the archive's own checksum
    and member list."""
    out = tmp_path / "out"
    run = out / "run"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"architectures": ["Qwen3"]}))
    weights = b"\x00\xff" * (2 << 20)
    (run / "model.safetensors").write_bytes(weights)
    (run / "optimizer.pt").write_bytes(b"training state, not the model")
    (run / "trainer_state.json").write_text("{}")
    monkeypatch.setattr(entrypoint, "OUT_DIR", out)

    info = entrypoint.collect_artifacts("full")

    assert info["artifact_path"] == "model.tar.gz"
    assert info["artifact_format"] == "tar.gz"
    assert info["artifact_source"] == "final"
    assert info["artifact_members"] == ["config.json", "model.safetensors"]
    tar_path = out / "model.tar.gz"
    assert info["artifact_bytes"] == tar_path.stat().st_size
    assert (
        info["artifact_sha256"]
        == hashlib.sha256(tar_path.read_bytes()).hexdigest()
    )
    with tarfile.open(tar_path, "r:gz") as tar:
        names = sorted(tar.getnames())
    assert names == ["model/config.json", "model/model.safetensors"]


def test_a_full_model_artifact_prefers_the_final_model_over_checkpoints(
    tmp_path, monkeypatch
):
    """Same explicit rule as the adapter: the end-of-training model wins;
    failing that, the numerically highest checkpoint."""
    out = tmp_path / "out"
    run = out / "run"
    ckpt = run / "checkpoint-3"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"ckpt3")
    monkeypatch.setattr(entrypoint, "OUT_DIR", out)

    info = entrypoint.collect_artifacts("full")
    assert info["artifact_source"] == "checkpoint-3"

    (run / "config.json").write_text("{}")
    (run / "model.safetensors").write_bytes(b"final")
    info = entrypoint.collect_artifacts("full")
    assert info["artifact_source"] == "final"


def test_no_full_model_means_no_artifact(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "run").mkdir(parents=True)
    monkeypatch.setattr(entrypoint, "OUT_DIR", out)
    info = entrypoint.collect_artifacts("full")
    assert "artifact_path" not in info


def test_a_full_model_artifact_hashes_without_holding_the_file(
    tmp_path, monkeypatch
):
    """The full-model archive is hashed in bounded blocks like the adapter's:
    a whole model is larger than any adapter, and the flat-memory rule holds
    here or nowhere."""
    out = tmp_path / "out"
    run = out / "run"
    run.mkdir(parents=True)
    payload = b"\x00\xff" * (2 << 20)
    (run / "config.json").write_text("{}")
    (run / "model.safetensors").write_bytes(payload)
    monkeypatch.setattr(entrypoint, "OUT_DIR", out)

    info = entrypoint.collect_artifacts("full")
    with (out / "model.tar.gz").open("rb") as f:
        assert info["artifact_sha256"] == hashlib.sha256(f.read()).hexdigest()
