"""Resolution happens before launch, once, outside this trainer.

Issue #83. The trainer used to hold its own `DEFAULTS` and resolve overrides a
second time after the control plane already had -- two resolvers that could
disagree, with the one that ran being the one nobody could see. Now the job
specification arrives carrying the full resolved set, and these tests pin the
remaining half of the contract: the trainer applies exactly what it is given,
invents nothing when something is missing, and still refuses unknown keys at
both levels.
"""

from __future__ import annotations

import entrypoint
import pytest

from temper_core import hyperparams


def complete_job(**hp_changes) -> dict:
    """A job spec shaped like the one the control plane writes at launch.

    Built on the resolver's own output rather than retyped literals, so this
    fixture cannot drift from what actually launches.
    """
    hp = hyperparams.effective({})
    hp.update(hp_changes)
    return {
        "job_id": "example-0001",
        "base_model": "Qwen/Qwen3-4B",
        "hyperparameters": hp,
    }


def test_every_value_comes_from_the_spec():
    job = complete_job(learning_rate=9e-5, num_epochs=7)
    cfg, _rejected = entrypoint.build_config(job)
    assert cfg["learning_rate"] == 9e-5
    assert cfg["num_epochs"] == 7
    assert cfg["lora_r"] == 16


def test_a_missing_required_value_fails_loudly():
    """No silent fallback: a value the spec omits stops the job here, naming
    the key, rather than training with a number nobody chose."""
    job = complete_job()
    del job["hyperparameters"]["sequence_len"]
    with pytest.raises(entrypoint.IncompleteJobSpec) as excinfo:
        entrypoint.build_config(job)
    assert "sequence_len" in str(excinfo.value)


def test_a_missing_hyperparameters_block_fails_loudly():
    with pytest.raises(entrypoint.IncompleteJobSpec):
        entrypoint.build_config({"job_id": "j", "base_model": "m"})


def test_the_trainer_derives_nothing():
    """α tracking r and rsLoRA-at-high-rank are resolver decisions. Whatever
    the spec pairs together is what trains, even when this code can see it is
    a strange pairing -- second-guessing the spec is the second resolver back."""
    job = complete_job(lora_r=64, lora_alpha=16, lora_use_rslora=False)
    cfg, _rejected = entrypoint.build_config(job)
    assert cfg["lora_alpha"] == 16
    assert cfg["lora_use_rslora"] is False


def test_unknown_hyperparameter_keys_are_refused_and_echoed():
    job = complete_job(maxSteps=5)
    cfg, rejected = entrypoint.build_config(job)
    assert "maxSteps" not in cfg
    assert rejected == {"maxSteps": 5}


def test_unknown_top_level_keys_are_refused_and_echoed():
    job = complete_job()
    job["maxSteps"] = 5
    cfg, rejected = entrypoint.build_config(job)
    assert "maxSteps" not in cfg
    assert rejected == {"maxSteps": 5}


def test_optional_keys_still_apply_when_present():
    job = complete_job()
    job["max_steps"] = 4
    job["resume_from_checkpoint"] = "/out/run/checkpoint-2"
    cfg, rejected = entrypoint.build_config(job)
    assert cfg["max_steps"] == 4
    assert cfg["resume_from_checkpoint"] == "/out/run/checkpoint-2"
    assert rejected == {}
