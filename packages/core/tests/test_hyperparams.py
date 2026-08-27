"""Tests for the effective job specification shown before launch.

The create-job page promises the user a set of hyperparameters. Since #83 that
promise is what launches: the orchestrator writes `effective()` into the job
spec and the trainer applies it without resolving anything, so these tests
define the contract both ends rely on -- defaults, the allowed overrides,
alpha tracking rank, rsLoRA inferred. The table that defines it is now
`packages/contracts/trainer-defaults.json` (#82), read by `hyperparams`; the
values themselves are pinned below so an edit to the data file fails a test
instead of silently changing every launch. The trainer's required-key set is
pinned against `effective({})` by
`apps/trainer/tests/test_agreement_with_the_domain.py`.
"""

from pathlib import Path

from temper_core import hyperparams


def test_the_table_is_loaded_from_the_contract_file():
    """Read, not declared. The path is asserted rather than trusted: if this
    module ever resolves somewhere else, trainer and page read different
    tables and nothing else in the suite would notice."""
    assert Path(hyperparams.CONTRACT_PATH).name == "trainer-defaults.json"
    assert Path(hyperparams.CONTRACT_PATH).is_file()


def test_defaults_are_what_the_trainer_will_use():
    eff = hyperparams.effective({})
    assert eff["lora_r"] == 16
    assert eff["lora_alpha"] == 32  # alpha = 2r
    assert eff["num_epochs"] == 3
    assert eff["learning_rate"] == 2e-4
    assert eff["sequence_len"] == 2048
    assert eff["gradient_accumulation_steps"] == 8


def test_alpha_tracks_rank_when_only_rank_is_moved():
    eff = hyperparams.effective({"lora_r": 64})
    assert eff["lora_r"] == 64
    assert eff["lora_alpha"] == 128, (
        "a new rank must not pair with a stale scale"
    )


def test_explicit_alpha_is_not_overwritten():
    eff = hyperparams.effective({"lora_r": 64, "lora_alpha": 16})
    assert eff["lora_alpha"] == 16


def test_rslora_is_inferred_from_the_rank():
    assert hyperparams.effective({})["lora_use_rslora"] is False
    assert hyperparams.effective({"lora_r": 32})["lora_use_rslora"] is True


def test_locked_settings_ignore_overrides():
    """Correctness settings are default-locked: an override outside
    ALLOWED_OVERRIDES never reaches the spec the page shows."""
    eff = hyperparams.effective(
        {"warmup_ratio": 0.9, "lr_scheduler": "constant"}
    )
    assert eff["warmup_ratio"] == 0.1
    assert eff["lr_scheduler"] == "cosine"


def test_overrides_are_coerced_to_the_schema_type():
    """An override typed in a browser arrives as a string; the resolver
    coerces it to the schema's type so the trainer's spec never carries a
    string where a number belongs (issue #80)."""
    eff = hyperparams.effective(
        {"lora_r": "32", "learning_rate": "0.0001", "num_epochs": "5"}
    )
    assert eff["lora_r"] == 32
    assert isinstance(eff["lora_r"], int)
    assert eff["learning_rate"] == 0.0001
    assert isinstance(eff["learning_rate"], float)
    assert eff["num_epochs"] == 5.0
    assert isinstance(eff["num_epochs"], float)
