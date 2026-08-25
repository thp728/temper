"""Tests for the effective job specification shown before launch.

The create-job page promises the user a set of hyperparameters. That promise
is only honest if it is computed the same way the trainer computes its config,
so this module mirrors `trainer/entrypoint.py`'s resolution -- defaults, the
allowed overrides, alpha tracking rank, rsLoRA inferred -- and the final test
pins the mirror against the original so the two cannot drift apart silently.

The control plane does not import trainer code (the same rule that made
feasibility.py duplicate DEFAULT_EPOCHS), so the pin is what stands in for a
shared constant.
"""

from temper_core import hyperparams  # noqa: E402


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


def test_mirror_matches_the_trainer():
    """The page's promise and the trainer's behaviour come from two copies of
    one table. This test is why that is safe: if either copy changes alone,
    this fails and the drift is caught before a user is told something false."""
    import entrypoint

    assert hyperparams.DEFAULTS == entrypoint.DEFAULTS
    assert hyperparams.ALLOWED_OVERRIDES == entrypoint.ALLOWED_OVERRIDES
