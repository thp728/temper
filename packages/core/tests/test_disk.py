"""Disk sizing -- spec 005, issue #64.

Property-shaped, like `test_selection.py`: the arithmetic is what is under
test (weights plus retained checkpoints plus fixed overhead, floored and
capped), never an internal breakdown that would break on every refinement of
the model.
"""

from __future__ import annotations

import pytest

from temper_core import disk, memory
from temper_core.models import ModelFacts

# Qwen3-4B, matching test_memory.py and test_selection.py.
QWEN3_4B = ModelFacts(
    architecture="qwen3",
    hidden_size=2560,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=9728,
    vocab_size=151936,
    tie_word_embeddings=True,
    is_moe=False,
    has_chat_template=True,
    pad_eos_distinct=True,
    context_length=40960,
    license="apache-2.0",
)

# Qwen3-8B, matching test_memory.py.
QWEN3_8B = ModelFacts(
    architecture="qwen3",
    hidden_size=4096,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=12288,
    vocab_size=151936,
    tie_word_embeddings=False,
    is_moe=False,
    has_chat_template=True,
    pad_eos_distinct=True,
    context_length=40960,
    license="apache-2.0",
)

# A synthetic ~70B-class model, dimensioned like Llama-2-70B's published
# config -- not in the catalog, which is the point: a stored figure cannot
# describe a model the catalog has never seen, and this is exactly the case
# spike 5 measured a large download against (its own Qwen3-8B download rate
# extrapolates to roughly this weight size at 70B).
LARGE_70B_CLASS = ModelFacts(
    architecture="llama",
    hidden_size=8192,
    num_hidden_layers=80,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=28672,
    vocab_size=32000,
    tie_word_embeddings=False,
    is_moe=False,
    has_chat_template=True,
    pad_eos_distinct=True,
    context_length=4096,
    license="llama2",
)

DEFAULT_CHECKPOINTS = 3  # matches packages/contracts/trainer-defaults.json


# --- the sum ------------------------------------------------------------


def test_required_disk_is_weights_plus_checkpoints_plus_fixed_overhead():
    plan = disk.required_disk(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        retained_checkpoints=DEFAULT_CHECKPOINTS,
    )
    assert plan.required_gb == pytest.approx(
        plan.weights_gb
        + plan.checkpoints_gb
        + plan.merged_output_gb
        + plan.image_gb
        + plan.working_space_gb
    )


def test_weights_are_priced_at_download_precision_not_training_precision():
    """QLoRA trains at NF4 (0.5 bytes/param) but downloads bf16 (2
    bytes/param) -- the trainer quantises on-device, never fetches an
    already-quantised checkpoint. Disk must reflect what crossed the wire."""
    plan = disk.required_disk(
        QWEN3_4B, method="qlora", lora_r=16, retained_checkpoints=0
    )
    assert plan.weights_gb == pytest.approx(
        QWEN3_4B.params * 2.0 / disk.BYTES_PER_GB
    )


def test_a_bigger_model_never_predicts_less_disk():
    small = disk.required_disk(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        retained_checkpoints=DEFAULT_CHECKPOINTS,
    )
    big = disk.required_disk(
        QWEN3_8B,
        method="qlora",
        lora_r=16,
        retained_checkpoints=DEFAULT_CHECKPOINTS,
    )
    assert big.required_gb > small.required_gb


def test_more_retained_checkpoints_never_predicts_less_disk():
    fewer = disk.required_disk(
        QWEN3_4B, method="qlora", lora_r=16, retained_checkpoints=1
    )
    more = disk.required_disk(
        QWEN3_4B, method="qlora", lora_r=16, retained_checkpoints=5
    )
    assert more.required_gb > fewer.required_gb


def test_full_fine_tune_checkpoints_are_never_smaller_than_an_adapters():
    """A retained full-fine-tune checkpoint holds every parameter; an
    adapter checkpoint holds only the trainable slice -- the full model can
    never cost less disk per checkpoint."""
    adapter = disk.required_disk(
        QWEN3_4B, method="qlora", lora_r=16, retained_checkpoints=1
    )
    full = disk.required_disk(
        QWEN3_4B, method="full", lora_r=16, retained_checkpoints=1
    )
    assert full.checkpoints_gb > adapter.checkpoints_gb


def test_no_method_this_trainer_runs_produces_a_merged_output():
    for method in ("qlora", "lora", "full"):
        plan = disk.required_disk(
            QWEN3_4B, method=method, lora_r=16, retained_checkpoints=1
        )
        assert plan.merged_output_gb == 0.0


# --- floor and ceiling ----------------------------------------------------


def test_a_small_job_is_floored_at_the_platform_minimum():
    plan = disk.required_disk(
        QWEN3_4B, method="qlora", lora_r=16, retained_checkpoints=0
    )
    assert plan.required_gb < disk.PLATFORM_MIN_DISK_GB
    assert plan.provisioned_gb == disk.PLATFORM_MIN_DISK_GB


def test_a_large_model_is_provisioned_above_the_platform_minimum():
    plan = disk.required_disk(
        LARGE_70B_CLASS,
        method="qlora",
        lora_r=16,
        retained_checkpoints=DEFAULT_CHECKPOINTS,
    )
    assert plan.provisioned_gb > disk.PLATFORM_MIN_DISK_GB
    assert plan.provisioned_gb <= disk.PLATFORM_MAX_DISK_GB


def test_a_job_needing_more_than_the_ceiling_is_refused_with_the_shortfall_named():
    with pytest.raises(disk.DiskExceedsCeilingError) as exc_info:
        disk.required_disk(
            QWEN3_4B,
            method="full",
            lora_r=16,
            retained_checkpoints=1000,  # absurd on purpose: forces the ceiling
        )
    err = exc_info.value
    assert err.shortfall_gb > 0
    assert err.required_gb - err.ceiling_gb == pytest.approx(err.shortfall_gb)
    assert "GB more" in str(err)


def test_refusal_happens_before_the_floor_not_after():
    """A raw requirement above the ceiling refuses outright -- it is never
    quietly rounded down to something provisionable."""
    with pytest.raises(disk.DiskExceedsCeilingError):
        disk.required_disk(
            QWEN3_4B,
            method="full",
            lora_r=16,
            retained_checkpoints=1000,  # absurd on purpose: forces the ceiling
        )


# --- an override (issue #79) --------------------------------------------------
# A disk override requests an exact size instead of the predictor's floor. A
# larger disk is honoured; a smaller one is refused with the need named --
# taking control must not be able to create a machine that cannot hold its
# own download.


def test_a_larger_disk_override_is_honoured_exactly():
    plan = disk.required_disk(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        retained_checkpoints=0,
        provisioned_gb=500,
    )
    assert plan.provisioned_gb == 500
    assert plan.required_gb < 500


def test_a_disk_override_below_the_need_is_refused_with_the_arithmetic():
    with pytest.raises(disk.DiskBelowNeedError) as exc:
        disk.required_disk(
            LARGE_70B_CLASS,
            method="qlora",
            lora_r=16,
            retained_checkpoints=DEFAULT_CHECKPOINTS,
            provisioned_gb=150,  # less than the ~140 GB raw need
        )
    assert exc.value.requested_gb == 150
    assert exc.value.required_gb > 150
    assert "cannot hold it" in str(exc.value)


def test_a_disk_override_below_the_platform_minimum_is_refused():
    with pytest.raises(disk.DiskBelowMinimumError) as exc:
        disk.required_disk(
            QWEN3_4B,
            method="qlora",
            lora_r=16,
            retained_checkpoints=0,
            provisioned_gb=50,
        )
    assert exc.value.minimum_gb == disk.PLATFORM_MIN_DISK_GB


def test_a_disk_override_above_the_ceiling_is_refused():
    with pytest.raises(disk.DiskExceedsCeilingError):
        disk.required_disk(
            QWEN3_4B,
            method="qlora",
            lora_r=16,
            retained_checkpoints=0,
            provisioned_gb=disk.PLATFORM_MAX_DISK_GB + 1,
        )


def test_a_bigger_disk_override_bills_more_storage():
    """The USD storage line is priced off what is actually provisioned, so a
    user who asks for more disk sees it in the quote rather than as an
    invisible line."""
    default = disk.required_disk(
        QWEN3_4B, method="qlora", lora_r=16, retained_checkpoints=0
    )
    bigger = disk.required_disk(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        retained_checkpoints=0,
        provisioned_gb=4000,
    )
    assert bigger.provisioned_gb == 4000
    assert bigger.storage_cost_usd_per_hour > default.storage_cost_usd_per_hour


# --- cost travels with the plan -------------------------------------------


def test_storage_cost_is_never_hidden():
    plan = disk.required_disk(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        retained_checkpoints=DEFAULT_CHECKPOINTS,
    )
    assert plan.storage_cost_usd_per_hour > 0


def test_more_disk_never_predicts_less_storage_cost():
    small = disk.required_disk(
        QWEN3_4B, method="qlora", lora_r=16, retained_checkpoints=0
    )
    big = disk.required_disk(
        LARGE_70B_CLASS,
        method="qlora",
        lora_r=16,
        retained_checkpoints=DEFAULT_CHECKPOINTS,
    )
    assert big.storage_cost_usd_per_hour > small.storage_cost_usd_per_hour


# --- validation -------------------------------------------------------------


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown method"):
        disk.required_disk(
            QWEN3_4B,
            method="not-a-method",
            lora_r=16,
            retained_checkpoints=1,
        )


def test_negative_retained_checkpoints_is_refused():
    with pytest.raises(ValueError, match="retained_checkpoints"):
        disk.required_disk(
            QWEN3_4B, method="qlora", lora_r=16, retained_checkpoints=-1
        )


# --- anchor: the checkpoint pool matches the one real measurement ----------


def test_a_qlora_checkpoint_matches_the_real_runs_adapter_size():
    """apps/trainer/README.md: 33,030,144 trainable params at r=16 shipped
    as a 132.2 MB fp32 adapter. One retained checkpoint should land there."""
    plan = disk.required_disk(
        QWEN3_4B, method="qlora", lora_r=16, retained_checkpoints=1
    )
    measured_gb = 132.2 * 1e6 / disk.BYTES_PER_GB
    assert plan.checkpoints_gb == pytest.approx(measured_gb, rel=0.05)
    assert memory.trainable_params(QWEN3_4B, lora_r=16) == 33_030_144
