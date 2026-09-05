"""The peak-VRAM predictor -- spec 005, issue #48.

Two anchors, both real: the 4B QLoRA run's trainable-parameter count, which
must match exactly, and its peak VRAM, which must land inside
`memory.PEAK_TOLERANCE`. Everything else is a property of the arithmetic
(more GPUs of parameters never predict less memory; a full fine-tune's
trainable set is never smaller than LoRA's), asserted on the returned
numbers, never on the internal breakdown -- per spec 005's testing
decisions, a test that pins an intermediate term breaks on every refinement
of the model, which is exactly when it should stay green.
"""

from __future__ import annotations

from temper_core import memory
from temper_core.models import ModelFacts

# Qwen3-4B, from its own published config.json -- see
# apps/control-plane/src/temper_control_plane/fake_models.py for the
# from-source confirmation. tie_word_embeddings=True.
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

# Qwen3-8B. tie_word_embeddings=False (separate input/output tables).
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

# The one real run this project has: Qwen3-4B, QLoRA, r=16, all-linear,
# sequence length 2048, micro-batch 1, single L4. apps/trainer/README.md,
# 2026-08-19.
MEASURED_TRAINABLE_PARAMS = 33_030_144
MEASURED_PEAK_GB = 5.31


# --- the two anchors ---------------------------------------------------------


def test_trainable_params_matches_the_real_run_exactly():
    """Not "close" -- exact. This is the anchor spec 005 names by name."""
    assert memory.trainable_params(QWEN3_4B, lora_r=16) == (
        MEASURED_TRAINABLE_PARAMS
    )


def test_qwen3_8b_trainable_params_matches_the_by_hand_arithmetic():
    """Derives 43,646,976 for Qwen3-8B at r=16, all-linear, by hand from the
    same config. A second independent confirmation of the formula, not just
    the first model."""
    assert memory.trainable_params(QWEN3_8B, lora_r=16) == 43_646_976


def test_qwen3_8b_total_params_matches_its_advertised_size():
    """wiki/foundations.md's whole point: the arithmetic reproduces the
    model's advertised 8.19B exactly, untied embeddings counted twice."""
    assert QWEN3_8B.params == 8_190_427_136


def test_predicted_peak_is_within_the_stated_tolerance_of_the_measured_run():
    peak = memory.predict_peak(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
    )
    error = abs(peak.total_gb - MEASURED_PEAK_GB) / MEASURED_PEAK_GB
    assert error <= memory.PEAK_TOLERANCE, (
        f"predicted {peak.total_gb:.2f} GB vs measured {MEASURED_PEAK_GB} GB "
        f"is {error:.1%} off, outside the stated {memory.PEAK_TOLERANCE:.0%}"
    )


def test_predicted_peak_reports_the_trainable_params_it_used():
    peak = memory.predict_peak(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
    )
    assert peak.trainable_params == MEASURED_TRAINABLE_PARAMS


# --- properties, not internals -----------------------------------------------


def test_more_parameters_never_predicts_less_peak_memory():
    small = memory.predict_peak(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
    )
    big = memory.predict_peak(
        QWEN3_8B,
        method="qlora",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
    )
    assert big.total_gb > small.total_gb


def test_full_fine_tune_trains_more_parameters_than_lora_on_the_same_model():
    lora = memory.predict_peak(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
    )
    full = memory.predict_peak(
        QWEN3_4B,
        method="full",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
    )
    assert full.trainable_params > lora.trainable_params
    assert full.total_gb > lora.total_gb


def test_a_larger_micro_batch_never_predicts_less_peak_memory():
    one = memory.predict_peak(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
    )
    four = memory.predict_peak(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=4,
    )
    assert four.total_gb > one.total_gb


def test_more_devices_never_predicts_less_aggregate_memory():
    """The property spec 005 names: adding GPUs must never reduce predicted
    total memory. Sharding (full fine-tune only) divides weights, gradients
    and optimizer state across devices, so the PER-DEVICE peak can shrink --
    but activations and the fixed overhead are paid on every device, so the
    CLUSTER-WIDE total (per-device peak times device count) never shrinks."""
    one = memory.predict_peak(
        QWEN3_4B,
        method="full",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
        device_count=1,
    )
    two = memory.predict_peak(
        QWEN3_4B,
        method="full",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
        device_count=2,
    )
    assert two.total_gb * 2 >= one.total_gb * 1


def test_sharding_reduces_full_fine_tunes_per_device_peak():
    one = memory.predict_peak(
        QWEN3_4B,
        method="full",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
        device_count=1,
    )
    two = memory.predict_peak(
        QWEN3_4B,
        method="full",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
        device_count=2,
    )
    assert two.total_gb < one.total_gb


def test_sharding_does_not_change_qlora_or_lora_per_device_peak():
    """Only full fine-tuning's FSDP path has ever run sharded (spike 6).
    LoRA and QLoRA's trainable set is tiny, so device_count is a throughput
    lever for them, not a memory one -- more devices replicate the same
    per-device footprint rather than dividing it."""
    for method in ("qlora", "lora"):
        one = memory.predict_peak(
            QWEN3_4B,
            method=method,
            lora_r=16,
            sequence_len=2048,
            micro_batch_size=1,
            device_count=1,
        )
        four = memory.predict_peak(
            QWEN3_4B,
            method=method,
            lora_r=16,
            sequence_len=2048,
            micro_batch_size=1,
            device_count=4,
        )
        assert four.total_gb == one.total_gb


def test_device_count_below_one_is_refused():
    import pytest

    with pytest.raises(ValueError, match="device_count"):
        memory.predict_peak(
            QWEN3_4B,
            method="qlora",
            lora_r=16,
            sequence_len=2048,
            micro_batch_size=1,
            device_count=0,
        )


def test_unknown_method_is_refused():
    import pytest

    with pytest.raises(ValueError, match="unknown method"):
        memory.predict_peak(
            QWEN3_4B,
            method="not-a-method",
            lora_r=16,
            sequence_len=2048,
            micro_batch_size=1,
        )


# --- headroom -----------------------------------------------------------------


def test_headroom_is_capacity_minus_predicted_peak():
    peak = memory.predict_peak(
        QWEN3_4B,
        method="qlora",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
    )
    assert memory.headroom_gb(peak, card_capacity_gb=24.0) == (
        24.0 - peak.total_gb
    )


def test_headroom_goes_negative_rather_than_clamping_when_a_job_does_not_fit():
    peak = memory.predict_peak(
        QWEN3_8B,
        method="full",
        lora_r=16,
        sequence_len=2048,
        micro_batch_size=1,
    )
    assert memory.headroom_gb(peak, card_capacity_gb=24.0) < 0
