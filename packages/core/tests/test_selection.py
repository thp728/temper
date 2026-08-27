"""Hardware selection -- spec 005, issue #55.

Property-shaped, per spec 005's testing decisions: the rules under test are
comparative (cheaper never loses to pricier, fewer devices break a tie,
adding an option never makes the answer worse), not a pinned "an L4 is
chosen" that would need rewriting on every catalog or pricing change.
"""

from __future__ import annotations

import pytest

from temper_core import selection
from temper_core.models import ModelFacts
from temper_core.selection import GpuAvailability, select_hardware

# Qwen3-4B, from its own published config.json (matches test_memory.py).
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

DEFAULT_SHAPE = dict(lora_r=16, sequence_len=2048, micro_batch_size=1)


def plan(availability, **overrides):
    kwargs = {**DEFAULT_SHAPE, **overrides}
    return select_hardware(
        QWEN3_4B, availability=availability, currency="INR", **kwargs
    )


# --- the cheapest fit always wins --------------------------------------------


def test_a_cheaper_card_that_fits_is_never_passed_over_for_a_pricier_one():
    cheap_and_pricey = [
        GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1),
        GpuAvailability("H100", price_per_hour=250.0, num_free_devices=1),
    ]
    result = plan(cheap_and_pricey)
    assert result.gpu_type == "L4"


def test_the_pricier_card_is_chosen_when_it_is_the_only_one_that_fits():
    """Cheapest-that-fits, not cheapest full stop: a card too small for the
    job is not a cheaper answer, it is not an answer."""
    too_small_and_big_enough = [
        GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1),
        GpuAvailability("H100", price_per_hour=250.0, num_free_devices=1),
    ]
    result = plan(
        too_small_and_big_enough,
        methods=("full",),  # full fine-tune of a 4B model does not fit an L4
    )
    assert result.gpu_type == "H100"


def test_adding_a_pricier_option_never_changes_a_cheaper_winner():
    """A property of the search itself: an alternative that cannot win must
    not be able to change the answer by merely existing."""
    without_extra = plan(
        [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)]
    )
    with_extra = plan(
        [
            GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1),
            GpuAvailability(
                "RTX-PRO6000", price_per_hour=90.0, num_free_devices=1
            ),
        ]
    )
    assert with_extra.gpu_type == without_extra.gpu_type
    assert with_extra.price_per_hour == without_extra.price_per_hour


def test_an_unavailable_type_is_never_chosen():
    result = plan(
        [
            GpuAvailability("L4", price_per_hour=41.31, num_free_devices=0),
            GpuAvailability("H100", price_per_hour=250.0, num_free_devices=1),
        ]
    )
    assert result.gpu_type == "H100"


def test_a_gpu_type_with_no_known_capacity_is_never_chosen():
    """A price with no capacity figure behind it is not a fact this search
    can predict a fit from -- skipped, not guessed."""
    result = plan(
        [
            GpuAvailability(
                "MYSTERY-GPU", price_per_hour=1.0, num_free_devices=4
            ),
            GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1),
        ]
    )
    assert result.gpu_type == "L4"


# --- fewer, larger devices over more, smaller ones ---------------------------


def test_a_tied_price_prefers_fewer_devices():
    """A single H100 and four sharded L4s cost the same here and both fit a
    full fine-tune of the 4B model -- the tie is broken toward the one card,
    because four GPUs pay interconnect overhead one does not."""
    tied_price = [
        GpuAvailability("H100", price_per_hour=60.0, num_free_devices=1),
        GpuAvailability("L4", price_per_hour=15.0, num_free_devices=4),
    ]
    result = plan(tied_price, methods=("full",))
    assert result.device_count == 1
    assert result.gpu_type == "H100"


def test_more_devices_is_chosen_only_when_nothing_smaller_fits():
    only_sharded_fits = [
        GpuAvailability("L4", price_per_hour=41.31, num_free_devices=4),
    ]
    result = plan(
        only_sharded_fits,
        methods=("full",),  # a full 4B fine-tune needs sharding to fit an L4
    )
    assert result.device_count == 4


# --- method falls out of the same decision -----------------------------------


def test_a_method_that_does_not_fit_is_never_chosen_over_one_that_does():
    result = plan(
        [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)],
        methods=("full", "lora", "qlora"),
    )
    assert result.method in ("lora", "qlora")  # full does not fit a 24 GB L4


def test_the_more_capable_method_wins_a_tied_price():
    """Price never depends on method, so at the same price the better
    method that still fits is a strictly better answer."""
    result = plan(
        [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)],
        methods=("lora", "qlora"),
    )
    assert result.method == "lora"


def test_default_methods_are_only_what_the_trainer_can_run():
    result = plan(
        [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)]
    )
    assert result.method in selection.EXECUTABLE_METHODS


# --- currency travels with the price -----------------------------------------


def test_currency_is_read_from_the_account_not_assumed():
    result = select_hardware(
        QWEN3_4B,
        **DEFAULT_SHAPE,
        availability=[
            GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)
        ],
        currency="USD",
    )
    assert result.currency == "USD"


# --- refusal -------------------------------------------------------------


def test_nothing_available_is_refused_rather_than_guessed():
    with pytest.raises(selection.NoFittingHardwareError):
        plan([])


def test_nothing_big_enough_is_refused_rather_than_guessed():
    with pytest.raises(selection.NoFittingHardwareError):
        plan(
            [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)],
            methods=("full",),
            micro_batch_size=64,
        )


# --- the alternatives it considered and rejected ---------------------------
# A quote must say what the user gave up to take the chosen card (issue #76),
# so the search returns the fitting configurations that lost -- never a
# re-derived guess, the same fit checks the winner passed.


def test_the_fitting_configurations_it_rejected_are_returned():
    result = plan(
        [
            GpuAvailability("L4", price_per_hour=41.31, num_free_devices=2),
            GpuAvailability("H100", price_per_hour=250.0, num_free_devices=1),
        ]
    )
    assert result.gpu_type == "L4"
    types = {a.gpu_type for a in result.alternatives}
    assert "H100" in types  # the pricier card that also fits
    counts = {
        a.device_count for a in result.alternatives if a.gpu_type == "L4"
    }
    assert 2 in counts  # a second L4 also fits, and costs twice as much


def test_the_chosen_configuration_is_not_in_the_alternatives():
    result = plan(
        [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=2)]
    )
    assert all(
        not (
            a.gpu_type == result.gpu_type
            and a.device_count == result.device_count
        )
        for a in result.alternatives
    )


def test_alternatives_are_never_cheaper_than_the_chosen_configuration():
    result = plan(
        [
            GpuAvailability("L4", price_per_hour=41.31, num_free_devices=2),
            GpuAvailability("H100", price_per_hour=250.0, num_free_devices=1),
        ]
    )
    for a in result.alternatives:
        assert a.price_per_hour >= result.price_per_hour


def test_nothing_fits_means_no_alternatives():
    """With only an L4 available and a full fine-tune requested, nothing
    fits -- the refusal carries no alternatives, because there were none."""
    with pytest.raises(selection.NoFittingHardwareError):
        plan(
            [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)],
            methods=("full",),
        )
