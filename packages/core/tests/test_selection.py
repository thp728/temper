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


def test_full_fine_tuning_is_executable_and_chosen_when_it_is_the_right_call():
    """Issue #66: the trainer now runs full fine-tuning, so an unpinned search
    ranges over it -- and when the only card that can hold anything is one a
    full fine-tune fits (the 4B model on an 80 GB H100), the predictor picks
    full, because price never depends on method and the more capable method
    that still fits is a strictly better answer."""
    only_big_card = [
        GpuAvailability("H100", price_per_hour=250.0, num_free_devices=1)
    ]
    result = plan(only_big_card)
    assert result.method == "full"
    assert "full" in selection.EXECUTABLE_METHODS


def test_full_loses_to_qlora_when_a_cheaper_adapter_configuration_fits():
    """Full fine-tuning is chosen when it is the right call, not always: on a
    card where qlora fits, qlora is cheaper and wins -- the cheapest
    configuration that fits is the rule, and full pays its price only when no
    cheaper adapter fits."""
    l4_and_h100 = [
        GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1),
        GpuAvailability("H100", price_per_hour=250.0, num_free_devices=1),
    ]
    result = plan(l4_and_h100)
    assert result.method == "qlora"
    assert result.gpu_type == "L4"


def test_the_lightest_method_is_the_one_a_refusal_names():
    """A refusal with a card free names the lightest executable method's
    arithmetic: if even the cheapest-to-fit method cannot fit, nothing can,
    and naming full (the heaviest) would overstate what the user has to beat."""
    assert selection.lightest_method(selection.EXECUTABLE_METHODS) == "qlora"
    assert selection.lightest_method(("full", "lora", "qlora")) == "qlora"


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


# --- pinned decisions (issue #79) -------------------------------------------
# An override pins one decision and recomputes the rest. A pinned decision is
# a hard constraint, never a preference: the search considers only matching
# configurations and refuses rather than falling back to the predictor's pick.


def test_pinning_a_gpu_type_restricts_the_search_to_it():
    with_h100 = [
        GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1),
        GpuAvailability("H100", price_per_hour=250.0, num_free_devices=1),
    ]
    result = plan(with_h100, gpu_type="H100")
    assert result.gpu_type == "H100"
    assert result.price_per_hour == 250.0


def test_pinning_a_method_searches_it_even_outside_the_executable_set():
    """A pinned method is searched even when it is not executable -- the
    caller that pinned it decides executability; selection prices the fit."""
    result = plan(
        [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)],
        method="lora",  # not in EXECUTABLE_METHODS, but describable
    )
    assert result.method == "lora"
    assert result.gpu_type == "L4"  # lora fits an L4


def test_pinning_a_device_count_prices_that_count():
    result = plan(
        [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=4)],
        device_count=2,
    )
    assert result.device_count == 2
    assert result.price_per_hour == pytest.approx(82.62)


def test_a_pinned_card_that_cannot_hold_the_job_is_refused_with_the_arithmetic():
    """An override that makes the job infeasible on memory grounds is refused
    with the same arithmetic that did the refusing -- the peak against the
    card's capacity, never a bare 'nothing fits' (issue #79)."""
    with pytest.raises(selection.NoFittingHardwareError) as exc:
        plan(
            [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)],
            method="full",  # full 4B fine-tune does not fit a 24 GB L4
        )
    assert exc.value.peak_gb is not None
    assert exc.value.gpu_type == "L4"
    assert exc.value.capacity_gb == 24.0
    assert exc.value.peak_gb > exc.value.capacity_gb


def test_a_pinned_card_with_nothing_free_is_refused_with_availability():
    """A card that *would* hold the job but has nothing free is a different
    refusal from one that cannot hold it -- the arithmetic keeps the
    distinction visible."""
    with pytest.raises(selection.NoFittingHardwareError) as exc:
        plan(
            [
                GpuAvailability(
                    "H100", price_per_hour=250.0, num_free_devices=0
                )
            ],
            gpu_type="H100",
        )
    assert exc.value.peak_gb is not None
    assert exc.value.gpu_type == "H100"
    assert exc.value.peak_gb <= exc.value.capacity_gb
    # A fit that merely lacks availability is not a memory shortfall: the
    # peak sits inside capacity, so there is nothing to be short by.
    assert exc.value.shortfall_gb is not None
    assert exc.value.shortfall_gb <= 0


def test_the_shortfall_is_peak_over_capacity_and_none_without_arithmetic():
    """The refusal names where the shortfall is (issue #54): the peak over
    the capacity, derived once on the error so the search and the refusal
    surface cannot disagree. Without a single configuration to price (nothing
    free at all) there is no shortfall to name."""
    with pytest.raises(selection.NoFittingHardwareError) as exc:
        plan(
            [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)],
            method="full",
        )
    assert exc.value.peak_gb > exc.value.capacity_gb
    assert exc.value.shortfall_gb == pytest.approx(
        exc.value.peak_gb - exc.value.capacity_gb
    )
    assert exc.value.shortfall_gb > 0
    with pytest.raises(selection.NoFittingHardwareError) as bare:
        plan([])
    assert bare.value.shortfall_gb is None


def test_pinning_more_devices_than_a_node_offers_is_refused():
    with pytest.raises(selection.NoFittingHardwareError):
        plan(
            [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=2)],
            device_count=4,
        )


def test_an_unpinned_refusal_still_carries_no_arithmetic():
    """The unpinned 'nothing fits' keeps its bare shape: there was no single
    configuration to price, so none is named."""
    with pytest.raises(selection.NoFittingHardwareError) as exc:
        plan([])
    assert exc.value.peak_gb is None
    assert exc.value.capacity_gb is None


def test_an_unpinned_search_with_free_cards_names_the_arithmetic():
    """An unpinned search that fails *despite* a card being free names the
    arithmetic (issue #80): the configuration was described by the user's
    hyperparameter overrides, and a bare 'nothing fits' would not say which
    peak lost against which capacity. The cheapest executable configuration
    (qlora on one device) is what the search tried first, so it is what is
    named."""
    with pytest.raises(selection.NoFittingHardwareError) as exc:
        plan(
            [GpuAvailability("L4", price_per_hour=41.31, num_free_devices=1)],
            micro_batch_size=64,
        )
    assert exc.value.peak_gb is not None
    assert exc.value.gpu_type == "L4"
    assert exc.value.capacity_gb == 24.0
    assert exc.value.peak_gb > exc.value.capacity_gb
    assert exc.value.method == "qlora"
    assert exc.value.device_count == 1
