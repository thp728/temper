"""Decision records -- spec 005, issue #76.

Every choice the predictor makes for the user is returned as a record: the
decision, the value chosen, the constraint that forced it, and the
alternatives with what each would have cost. Method, hardware, device count,
disk, precision and sequence length each carry one.

A good test here asserts on what `decide` returns -- that the chosen value is
the executable method, that the chosen card is the cheapest that fits, that
the alternatives cost what the arithmetic says they cost -- never on the
internal phrasing that would change every time the model is refined. Where a
reason names a number (peak memory, price per hour), the number is the
assertion, because that is the part a reader can check against the memory and
selection arithmetic.
"""

from __future__ import annotations

from temper_core import decisions, disk, hyperparams, overrides, selection
from temper_core.models import ModelFacts
from temper_core.selection import GpuAvailability

# Qwen3-4B, matching test_memory.py / test_selection.py.
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

# The L4 rate spike 5 measured; price and currency are inputs, not assumptions.
L4_INR_PER_HOUR = 41.31

# A single L4 with plenty free -- the default availability shape for these
# tests, so the chosen configuration is deterministic.
L4_ONLY = [GpuAvailability("L4", L4_INR_PER_HOUR, 8)]


def records(availability, **overrides):
    hp = hyperparams.effective({})
    p = selection.select_hardware(
        QWEN3_4B,
        lora_r=hp["lora_r"],
        sequence_len=hp["sequence_len"],
        micro_batch_size=hp["micro_batch_size"],
        availability=availability,
        currency="INR",
        **overrides,
    )
    dp = disk.required_disk(
        QWEN3_4B,
        method=p.method,
        lora_r=hp["lora_r"],
        retained_checkpoints=hp["save_total_limit"],
    )
    return decisions.decide(QWEN3_4B, hyperparameters=hp, plan=p, disk_plan=dp)


def decision(named, availability=L4_ONLY, **overrides):
    for d in records(availability, **overrides):
        if d.decision == named:
            return d
    raise AssertionError(f"no decision named {named!r}")


# --- the six decisions, in the criteria's order ------------------------------


def test_all_six_decisions_are_recorded_in_the_criteria_order():
    ds = records(L4_ONLY)
    assert [d.decision for d in ds] == [
        "method",
        "hardware",
        "device count",
        "disk",
        "precision",
        "sequence length",
    ]


def test_every_decision_carries_a_chosen_value_and_a_constraint():
    for d in records(L4_ONLY):
        assert d.chosen
        assert d.constraint
        # Every alternative is a value with a cost and a reason, never a
        # bare name -- the whole point of recording them.
        for a in d.alternatives:
            assert a.value
            assert a.cost
            assert a.constraint


# --- method -------------------------------------------------------------------


def test_method_chooses_the_executable_one_and_names_the_alternatives():
    m = decision("method")
    assert m.chosen in selection.EXECUTABLE_METHODS
    values = [a.value for a in m.alternatives]
    assert "lora" in values
    assert "full fine-tune" in values
    # The alternatives cost in memory -- the reason the chosen method wins.
    # lora fits the 24 GB L4; full does not.
    lora = m.alternatives[values.index("lora")]
    assert "GB" in lora.cost
    full = m.alternatives[values.index("full fine-tune")]
    assert "GB" in full.cost
    # The reason names the executable-only bound, not a price one.
    assert "execute" in m.constraint.lower()


H100_ONLY = [GpuAvailability("H100", 250.0, 2)]


def test_the_full_alternative_shows_what_it_would_have_cost():
    """The reasoning against the adapter alternative, with its price (issue
    #66): on the chosen L4 a full fine-tune of the 4B model cannot fit, and the
    reason says the memory it would have needed -- the cost of the more capable
    method, in the same numbers the predictor priced with."""
    m = decision("method")
    full = next(a for a in m.alternatives if a.value == "full fine-tune")
    assert "GB" in full.cost
    assert (
        "beyond" in full.constraint.lower()
        or "bigger" in full.constraint.lower()
    )


def test_full_is_chosen_and_the_adapter_alternative_is_priced_when_it_is_the_right_call():
    """When full fine-tuning is the cheapest configuration that fits (only an
    H100 is free), the method decision chooses it and the qlora alternative
    carries what it would have cost -- the mirror image of the L4 case."""
    m = decision("method", H100_ONLY)
    assert m.chosen == "full"
    values = [a.value for a in m.alternatives]
    assert "qlora" in values
    qlora = m.alternatives[values.index("qlora")]
    assert (
        "GB" in qlora.cost
    )  # the adapter's own footprint is the price of taking it
    assert (
        "prefers the more capable" in m.constraint
        or "cheapest executable" in m.constraint
    )


def test_a_full_device_count_shrinks_the_footprint_and_is_not_shipped():
    """Full fine-tuning's sharding is real (memory.py divides weights,
    gradients and optimizer state by device count; spike 6 proved the
    mechanism), so the device-count reason must not claim extra cards merely
    replicate the footprint -- while also stating that multi-device execution
    is not shipped, so a launch of more than one device is still refused."""
    n = decision("device count", H100_ONLY)
    assert n.chosen == "1"
    assert "shard" in n.constraint.lower()
    assert (
        "not shipped" in n.constraint.lower()
        or "refused" in n.constraint.lower()
    )
    # The qlora lie is gone: extra devices are never claimed to merely
    # replicate the footprint of a full fine-tune.
    assert "replicate the same per-device footprint" not in n.constraint


# --- hardware -----------------------------------------------------------------


def test_hardware_chooses_the_cheapest_card_that_fits_and_shows_the_rest():
    with_h100 = [
        GpuAvailability("L4", L4_INR_PER_HOUR, 8),
        GpuAvailability("H100", 250.0, 2),
    ]
    h = decision("hardware", with_h100)
    assert h.chosen == "L4"
    assert "cheapest card" in h.constraint
    # The alternative that lost carries its price, and it is higher than the
    # chosen card's.
    alt = next(a for a in h.alternatives if a.value == "H100")
    assert "INR" in alt.cost
    assert "250" in alt.cost
    assert "costs more than the L4" in alt.constraint


def test_hardware_with_no_alternative_still_records_the_choice():
    """When the only available card is the chosen one, the alternatives are
    empty -- an honest 'nothing else fit' is a reason, not an omission."""
    h = decision("hardware", L4_ONLY)
    assert h.chosen == "L4"
    assert h.alternatives == ()


# --- device count --------------------------------------------------------------


def test_device_count_is_one_and_more_devices_cost_more():
    n = decision("device count")
    assert n.chosen == "1"
    first = n.alternatives[0]
    assert first.value == "2 × L4"
    assert "82.62" in first.cost  # 2 × the single-card rate
    assert "sharded" in first.constraint


# --- disk ----------------------------------------------------------------------


def test_disk_is_floored_at_the_platform_minimum():
    d = decision("disk")
    assert d.chosen == "100 GB"
    assert "platform minimum" in d.constraint
    assert "floor" in d.constraint.lower()
    # The alternative that lost is the job's true need, which the provider
    # will not create.
    raw = next(a for a in d.alternatives if "raw need" in a.value)
    assert "GB" in raw.value
    assert "will not create" in raw.cost


# --- precision -----------------------------------------------------------------


def test_precision_chooses_nf4_and_bf16_pays_four_times_the_weights():
    p = decision("precision")
    assert p.chosen == "nf4 (4-bit)"
    alt = p.alternatives[0]
    assert alt.value == "bf16 (no quantisation)"
    # bf16 weights are 4× the NF4 pool (2.0 vs 0.5 bytes/param): the numbers
    # in the reason are the assertion.
    assert "2 bytes/param" in alt.cost
    assert "4×" in alt.constraint


# --- sequence length -------------------------------------------------------------


def test_sequence_length_uses_the_trainer_default_and_doubling_costs_activations():
    s = decision("sequence length")
    assert s.chosen == "2048"
    assert "trainer default" in s.constraint
    alt = s.alternatives[0]
    assert alt.value == "4096"
    assert "activation memory" in alt.cost


# --- the records serialize once, for the API and the frozen quote ---------------


def test_to_dict_serializes_a_decision_for_the_contract():
    s = decision("hardware")
    as_dict = decisions.to_dict(s)
    assert set(as_dict) == {
        "decision",
        "chosen",
        "constraint",
        "alternatives",
        "overridden",
    }
    assert as_dict["decision"] == "hardware"
    assert as_dict["overridden"] is False
    for a in as_dict["alternatives"]:
        assert set(a) == {"value", "cost", "constraint"}


# --- overrides (issue #79) -----------------------------------------------------
# An overridden decision is marked, its chosen value reflects the override,
# and the other decisions recompute around it -- the reason never drifts from
# the configuration it explains.


def overridden_records(availability, *override_list):
    """The six records after applying decision overrides, resolved exactly as
    the quote path resolves them: pins -> selection -> disk -> decide."""
    hp = hyperparams.effective({})
    r = overrides.resolve(hp, list(override_list))
    p = selection.select_hardware(
        QWEN3_4B,
        lora_r=r.hyperparameters["lora_r"],
        sequence_len=r.hyperparameters["sequence_len"],
        micro_batch_size=r.hyperparameters["micro_batch_size"],
        availability=availability,
        currency="INR",
        method=r.method,
        gpu_type=r.gpu_type,
        device_count=r.device_count,
    )
    dp = disk.required_disk(
        QWEN3_4B,
        method=p.method,
        lora_r=r.hyperparameters["lora_r"],
        retained_checkpoints=r.hyperparameters["save_total_limit"],
        provisioned_gb=r.disk_gb,
    )
    return decisions.decide(
        QWEN3_4B,
        hyperparameters=r.hyperparameters,
        plan=p,
        disk_plan=dp,
        overridden=r.overridden,
    )


def overridden_decision(named, *override_list, availability=L4_ONLY):
    for d in overridden_records(availability, *override_list):
        if d.decision == named:
            return d
    raise AssertionError(f"no decision named {named!r}")


def test_with_no_overrides_nothing_is_marked_overridden():
    for d in records(L4_ONLY):
        assert d.overridden is False


def test_an_overridden_decision_is_marked_and_the_rest_recompute():
    """Overriding sequence length marks that decision, recomputes hardware
    around the longer window, and leaves the untouched decisions unmarked."""
    s = overridden_decision(
        "sequence length", overrides.Override("sequence length", "4096")
    )
    assert s.overridden is True
    assert s.chosen == "4096"
    h = overridden_decision(
        "hardware", overrides.Override("sequence length", "4096")
    )
    assert h.overridden is False  # the rest recompute, they are not overridden
    assert "chose" in s.constraint  # the reason acknowledges the override


def test_overriding_precision_to_bf16_moves_the_method_to_lora():
    """Precision and method are one coupled decision: naming bf16 *means*
    lora, so the method decision recomputes to it and the precision decision
    shows bf16 as its own choice."""
    r = overrides.resolve(
        hyperparams.effective({}),
        [overrides.Override("precision", overrides.PRECISION_BF16)],
    )
    assert r.method == "lora"
    m = overridden_decision(
        "method", overrides.Override("precision", overrides.PRECISION_BF16)
    )
    assert m.chosen == "lora"
    p = overridden_decision(
        "precision", overrides.Override("precision", overrides.PRECISION_BF16)
    )
    assert p.chosen == overrides.PRECISION_BF16
    assert p.overridden is True


def test_the_nf4_alternative_for_a_lora_plan_uses_nf4s_own_pool():
    """The precision reason's numbers stay honest when method is overridden:
    the NF4 alternative's pool is qlora's own 0.5 bytes/param arithmetic
    (about 2.0 GB for the 4B model), never the bf16 pool dressed up as
    NF4."""
    p = overridden_decision("precision", overrides.Override("method", "lora"))
    assert p.chosen == overrides.PRECISION_BF16
    alt = p.alternatives[0]
    assert alt.value == overrides.PRECISION_NF4
    assert "0.5 bytes/param" in alt.cost
    assert "shrinks to 2.0 GB" in alt.cost


def test_overriding_hardware_picks_that_card_and_marks_it():
    with_h100 = [
        GpuAvailability("L4", L4_INR_PER_HOUR, 8),
        GpuAvailability("H100", 250.0, 2),
    ]
    h = overridden_decision(
        "hardware",
        overrides.Override("hardware", "H100"),
        availability=with_h100,
    )
    assert h.overridden is True
    assert h.chosen == "H100"


def test_overriding_the_disk_marks_it_and_prices_the_extra():
    d = overridden_decision("disk", overrides.Override("disk", "500 GB"))
    assert d.overridden is True
    assert d.chosen == "500 GB"
    assert "chose" in d.constraint


def test_overriding_device_count_offers_the_single_card_as_an_alternative():
    n = overridden_decision(
        "device count", overrides.Override("device count", "2")
    )
    assert n.overridden is True
    assert n.chosen == "2"
    values = [a.value for a in n.alternatives]
    assert "1 × L4" in values


def test_an_overridden_method_names_what_the_trainer_cannot_run():
    m = overridden_decision("method", overrides.Override("method", "lora"))
    assert m.chosen == "lora"
    assert m.overridden is True
    assert "executes only" in m.constraint
    assert "refused" in m.constraint
