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

from temper_core import decisions, disk, hyperparams, selection
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
    assert set(as_dict) == {"decision", "chosen", "constraint", "alternatives"}
    assert as_dict["decision"] == "hardware"
    for a in as_dict["alternatives"]:
        assert set(a) == {"value", "cost", "constraint"}
