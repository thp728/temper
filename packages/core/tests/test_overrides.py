"""Plan overrides -- spec 005, issue #79.

The experienced-user half of the predictor: any decision it made can be
changed, and changing one re-requests the plan rather than mutating it
locally. This module owns the vocabulary -- what each of the six decisions can
be overridden to, and how method and precision move together -- so a good test
here asserts on what `resolve` returns for a set of `{decision, value}`
pairs: the typed pins the rest of the predictor recomputes from, and the
refusals for a value or pairing that describes no real configuration.

A good test asserts on the resolved pins and the stable refusal codes, never
on wording. The coupling (qlora <-> nf4, lora/full <-> bf16) is the
load-bearing rule, and refusing an inconsistent pair rather than silently
resolving it is the safety property.
"""

from __future__ import annotations

import pytest

from temper_core import hyperparams, overrides
from temper_core.overrides import (
    PRECISION_BF16,
    PRECISION_NF4,
    InconsistentOverridesError,
    InvalidOverrideValueError,
    Override,
    UnknownDecisionError,
    UnknownOverrideValueError,
    resolve,
)


def ov(decision, value):
    return Override(decision=decision, value=value)


def base_hp():
    return hyperparams.effective({})


# --- the vocabulary -----------------------------------------------------------


def test_the_six_decisions_are_the_acceptance_criteria_order():
    assert overrides.DECISIONS == (
        "method",
        "hardware",
        "device count",
        "disk",
        "precision",
        "sequence length",
    )


def test_method_and_precision_are_one_coupled_decision():
    """The coupling is load-bearing: qlora quantises to NF4, lora and full
    hold bf16. A method that 'chooses' the other precision describes no real
    configuration."""
    assert overrides.METHOD_PRECISION == {
        "qlora": PRECISION_NF4,
        "lora": PRECISION_BF16,
        "full": PRECISION_BF16,
    }


# --- resolution ---------------------------------------------------------------


def test_an_empty_override_list_pins_nothing():
    r = resolve(base_hp(), [])
    assert r.method is None
    assert r.gpu_type is None
    assert r.device_count is None
    assert r.disk_gb is None
    assert r.precision is None
    assert r.overridden == frozenset()
    assert r.hyperparameters["sequence_len"] == base_hp()["sequence_len"]


def test_each_decision_resolves_to_its_typed_pin():
    r = resolve(
        base_hp(),
        [
            ov("method", "lora"),
            ov("hardware", "H100"),
            ov("device count", "2"),
            ov("disk", "200 GB"),
            ov("precision", "bf16 (no quantisation)"),
            ov("sequence length", "4096"),
        ],
    )
    assert r.method == "lora"
    assert r.gpu_type == "H100"
    assert r.device_count == 2
    assert r.disk_gb == 200
    assert r.precision == PRECISION_BF16
    assert r.hyperparameters["sequence_len"] == 4096
    assert r.overridden == frozenset(overrides.DECISIONS)


def test_sequence_length_override_becomes_the_effective_hyperparameter():
    """The rest of the plan recomputes around a sequence-length change because
    it lands in the same effective spec the predictor reads -- one resolver."""
    r = resolve(base_hp(), [ov("sequence length", "1024")])
    assert r.hyperparameters["sequence_len"] == 1024
    assert r.hyperparameters["lora_r"] == base_hp()["lora_r"]


def test_naming_only_the_precision_derives_the_method():
    """'bf16' *means* LoRA; 'nf4' *means* QLoRA. The user named the dtype, so
    the method follows rather than being left stale."""
    assert (
        resolve(base_hp(), [ov("precision", PRECISION_BF16)]).method == "lora"
    )
    assert (
        resolve(base_hp(), [ov("precision", PRECISION_NF4)]).method == "qlora"
    )


def test_naming_only_the_method_leaves_the_precision_derivable():
    r = resolve(base_hp(), [ov("method", "full")])
    assert r.method == "full"
    assert r.precision is None  # derived by decisions.decide from the method


def test_a_consistent_precision_and_method_pair_resolves():
    r = resolve(
        base_hp(),
        [ov("method", "lora"), ov("precision", PRECISION_BF16)],
    )
    assert r.method == "lora"
    assert r.precision == PRECISION_BF16


# --- refusals -----------------------------------------------------------------


def test_an_unknown_decision_is_refused_with_a_stable_code():
    with pytest.raises(UnknownDecisionError) as exc:
        resolve(base_hp(), [ov("quantisation", "nf4")])
    assert exc.value.code == "unknown_decision"
    assert exc.value.decision == "quantisation"
    assert set(exc.value.fields["known"]) == set(overrides.DECISIONS)


def test_an_unknown_method_value_is_refused():
    with pytest.raises(UnknownOverrideValueError) as exc:
        resolve(base_hp(), [ov("method", "deepseek-r1")])
    assert exc.value.code == "unknown_override_value"
    assert "allowed" in exc.value.fields


def test_an_unknown_gpu_type_is_refused():
    with pytest.raises(UnknownOverrideValueError):
        resolve(base_hp(), [ov("hardware", "T4")])


def test_a_non_integer_device_count_is_refused():
    with pytest.raises(UnknownOverrideValueError):
        resolve(base_hp(), [ov("device count", "two")])


def test_a_zero_device_count_is_refused():
    with pytest.raises(InvalidOverrideValueError) as exc:
        resolve(base_hp(), [ov("device count", "0")])
    assert exc.value.code == "invalid_override_value"


def test_a_disk_value_not_in_gb_is_refused():
    with pytest.raises(UnknownOverrideValueError):
        resolve(base_hp(), [ov("disk", "200 megabytes")])


def test_an_unknown_precision_is_refused():
    with pytest.raises(UnknownOverrideValueError):
        resolve(base_hp(), [ov("precision", "fp8 (8-bit)")])


def test_an_inconsistent_method_and_precision_pair_is_refused():
    """Naming both and having them disagree is refused, never silently
    resolved to one side -- a configuration that contradicts itself describes
    a job nobody intended."""
    with pytest.raises(InconsistentOverridesError) as exc:
        resolve(
            base_hp(),
            [ov("method", "qlora"), ov("precision", PRECISION_BF16)],
        )
    assert exc.value.code == "inconsistent_overrides"
    assert exc.value.method == "qlora"
    assert exc.value.precision == PRECISION_BF16


# --- executability ------------------------------------------------------------


def test_only_what_the_trainer_runs_today_is_executable():
    assert overrides.executable("qlora", 1) is True
    assert overrides.executable("full", 1) is True  # taught as of issue #66
    assert overrides.executable("lora", 1) is False  # not taught yet
    assert overrides.executable("qlora", 2) is False  # multi-GPU never shipped
    assert overrides.executable("full", 2) is False
    assert overrides.executable("lora", 2) is False


def test_the_default_method_is_the_lightest_executable_one():
    """A job with no method override is picked by the predictor, so the
    'default' the launch gate names when only a device count was overridden is
    the lightest executable method -- the configuration closest to what such a
    job would actually run, never full (the heaviest)."""
    assert overrides.executable_default() == "qlora"


def test_executable_is_defined_once_not_retyped():
    """The executable set is selection's, read not retyped -- the day spec 009
    teaches another method there is one place that says so."""
    from temper_core import selection as selection_mod

    assert overrides.executable("qlora", 1) == (
        "qlora" in selection_mod.EXECUTABLE_METHODS
    )


# --- serialization ------------------------------------------------------------


def test_an_override_round_trips_through_its_dict():
    original = Override("hardware", "H100")
    assert overrides.from_dict(overrides.to_dict(original)) == original
    assert overrides.to_dict(original) == {
        "decision": "hardware",
        "value": "H100",
    }
