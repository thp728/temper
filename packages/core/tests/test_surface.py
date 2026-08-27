"""The generated advanced surface: completeness, determinism, the refusal gate.

Spec 009 / issue #33. The exposed surface is generated from the pinned
trainer's configuration schema, the tier classification is a checked-in data
file, a field in no tier fails the completeness check, and generation is
deterministic for a given pinned image. These tests assert the generated
surface's contract -- the tiers, the completeness guarantee, the refusal
vocabulary -- rather than the mechanism that produced it.

Two things are pinned against the real data files so an edit to either fails a
test instead of silently changing every launch: the overrideable set (the
reachable surface) and the schema snapshot's source image (so a pinned-image
move without a regenerated schema fails the gate).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from temper_core import surface

REPO_ROOT = Path(surface.SCHEMA_PATH).parents[2]


def _synthetic(
    fields: list[str],
    entries: list[dict],
) -> tuple[dict, dict]:
    """Schema + tier documents for driving `check_complete` without touching
    the checked-in data."""
    schema = {"fields": [{"name": n, "type": "int"} for n in fields]}
    tiers = {"fields": entries}
    return schema, tiers


# --- completeness ------------------------------------------------------------


def test_every_schema_field_lands_in_exactly_one_tier():
    """The completeness assertion rather than a sample: the checked-in tier
    file covers every one of the schema's fields, and the module itself
    refuses to import with a hole."""
    tiers = {e["name"]: e["tier"] for e in surface.TIERS_DOC["fields"]}
    assert len(tiers) == len(surface.FIELDS)
    assert set(tiers) == set(surface.FIELDS)
    assert all(t in surface.TIERS for t in tiers.values())


def test_the_three_tiers_are_the_specs_three():
    assert set(surface.TIERS) == {
        "calculated",
        "exposed_with_named_failure_mode",
        "known_but_unsupported",
    }


def test_a_field_in_no_tier_fails_the_completeness_check():
    schema, _ = _synthetic(["a", "b"], [])
    with pytest.raises(
        surface.CompletenessError, match="no tier classification"
    ):
        surface.check_complete(schema, {"fields": []})
    with pytest.raises(
        surface.CompletenessError, match="no tier classification: b"
    ):
        surface.check_complete(
            schema, {"fields": [{"name": "a", "tier": "calculated"}]}
        )


def test_a_tier_entry_for_a_non_field_fails():
    schema, _ = _synthetic(["a"], [])
    with pytest.raises(surface.CompletenessError, match="not trainer schema"):
        surface.check_complete(
            schema,
            {
                "fields": [
                    {"name": "a", "tier": "calculated"},
                    {"name": "ghost", "tier": "calculated"},
                ]
            },
        )


def test_a_field_in_two_tiers_fails():
    schema, _ = _synthetic(["a"], [])
    with pytest.raises(surface.CompletenessError, match="more than once"):
        surface.check_complete(
            schema,
            {
                "fields": [
                    {"name": "a", "tier": "calculated"},
                    {"name": "a", "tier": "known_but_unsupported"},
                ]
            },
        )


def test_an_unknown_tier_fails():
    schema, _ = _synthetic(["a"], [])
    with pytest.raises(surface.CompletenessError, match="unknown tier"):
        surface.check_complete(
            schema, {"fields": [{"name": "a", "tier": "exposed"}]}
        )


def test_every_entry_carries_a_reason_and_every_exposed_a_failure_mode():
    for entry in surface.TIER_ENTRIES.values():
        assert entry["reason"].strip(), entry["name"]
    for name, entry in surface.exposed_fields().items():
        assert entry["failure_mode"].strip(), name


# --- determinism -------------------------------------------------------------


def test_generation_is_deterministic_for_the_pinned_image():
    first = surface.surface_document()
    second = surface.surface_document()
    assert first == second


def test_writing_the_surface_is_deterministic(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    surface.write_surface(a)
    surface.write_surface(b)
    assert a.read_bytes() == b.read_bytes()


def test_the_surface_names_its_source_image_and_versions():
    doc = surface.surface_document()
    assert doc["_source_image"].startswith("axolotlai/axolotl:")
    assert "@sha256:" in doc["_source_image"]
    assert doc["_axolotl_version"]
    assert (
        doc["_config_model"]
        == "axolotl.utils.schemas.config.AxolotlInputConfig"
    )


def test_the_schema_snapshot_matches_the_pinned_image_digest():
    """The surface cannot drift when the pinned image moves: the schema
    snapshot must be for the same digest the Dockerfile pins. A digest change
    (#44) without a regenerated schema fails here, making the change a
    deliberate, visible act rather than a silent one."""
    dockerfile = (REPO_ROOT / "apps" / "trainer" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^FROM\s+(\S+@sha256:\S+)", dockerfile, re.M)
    assert match, "no pinned FROM line found in the trainer Dockerfile"
    assert surface.SCHEMA["_source_image"] == match.group(1)


# --- the reachable surface ---------------------------------------------------


def test_the_overrideable_set_is_the_exposed_tier_plus_platform_internal():
    expected = set(surface.exposed_fields()) | surface.PLATFORM_INTERNAL_KEYS
    assert surface.overrideable_keys() == frozenset(expected)


def test_the_reachable_surface_is_the_current_override_set():
    """The reachable set is pinned so a change to it is a deliberate diff in
    the tier data file, not a silent behavioural shift on the create path."""
    assert surface.overrideable_keys() == {
        "lora_r",
        "lora_alpha",
        "learning_rate",
        "num_epochs",
        "max_steps",
        "sequence_len",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "val_set_size",
        "save_steps",
        "simulated_failure_code",
    }


def test_the_resolver_accepts_exactly_the_reachable_surface():
    """`hyperparams.effective` applies only what the surface exposes, so the
    page, the gate and the frozen spec agree on what is reachable."""
    from temper_core import hyperparams

    assert hyperparams.ALLOWED_OVERRIDES == set(surface.overrideable_keys())


def test_known_keys_are_the_schema_field_names():
    assert surface.known_keys() == frozenset(surface.FIELDS)
    # The defaults the resolver ships are all trainer-known keys.
    from temper_core import hyperparams

    assert set(hyperparams.DEFAULTS) <= surface.known_keys()


# --- the refusal gate --------------------------------------------------------


def test_an_unknown_key_is_refused_and_echoed_back():
    refusals = surface.validate_overrides({"lora_r": 32, "maxSteps": 5})
    assert refusals[0]["code"] == "unknown_hyperparameter"
    assert refusals[0]["unknown"] == ["maxSteps"]


def test_a_known_but_unsupported_key_is_refused_with_its_reason():
    refusals = surface.validate_overrides({"wandb_project": "x"})
    assert refusals[0]["code"] == "unsupported_hyperparameter"
    assert refusals[0]["name"] == "wandb_project"
    assert "reason" in refusals[0]
    assert (
        refusals[0]["reason"] == surface.unsupported_fields()["wandb_project"]
    )


def test_a_calculated_platform_field_is_refused_with_its_reason():
    """A field the platform sets (a correctness setting) is refused as an
    override, and the refusal says why rather than 'unknown key'."""
    refusals = surface.validate_overrides({"train_on_inputs": True})
    assert refusals[0]["code"] == "unsupported_hyperparameter"
    assert (
        refusals[0]["reason"] == surface.calculated_fields()["train_on_inputs"]
    )


def test_an_exposed_override_is_accepted():
    assert surface.validate_overrides({"lora_r": 32}) == []
    assert surface.validate_overrides({"sequence_len": 4096}) == []
    assert (
        surface.validate_overrides({"simulated_failure_code": "gpu_stalled"})
        == []
    )


def test_a_value_outside_the_schema_bounds_is_refused():
    """The combinations the schema *does* express are validated before launch.
    `relora_prune_ratio` carries a ge=0 / le=1 constraint in the snapshot."""
    violation = surface._schema_constraint_violation("relora_prune_ratio", 2.0)
    assert violation is not None and "maximum" in violation
    violation = surface._schema_constraint_violation("relora_prune_ratio", -1)
    assert violation is not None and "minimum" in violation


def test_a_value_outside_the_schema_enum_is_refused():
    violation = surface._schema_constraint_violation(
        "optimizer", "not_an_optimizer"
    )
    assert violation is not None and "not one of" in violation


def test_an_empty_override_dict_is_accepted():
    assert surface.validate_overrides({}) == []
    assert surface.validate_overrides(None) == []


# --- the combinations that cannot be known ------------------------------------


def test_the_runtime_only_validators_are_named_not_estimated():
    rv = surface.runtime_only_validators()
    assert len(rv["model_validators"]) >= 100
    assert rv["field_validator_count"] >= 20
    assert "check_fsdp_deepspeed" in rv["model_validators"]
    # The generated document carries them too, so the interface can name them.
    assert surface.surface_document()["runtime_only_validators"] == rv
