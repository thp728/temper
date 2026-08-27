"""The advanced configuration surface, generated from the trainer's own schema.

Spec 009 / issue #33. The exposed set is derived from the configuration schema
of the pinned trainer image rather than hand-curated, which reconciles *expose
every dial* with *refuse unknown keys loudly*: the surface is exactly as wide
as the trainer and no wider; it cannot claim support for something the trainer
lacks; it cannot silently drop something the trainer has; and it cannot drift
when the pinned image moves, because it is derived from that image.

Three things this module reads, all data, never code:

* `packages/contracts/axolotl-schema.json` -- the schema snapshot of the pinned
  image (the universe: a key not here is a key the trainer does not know).
* `packages/contracts/axolotl-field-tiers.json` -- the classification: every
  field lands in exactly one of three tiers, and a field in no tier fails the
  completeness check below. It is data rather than code so a classification is
  reviewable in a diff.

The tiers (Spec 009):

* **calculated** -- the platform sets it (the resolver, the entrypoint or the
  orchestrator); the common path never sees it, and an override is refused with
  the reason.
* **exposed with a named failure mode** -- reachable, with the specific thing
  that goes wrong written beside it, not a general warning.
* **known but unsupported here** -- present in the trainer, not offered, with
  the reason stated. Refused loudly if passed -- the refusal can say *why*
  rather than 'unknown key'.

`validate_overrides` is the pre-launch gate. It refuses, before anything is
priced or provisioned: keys unknown to the trainer (echoed back, per the
unknown-key rule), keys the trainer knows but the platform does not expose
(with the reason), and exposed values that violate the constraints the schema
*does* express (enum membership, bounds). The constraints the schema does not
express are the risk Spec 009 names: the schema snapshot also carries the
runtime-only validators Axolotl enforces in arbitrary Python, surfaced here as
the combinations that cannot be known ahead of launch, because a generated form
cannot know what those will refuse until the job is already running.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The three tiers, named once. The tier file uses these strings; everything
# that branches on tier reads these constants rather than retyping them.
TIERS: tuple[str, ...] = (
    "calculated",
    "exposed_with_named_failure_mode",
    "known_but_unsupported",
)

CALCULATED_TIER = "calculated"
EXPOSED_TIER = "exposed_with_named_failure_mode"
UNSUPPORTED_TIER = "known_but_unsupported"

# Keys the platform itself carries that are not trainer fields. They travel in
# the same `hyperparameters` dict (so the create path validates them like any
# other key) but they are not part of the trainer's schema, so they live here,
# not in the tier file: `simulated_failure_code` is how a journey asks the
# simulated machine to fail on demand (ADR-0026).
PLATFORM_INTERNAL_KEYS: frozenset[str] = frozenset(
    {
        "simulated_failure_code",
    }
)


class CompletenessError(Exception):
    """A field in the schema has no tier, or a tier entry is not a field.

    A field in no tier is a hole through which an unclassified control reaches
    a user; a tier entry that names no schema field is a classification that
    drifted away from the trainer. Both are build failures, not runtime noise:
    the tier file and the schema snapshot must agree field-for-field.
    """


def _contract_path(name: str) -> Path:
    """Find a contract file by walking up to the workspace root.

    The same lookup `temper_core.hyperparams` uses for the defaults contract,
    so a wrong resolve would be caught by the same reasoning: this package
    reads data files out of `packages/contracts`, and a missing one stops the
    process rather than defaulting (the config.py rule).
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "packages" / "contracts" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"packages/contracts/{name} not found above {__file__}; the workspace "
        "tree is incomplete"
    )


SCHEMA_PATH = _contract_path("axolotl-schema.json")
TIERS_PATH = _contract_path("axolotl-field-tiers.json")

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
TIERS_DOC = json.loads(TIERS_PATH.read_text(encoding="utf-8"))

# name -> the schema field record
FIELDS: dict[str, dict[str, Any]] = {f["name"]: f for f in SCHEMA["fields"]}
# name -> the tier entry (tier, reason, optional failure_mode)
TIER_ENTRIES: dict[str, dict[str, Any]] = {
    e["name"]: e for e in TIERS_DOC["fields"]
}


def check_complete(schema: dict[str, Any], tiers_doc: dict[str, Any]) -> None:
    """The completeness check: the schema and the classification agree.

    Every schema field lands in exactly one known tier; a tier entry names a
    schema field; nothing is duplicated. A hole -- a field in no tier -- is a
    control that would reach a user unclassified, and an extra is a
    classification that drifted away from the trainer, so both are build
    failures. The module runs this at import; tests run it against synthetic
    documents to prove each hole shape is caught.
    """
    known_tiers = set(TIERS)
    fields = {f["name"]: f for f in schema["fields"]}
    _counts: dict[str, int] = {}
    for entry in tiers_doc["fields"]:
        _counts[entry["name"]] = _counts.get(entry["name"], 0) + 1
    entries = {e["name"]: e for e in tiers_doc["fields"]}

    unknown_tier = [
        f"{name}: {entry['tier']}"
        for name, entry in entries.items()
        if entry["tier"] not in known_tiers
    ]
    if unknown_tier:
        raise CompletenessError(
            "tier entries with unknown tiers: " + ", ".join(unknown_tier)
        )
    duplicate = sorted(
        {name for name, _count in _counts.items() if _count > 1}
    )
    if duplicate:
        raise CompletenessError(
            "fields classified more than once: " + ", ".join(duplicate)
        )
    missing = sorted(set(fields) - set(entries))
    if missing:
        raise CompletenessError(
            "fields in the trainer schema with no tier classification: "
            + ", ".join(missing)
        )
    extra = sorted(set(entries) - set(fields))
    if extra:
        raise CompletenessError(
            "tier entries that are not trainer schema fields: "
            + ", ".join(extra)
        )


# Classification, computed once and completeness-checked. A hole here is a
# build failure, so the module refuses to import with one.
check_complete(SCHEMA, TIERS_DOC)


def known_keys() -> frozenset[str]:
    """Every key the trainer knows, from its own schema.

    'Unknown' now means unknown to the trainer rather than absent from a
    hand-written list: this set is the trainer's schema field names, which is
    what both the control-plane gate and the trainer's own guard read.
    """
    return frozenset(FIELDS)


def tier_of(name: str) -> str:
    """The tier a field lands in, or CompletenessError for an unknown field."""
    entry = TIER_ENTRIES.get(name)
    if entry is None:
        raise CompletenessError(
            f"'{name}' is not a trainer schema field"
        ) from None
    return str(entry["tier"])


def classify() -> dict[str, str]:
    """name -> tier for every schema field. The completeness guarantee, as a
    mapping: every field appears exactly once because the module refuses to
    import with a hole, a duplicate or an extra."""
    return {name: tier_of(name) for name in sorted(FIELDS)}


def exposed_fields() -> dict[str, dict[str, str]]:
    """The exposed tier: name -> {reason, failure_mode}. What a user may
    override, and the specific thing that goes wrong if they get it wrong."""
    out: dict[str, dict[str, str]] = {}
    for name, entry in TIER_ENTRIES.items():
        if entry["tier"] == EXPOSED_TIER:
            fm = entry.get("failure_mode")
            out[name] = {
                "reason": entry["reason"],
                "failure_mode": fm if isinstance(fm, str) else "",
            }
    return out


def unsupported_fields() -> dict[str, str]:
    """The known-but-unsupported tier: name -> reason. Visible in the surface
    so a field is not wondered about as overlooked; refused if passed."""
    return {
        name: entry["reason"]
        for name, entry in TIER_ENTRIES.items()
        if entry["tier"] == UNSUPPORTED_TIER
    }


def calculated_fields() -> dict[str, str]:
    """The calculated tier: name -> reason. The platform sets these; the
    common path never sees them."""
    return {
        name: entry["reason"]
        for name, entry in TIER_ENTRIES.items()
        if entry["tier"] == CALCULATED_TIER
    }


def overrideable_keys() -> frozenset[str]:
    """The keys the create path accepts: the exposed tier plus the platform's
    own internal keys. Everything else in the trainer's schema is refused with
    its reason before launch."""
    return frozenset(exposed_fields()) | PLATFORM_INTERNAL_KEYS


def value_kind(name: str) -> str:
    """How a field's value is rendered and typed: 'int', 'float' or 'string'.

    Derived from the schema snapshot's own type string, not retyped: the type
    tokens are split on '|', and an int or float anywhere in the union wins
    (a field typed `str | float`, like `learning_rate`, is a float). The
    interface generates its inputs from this, and `coerce_value` uses it, so
    the two cannot disagree about what an exposed value is.
    """
    type_str = str(FIELDS[name].get("type", ""))
    tokens = {t.strip() for t in type_str.split("|")}
    if "int" in tokens:
        return "int"
    if "float" in tokens:
        return "float"
    return "string"


def coerce_value(name: str, raw: Any) -> Any:
    """`raw` as the schema's type for `name`, or unchanged when it cannot be.

    An override arrives as a string from a browser form; the schema knows it
    is an int or a float, so this normalises it to the typed value before it
    reaches the resolver, the memory/disk arithmetic and the trainer's own
    config -- a string that reaches a numeric field is a silent type drift.
    A value the type cannot express is left as-is: refusing it is the
    schema's enum/bounds gate's job (`validate_overrides`), not this one's.
    """
    if raw is None or not isinstance(raw, str):
        return raw
    kind = value_kind(name)
    try:
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
    except ValueError:
        return raw
    return raw


def runtime_only_validators() -> dict[str, list[str] | int]:
    """The combinations that cannot be known ahead of launch, named as such.

    Axolotl enforces these in arbitrary Python (`model_validator` /
    `field_validator`), so a generated form cannot know what they will refuse
    until the job is running. Spec 009 says they must be named rather than
    pretended away; this is the name, read from the schema snapshot. The model
    validators are named; the field-validator names were not captured by the
    introspection (pydantic exposes them as a count), so the count is
    recorded, with the schema snapshot's note saying so.
    """
    rv = SCHEMA.get("runtime_validators", {})
    return {
        "model_validators": list(rv.get("model_validator_names", [])),
        "field_validator_count": int(rv.get("field_validator_count", 0)),
    }


# Tokens in a field's type string that mean its captured enum is a PARTIAL
# vocabulary rather than the exhaustive one. A field typed
# `Union[Literal['auto'], bool]` (bf16, tf32, torch_compile, ...) or
# `str | SomeEnum` (optimizer, chat_template) legitimately accepts values
# outside the captured list, so enforcing the list as exhaustive would refuse
# valid values -- `bf16: true` is the platform's own config, and refusing it
# would be a gate that blocks what the trainer accepts. Enforcement only
# applies where the type is a bare Literal/Enum (optionally optional).
_PARTIAL_ENUM_TOKENS = (
    "bool",
    "str",
    "list",
    "dict",
    "Any",
    "PIL.",
)


def _schema_constraint_violation(name: str, value: Any) -> str | None:
    """A reason the value violates what the schema *does* express, or None.

    The schema snapshot records enum membership and the few numeric bounds
    (ge/le) Axolotl declares. These are the combinations that can be validated
    ahead of launch; the runtime-only validators (see
    `runtime_only_validators`) are the ones that cannot, and are named as such.
    A partial enum (a Literal unioned with bool/str/list/... or another enum)
    is not enforced, because the captured list is not the whole vocabulary and
    refusing outside it would block values the trainer accepts.
    """
    field = FIELDS[name]
    enums = field.get("enum_values") or []
    partial = any(
        tok in str(field.get("type", "")) for tok in _PARTIAL_ENUM_TOKENS
    )
    if enums and value is not None and not partial:
        if str(value) not in enums:
            return f"'{value}' is not one of: {', '.join(sorted(enums))}."
    for raw in field.get("constraints") or []:
        kind, _, bound = raw.partition("=")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if kind == "ge" and numeric < float(bound):
            return f"{value} is below the schema minimum of {bound}."
        if kind == "le" and numeric > float(bound):
            return f"{value} is above the schema maximum of {bound}."
    return None


def validate_overrides(
    overrides: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """The pre-launch gate over a user's override dict.

    Returns refusals, newest vocabulary first: keys unknown to the trainer
    (one combined refusal, echoed back, code `unknown_hyperparameter`); keys
    the trainer knows but the platform does not expose (one per key, carrying
    the reason, code `unsupported_hyperparameter`); and exposed values that
    violate the schema's expressed constraints (code `invalid_hyperparameter`).
    An empty list means the overrides are all acceptable, which is not the same
    as a promise they will train: the runtime-only validators named in
    `runtime_only_validators` are beyond what a generated form can know.
    """
    if not overrides:
        return []
    refusals: list[dict[str, Any]] = []

    unknown = sorted(
        k
        for k in overrides
        if k not in FIELDS and k not in PLATFORM_INTERNAL_KEYS
    )
    if unknown:
        refusals.append(
            {
                "code": "unknown_hyperparameter",
                "message": (
                    "Unknown hyperparameter keys are refused: "
                    f"{unknown}. Nothing was launched."
                ),
                "unknown": unknown,
            }
        )

    for name in sorted(overrides):
        if name not in FIELDS:
            continue
        if name in PLATFORM_INTERNAL_KEYS:
            continue
        if tier_of(name) != EXPOSED_TIER:
            reason = TIER_ENTRIES[name]["reason"]
            refusals.append(
                {
                    "code": "unsupported_hyperparameter",
                    "message": (
                        f"'{name}' is a setting the trainer supports but this "
                        f"platform does not expose: {reason}"
                    ),
                    "name": name,
                    "reason": reason,
                }
            )
            continue
        violation = _schema_constraint_violation(name, overrides[name])
        if violation is not None:
            refusals.append(
                {
                    "code": "invalid_hyperparameter",
                    "message": f"'{name}': {violation}",
                    "name": name,
                    "reason": violation,
                }
            )
    return refusals


def surface_document() -> dict[str, Any]:
    """The generated advanced surface, as one deterministic document.

    Everything the interface needs to render the advanced surface -- the known
    keys, the overrideable set, each field's tier with its reason and failure
    mode, and the runtime-only validators named as not-pre-checkable -- derived
    from the schema snapshot and the tier data file. Deterministic for a given
    pinned image: it is a pure function of two checked-in data files.
    """
    tiers_doc: dict[str, Any] = {}
    for tier in TIERS:
        tiers_doc[tier] = {}
    for name, entry in TIER_ENTRIES.items():
        doc: dict[str, Any] = {
            "tier": entry["tier"],
            "reason": entry["reason"],
        }
        if entry["tier"] == EXPOSED_TIER:
            if isinstance(entry.get("failure_mode"), str):
                doc["failure_mode"] = entry["failure_mode"]
            # The render/coerce type, derived from the schema snapshot (see
            # `value_kind`): the interface generates its input from this.
            doc["type"] = value_kind(name)
        tiers_doc[entry["tier"]][name] = doc

    counts = {tier: len(tiers_doc[tier]) for tier in TIERS}
    return {
        "_source_image": SCHEMA["_source_image"],
        "_axolotl_version": SCHEMA["_axolotl_version"],
        "_config_model": SCHEMA["_config_model"],
        "known_keys": sorted(FIELDS),
        "platform_internal_keys": sorted(PLATFORM_INTERNAL_KEYS),
        "overrideable_keys": sorted(overrideable_keys()),
        "tiers": tiers_doc,
        "counts": counts,
        "runtime_only_validators": runtime_only_validators(),
    }


def write_surface(destination: Path) -> Path:
    """Materialise `surface_document` for review and for the interface.

    The checked-in `packages/contracts/advanced-surface.json` is generated by
    `just contracts` and drift-checked by `just contracts-check`, so a change
    to the schema snapshot or the tier file that is not regenerated here fails
    the gate.
    """
    document = surface_document()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def main() -> None:
    print(write_surface(TIERS_PATH.parent / "advanced-surface.json"))


if __name__ == "__main__":
    main()
