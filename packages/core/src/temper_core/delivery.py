"""Delivery formats: the forms an artifact is offered in (issue #74).

The product offers more than one form of the finished result (spec 011): the
trained change as it is, a merged single-file model for serving, and a
quantised local-inference format for running on your own machine. This module
is the single definition of that vocabulary -- what a format is called, what
artifact kind it maps to, what it is for in plain language, and which
conversion produces it -- read by the control plane (through `delivery.py`) and
the trainer image (through `packages/contracts/delivery-formats.json`, since
the trainer never installs this package, ADR-0010). A value two components
must agree on is defined once and read, never retyped.

The load-bearing property is **order**. The issue states it as the difference
between a usable local model and a subtly degraded one: merging into an
already-quantised base compounds error, and a model merged in the wrong order
still loads and answers, only slightly worse. So `production_steps` cannot
produce an order that quantises before it merges -- the function *is* the
assertion, and the property test proves it over every subset of the vocabulary.

Everything here is pure data over strings -- no I/O beyond the read of the
contract at import (the same read `hyperparams` makes), no framework -- because
the control plane, the trainer, the manifest and the interface must all read
the same vocabulary from one source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The as-trained format needs no conversion; the other two name the step that
# produces them. These strings are the vocabulary the API, the trainer and the
# interface share; a format id is never re-spelled anywhere else.
DELIVERY_FORMAT_ADAPTER = "adapter"
DELIVERY_FORMAT_MERGED = "merged"
DELIVERY_FORMAT_QUANTISED = "quantised"

# The conversions, in the only order they are legal in. `quantise` must follow
# `merge` at full precision; a quantised output that skipped the merge would be
# the doubly-approximated model the issue names. This tuple is the order
# property: `production_steps` builds from it and cannot put `quantise` first.
CONVERSION_ORDER: tuple[str, ...] = ("merge", "quantise")

# A format whose production requires the converted source. quantised is produced
# from the merged model (never from the adapter or a quantised base), so
# requesting it implies the merge step runs first.
CONVERSION_SOURCE: dict[str, str] = {
    DELIVERY_FORMAT_QUANTISED: DELIVERY_FORMAT_MERGED,
}


def _contract_path() -> Path:
    """Find the delivery-format contract by walking up to the workspace root.

    The same lookup `hyperparams` and the trainer's own contract reads use:
    marked on `pyproject.toml` beside `packages/` rather than depth-coded, so a
    move does not break silently.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "packages" / "contracts" / "delivery-formats.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "packages/contracts/delivery-formats.json not found above "
        f"{__file__}; the workspace tree is incomplete"
    )


_CONTRACT = json.loads(_contract_path().read_text(encoding="utf-8"))


@dataclass(frozen=True)
class DeliveryFormat:
    """One delivery format: id, the artifact kind it maps to, its purpose, and
    the conversion (if any) that produces it."""

    id: str
    kind: str
    conversion: str | None
    what_for: str


def _formats() -> dict[str, DeliveryFormat]:
    out: dict[str, DeliveryFormat] = {}
    for row in _CONTRACT["formats"]:
        out[row["id"]] = DeliveryFormat(
            id=row["id"],
            kind=row["kind"],
            conversion=row.get("conversion"),
            what_for=row["what_for"],
        )
    return out


FORMATS: dict[str, DeliveryFormat] = _formats()

# The artifact kind each delivery format maps to. Defined here because a format
# is a delivery-time concept and a kind is an artifact-time concept; the map is
# the one place the two vocabularies meet.
KIND_BY_FORMAT: dict[str, str] = {fmt.id: fmt.kind for fmt in FORMATS.values()}


class UnknownDeliveryFormat(ValueError):
    """A format id this module does not define.

    Raised rather than guessed: a format that does not exist cannot be
    produced, verified, or explained, and inventing a purpose for one would
    hand a user a choice the platform does not offer.
    """


class IllegalProductionOrder(ValueError):
    """A requested production order violates the merge-then-quantise property.

    Raised rather than silently corrected: the issue says the failure this
    prevents is silent, so the order must be enforced, not narrated. A caller
    that asks to quantise without first merging has asked for the doubly
    approximated model, and this is the refusal.
    """


def known_format(format_id: str) -> bool:
    """Whether `format_id` is a defined delivery format."""
    return format_id in FORMATS


def require_known(format_id: str) -> DeliveryFormat:
    try:
        return FORMATS[format_id]
    except KeyError:
        raise UnknownDeliveryFormat(
            f"{format_id!r} is not a defined delivery format. "
            f"Known formats: {', '.join(sorted(FORMATS))}."
        ) from None


def what_for(format_id: str) -> str:
    """What `format_id` is for, in plain language without the internals.

    This is the sentence the interface shows beside the download, so a user
    can choose a format without knowing what a merge or a quantisation is.
    """
    return require_known(format_id).what_for


def kind_for(format_id: str) -> str:
    """The artifact kind a delivery format is delivered as."""
    return require_known(format_id).kind


def production_steps(requested: Any) -> list[str]:
    """The ordered conversions that produce `requested`, or an empty list.

    `requested` is the set of delivery format ids the job asked for. The
    as-trained format (`adapter`) needs no conversion and is never returned.
    Every other format is resolved to its conversion, in `CONVERSION_ORDER`,
    with dependencies included: requesting `quantised` alone still runs the
    merge first, because quantisation consumes the correctly merged model.

    This function *is* the merge-then-quantise order property: it cannot
    return a step list where `quantise` precedes `merge`, and it refuses a
    request it cannot satisfy rather than guessing. The property test asserts
    the invariant over every subset of the vocabulary.
    """
    if requested is None:
        return []
    if isinstance(requested, str):
        raise UnknownDeliveryFormat(
            f"delivery request must be a list of format ids, got a string "
            f"{requested!r}"
        )
    ids = [require_known(str(f)).id for f in requested]

    conversions: set[str] = set()
    for fmt_id in ids:
        if fmt_id == DELIVERY_FORMAT_ADAPTER:
            continue
        fmt = require_known(fmt_id)
        if fmt.conversion is None:
            # A non-adapter format with no conversion is a vocabulary error;
            # the adapter is the only conversion-free format.
            continue
        conversions.add(fmt.conversion)
        # A conversion whose source is itself a delivery format brings that
        # format's conversion along: quantised needs merged, merged needs the
        # merge step.
        source = CONVERSION_SOURCE.get(fmt_id)
        if source is not None:
            source_fmt = require_known(source)
            if source_fmt.conversion is not None:
                conversions.add(source_fmt.conversion)

    steps = [c for c in CONVERSION_ORDER if c in conversions]
    _assert_order(steps, ids)
    return steps


def _assert_order(steps: list[str], requested: list[str]) -> None:
    """The order invariant, stated once.

    `steps` is always built from `CONVERSION_ORDER`, so it cannot violate the
    property by construction; this check exists so the invariant is visible
    and testable rather than implicit. It refuses an order that would quantise
    without a prior merge.
    """
    if "quantise" in steps and "merge" not in steps:
        raise IllegalProductionOrder(
            "quantised delivery requires the merge step first: quantisation "
            f"must follow a full-precision merge. Requested: {requested}"
        )
    if "quantise" in steps:
        m = steps.index("merge")
        q = steps.index("quantise")
        if m >= q:
            raise IllegalProductionOrder(
                f"merge (index {m}) must precede quantise (index {q}); "
                f"requested: {requested}"
            )
