"""Delivery formats (issue #74): vocabulary, purposes, and the order property.

Spec 011 / issue #74. The product offers more than one form of the finished
result: the trained change as it is, a merged single-file model for serving,
and a quantised local-inference format for running locally. The load-bearing
property is order -- merging into an already-quantised base compounds error,
and the failure is silent -- so `production_steps` must be unable to produce a
step list that quantises before it merges, and the property test asserts that
over every subset of the vocabulary.
"""

from __future__ import annotations

import itertools

import pytest

from temper_core import artifacts, delivery
from temper_core.delivery import (
    IllegalProductionOrder,
    UnknownDeliveryFormat,
    production_steps,
)


def test_the_three_delivery_formats_are_defined():
    """The vocabulary has exactly the as-trained, merged and quantised forms,
    in that stable order, each with a plain-language purpose (a user chooses a
    format without knowing what a merge is)."""
    assert set(delivery.FORMATS) == {
        delivery.DELIVERY_FORMAT_ADAPTER,
        delivery.DELIVERY_FORMAT_MERGED,
        delivery.DELIVERY_FORMAT_QUANTISED,
    }
    # Every format states what it is for, without requiring internals: the
    # sentence names the user's outcome (move it, serve it, run it locally),
    # not the machinery that produced it.
    assert set(delivery.FORMATS) == {
        delivery.DELIVERY_FORMAT_ADAPTER,
        delivery.DELIVERY_FORMAT_MERGED,
        delivery.DELIVERY_FORMAT_QUANTISED,
    }
    for fmt in delivery.FORMATS.values():
        assert fmt.what_for.strip()
        # The purpose is a sentence a user acts on, not a recipe of internal
        # steps: no step vocabulary, no schema jargon.
        assert "step" not in fmt.what_for.lower()
        assert "conversion" not in fmt.what_for.lower()
        assert "hyperparameter" not in fmt.what_for.lower()


def test_the_adapter_format_needs_no_conversion():
    """The as-trained artifact is the canonical one: requesting it alone
    produces nothing extra."""
    assert (
        delivery.FORMATS[delivery.DELIVERY_FORMAT_ADAPTER].conversion is None
    )
    assert production_steps([delivery.DELIVERY_FORMAT_ADAPTER]) == []


def test_requesting_merged_produces_only_the_merge_step():
    assert production_steps([delivery.DELIVERY_FORMAT_MERGED]) == ["merge"]


def test_requesting_quantised_implies_the_merge_first():
    """THE order property: a quantised local format is produced from the
    correctly merged model, so requesting it alone still runs the merge first.
    The step list cannot be [quantise]."""
    assert production_steps([delivery.DELIVERY_FORMAT_QUANTISED]) == [
        "merge",
        "quantise",
    ]


def test_requesting_everything_yields_the_canonical_order():
    assert production_steps(
        [
            delivery.DELIVERY_FORMAT_ADAPTER,
            delivery.DELIVERY_FORMAT_MERGED,
            delivery.DELIVERY_FORMAT_QUANTISED,
        ]
    ) == ["merge", "quantise"]


def test_the_merge_then_quantise_order_holds_for_every_subset():
    """The property, over the whole vocabulary: for every subset of requested
    formats, `quantise` never precedes `merge`, and `quantise` only appears
    when `merge` does. The failure this prevents is silent, so the assertion
    over all subsets is the deliverable."""
    formats = list(delivery.FORMATS)
    for size in range(len(formats) + 1):
        for subset in itertools.combinations(formats, size):
            steps = production_steps(list(subset))
            if "quantise" in steps:
                assert "merge" in steps
                assert steps.index("merge") < steps.index("quantise")


def test_an_unknown_format_is_refused_not_guessed():
    with pytest.raises(UnknownDeliveryFormat):
        production_steps(["wat"])
    with pytest.raises(UnknownDeliveryFormat):
        delivery.what_for("wat")
    with pytest.raises(UnknownDeliveryFormat):
        delivery.kind_for("wat")


def test_a_string_request_is_refused():
    """A delivery request is a list of format ids; a bare string is a caller
    mistake and is refused loudly rather than iterated as characters."""
    with pytest.raises(UnknownDeliveryFormat):
        production_steps("merged")


def test_each_format_maps_to_a_defined_artifact_kind():
    """Every delivery format's kind is a kind the artifact vocabulary knows,
    so a download's manifest can describe it without inventing a kind."""
    for fmt in delivery.FORMATS.values():
        assert fmt.kind in artifacts.ARTIFACT_KINDS
    assert (
        delivery.kind_for(delivery.DELIVERY_FORMAT_MERGED)
        == artifacts.ARTIFACT_KIND_MERGED_MODEL
    )


def test_every_contract_format_round_trips_through_known_format():
    """The contract file is the single definition; every row it carries is
    loadable by the module that reads it (the trainer reads the same file, so
    the two sides cannot drift)."""
    for fmt in delivery.FORMATS.values():
        assert delivery.require_known(fmt.id) == fmt


def test_illegal_order_cannot_be_constructed():
    """The enforcement is not just the property test's reading of the result:
    the module itself refuses to build an illegal step list."""
    with pytest.raises(IllegalProductionOrder):
        delivery._assert_order(["quantise"], ["quantised"])
