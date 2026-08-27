"""The held-out split: dedup first, split second, deterministic under a seed.

Issue #53. The ordering is the substance: splitting before deduplication lets a
duplicated row land on both sides of the split, which makes held-out loss
optimistic in a way nothing downstream can detect. That ordering is proven
below rather than asserted in a comment.
"""

from __future__ import annotations

import json

from temper_core.split import (
    MIN_TRAIN_ROWS,
    deduplicate,
    held_out_split,
    normalise_text,
    split,
)


def rows(pairs):
    return [
        {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }
        for q, a in pairs
    ]


def _conversation_of(row) -> str:
    return json.dumps(row["messages"], sort_keys=True)


# --- deduplication ----------------------------------------------------------


def test_duplicate_rows_are_removed_keeping_the_first_occurrence():
    ds = rows([("q1", "a1"), ("q1", "a1"), ("q2", "a2")])
    unique, removed = deduplicate(ds)
    assert removed == 1
    assert [_conversation_of(r) for r in unique] == [
        _conversation_of(ds[0]),
        _conversation_of(ds[2]),
    ]
    # The first occurrence is the one kept: it is the copy the trainer would
    # have seen first, and which one survives must not depend on the split.
    assert unique[0] is ds[0]


def test_rows_that_differ_only_by_encoding_are_duplicates():
    """The same visual conversation under two encodings is one row.

    NFC versus a decomposed form of the same string tokenises differently,
    which is exactly the case where a duplicated row would not look duplicated
    -- and where splitting before dedup lets it appear on both sides.
    """
    decomposed = "\u0065\u0301"  # e + combining acute = é
    composed = "\u00e9"  # é
    assert normalise_text(decomposed) == normalise_text(composed)
    ds = rows([("q", decomposed), ("q", composed)])
    unique, removed = deduplicate(ds)
    assert removed == 1
    assert len(unique) == 1


def test_rows_with_distinct_conversations_are_all_kept():
    ds = rows([(f"q{i}", f"a{i}") for i in range(40)])
    unique, removed = deduplicate(ds)
    assert removed == 0
    assert len(unique) == 40


def test_a_row_without_messages_keeps_its_whole_json_identity():
    ds = [
        {"text": "x"},
        {"text": "x"},
        {"messages": [{"role": "user", "content": "q"}]},
    ]
    unique, removed = deduplicate(ds)
    assert removed == 1


# --- the split --------------------------------------------------------------


def test_split_is_deterministic_under_a_seed():
    ds = rows([(f"q{i}", f"a{i}") for i in range(50)])
    t1, h1 = split(ds, fraction=0.2, seed=7)
    t2, h2 = split(ds, fraction=0.2, seed=7)
    assert [_conversation_of(r) for r in t1] == [
        _conversation_of(r) for r in t2
    ]
    assert [_conversation_of(r) for r in h1] == [
        _conversation_of(r) for r in h2
    ]


def test_a_different_seed_produces_a_different_held_out_set():
    ds = rows([(f"q{i}", f"a{i}") for i in range(200)])
    _, h1 = split(ds, fraction=0.1, seed=7)
    _, h2 = split(ds, fraction=0.1, seed=8)
    assert [_conversation_of(r) for r in h1] != [
        _conversation_of(r) for r in h2
    ]


def test_the_split_is_disjoint_and_proportional():
    ds = rows([(f"q{i}", f"a{i}") for i in range(100)])
    train, held = split(ds, fraction=0.1, seed=3)
    train_keys = {_conversation_of(r) for r in train}
    held_keys = {_conversation_of(r) for r in held}
    assert train_keys.isdisjoint(held_keys)
    assert len(held) == 10
    assert len(train) == 90


def test_every_row_lands_on_exactly_one_side():
    ds = rows([(f"q{i}", f"a{i}") for i in range(31)])
    train, held = split(ds, fraction=0.2, seed=11)
    keys = {_conversation_of(r) for r in [*train, *held]}
    assert len(keys) == 31  # nothing dropped, nothing duplicated


def test_held_out_never_drops_training_below_the_floor():
    """A small dataset holds out as much as the training floor allows, and
    never so much that training falls below it."""
    for n in range(MIN_TRAIN_ROWS, MIN_TRAIN_ROWS + 20):
        ds = rows([(f"q{i}", f"a{i}") for i in range(n)])
        train, _ = split(ds, fraction=0.5, seed=1)
        assert len(train) >= MIN_TRAIN_ROWS, (
            f"n={n} dropped training below the floor"
        )


def test_zero_fraction_holds_out_nothing():
    ds = rows([(f"q{i}", f"a{i}") for i in range(20)])
    train, held = split(ds, fraction=0, seed=1)
    assert held == []
    assert len(train) == 20


def test_a_fraction_above_one_is_an_absolute_count():
    """`val_set_size` semantics carried over: below 1 it is a fraction, at or
    above 1 it is a count of rows."""
    ds = rows([(f"q{i}", f"a{i}") for i in range(100)])
    train, held = split(ds, fraction=5, seed=1)
    assert len(held) == 5
    assert len(train) == 95


# --- the ordering constraint ------------------------------------------------


def test_splitting_before_dedup_can_put_a_duplicate_on_both_sides():
    """The ordering constraint, proven rather than asserted.

    A duplicated row split *without* deduplication can land in both the
    training set and the held-out set -- the exact failure the issue exists
    to prevent, because a row that appears on both sides makes held-out loss
    optimistic in a way nothing downstream can detect. Dedup-then-split never
    produces that overlap.
    """
    ds = rows([("q0", "a0")]) * 2 + rows(
        [(f"q{i}", f"a{i}") for i in range(1, 22)]
    )
    dup = _conversation_of(ds[0])

    def overlaps(msgs: set[str]) -> bool:
        return dup in msgs

    # Find a seed under which the raw splitter puts the duplicate on both
    # sides. The danger is real; the test pins a concrete instance of it.
    guilty_seed = next(
        seed
        for seed in range(100)
        if (
            overlaps(
                {
                    _conversation_of(r)
                    for r in split(ds, fraction=0.5, seed=seed)[0]
                }
            )
            and overlaps(
                {
                    _conversation_of(r)
                    for r in split(ds, fraction=0.5, seed=seed)[1]
                }
            )
        )
    )

    train, held = split(ds, fraction=0.5, seed=guilty_seed)
    assert dup in {_conversation_of(r) for r in train}
    assert dup in {_conversation_of(r) for r in held}

    # The pipeline the product calls -- dedup first, then split -- never puts
    # any row on both sides, whatever the seed.
    for seed in range(20):
        t, h, _ = held_out_split(ds, fraction=0.5, seed=seed)
        t_keys = {_conversation_of(r) for r in t}
        h_keys = {_conversation_of(r) for r in h}
        assert t_keys.isdisjoint(h_keys)


# --- the pipeline record ----------------------------------------------------


def test_held_out_split_records_the_split_size_and_seed():
    ds = rows(
        [("q0", "a0"), ("q0", "a0")]
        + [(f"q{i}", f"a{i}") for i in range(1, 11)]
    )
    train, held, record = held_out_split(ds, fraction=0.2, seed=42)
    assert record.rows_in == 12
    assert record.rows_removed_duplicates == 1
    assert record.train_rows == len(train)
    assert record.held_out_rows == len(held)
    assert record.fraction == 0.2
    assert record.seed == 42
    d = record.to_dict()
    assert d == {
        "rows_in": 12,
        "rows_removed_duplicates": 1,
        "train_rows": len(train),
        "held_out_rows": len(held),
        "fraction": 0.2,
        "seed": 42,
    }
