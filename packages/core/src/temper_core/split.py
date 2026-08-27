"""The held-out split: dedup first, split second, deterministic under a seed.

Issue #53. A portion of the dataset is held out automatically so a job gets an
eval loss it can chart against training loss. Two properties are load-bearing,
and each is pinned by a test in this module:

* **The order is the substance.** The split runs *after* deduplication.
  Splitting before it lets a duplicated row land on both sides of the split,
  which makes held-out loss optimistic in a way nothing downstream can detect.
  The pipeline `held_out_split` therefore dedups first and then splits; the raw
  `split` is exposed only so the ordering can be proven by a test rather than
  asserted in a comment.
* **The split is deterministic under a seed, and its size is recorded.** The
  trainer passes the seed (the run's own) and the held-out fraction (the
  effective `val_set_size`) and records the resulting counts in the run's
  result document, so a run can say exactly how it was split.

The module is deliberately self-contained -- no imports from `temper_core`
siblings -- because it is one of the files shipped flat into the trainer image
(ADR-0010's rule that a value two components must agree on is defined once):
the control plane's validation and the machine's trainer both need the same
dedup-and-split behaviour, and a second copy beside the entrypoint would be the
hand-mirrored definition ADR-0010 forbids. It owns `normalise_text` for the
same reason: validation's content normalisation and the dedup key are one rule.
"""

from __future__ import annotations

import json
import random
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# The floor a job's training set must not fall below. Mirrors
# `temper_core.validation.MIN_ROWS` (the platform's adopted minimum for a
# meaningful adapter): the held-out set may be as large as the fraction asks,
# but never so large that what remains to train on is below this floor. One
# copy of the number is kept here because this module ships flat into the
# trainer image and cannot import the validator.
MIN_TRAIN_ROWS = 10


def normalise_text(text: str) -> str:
    """NFC, and strip control characters except tab and newline.

    Not cosmetic: the same visual string existing as two different token
    sequences distorts length statistics and defeats deduplication. This is the
    rule validation warns about when a row needs it; the dedup key applies the
    same rule so that two encodings of one conversation are one row.
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )


def _dedup_key(row: dict[str, Any], messages_field: str) -> str:
    """The identity two rows must agree on to be the same training example.

    The trainer learns from `messages`, so that is what defines a duplicate;
    a row that differs only outside its conversation (metadata the trainer
    never reads) is the same example twice. Content is normalised before the
    key is built, so two encodings of one conversation collapse into one row.
    """
    msgs = row.get(messages_field)
    if isinstance(msgs, list):
        normalised = []
        for m in msgs:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                m = {**m, "content": normalise_text(m["content"])}
            normalised.append(m)
        return json.dumps(normalised, sort_keys=True)
    return json.dumps(row, sort_keys=True)


def deduplicate(
    rows: Iterable[dict[str, Any]], messages_field: str = "messages"
) -> tuple[list[dict[str, Any]], int]:
    """Rows with one entry per distinct conversation, first occurrence kept.

    Returns (unique rows, number removed). Order is preserved, so the split
    and the run both see the data in the order it was uploaded.
    """
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed = 0
    for row in rows:
        key = _dedup_key(row, messages_field)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, removed


def split(
    rows: Sequence[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split `rows` into (train, held_out), disjoint.

    `fraction` follows `val_set_size` semantics: below 1 it is a fraction of
    the rows, at or above 1 it is a count. The held-out size is floored at one
    row (so the signal exists) and capped so training never drops below
    `min_train_rows`. `seed` fixes the assignment: the same rows, fraction and
    seed produce the same two sets, which is what lets a run's split be
    reproduced from its record.

    Both sides preserve the original relative order of the rows they keep, and
    every row lands on exactly one side. This function does **not** deduplicate
    -- call `held_out_split` unless the input is already unique.
    """
    if fraction <= 0:
        return list(rows), []
    n = len(rows)
    if fraction >= 1:
        raw = int(fraction)
    else:
        raw = round(n * fraction)
    held_count = min(max(1, raw), n - min_train_rows)
    if held_count <= 0:
        return list(rows), []

    order = list(range(n))
    random.Random(seed).shuffle(order)
    held_indices = set(order[:held_count])
    train = [rows[i] for i in range(n) if i not in held_indices]
    held = [rows[i] for i in range(n) if i in held_indices]
    return train, held


@dataclass(frozen=True)
class SplitRecord:
    """What a run can say about how its dataset was split.

    Recorded in the run's result document so the split is legible and
    reproducible: how many rows arrived, how many were duplicates, how many
    trained, how many were held out, and the fraction and seed the assignment
    used.
    """

    rows_in: int
    rows_removed_duplicates: int
    train_rows: int
    held_out_rows: int
    fraction: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_removed_duplicates": self.rows_removed_duplicates,
            "train_rows": self.train_rows,
            "held_out_rows": self.held_out_rows,
            "fraction": self.fraction,
            "seed": self.seed,
        }


def held_out_split(
    rows: Sequence[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
    messages_field: str = "messages",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SplitRecord]:
    """The product's split: deduplicate, then split. The order is the substance.

    Splitting before deduplication would let a duplicated row appear on both
    sides, making held-out loss optimistic in a way nothing downstream can
    detect. Returns (train, held_out, record); `record` states the sizes and
    the seed the split was fixed under.
    """
    unique, removed = deduplicate(rows, messages_field)
    train, held = split(unique, fraction=fraction, seed=seed)
    record = SplitRecord(
        rows_in=len(rows),
        rows_removed_duplicates=removed,
        train_rows=len(train),
        held_out_rows=len(held),
        fraction=fraction,
        seed=seed,
    )
    return train, held, record
