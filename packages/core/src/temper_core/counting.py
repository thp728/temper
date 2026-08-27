"""Token counting, as its own streaming pass.

Issue #42. Cost is quoted per training token, so a dataset's token count is a
hard prerequisite of the quote -- and tokenising is the expensive half of the
validation work. Spike 9 measured that expense at roughly six times the
validate-only pass (3.4 MB/s tokenised vs 21.6 MB/s), and the re-measurement
against the shipped streaming validator came back the same way (~7x slower:
1.8 vs 12.4 MB/s, `.scratch42/findings-measure42.json`). The ticket's fork --
*"if measurement showed tokenising dominates, counting runs as its own phase
with its own state rather than blocking the report"* -- therefore lands on
**SPLIT**: this pass runs after validation, never blocking the report, and its
output is recorded with the dataset version so the quote reads it without
recomputation.

Two guarantees carry over from the streaming validator (#31) and are pinned by
the same tests:

* **One row at a time.** The pass reads bounded chunks and decides each row
  before the next is read, so peak memory stays flat as the dataset grows.
* **Everything retained is bounded.** The one per-row object that *could* grow
  with the file -- the distribution of token counts across rows -- is a
  histogram over a fixed set of edges (`HISTOGRAM_EDGES`), not one entry per
  row. A per-row list of counts would be exactly the materialisation the
  streaming guarantee exists to forbid.

The tokeniser is a seam, not a dependency: `count_row` is a `dict -> int`
callable the caller supplies (the control plane, which owns the network access
and the `tokenizers` dependency). `temper_core` stays pure -- no tokenizer
library is imported here, exactly as it imports no web framework.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from temper_core.validation import (
    ValidationProgress,
    _chunks_of,
    _iter_lines,
)

# The edges of the per-row token-count histogram, in tokens. Bucket `i` counts
# rows with `edges[i] <= tokens < edges[i+1]` for `i < len(edges)-1`; the last
# bucket counts rows with `tokens >= edges[-1]`. The edges are a constant, not
# a tunable: the bucket count is what keeps peak memory independent of the
# file, and a deployment knob on a safety property is a knob on the property.
# 2048 is an edge on purpose -- it is the trainer's default sequence length, so
# the histogram itself answers "how many rows reach the context window" at the
# length the quote is priced against.
HISTOGRAM_EDGES: tuple[int, ...] = (
    0,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
)

# How often `on_progress` may fire, in bytes read. Same cadence as validation:
# a progress callback a user can watch without being drowned.
PROGRESS_EVERY_BYTES = 1024 * 1024


@dataclass(frozen=True)
class TokenCounts:
    """The count for one dataset, under one tokenizer and sequence length.

    Every field is bounded: nothing per-row survives the pass. `histogram` is
    the acceptance criterion's "distribution across rows" -- counts per
    `HISTOGRAM_EDGES` bucket rather than one entry per row, so it cannot grow
    with the file. `truncated_rows` is exact (the counter ran during the pass,
    not derived from the histogram afterwards); the histogram is what lets a
    future surface re-derive a shape for any sequence length without a second
    tokenising pass.
    """

    total_tokens: int
    rows_counted: int
    sequence_len: int
    truncated_rows: int
    max_row_tokens: int
    histogram: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "rows_counted": self.rows_counted,
            "sequence_len": self.sequence_len,
            "truncated_rows": self.truncated_rows,
            "max_row_tokens": self.max_row_tokens,
            "histogram": list(self.histogram),
            "histogram_edges": list(HISTOGRAM_EDGES),
        }


def _bucket(token_count: int) -> int:
    """The histogram bucket a row's token count falls into.

    Bucket `i` covers `[edges[i], edges[i+1])`; the last bucket covers
    everything at or above `edges[-1]`. A row of exactly `sequence_len` tokens
    sits at the top of its own bucket, which is why 2048 is an edge: the count
    of rows reaching the context window reads straight off the histogram.
    """
    for i in range(1, len(HISTOGRAM_EDGES)):
        if token_count < HISTOGRAM_EDGES[i]:
            return i - 1
    return len(HISTOGRAM_EDGES) - 1


def count_tokens_chunks(
    chunks: Iterable[bytes],
    count_row: Callable[[dict[str, Any]], int],
    *,
    sequence_len: int,
    on_progress: Callable[[ValidationProgress], None] | None = None,
    total_bytes: int | None = None,
) -> TokenCounts:
    """Count tokens as the dataset's bytes arrive, one row at a time.

    `chunks` is any iterable of bytes -- a file's chunks, a stored object's
    stream. `count_row` is the tokeniser seam: it takes one parsed row and
    returns its token count under the caller's tokenizer, so this module (and
    `temper_core`) never imports a tokenizer library. `sequence_len` is the
    length truncation is measured against: rows whose count exceeds it are the
    ones a training run would truncate, and the exact number is counted here.

    This pass deliberately does **not** re-validate: the report already named
    every problem with its line, and validation ran at upload. Rows that do
    not parse are skipped (they contribute zero) rather than re-reported --
    counting is the second, focused phase the measurement said validation
    must not block on.
    """
    total_tokens = 0
    rows_counted = 0
    truncated_rows = 0
    max_row_tokens = 0
    histogram = [0] * len(HISTOGRAM_EDGES)
    bytes_read = 0
    next_report = PROGRESS_EVERY_BYTES

    def report() -> None:
        if on_progress is not None:
            on_progress(
                ValidationProgress(bytes_read, total_bytes, rows_counted)
            )

    for _line_no, raw, bytes_read in _iter_lines(chunks):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue  # the validator named this line; counting skips it
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        tokens = count_row(obj)
        total_tokens += tokens
        rows_counted += 1
        histogram[_bucket(tokens)] += 1
        if tokens > max_row_tokens:
            max_row_tokens = tokens
        if tokens > sequence_len:
            truncated_rows += 1

        if bytes_read >= next_report:
            report()
            next_report = bytes_read + PROGRESS_EVERY_BYTES

    report()
    return TokenCounts(
        total_tokens=total_tokens,
        rows_counted=rows_counted,
        sequence_len=sequence_len,
        truncated_rows=truncated_rows,
        max_row_tokens=max_row_tokens,
        histogram=tuple(histogram),
    )


def count_tokens(
    path: Path,
    count_row: Callable[[dict[str, Any]], int],
    *,
    sequence_len: int,
    on_progress: Callable[[ValidationProgress], None] | None = None,
    total_bytes: int | None = None,
) -> TokenCounts:
    """Count tokens in the dataset at `path`. The file convenience over
    `count_tokens_chunks`; the control plane feeds stored-object streams to
    the chunks form directly."""
    return count_tokens_chunks(
        _chunks_of(path),
        count_row,
        sequence_len=sequence_len,
        on_progress=on_progress,
        total_bytes=total_bytes,
    )
