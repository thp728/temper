"""The streaming token-counting pass's promises, pinned.

Counting runs as its own phase after validation (issue #42), but it inherits
the streaming validator's two guarantees and pins them the same way: a pass
that quietly accumulates per-row state is exactly the materialisation #31's
flat-memory rewrite exists to forbid.

The one per-row object that *could* grow with the file -- the distribution of
token counts across rows -- is a histogram over a fixed set of edges, and
`TokenCounts` is a fixed shape. A good test here asserts that the count is
exact and that what is retained does not grow with the file; `retained
structure` is the part a refactor breaks, the same reasoning that made
`Report.retained_objects()` the guard for validation.
"""

from __future__ import annotations

import json
from pathlib import Path

from temper_core.counting import (
    HISTOGRAM_EDGES,
    TokenCounts,
    count_tokens,
    count_tokens_chunks,
)


def chat(i: int) -> dict:
    """A chat row whose token count under `char_count_row` is predictable."""
    return {
        "messages": [
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}" + "x" * (i % 3)},
        ]
    }


def char_count_row(obj: dict) -> int:
    """The test's stand-in for the control plane's tokenizer seam: one row's
    token count is the number of characters in its message contents, which the
    tests can compute exactly without a tokenizer."""
    total = 0
    for m in obj.get("messages") or []:
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            total += len(m["content"])
    return total


def write(tmp_path: Path, rows: list, name: str = "d.jsonl") -> Path:
    p = tmp_path / name
    p.write_text(
        "\n".join(json.dumps(r) for r in rows),
        encoding="utf-8",
    )
    return p


def tiny_chunks(raw: bytes, size: int):
    for i in range(0, len(raw), size):
        yield raw[i : i + size]


def assert_counts(c: TokenCounts, expected_total: int, rows: int) -> None:
    assert c.total_tokens == expected_total
    assert c.rows_counted == rows
    assert len(c.histogram) == len(HISTOGRAM_EDGES)
    assert sum(c.histogram) == rows


# --- the same pass through every door -------------------------------------


def test_count_via_path_and_bytes_and_tiny_chunks_agree(tmp_path):
    rows = [chat(i) for i in range(40)]
    path = write(tmp_path, rows)
    raw = path.read_bytes()
    kw = {"count_row": char_count_row, "sequence_len": 2048}

    expected = sum(char_count_row(r) for r in rows)
    assert (
        count_tokens(path, **kw).to_dict()
        == count_tokens_chunks((raw,), **kw).to_dict()
    )
    assert count_tokens_chunks(tiny_chunks(raw, 5), **kw).to_dict() == (
        count_tokens(path, **kw).to_dict()
    )
    assert count_tokens(path, **kw).total_tokens == expected


# --- exactness -------------------------------------------------------------


def test_total_is_the_sum_of_individual_rows(tmp_path):
    rows = [chat(i) for i in range(50)]
    counts = count_tokens(
        write(tmp_path, rows), char_count_row, sequence_len=2048
    )
    assert counts.total_tokens == sum(char_count_row(r) for r in rows)
    assert counts.rows_counted == 50
    assert counts.max_row_tokens == max(char_count_row(r) for r in rows)


def test_truncation_is_exact_at_the_sequence_length(tmp_path):
    """A row of exactly `sequence_len` tokens is not truncated; one token over
    is. The boundary is where the counting has to be exact, because it is the
    number a user acts on."""
    rows = [
        {
            "messages": [{"role": "assistant", "content": "a" * 99}]
        },  # 99 tokens
        {
            "messages": [{"role": "assistant", "content": "a" * 100}]
        },  # exactly 100
        {"messages": [{"role": "assistant", "content": "a" * 101}]},  # 1 over
    ]
    counts = count_tokens(
        write(tmp_path, rows), char_count_row, sequence_len=100
    )
    assert counts.truncated_rows == 1
    assert counts.max_row_tokens == 101


def test_max_row_tokens_tracks_the_longest_row(tmp_path):
    rows = [
        {"messages": [{"role": "assistant", "content": "a" * 5}]},
        {"messages": [{"role": "assistant", "content": "a" * 9000}]},
    ]
    counts = count_tokens(
        write(tmp_path, rows), char_count_row, sequence_len=100
    )
    assert counts.max_row_tokens == 9000
    assert counts.truncated_rows == 1


# --- the histogram ---------------------------------------------------------


def test_histogram_bucket_boundaries(tmp_path):
    """2048 is an edge on purpose: a row at exactly the default sequence length
    must sit in the bucket the "reaches the context window" reading expects."""
    rows = [
        {"messages": [{"role": "assistant", "content": "a" * 2047}]},
        {"messages": [{"role": "assistant", "content": "a" * 2048}]},
        {"messages": [{"role": "assistant", "content": "a" * 2049}]},
    ]
    counts = count_tokens(
        write(tmp_path, rows), char_count_row, sequence_len=2048
    )
    # The bucket for [2048, 3072) is at the index of the 2048 edge.
    edge_index = HISTOGRAM_EDGES.index(2048)
    assert counts.histogram[edge_index - 1] == 1  # 2047
    assert counts.histogram[edge_index] == 2  # 2048 and 2049 share a bucket
    assert counts.truncated_rows == 1  # only the 2049 row is truncated


# --- flat memory: the guard a later merge must not break ------------------


def test_retained_structure_does_not_grow_with_row_count(tmp_path):
    """Ten times the rows must not mean a bigger TokenCounts: the distribution
    is a fixed histogram, not one entry per row. A refactor that collects every
    row's count into a list fails here."""
    small = count_tokens(
        write(tmp_path, [chat(i) for i in range(100)], "s.jsonl"),
        char_count_row,
        sequence_len=2048,
    )
    large = count_tokens(
        write(tmp_path, [chat(i) for i in range(10_000)], "l.jsonl"),
        char_count_row,
        sequence_len=2048,
    )
    assert large.rows_counted == 100 * small.rows_counted
    assert len(large.histogram) == len(small.histogram) == len(HISTOGRAM_EDGES)
    # The published shape is fixed, whatever the file holds.
    assert set(small.to_dict()) == set(large.to_dict())


def test_peak_python_heap_does_not_scale_with_row_count(tmp_path):
    """The streaming claim asserted rather than only derived from the shape:
    counting 10,000 rows must not hold proportionally more than counting 100.
    tracemalloc measures this process's own Python allocations -- the counting
    pass's retention -- which is the failure mode a per-row list would show."""

    def count(rows: int):
        import tracemalloc

        path = write(tmp_path, [chat(i) for i in range(rows)], f"{rows}.jsonl")
        tracemalloc.start()
        try:
            count_tokens(path, char_count_row, sequence_len=2048)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    peak_small = count(100)
    peak_large = count(10_000)
    # 100x the rows, comfortably less than 100x the peak: if the pass retained
    # a per-row structure, the ratio would be ~100. The bound is generous to
    # keep the assertion robust to allocator noise, not to hide accumulation.
    assert peak_large < peak_small * 8, (
        f"peak grew {peak_small} -> {peak_large} for 100x the rows; "
        "the counting pass is retaining per-row state"
    )


# --- rows the validator would reject --------------------------------------


def test_rows_that_do_not_parse_are_skipped_not_reported(tmp_path):
    """Counting is the second, focused phase: it never re-validates. A row that
    does not parse contributes nothing and is not counted -- the report already
    named it with its line."""
    rows = [
        chat(0),
        "{not json}",
        '"a bare string"',
        chat(1),
    ]
    counts = count_tokens(
        write(tmp_path, rows), char_count_row, sequence_len=2048
    )
    assert counts.rows_counted == 2
    assert counts.total_tokens == char_count_row(chat(0)) + char_count_row(
        chat(1)
    )


# --- progress --------------------------------------------------------------


def test_counting_reports_progress_in_bytes_and_rows(tmp_path):
    rows = [chat(i) for i in range(100)]
    path = write(tmp_path, rows)
    total = path.stat().st_size
    seen = []
    count_tokens(
        path,
        char_count_row,
        sequence_len=2048,
        on_progress=lambda p: seen.append(p),
        total_bytes=total,
    )
    assert seen, "a counting pass must report progress, however small"
    assert all(
        0 <= b <= total
        for b, _, _ in [(p.bytes_read, p.bytes_total, p.rows) for p in seen]
    )
    assert seen[-1].bytes_read == total
    assert seen[-1].rows == 100
