"""The streaming validator's promises, pinned.

Spike 9 measured what streaming validation costs at gigabyte scale: peak RSS
flat from 1 GB to 20 GB, and faster than the in-memory path it replaces
(`findings-spike9.json`). These tests pin the same claims at the size a test
can afford, where the failure mode is a refactor that quietly re-accumulates
-- a list of every row, an uncapped error list, sample lines that grow with
the file. RSS at gigabyte scale tells you the claim held for one file;
`retained_objects` and the caps tell you which change broke it.

The other half of the deal is that a streaming pass is still the same
validator: every entry point -- path, already-held bytes, and chunks as small
as one byte -- must reach identical reports for identical data, including
line numbers, or the throughput number is not the cost of validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from temper_core.validation import (
    MAX_ERRORS,
    MAX_PREVIEW,
    MAX_SAMPLE,
    validate,
    validate_bytes,
    validate_chunks,
)


def chat(i: int, think: bool = False) -> dict:
    # The thinking marker is the Qwen chat-template token `<think>...</think>`,
    # built from chr() so the literal angle brackets survive every file write.
    answer = (
        f"{chr(60)}think{chr(62)}r{i}{chr(60)}/think{chr(62)}a{i}"
        if think
        else f"a{i}"
    )
    return {
        "messages": [
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": answer},
        ]
    }


def write(tmp_path: Path, rows: list, name: str = "d.jsonl") -> Path:
    p = tmp_path / name
    p.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows),
        encoding="utf-8",
    )
    return p


def tiny_chunks(raw: bytes, size: int):
    for i in range(0, len(raw), size):
        yield raw[i : i + size]


# --- the same validator through every door ---------------------------------


def test_path_and_bytes_and_tiny_chunks_agree(tmp_path):
    rows = [chat(i) for i in range(60)]
    rows[17] = {"messages": [{"role": "user", "content": "orphan"}]}
    path = write(tmp_path, rows)
    raw = path.read_bytes()

    assert validate(path).to_dict() == validate_bytes(raw).to_dict()
    assert (
        validate_chunks(tiny_chunks(raw, 7)).to_dict()
        == validate(path).to_dict()
    )


@pytest.mark.parametrize("chunk", [1, 7, 64, 8192, 1 << 20])
def test_a_row_split_across_reads_still_parses(tmp_path, chunk):
    """A row straddling two reads is the classic streaming bug."""
    path = write(tmp_path, [chat(i) for i in range(60)])
    rep = validate_chunks(tiny_chunks(path.read_bytes(), chunk))
    assert rep.valid
    assert rep.usable_rows == 60
    assert rep.errors == []


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 8192])
def test_a_bom_survives_being_split_across_reads(tmp_path, chunk):
    """A one-byte read splits the BOM itself. The first version of the line
    iterator tested the CHUNK for the BOM rather than the accumulated buffer,
    so at size 1 two of the BOM's three bytes stayed at the head of line 1
    and every row failed to parse."""
    p = tmp_path / "bom.jsonl"
    body = "\n".join(json.dumps(chat(i)) for i in range(20))
    p.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    rep = validate_chunks(tiny_chunks(p.read_bytes(), chunk))
    assert rep.row_count == 20
    assert rep.usable_rows == 20
    assert rep.errors == [], "the BOM must not become part of line 1"


def test_encoding_errors_name_a_line(tmp_path):
    """The in-memory path decoded the whole file at once and could only give
    a byte offset; streaming decodes a line at a time, so it names the line.
    A divergence recorded rather than hidden, and strictly better for the
    user."""
    p = tmp_path / "bad.jsonl"
    p.write_bytes(b'{"messages": []}\n\xff\xfe not utf-8\n')
    rep = validate(p)
    assert not rep.valid
    assert [e.code for e in rep.errors] == ["encoding"]
    assert rep.errors[0].line == 2


# --- the shape of the report -------------------------------------------------


def test_report_dict_is_exactly_the_documented_contract(tmp_path):
    """The report's contents -- counts, thinking mode, preview, line-numbered
    problems -- are unchanged by streaming; the only additions are the
    suppression totals that keep a capped report honest, and the per-code
    totals that let a capped report attribute its suppression to a cause."""
    path = write(tmp_path, [chat(i) for i in range(12)])
    assert set(validate(path).to_dict()) == {
        "valid",
        "row_count",
        "usable_rows",
        "schema_type",
        "enable_thinking",
        "errors",
        "warnings",
        "preview",
        "error_count",
        "warning_count",
        "errors_suppressed",
        "warnings_suppressed",
        "error_code_counts",
        "warning_code_counts",
    }


# --- flat memory: the guard a later merge must not break --------------------


def test_peak_memory_does_not_grow_with_row_count(tmp_path):
    """The claim, asserted rather than only measured in the spike run.

    Ten times the rows must not mean ten times the retained objects. This
    checks the shape of what is kept, which is the part a refactor breaks;
    the spike measured actual RSS at gigabyte scale. A later PR that
    reintroduces materialisation -- a list of every row, an uncapped error
    list, thinking samples collected before capping -- fails here.
    """
    small = validate(write(tmp_path, [chat(i) for i in range(100)], "s.jsonl"))
    large = validate(
        write(tmp_path, [chat(i) for i in range(10_000)], "l.jsonl")
    )
    assert large.usable_rows == 100 * small.usable_rows
    assert len(large.preview) == len(small.preview) == MAX_PREVIEW
    assert len(large.errors) == len(small.errors)
    assert len(large.warnings) == len(small.warnings)
    assert large.retained_objects() == small.retained_objects()


def test_error_list_is_capped_but_the_count_is_not(tmp_path):
    """A large file of broken JSON must not become a large error list. The
    count is what tells the user it is broken on every line; the individual
    messages past the first few are noise."""
    rows = ['{"messages": [broken'] * 500
    rep = validate(write(tmp_path, rows))
    assert len(rep.errors) == MAX_ERRORS
    # 500 unparseable rows, plus the schema verdict that follows from none of
    # them parsing. The count is exact; the list is capped.
    assert rep.error_count == 501
    assert rep.errors_suppressed == 501 - MAX_ERRORS
    assert not rep.valid


def test_error_code_counts_are_exact_even_past_the_cap(tmp_path):
    """The per-code totals are what let a capped report point at a cause
    instead of leaving a bare 'N more, not shown': every suppressed error
    here is a duplicate of `invalid_json`, which the total says plainly."""
    rows = ['{"messages": [broken'] * 500
    rep = validate(write(tmp_path, rows))
    assert rep.error_code_counts["invalid_json"] == 500
    assert rep.error_code_counts["unrecognised_schema"] == 1
    assert sum(rep.error_code_counts.values()) == rep.error_count


def test_thinking_sample_lines_are_capped(tmp_path):
    """Thinking detection is where the streaming path leaks if it does: the
    old detector accumulated one line number per assistant turn before capping.
    The streaming path caps as it goes."""
    rows = [chat(i, think=i % 2 == 0) for i in range(2000)]
    rep = validate(write(tmp_path, rows))
    assert not rep.valid
    assert [e.code for e in rep.errors] == ["mixed_thinking"]
    assert len(rep.thinking_lines_with) == MAX_SAMPLE
    assert len(rep.thinking_lines_without) == MAX_SAMPLE
    assert rep.turns_with_think == 1000
    assert rep.turns_without_think == 1000


# --- progress ---------------------------------------------------------------


def test_progress_is_reported_in_bytes_and_rows(tmp_path):
    path = write(tmp_path, [chat(i) for i in range(100)])
    total = path.stat().st_size
    seen: list[tuple[int, int | None, int]] = []

    validate(
        path,
        on_progress=lambda p: seen.append(
            (p.bytes_read, p.bytes_total, p.rows)
        ),
        total_bytes=total,
    )

    assert seen, "a pass must report progress, however small"
    assert all(r >= 0 for _, _, r in seen)
    assert all(b >= 0 and b <= total for b, _, _ in seen)
    # The final report reaches the end exactly: a progress bar that stops
    # short of the file it is describing is a lie.
    assert seen[-1][0] == total
    assert seen[-1][1] == total
    assert seen[-1][2] == 100


def test_progress_reaches_exactly_100_percent_on_a_bom_file(tmp_path):
    """The BOM is input too: progress must not under-count the three bytes it
    strips, or a BOM-prefixed file would cap its bar below 100%."""
    p = tmp_path / "bom.jsonl"
    body = "\n".join(json.dumps(chat(i)) for i in range(50))
    p.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    total = p.stat().st_size
    seen = []
    validate(p, on_progress=lambda pr: seen.append(pr), total_bytes=total)
    assert seen[-1].bytes_read == total


def test_progress_can_report_without_a_total(tmp_path):
    path = write(tmp_path, [chat(i) for i in range(10)])
    seen = []
    validate(path, on_progress=lambda p: seen.append(p))
    assert seen and all(p.bytes_total is None for p in seen)


def test_progress_reports_rows_as_they_are_counted(tmp_path):
    """A large pass reports a monotone byte position, so the client can draw
    a bar rather than a spinner."""
    path = write(tmp_path, [chat(i) for i in range(2000)])
    validate(path, on_progress=lambda p: None)  # must not raise


def test_report_retained_objects_is_bounded_on_a_huge_broken_file(tmp_path):
    """The pathological case the caps exist for: a file broken on every line
    must not grow the report with the file."""
    rows = ['{"messages": [broken'] * 20_000
    rep = validate(write(tmp_path, rows))
    assert rep.error_count == 20_001
    assert len(rep.errors) == MAX_ERRORS
    assert rep.retained_objects() == MAX_ERRORS
