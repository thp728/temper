"""The streaming validator has to agree with the one that already ships.

Spike 9 measures how fast validation runs when it does not hold the file. That
measurement is worthless if the streaming pass is fast because it checks less,
so the seam is pinned here first: for any dataset small enough to validate both
ways, `stream_validate` and `api.validation.validate` must reach the same
verdict, the same counts, and the same line numbers.

These tests are the reason the throughput number in `findings-spike9.json` can
be quoted as the cost of *validation* rather than the cost of *reading*.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.validation import validate  # noqa: E402
from streaming import stream_validate  # noqa: E402


def write(tmp_path: Path, rows: list, name: str = "d.jsonl") -> Path:
    p = tmp_path / name
    p.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows),
        encoding="utf-8",
    )
    return p


def good_row(i: int = 0, think: bool = False) -> dict:
    answer = f"<think>because {i}</think>the answer" if think else f"the answer {i}"
    return {"messages": [
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": answer},
    ]}


def assert_agrees(path: Path) -> None:
    """The two validators must not merely both fail -- they must agree on why."""
    memory = validate(path).to_dict()
    streamed = stream_validate(path).to_dict()
    assert streamed["valid"] == memory["valid"]
    assert streamed["row_count"] == memory["row_count"]
    assert streamed["usable_rows"] == memory["usable_rows"]
    assert streamed["schema_type"] == memory["schema_type"]
    assert streamed["enable_thinking"] == memory["enable_thinking"]
    for kind in ("errors", "warnings"):
        assert ([(i["line"], i["code"]) for i in streamed[kind]]
                == [(i["line"], i["code"]) for i in memory[kind]]), kind
    assert streamed["preview"] == memory["preview"]


def test_agrees_on_a_clean_dataset(tmp_path):
    assert_agrees(write(tmp_path, [good_row(i) for i in range(60)]))


def test_agrees_on_a_thinking_dataset(tmp_path):
    assert_agrees(write(tmp_path, [good_row(i, think=True) for i in range(60)]))


def test_agrees_on_a_mixed_thinking_dataset(tmp_path):
    rows = [good_row(i, think=i < 30) for i in range(60)]
    assert_agrees(write(tmp_path, rows))


def test_agrees_on_malformed_json_and_names_the_same_line(tmp_path):
    rows = [json.dumps(good_row(i)) for i in range(60)]
    rows[17] = '{"messages": [broken'
    path = write(tmp_path, rows)
    assert_agrees(path)
    err = stream_validate(path).errors[0]
    assert err.line == 18, "line numbers are 1-indexed and count blank lines"
    assert err.code == "invalid_json"


def test_agrees_on_rows_that_teach_nothing(tmp_path):
    rows = [good_row(i) for i in range(60)]
    rows[3]["messages"] = [{"role": "user", "content": "orphan"}]     # no assistant
    rows[4]["messages"][1]["content"] = "   "                          # empty target
    rows[5]["messages"] = [{"role": "assistant", "content": "reply"}]  # no user
    assert_agrees(write(tmp_path, rows))


def test_agrees_on_a_non_chat_schema(tmp_path):
    assert_agrees(write(tmp_path, [{"prompt": "a", "completion": "b"}] * 20))


def test_agrees_when_there_are_too_few_rows(tmp_path):
    assert_agrees(write(tmp_path, [good_row(i) for i in range(4)]))


def test_agrees_on_an_empty_file(tmp_path):
    assert_agrees(write(tmp_path, []))


def test_agrees_on_blank_lines_between_rows(tmp_path):
    rows: list = []
    for i in range(60):
        rows.append(good_row(i))
        rows.append("")
    assert_agrees(write(tmp_path, rows))


def test_agrees_on_a_bom(tmp_path):
    p = tmp_path / "bom.jsonl"
    body = "\n".join(json.dumps(good_row(i)) for i in range(60))
    p.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    assert_agrees(p)


def test_rejects_invalid_utf8_without_reading_the_whole_file(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_bytes(b'{"messages": []}\n\xff\xfe not utf-8\n')
    rep = stream_validate(p)
    assert not rep.valid
    assert [e.code for e in rep.errors] == ["encoding"]
    assert rep.errors[0].line == 2, "a decode failure still names its line"


# --- the properties that make it a *streaming* validator --------------------

def test_error_list_is_capped_but_the_count_is_not(tmp_path):
    """A 5 GB file of broken JSON must not become a 5 GB error list.

    The count is what the user needs to understand the scale of the problem;
    the individual messages past the first few are noise. Capping the list is
    what keeps peak memory flat -- see findings-spike9.json.
    """
    rows = ['{"messages": [broken'] * 500
    rep = stream_validate(write(tmp_path, rows), error_cap=10)
    assert len(rep.errors) == 10
    # 500 unparseable rows, plus the schema verdict that follows from none of
    # them parsing. The count is exact; the list is not.
    assert rep.error_count == 501
    assert rep.errors_suppressed == rep.error_count - len(rep.errors)


def test_thinking_sample_lines_are_capped(tmp_path):
    """The line-number guarantee is where the streaming path leaks if it does.

    `trainer.thinking.detect` accumulates one line number per assistant turn
    before capping, which is fine for a file held in memory and unbounded for
    one that is not. The streaming path caps as it goes.
    """
    rows = [good_row(i, think=i % 2 == 0) for i in range(2000)]
    rep = stream_validate(write(tmp_path, rows), sample_cap=5)
    assert not rep.valid
    assert [e.code for e in rep.errors] == ["mixed_thinking"]
    assert len(rep.thinking_lines_with) == 5
    assert len(rep.thinking_lines_without) == 5
    assert rep.turns_with_think == 1000
    assert rep.turns_without_think == 1000


def test_line_number_tracking_can_be_turned_off_for_measurement(tmp_path):
    """Only so the spike can price the guarantee. The product never does this."""
    path = write(tmp_path, [good_row(i) for i in range(60)])
    rep = stream_validate(path, track_lines=False)
    assert rep.valid
    assert rep.usable_rows == 60
    assert all(e.line is None for e in rep.errors)


def test_token_counting_is_optional_and_reports_a_total(tmp_path):
    path = write(tmp_path, [good_row(i) for i in range(60)])
    assert stream_validate(path).token_count is None
    # A stand-in for a real tokeniser: the seam is that it takes text and
    # returns a count, so the spike can swap in Qwen's without touching this.
    rep = stream_validate(path, count_tokens=lambda text: len(text.split()))
    assert rep.token_count > 0
    assert rep.token_count == sum(
        len(m["content"].split())
        for i in range(60) for m in good_row(i)["messages"]
    )


def test_peak_memory_does_not_grow_with_row_count(tmp_path):
    """The claim, asserted rather than only measured in the spike run.

    Ten times the rows must not mean ten times the retained objects. This
    checks the shape of what is kept, which is the part a refactor breaks;
    the spike measures actual RSS at gigabyte scale.
    """
    small = stream_validate(write(tmp_path, [good_row(i) for i in range(100)], "s.jsonl"))
    large = stream_validate(write(tmp_path, [good_row(i) for i in range(10_000)], "l.jsonl"))
    assert large.usable_rows == 100 * small.usable_rows
    assert len(large.preview) == len(small.preview)
    assert len(large.errors) == len(small.errors)
    assert len(large.warnings) == len(small.warnings)
    assert large.retained_objects() == small.retained_objects()


@pytest.mark.parametrize("chunk", [1, 7, 64, 8192])
def test_agrees_regardless_of_read_buffer_size(tmp_path, chunk):
    """A row split across two reads is the classic streaming bug."""
    path = write(tmp_path, [good_row(i) for i in range(60)])
    assert stream_validate(path, chunk_bytes=chunk).to_dict() == \
        stream_validate(path, chunk_bytes=8 * 1024 * 1024).to_dict()


def test_a_byte_limited_pass_stops_early_and_refuses_a_verdict(tmp_path):
    """Truncation is a rate measurement, never a verdict.

    Spike 9 needs a tokenisation rate at 20 GB and cannot afford to tokenise
    20 GB. Measuring a prefix is fine; reporting the prefix as `valid` would
    be a claim about rows nobody read.
    """
    path = write(tmp_path, [good_row(i) for i in range(5000)])
    full = stream_validate(path)
    part = stream_validate(path, byte_limit=50_000)

    assert full.valid and not full.truncated
    assert part.truncated
    assert not part.valid, "a partial read cannot pronounce a file valid"
    assert 0 < part.row_count < full.row_count
    assert part.bytes_read <= 50_000 + 1000


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 8192])
def test_a_bom_survives_being_split_across_reads(tmp_path, chunk):
    """A one-byte read splits the BOM itself.

    The first version of `_iter_lines` tested the CHUNK for the BOM rather than
    the accumulated buffer, so at chunk_bytes=1 two of the BOM's three bytes
    stayed at the head of line 1 and every row failed to parse. Found by
    reading the code rather than by a test failing, which is why the parameter
    list now goes down to 1.
    """
    p = tmp_path / "bom.jsonl"
    body = "\n".join(json.dumps(good_row(i)) for i in range(20))
    p.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    rep = stream_validate(p, chunk_bytes=chunk)
    assert rep.row_count == 20
    assert rep.usable_rows == 20
    assert rep.errors == [], "the BOM must not become part of line 1"
