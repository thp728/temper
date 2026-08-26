"""Validation's two front doors agree, because they are one validator.

`validate` reads a path; `validate_bytes` takes the bytes already held -- which
is what an upload has before anything is stored. They must reach identical
reports for identical bytes, because a verdict that depends on how the dataset
was reached is not a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

from temper_core.validation import validate, validate_bytes


def chat(i: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    }


def dataset_bytes() -> bytes:
    rows = [json.dumps(chat(i)) for i in range(12)]
    rows[5] = "{broken"
    return ("\n".join(rows)).encode("utf-8")


def test_validate_bytes_and_validate_path_report_identically(tmp_path):
    raw = dataset_bytes()
    p = tmp_path / "d.jsonl"
    p.write_bytes(raw)
    assert validate_bytes(raw).to_dict() == validate(p).to_dict()


def test_a_bom_is_honoured_whichever_door_the_bytes_arrive_through():
    raw = b"\xef\xbb\xbf" + (
        "\n".join(json.dumps(chat(i)) for i in range(12))
    ).encode("utf-8")
    report = validate_bytes(raw)
    assert report.valid
    assert report.row_count == 12


def test_invalid_utf8_names_the_problem_not_the_file():
    report = validate_bytes(b'{"messages": []}\n\xff\xfe\n')
    assert not report.valid
    assert [e.code for e in report.errors] == ["encoding"]


def test_the_path_entry_point_delegates_to_the_bytes_one(tmp_path):
    """One validator, two doors: the path form exists so callers holding a
    file keep working, and it must stay a thin read over the pure core."""
    p = Path(tmp_path) / "d.jsonl"
    p.write_bytes(dataset_bytes())
    assert validate(p).to_dict() == validate_bytes(dataset_bytes()).to_dict()
