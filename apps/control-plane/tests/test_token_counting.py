"""The counting phase, end to end: an accepted dataset is counted in the
background after validation, the count lands on the record with its bounded
distribution, and the quote reads it without recomputation.

Counting runs as its own phase with its own state (issue #42) -- the report
stands the moment validation finishes, and the count arrives afterwards. The
tests below poll for the phase's `done` state exactly as they poll for the
validation verdict, and the conftest `no_real_tokenizer` fixture replaces the
tokenizer seam with the deterministic fake (one token per character), so every
count here is exact arithmetic.
"""

from __future__ import annotations

import json

from helpers import wait_counted, wait_validated

from temper_control_plane import db
from temper_core.counting import HISTOGRAM_EDGES


def jsonl(tmp_path, rows, name="d.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def chat(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def upload(client, path):
    with open(path, "rb") as f:
        return client.post("/v1/datasets", files={"file": (path.name, f)})


def chars(row) -> int:
    """The token count the fake tokenizer reports: one per character."""
    return sum(
        len(m["content"])
        for m in row["messages"]
        if isinstance(m.get("content"), str)
    )


def test_accepted_dataset_is_counted_after_validation(client, tmp_path):
    rows = [chat(f"q{i}", f"a{i}") for i in range(12)]
    r = upload(client, jsonl(tmp_path, rows))
    assert r.status_code == 202
    ds_id = r.json()["id"]

    wait_validated(client, ds_id)
    record = wait_counted(client, ds_id)

    # The report lands the moment validation finishes, and the count arrives
    # with it on the record: the phase separation itself is pinned
    # deterministically in test_db (the count is produced after the report and
    # merged into it), where it cannot be raced by an instant test tokenizer.
    assert record["token_count_status"] == db.COUNT_PHASE_DONE
    assert record["status"] == "valid"
    expected = sum(chars(row) for row in rows)
    assert record["report"]["token_count"] == expected

    dist = record["report"]["token_distribution"]
    assert dist["rows_counted"] == 12
    assert dist["total_tokens"] == expected
    assert dist["sequence_len"] == 2048
    assert dist["truncated_rows"] == 0
    assert dist["max_row_tokens"] == max(chars(row) for row in rows)
    assert len(dist["histogram"]) == len(HISTOGRAM_EDGES)
    assert sum(dist["histogram"]) == 12


def test_rows_longer_than_the_sequence_length_are_counted(client, tmp_path):
    """Rows that would be truncated at the trainer's default sequence length
    are counted exactly, and surfaced on the report."""
    long_row = chat("x", "y" * 5000)  # 5001 chars > 2048
    rows = [chat("short", "short") for _ in range(11)] + [long_row]
    r = upload(client, jsonl(tmp_path, rows))
    ds_id = r.json()["id"]
    wait_validated(client, ds_id)
    record = wait_counted(client, ds_id)

    dist = record["report"]["token_distribution"]
    assert dist["truncated_rows"] == 1
    assert dist["max_row_tokens"] == chars(long_row)


def test_invalid_dataset_is_not_counted(client, tmp_path):
    """Counting is spent only on a dataset that can launch: an invalid one
    would waste the expensive pass on a file nobody can quote."""
    rows = [chat("q", "a") for _ in range(3)]  # below MIN_ROWS -> invalid
    r = upload(client, jsonl(tmp_path, rows))
    ds_id = r.json()["id"]
    record = wait_validated(client, ds_id)
    assert record["status"] == "invalid"
    assert record["token_count_status"] is None
    assert record["report"]["token_count"] is None


def test_the_quote_reads_the_recorded_count_without_recomputation(
    client,
    tmp_path,
):
    """The quote's `token_count` comes from the stored count, not a fresh
    tokenising pass: waiting for the phase then creating a job carries the
    exact number the report shows."""
    rows = [chat(f"q{i}", f"a{i}") for i in range(12)]
    r = upload(client, jsonl(tmp_path, rows))
    ds_id = r.json()["id"]
    wait_validated(client, ds_id)
    record = wait_counted(client, ds_id)
    expected = record["report"]["token_count"]
    assert expected

    created = client.post(
        "/v1/jobs", json={"dataset_id": ds_id, "base_model": "qwen3-4b"}
    )
    assert created.status_code == 201
    assert created.json()["quote"]["token_count"] == expected


def test_a_count_that_cannot_be_produced_leaves_the_dataset_valid(
    client,
    tmp_path,
    monkeypatch,
):
    """A tokenizer that fails is an absent estimate, not a broken dataset: the
    phase records `failed`, the report's count stays null, and the quote
    renders the count absent (the quote still exists -- ADR-0031's posture)."""
    from temper_control_plane import tokenize

    def broken_count_row_for(messages_field="messages"):
        raise RuntimeError("tokenizer could not be loaded")

    monkeypatch.setattr(tokenize, "count_row_for", broken_count_row_for)

    rows = [chat(f"q{i}", f"a{i}") for i in range(12)]
    r = upload(client, jsonl(tmp_path, rows))
    ds_id = r.json()["id"]
    wait_validated(client, ds_id)
    record = wait_counted(client, ds_id)

    assert record["token_count_status"] == db.COUNT_PHASE_FAILED
    assert record["status"] == "valid"
    assert record["report"]["token_count"] is None

    created = client.post(
        "/v1/jobs", json={"dataset_id": ds_id, "base_model": "qwen3-4b"}
    )
    assert created.status_code == 201
    assert created.json()["quote"]["token_count"] is None


def test_the_record_exposes_the_phases_own_state(client, tmp_path):
    """The counting phase's state rides on the record beside the report, so a
    page can tell counting-in-progress from done from failed."""
    rows = [chat(f"q{i}", f"a{i}") for i in range(12)]
    r = upload(client, jsonl(tmp_path, rows))
    ds_id = r.json()["id"]
    record = wait_counted(client, ds_id)
    assert "token_count_status" in record
    assert "counting_progress" in record
