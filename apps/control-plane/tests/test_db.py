"""Stored objects are addressed by key, and the raising companions over rows.

Issue #22 renamed what the two location columns mean: `datasets.path` became
`object_key`, `jobs.adapter_path` became `artifact_key`. A database written by
the previous build must open cleanly -- renamed in place, with legacy values
that map onto their key rewritten rather than silently left pointing at
nothing.

Issue #85 adds `require_job` and `require_dataset`: `db.get_job` returns
`dict | None` but the orchestration path spends money, so a missing row must
arrive as a named error, not as `TypeError` on `None`.
"""

from __future__ import annotations

import sqlite3

import pytest

from temper_control_plane import db
from temper_core.errors import OrchestratorError

OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    path          TEXT NOT NULL,
    created_at    REAL NOT NULL,
    status        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    dataset_id    TEXT NOT NULL REFERENCES datasets(id),
    hyperparams_json TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    REAL NOT NULL,
    adapter_path  TEXT
);
"""


def write_old_database(path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute(
        "INSERT INTO datasets (id, filename, path, created_at, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            "ds_legacy01",
            "notes.jsonl",
            r"C:\dev\temper\data\uploads\ds_legacy01.jsonl",
            1.0,
            "valid",
        ),
    )
    conn.execute(
        "INSERT INTO jobs (id, dataset_id, hyperparams_json, status, "
        "created_at, adapter_path) VALUES (?,?,?,?,?,?)",
        (
            "job_legacy01",
            "ds_legacy01",
            "{}",
            "complete",
            1.0,
            r"C:\dev\temper\data\artifacts\job_legacy01"
            r"\adapter_model.safetensors",
        ),
    )
    conn.commit()
    conn.close()


def test_init_renames_the_location_columns_and_rewrites_legacy_values(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    write_old_database(db.DB_PATH)

    db.init()

    ds = db.get_dataset("ds_legacy01")
    assert ds["object_key"] == "datasets/ds_legacy01.jsonl"
    job = db.get_job("job_legacy01")
    assert (
        job["artifact_key"]
        == "artifacts/job_legacy01/adapter_model.safetensors"
    )
    with sqlite3.connect(db.DB_PATH) as conn:
        names = {
            r[1]
            for r in conn.execute("PRAGMA table_info(datasets)").fetchall()
        } | {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    assert "object_key" in names and "artifact_key" in names
    assert "path" not in names and "adapter_path" not in names


def test_a_legacy_value_that_maps_to_no_key_is_left_alone(
    tmp_path, monkeypatch
):
    """Rewriting requires the old layout's shape. Anything else is not
    invented into a key -- a wrong address beats a plausible-looking one."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    write_old_database(db.DB_PATH)
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute(
        "UPDATE datasets SET path=? WHERE id=?",
        ("somewhere.jsonl", "ds_legacy01"),
    )
    conn.commit()
    conn.close()

    db.init()

    assert db.get_dataset("ds_legacy01")["object_key"] == "somewhere.jsonl"


def test_a_fresh_database_has_the_key_columns_from_birth(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init()
    ds_id = db.create_dataset("d.jsonl", "datasets/ds_new.jsonl", "ds_new")
    assert db.get_dataset(ds_id)["object_key"] == "datasets/ds_new.jsonl"


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init()
    return db


def test_require_job_returns_the_row(temp_db):
    ds_id = temp_db.create_dataset(
        "d.jsonl", "datasets/ds_probe.jsonl", "ds_probe"
    )
    job_id = temp_db.create_job(ds_id, "qwen3-4b", {})
    assert temp_db.require_job(job_id)["id"] == job_id


def test_require_dataset_returns_the_row(temp_db):
    ds_id = temp_db.create_dataset(
        "d.jsonl", "datasets/ds_probe2.jsonl", "ds_probe2"
    )
    assert temp_db.require_dataset(ds_id)["id"] == ds_id


def test_a_missing_job_row_raises_a_coded_error(temp_db):
    with pytest.raises(OrchestratorError) as exc:
        temp_db.require_job("job_absent")
    assert exc.value.code == "job_not_found"
    assert "job_absent" in str(exc.value)


def test_a_missing_dataset_row_raises_a_coded_error(temp_db):
    with pytest.raises(OrchestratorError) as exc:
        temp_db.require_dataset("ds_absent")
    assert exc.value.code == "dataset_not_found"
    assert "ds_absent" in str(exc.value)
