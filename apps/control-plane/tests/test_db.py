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


def test_every_measured_stage_is_a_state_the_state_machine_has(temp_db):
    """The stages temper_core.actuals measures (issue #77) are a subset of
    the lifecycle states db defines -- a state renamed on either side fails
    this loudly rather than silently measuring nothing."""
    from temper_core.actuals import MEASURED_STAGES

    assert set(MEASURED_STAGES) <= set(temp_db.JOB_STATES)


def test_list_jobs_reads_every_row_when_limit_is_none(temp_db):
    """The job list caps at 50 for a page; an aggregate that says "across N
    runs" must not silently drop the runs older than its reader's page."""
    for i in range(3):
        ds_id = temp_db.create_dataset(
            "d.jsonl", f"datasets/ds_probe{i}.jsonl", f"ds_probe{i}"
        )
        temp_db.create_job(ds_id, "qwen3-4b", {})
    assert len(temp_db.list_jobs(limit=None)) == 3
    assert len(temp_db.list_jobs(limit=1)) == 1


# --- the counting phase (issue #42) ----------------------------------------


def test_token_count_is_merged_into_the_report_the_quote_reads(temp_db):
    """The count is produced by the phase that runs after validation, so it is
    stored separately and merged into `report` on read -- the seam the quote
    already reads. Before the phase runs, `token_count` is null."""
    ds_id = temp_db.create_dataset(
        "d.jsonl", "datasets/ds_count.jsonl", "ds_count"
    )
    temp_db.finish_dataset(
        ds_id,
        {
            "valid": True,
            "row_count": 2,
            "usable_rows": 2,
            "schema_type": "chat",
            "enable_thinking": False,
            "errors": [],
            "warnings": [],
            "preview": [],
        },
    )
    assert temp_db.get_dataset(ds_id)["report"]["token_count"] is None

    temp_db.finish_token_count(
        ds_id,
        {
            "total_tokens": 12345,
            "rows_counted": 2,
            "sequence_len": 2048,
            "truncated_rows": 0,
            "max_row_tokens": 7000,
            "histogram": [0, 1, 1],
        },
    )
    record = temp_db.get_dataset(ds_id)
    assert record["token_count_status"] == db.COUNT_PHASE_DONE
    assert record["report"]["token_count"] == 12345
    assert record["report"]["token_distribution"]["rows_counted"] == 2
    # The count does not leak onto the record itself: it lives in the report.
    assert "token_count" not in record


def test_token_count_phase_states_transition_on_the_row(temp_db):
    """The counting phase has its own state, recorded with the version: it is
    'counting' while the pass runs, 'done' when it lands, 'failed' when it
    cannot, and the dataset's own status never moves -- it was already
    `valid`."""
    ds_id = temp_db.create_dataset(
        "d.jsonl", "datasets/ds_count2.jsonl", "ds_count2"
    )
    temp_db.finish_dataset(
        ds_id,
        {
            "valid": True,
            "row_count": 1,
            "usable_rows": 1,
            "schema_type": "chat",
            "enable_thinking": False,
            "errors": [],
            "warnings": [],
            "preview": [],
        },
    )
    temp_db.begin_token_count(ds_id)
    record = temp_db.get_dataset(ds_id)
    assert record["token_count_status"] == db.COUNT_PHASE_COUNTING
    assert record["status"] == "valid"

    temp_db.set_counting_progress(
        ds_id, {"bytes_read": 10, "bytes_total": 100, "rows": 5}
    )
    assert temp_db.get_dataset(ds_id)["counting_progress"]["rows"] == 5

    temp_db.finish_token_count(
        ds_id,
        {
            "total_tokens": 7,
            "rows_counted": 1,
            "sequence_len": 2048,
            "truncated_rows": 0,
            "max_row_tokens": 7,
            "histogram": [1],
        },
    )
    record = temp_db.get_dataset(ds_id)
    assert record["token_count_status"] == db.COUNT_PHASE_DONE
    assert record["counting_progress"] is None
    assert record["report"]["token_count"] == 7


def test_token_count_failure_leaves_the_dataset_valid_and_launchable(
    temp_db,
):
    """A count that cannot be produced is an absent estimate, not a broken
    dataset: the phase records `failed` and the report's count stays null."""
    ds_id = temp_db.create_dataset(
        "d.jsonl", "datasets/ds_count3.jsonl", "ds_count3"
    )
    temp_db.finish_dataset(
        ds_id,
        {
            "valid": True,
            "row_count": 1,
            "usable_rows": 1,
            "schema_type": "chat",
            "enable_thinking": False,
            "errors": [],
            "warnings": [],
            "preview": [],
        },
    )
    temp_db.begin_token_count(ds_id)
    temp_db.fail_token_count(ds_id)
    record = temp_db.get_dataset(ds_id)
    assert record["token_count_status"] == db.COUNT_PHASE_FAILED
    assert record["status"] == "valid"
    assert record["report"]["token_count"] is None
    assert record["counting_progress"] is None


def test_init_with_reset_recreates_a_dirty_database(tmp_path, monkeypatch):
    """TEMPER_DB_RESET is the journeys' guarantee of a clean slate: a
    database left with a stale non-terminal job (an interrupted run) must not
    survive into the next run's startup, where it would crash the orphaned-job
    scan."""
    from temper_control_plane import config

    monkeypatch.setattr(config, "DB_RESET", True)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init()
    ds_id = db.create_dataset("d.jsonl", "datasets/ds_stale.jsonl", "ds_stale")
    job_id = db.create_job(
        ds_id, "qwen3-4b", {}
    )  # stays `queued` (non-terminal)

    # A second boot with the reset flag wipes the row and the tables are new.
    db.init()
    assert db.get_job(job_id) is None
    assert db.get_dataset(ds_id) is None
