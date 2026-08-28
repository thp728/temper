"""The raising companions over rows, and the counting phase's own state.

Issue #85 adds `require_job` and `require_dataset`: `db.get_job` returns
`dict | None` but the orchestration path spends money, so a missing row must
arrive as a named error, not as `TypeError` on `None`.

**A finding from issue #43** belongs here rather than in the tests it
replaced. Issue #22's `datasets.path` -> `object_key` and
`jobs.adapter_path` -> `artifact_key` rename shipped as a SQLite-specific
compatibility shim (`_rename_location_columns` /
`_rewrite_legacy_locations`, both now deleted): a database written by the
previous build opened cleanly because `db.init()` detected and rewrote the
old column names and values in place. Three tests pinned that shim by
constructing a SQLite file with the pre-#22 schema, running `db.init()`
over it, and asserting the rewrite. None of that has a PostgreSQL
equivalent: this migration's baseline (`migrations.py`'s `0001_baseline`)
starts every database at the post-#22 schema, `object_key` and
`artifact_key` from birth, because no PostgreSQL-backed deployment of this
product ever ran the pre-#22 column names to begin with. There is nothing
for a rewrite to find. Carrying the shim (and its tests) forward would mean
testing dead code against a database engine it was never written for --
the honest move is to delete both together, not to keep a compatibility
path unreachable in production and untested in truth. `db.py` still starts
every fresh row with the correct column names, which
`test_a_fresh_database_has_the_key_columns_from_birth` below still checks;
what is gone is only the one-time rewrite of a different engine's legacy
values.
"""

from __future__ import annotations

import pytest

from temper_control_plane import db
from temper_core.errors import OrchestratorError


def test_a_fresh_database_has_the_key_columns_from_birth(isolated):
    ds_id = db.create_dataset("d.jsonl", "datasets/ds_new.jsonl", "ds_new")
    assert db.get_dataset(ds_id)["object_key"] == "datasets/ds_new.jsonl"


@pytest.fixture()
def temp_db(isolated):
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


def test_init_with_reset_recreates_a_dirty_database(isolated, monkeypatch):
    """TEMPER_DB_RESET is the journeys' guarantee of a clean slate: a
    database left with a stale non-terminal job (an interrupted run) must not
    survive into the next run's startup, where it would crash the orphaned-job
    scan."""
    from temper_control_plane import config

    monkeypatch.setattr(config, "DB_RESET", True)
    ds_id = db.create_dataset("d.jsonl", "datasets/ds_stale.jsonl", "ds_stale")
    job_id = db.create_job(
        ds_id, "qwen3-4b", {}
    )  # stays `queued` (non-terminal)

    # A second boot with the reset flag wipes the row and the tables are new.
    db.init()
    assert db.get_job(job_id) is None
    assert db.get_dataset(ds_id) is None
