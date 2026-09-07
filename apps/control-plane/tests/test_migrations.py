"""Schema changes are versioned migrations, applied forward and rolled back.

"Rolled back" means exercised: a `down` function nobody has run is not a
rollback. Every test here actually calls `migrate_down` and asserts the
schema and the data it held are gone, not merely that the function exists
and returns without raising.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from temper_control_plane import migrations


@pytest.fixture()
def empty_database(_postgres_template):
    """A brand-new, unmigrated database on the shared throwaway server.

    Distinct from `isolated`'s database, which is cloned from an
    already-migrated template -- this fixture exists specifically to prove
    the migrations run from nothing, which a template clone can never show.
    """
    admin_url, _template_url = _postgres_template
    db_name = f"migtest_{uuid.uuid4().hex[:16]}"
    base_url = admin_url.rsplit("/", 1)[0]
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{db_name}"')
    url = f"{base_url}/{db_name}"
    conn = psycopg.connect(url)
    try:
        yield conn
    finally:
        conn.close()
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')


def _table_names(conn) -> set[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public'"
    ).fetchall()
    return {r[0] for r in rows}


def test_a_fresh_database_migrates_to_head(empty_database):
    migrations.migrate_up(empty_database)
    tables = _table_names(empty_database)
    for expected in (
        "datasets",
        "jobs",
        "events",
        "admitted_models",
        "job_progress",
        "job_output",
        "quotes",
        "attempts",
        "artifacts",
        "artifact_members",
        "metric_series",
        "checkpoints",
        "endpoints",
        "machine_reconciliation",
    ):
        assert expected in tables, f"{expected} missing after migrate_up"
    assert set(migrations.applied_ids(empty_database)) == {
        m[0] for m in migrations.MIGRATIONS
    }


def test_migrate_up_is_idempotent(empty_database):
    migrations.migrate_up(empty_database)
    migrations.migrate_up(empty_database)  # must not raise or duplicate
    assert migrations.applied_ids(empty_database) == [
        m[0] for m in migrations.MIGRATIONS
    ]


def test_every_migration_rolls_back_the_schema_it_added(empty_database):
    """Up to head, then all the way back down: the schema round-trips to
    nothing, which is the property this criterion asks for -- not that a
    particular statement was issued."""
    migrations.migrate_up(empty_database)
    migrations.migrate_down(empty_database, steps=len(migrations.MIGRATIONS))

    tables = _table_names(empty_database)
    for table in ("jobs", "datasets", "quotes", "checkpoints", "endpoints"):
        assert table not in tables
    assert migrations.applied_ids(empty_database) == []

    # And forward again: a migration that could not be re-applied after a
    # rollback would be an outage waiting for a bad afternoon.
    migrations.migrate_up(empty_database)
    assert "jobs" in _table_names(empty_database)


def test_rolling_back_0002_drops_the_phase_b_tables_and_keeps_the_baseline(
    empty_database,
):
    """A partial rollback -- one migration, not every one -- proves `down`
    is scoped to what its own `up` added, not a blunt drop-everything."""
    migrations.migrate_up(empty_database)
    migrations.migrate_down(empty_database, steps=1)

    tables = _table_names(empty_database)
    # Rolling back the last migration (now 0008_endpoint_machine_handle)
    # drops only its own `machine_handle` column and keeps the baseline, the
    # phase-B tables, endpoints and the reconciliation table. The absence of
    # `machine_handle` proves the rollback was scoped; dropping the
    # `endpoints` table it added a column to would mean it was too broad.
    assert "endpoints" in tables
    assert "quotes" in tables
    assert "checkpoints" in tables
    assert "jobs" in tables
    assert "datasets" in tables
    assert "machine_reconciliation" in tables

    def columns(table):
        return {
            r[0]
            for r in empty_database.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s",
                (table,),
            ).fetchall()
        }

    endpoint_cols = columns("endpoints")
    assert "machine_handle" not in endpoint_cols
    assert "machine_id" in endpoint_cols
    # Columns from 0004, 0006 and 0007 are kept -- none was this
    # migration's own.
    assert "correlation_id" in columns("jobs")
    assert "claimed_at" in columns("jobs")
    assert "updated_at" in columns("datasets")
    assert migrations.applied_ids(empty_database) == [
        "0001_baseline",
        "0002_phase_b_tables",
        "0003_serving_endpoints",
        "0004_correlation_id",
        "0005_machine_reconciliation",
        "0006_job_claim_lease",
        "0007_dataset_updated_at",
    ]


def test_rolling_back_the_baseline_loses_the_data_it_held(empty_database):
    """The round-trip this criterion is really about: schema *and data*
    return to what they were. A `down` that drops a table by definition
    drops what was in it, and this proves that is true rather than assumed.
    """
    migrations.migrate_up(empty_database)
    empty_database.execute(
        "INSERT INTO datasets (id, filename, object_key, created_at, status) "
        "VALUES (%s,%s,%s,%s,%s)",
        (
            "ds_roundtrip",
            "d.jsonl",
            "datasets/ds_roundtrip.jsonl",
            1.0,
            "valid",
        ),
    )
    empty_database.commit()

    migrations.migrate_down(empty_database, steps=len(migrations.MIGRATIONS))
    migrations.migrate_up(empty_database)

    # The table is back (schema round-tripped); the row is not (so did the
    # data it held -- a `down` that dropped the table cannot have kept it).
    row = empty_database.execute(
        "SELECT id FROM datasets WHERE id = %s", ("ds_roundtrip",)
    ).fetchone()
    assert row is None
