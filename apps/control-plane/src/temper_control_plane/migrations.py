"""Versioned schema migrations, applied forward and rolled back.

Each migration is a plain function pair over a psycopg connection: `up`
creates or alters, `down` undoes it exactly. Applied migrations are tracked in
`schema_migrations`, one row per id, so `migrate_up` is idempotent -- running
it against a database that already has some (or all) migrations applied
only runs what is missing.

**0001 is the baseline**: it recreates the schema `db.py` held in SQLite,
translated to PostgreSQL types, column for column. It exists so "the baseline
matches the current schema" is something this file states rather than
something asserted in prose.

**0002 lays down the Phase B tables** the spec names -- quote, attempt,
artifact and manifest, metric series, checkpoint -- as normalized tables
alongside the JSON columns `db.py`'s functions still read and write. They are
not wired into those functions in this migration: `db.py`'s bodies changed
engine, not shape, and wiring the orchestrator's per-attempt and per-checkpoint
writes onto new tables is a behavioural change to code outside this issue's
boundary (the orchestrator, not the persistence seam). Recorded in
ADR-0064 as a considered, bounded gap rather than a silent one.

Money on 0002's tables is an integer in the currency's smallest unit beside
the currency code, the same convention `temper_core.quote` and
`SPEND_CEILING_MINOR` (ADR-0063) already use for a quote's cost range --
carried into typed columns rather than invented fresh.

**What did not move to minor units, and why**: `jobs.price_per_hour` and
`jobs.storage_cost_usd_per_hour` stay `DOUBLE PRECISION`, unconverted, in the
baseline. `price_per_hour` is a billed rate `temper_core.actuals` multiplies
into a minor-unit total itself (`round(duration_s / 3600 * price_per_hour *
minor_unit)`); converting it here would move that rounding earlier without
being asked to. `storage_cost_usd_per_hour` is worse: `test_calibration.py`
seeds it at `0.0137` USD/hour, sub-cent precision by design (ADR-0030 prices
storage as a separate low-volume USD line). Rounding either to whole minor
units at rest would silently corrupt a rate a downstream computation still
needs at full precision -- a real conversion this migration considered and
rejected, not an oversight.
"""

from __future__ import annotations

from collections.abc import Callable

import psycopg

Migration = tuple[
    str, Callable[[psycopg.Cursor], None], Callable[[psycopg.Cursor], None]
]

MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    id          TEXT PRIMARY KEY,
    applied_at  DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now())
)
"""


def _up_0001(cur: psycopg.Cursor) -> None:
    cur.execute("""
        CREATE TABLE datasets (
            id            TEXT PRIMARY KEY,
            filename      TEXT NOT NULL,
            object_key    TEXT NOT NULL,
            created_at    DOUBLE PRECISION NOT NULL,
            row_count     INTEGER,
            schema_type   TEXT,
            enable_thinking INTEGER,
            status        TEXT NOT NULL,
            report_json   TEXT,
            progress_json TEXT,
            token_count_status TEXT,
            token_count       INTEGER,
            token_stats_json  TEXT,
            counting_progress_json TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE jobs (
            id            TEXT PRIMARY KEY,
            dataset_id    TEXT NOT NULL REFERENCES datasets(id),
            base_model    TEXT NOT NULL,
            base_revision TEXT,
            hyperparams_json TEXT NOT NULL,
            status        TEXT NOT NULL,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            created_at    DOUBLE PRECISION NOT NULL,
            started_at    DOUBLE PRECISION,
            finished_at   DOUBLE PRECISION,
            machine_id    INTEGER,
            gpu_type      TEXT,
            device_count  INTEGER,
            method        TEXT,
            price_per_hour DOUBLE PRECISION,
            currency      TEXT,
            disk_gb       INTEGER,
            storage_cost_usd_per_hour DOUBLE PRECISION,
            error_code    TEXT,
            error_message TEXT,
            warnings_json TEXT,
            quote_json    TEXT,
            overrides_json TEXT,
            actuals_json   TEXT,
            result_json   TEXT,
            artifact_key  TEXT,
            artifact_json TEXT,
            checkpoints_json TEXT,
            best_checkpoint_json TEXT,
            retry_from TEXT REFERENCES jobs(id),
            is_moe      INTEGER,
            delivery_request_json TEXT,
            delivery_json TEXT,
            attempts_json TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE events (
            id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            job_id   TEXT NOT NULL REFERENCES jobs(id),
            ts       DOUBLE PRECISION NOT NULL,
            kind     TEXT NOT NULL,
            message  TEXT,
            data_json TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE admitted_models (
            id          TEXT PRIMARY KEY,
            repo        TEXT NOT NULL,
            revision    TEXT NOT NULL,
            probe_json  TEXT NOT NULL,
            created_at  DOUBLE PRECISION NOT NULL,
            UNIQUE(repo, revision)
        )
    """)
    cur.execute("CREATE INDEX idx_events_job ON events(job_id, id)")
    cur.execute("""
        CREATE TABLE job_progress (
            job_id  TEXT NOT NULL REFERENCES jobs(id),
            phase   TEXT NOT NULL,
            done    DOUBLE PRECISION,
            total   DOUBLE PRECISION,
            rate    DOUBLE PRECISION,
            eta_s   DOUBLE PRECISION,
            ts      DOUBLE PRECISION NOT NULL,
            message TEXT,
            PRIMARY KEY (job_id, phase)
        )
    """)
    cur.execute("""
        CREATE TABLE job_output (
            id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            job_id  TEXT NOT NULL REFERENCES jobs(id),
            phase   TEXT NOT NULL,
            line    TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX idx_job_output_job ON job_output(job_id, id)")
    cur.execute("CREATE INDEX idx_jobs_status ON jobs(status)")


def _down_0001(cur: psycopg.Cursor) -> None:
    cur.execute("DROP TABLE job_output")
    cur.execute("DROP TABLE job_progress")
    cur.execute("DROP TABLE admitted_models")
    cur.execute("DROP TABLE events")
    cur.execute("DROP TABLE jobs")
    cur.execute("DROP TABLE datasets")


def _up_0002(cur: psycopg.Cursor) -> None:
    # The quote a job launched under, normalized. `jobs.quote_json` remains
    # the record `db.py`'s functions read and write; this table is the
    # typed Phase B shape the spec asks for, laid down beside it.
    cur.execute("""
        CREATE TABLE quotes (
            id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            job_id        TEXT NOT NULL REFERENCES jobs(id),
            cost_low_minor  BIGINT NOT NULL,
            cost_high_minor BIGINT NOT NULL,
            currency        TEXT NOT NULL,
            storage_cost_usd_total_low_minor  BIGINT,
            storage_cost_usd_total_high_minor BIGINT,
            created_at    DOUBLE PRECISION NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE attempts (
            id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            job_id        TEXT NOT NULL REFERENCES jobs(id),
            attempt_number INTEGER NOT NULL,
            machine_id    INTEGER,
            started_at    DOUBLE PRECISION,
            finished_at   DOUBLE PRECISION,
            outcome       TEXT,
            spec_json     TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE artifacts (
            id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            job_id        TEXT NOT NULL REFERENCES jobs(id),
            kind          TEXT NOT NULL,
            bytes         BIGINT,
            checksum      TEXT,
            created_at    DOUBLE PRECISION NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE artifact_members (
            id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            artifact_id   BIGINT NOT NULL REFERENCES artifacts(id),
            name          TEXT NOT NULL,
            storage_key   TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE metric_series (
            id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            job_id        TEXT NOT NULL REFERENCES jobs(id),
            name          TEXT NOT NULL,
            step          INTEGER NOT NULL,
            value         DOUBLE PRECISION NOT NULL,
            ts            DOUBLE PRECISION NOT NULL
        )
    """)
    cur.execute(
        "CREATE INDEX idx_metric_series_job ON metric_series(job_id, name, step)"
    )
    cur.execute("""
        CREATE TABLE checkpoints (
            id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            job_id        TEXT NOT NULL REFERENCES jobs(id),
            step          INTEGER NOT NULL,
            loss          DOUBLE PRECISION,
            storage_key   TEXT NOT NULL,
            verified_at   DOUBLE PRECISION NOT NULL
        )
    """)


def _down_0002(cur: psycopg.Cursor) -> None:
    cur.execute("DROP TABLE checkpoints")
    cur.execute("DROP TABLE metric_series")
    cur.execute("DROP TABLE artifact_members")
    cur.execute("DROP TABLE artifacts")
    cur.execute("DROP TABLE attempts")
    cur.execute("DROP TABLE quotes")


def _up_0003(cur: psycopg.Cursor) -> None:
    """Temporary authenticated endpoints (spec 011 / issue #78).

    One row per endpoint. The endpoint is the billed, warm machine the
    comparison and the "try it" story run on; it carries an expiry from the
    moment it starts, extends on use, and stops itself -- the forgotten warm
    machine is the loudest complaint against the commercial baseline, so
    stopping itself is the feature, not a convenience (ADR-0065).

    Keys are stored hashed (SHA-256 hex), never plaintext: a leak of the
    store is not a leak of the keys. The prefix is stored alongside for
    display (which key without revealing it).

    `max_expires_at` is the hard ceiling: even a busy endpoint dies there,
    so traffic cannot keep a machine alive forever. `expires_at` is the idle
    grace that extends on use. Both are doubles (epoch seconds) like every
    other timestamp in this schema.

    The table is the source of truth the expiry timer reads; the timers
    themselves live in the process (see `serving.py`). A process restart
    re-reads this table and re-arms timers for any still-running endpoints,
    or stops those already past their deadline -- an endpoint that survives
    a restart without a timer is an endpoint that never stops.
    """
    cur.execute("""
        CREATE TABLE endpoints (
            id              TEXT PRIMARY KEY,
            job_id          TEXT NOT NULL REFERENCES jobs(id),
            status          TEXT NOT NULL,
            api_key_hash    TEXT NOT NULL,
            api_key_prefix  TEXT NOT NULL,
            created_at      DOUBLE PRECISION NOT NULL,
            expires_at      DOUBLE PRECISION NOT NULL,
            last_used_at    DOUBLE PRECISION NOT NULL,
            max_expires_at  DOUBLE PRECISION NOT NULL,
            machine_id      INTEGER,
            price_per_hour  DOUBLE PRECISION,
            currency        TEXT,
            stopped_at      DOUBLE PRECISION,
            stop_reason     TEXT
        )
    """)
    cur.execute("CREATE INDEX idx_endpoints_job ON endpoints(job_id)")
    cur.execute("CREATE INDEX idx_endpoints_status ON endpoints(status)")


def _down_0003(cur: psycopg.Cursor) -> None:
    cur.execute("DROP TABLE endpoints")


MIGRATIONS: tuple[Migration, ...] = (
    ("0001_baseline", _up_0001, _down_0001),
    ("0002_phase_b_tables", _up_0002, _down_0002),
    ("0003_serving_endpoints", _up_0003, _down_0003),
)


def _ensure_tracking_table(conn: psycopg.Connection) -> None:
    conn.execute(MIGRATIONS_TABLE)


def applied_ids(conn: psycopg.Connection) -> list[str]:
    """Every migration id already recorded, oldest first.

    Reads positionally rather than by column name: callers pass connections
    built with either row factory (`db.connect()`'s `dict_row`, or the plain
    tuple rows a maintenance connection outside `db.py` opens), and a
    migration id is unambiguous by position either way.
    """
    _ensure_tracking_table(conn)
    rows = conn.execute(
        "SELECT id FROM schema_migrations ORDER BY id"
    ).fetchall()
    return [(r["id"] if isinstance(r, dict) else r[0]) for r in rows]


def migrate_up(conn: psycopg.Connection) -> None:
    """Apply every migration not yet recorded, in order, oldest first."""
    _ensure_tracking_table(conn)
    already = set(applied_ids(conn))
    for migration_id, up, _down in MIGRATIONS:
        if migration_id in already:
            continue
        with conn.cursor() as cur:
            up(cur)
            cur.execute(
                "INSERT INTO schema_migrations (id) VALUES (%s)",
                (migration_id,),
            )
        conn.commit()


def migrate_down(conn: psycopg.Connection, steps: int = 1) -> None:
    """Roll back the `steps` most recently applied migrations, newest first."""
    _ensure_tracking_table(conn)
    applied = applied_ids(conn)
    by_id = {m[0]: m for m in MIGRATIONS}
    for migration_id in reversed(applied[-steps:]):
        _id, _up, down = by_id[migration_id]
        with conn.cursor() as cur:
            down(cur)
            cur.execute(
                "DELETE FROM schema_migrations WHERE id = %s", (migration_id,)
            )
        conn.commit()


def reset(conn: psycopg.Connection) -> None:
    """Roll every applied migration back, then reapply to head.

    Backs `db.init()`'s `DB_RESET` behaviour: the e2e journeys' guarantee of a
    clean slate, translated from "delete the SQLite file" into "roll the
    schema back to nothing and forward again" -- the same end state, reached
    through the same migrations everything else runs through, rather than a
    second, untested way to build the schema.
    """
    applied = applied_ids(conn)
    if applied:
        migrate_down(conn, steps=len(applied))
    migrate_up(conn)
