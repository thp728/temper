"""SQLite persistence. Three tables, no ORM.

The reference architecture specifies PostgreSQL with Alembic, an outbox, and a
transactional launch. That is correct for a product with tenants; it is not
correct for a build with four evenings. SQLite with explicit SQL keeps the
state machine visible, and every column here earns its place by being read
somewhere.

What is kept from the architecture, because these are the parts that matter:

* **The run spec is immutable after launch.** Hyperparameters are frozen into
  the job row at creation, so a later edit to a dataset or a default cannot
  retroactively change what a completed run claims to have done.
* **Every state transition appends an event, in the same transaction.** A job
  whose status moved with no event recorded is a job you cannot explain, and
  explaining runs is the product.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any

from temper_core.errors import OrchestratorError

from . import config, storage

# Via config, not by counting directories up from this file. The counted form
# meant the repo root at `api/db.py` and `apps/control-plane/src/` after the
# move, which put the database outside the anchored `/data/` gitignore rule.
DB_PATH = config.REPO_ROOT / "data" / "temper.db"

# The full lifecycle. `preparing` covers image build and model download --
# separated from `training` because they fail for completely different reasons
# and a user staring at a progress bar deserves to know which one they are in.
JOB_STATES = (
    "queued",
    "provisioning",
    "preparing",
    "training",
    "packaging",
    "complete",
    "failed",
    "cancelled",
)
TERMINAL_STATES = {"complete", "failed", "cancelled"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    -- The storage seam's address for the dataset's object. Not a path: only
    -- storage.py resolves a key to a location (spec 006, issue #22).
    object_key    TEXT NOT NULL,
    created_at    REAL NOT NULL,
    row_count     INTEGER,
    schema_type   TEXT,
    enable_thinking INTEGER,      -- detected, not chosen; NULL until validated
    status        TEXT NOT NULL,  -- validating | valid | invalid
    report_json   TEXT,           -- the full validation report, errors included
    progress_json TEXT            -- validation progress while status is validating
);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    dataset_id    TEXT NOT NULL REFERENCES datasets(id),
    base_model    TEXT NOT NULL,
    base_revision TEXT,
    -- Frozen at creation. The run spec is immutable after launch.
    hyperparams_json TEXT NOT NULL,
    status        TEXT NOT NULL,
    -- The user has asked for this job to stop. Set by a request, read by the
    -- thread running the job: the two are not in the same call stack, and a
    -- row is the one place both can see.
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    machine_id    INTEGER,
    gpu_type      TEXT,
    device_count  INTEGER,
    method        TEXT,          -- qlora | lora | full; issue #55's decision
    price_per_hour REAL,
    currency      TEXT,
    disk_gb       INTEGER,        -- computed, not the old platform-minimum constant; issue #64
    storage_cost_usd_per_hour REAL,
    error_code    TEXT,
    error_message TEXT,
    -- Warnings attached at creation, frozen like the hyperparameters: what
    -- the user was told before launching is part of the run's record.
    warnings_json TEXT,
    -- The quote the job launched under, frozen at creation and never updated
    -- (issue #72): a completed job can say what it was predicted to cost and
    -- how long it was predicted to take, not only what actually happened.
    quote_json    TEXT,
    -- The decision overrides the launch committed to (issue #79): the
    -- {decision, value} pairs the user pinned, frozen beside the spec so a
    -- run says what it actually used rather than what it would have
    -- defaulted to -- and what the orchestrator re-provisions against.
    overrides_json TEXT,
    result_json   TEXT,
    -- The storage seam's address for this job's artifact weights.
    artifact_key  TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id   TEXT NOT NULL REFERENCES jobs(id),
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,   -- state | metric | log | error
    message  TEXT,
    data_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    # WAL so the orchestrator thread can write while the API reads.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns added after a database already existed somewhere. `CREATE TABLE IF
# NOT EXISTS` is a no-op against a table that is already there, so a column
# added to SCHEMA alone would exist on a fresh machine and be missing on the
# one that has been running all week — and the failure would be a stray
# `no such column` from inside a request handler. Alembic does this properly in
# Phase B; until then, four lines beat a silent divergence.
ADDED_COLUMNS = (
    ("jobs", "cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
    ("jobs", "warnings_json", "TEXT"),
    ("jobs", "base_revision", "TEXT"),
    ("jobs", "device_count", "INTEGER"),
    ("jobs", "method", "TEXT"),
    ("jobs", "disk_gb", "INTEGER"),
    ("jobs", "storage_cost_usd_per_hour", "REAL"),
    ("jobs", "quote_json", "TEXT"),
    ("jobs", "overrides_json", "TEXT"),
    ("datasets", "progress_json", "TEXT"),
)


def init() -> None:
    with connect() as c:
        c.executescript(SCHEMA)
        for table, column, decl in ADDED_COLUMNS:
            present = {
                r["name"]
                for r in c.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in present:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        _rename_location_columns(c)
        _rewrite_legacy_locations(c)


# Issue #22 renamed what these columns mean, not just their values: stored
# objects went from filesystem paths to keys. Renamed in place so a database
# written by the previous build opens cleanly.
RENAMED_COLUMNS = (
    ("datasets", "path", "object_key"),
    ("jobs", "adapter_path", "artifact_key"),
)


def _rename_location_columns(c) -> None:
    for table, old, new in RENAMED_COLUMNS:
        present = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
        if old in present and new not in present:
            c.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")


def _rewrite_legacy_locations(c) -> None:
    """Best-effort conversion of pre-seam absolute paths into their keys.

    The old layout named uploads `{id}.jsonl` and artifact directories after
    the job, so most legacy rows map onto exactly one key and are rewritten.
    A value whose shape does not match is left alone: an address that reads
    back as missing beats a plausible-looking wrong one. Dev databases hold
    disposable data; submission starts fresh.
    """
    rows = c.execute(
        "SELECT id, object_key FROM datasets "
        "WHERE object_key NOT LIKE 'datasets/%'"
    ).fetchall()
    for ds_id, location in rows:
        name = location.replace("\\", "/").rsplit("/", 1)[-1]
        if name == f"{ds_id}.jsonl":
            c.execute(
                "UPDATE datasets SET object_key=? WHERE id=?",
                (storage.dataset_key(ds_id), ds_id),
            )
    rows = c.execute(
        "SELECT id, artifact_key FROM jobs "
        "WHERE artifact_key IS NOT NULL AND artifact_key NOT LIKE 'artifacts/%'"
    ).fetchall()
    for job_id, location in rows:
        parts = location.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[-2] == job_id:
            c.execute(
                "UPDATE jobs SET artifact_key=? WHERE id=?",
                (storage.artifact_key(job_id, parts[-1]), job_id),
            )


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


def create_dataset(
    filename: str, object_key: str, ds_id: str | None = None
) -> str:
    """Insert a dataset row. `ds_id` lets the caller own the id -- the upload
    path derives the key from it, so the id must exist before the row."""
    ds_id = ds_id or new_id("ds")
    with connect() as c:
        c.execute(
            "INSERT INTO datasets (id, filename, object_key, created_at, status)"
            " VALUES (?,?,?,?,?)",
            (ds_id, filename, object_key, time.time(), "validating"),
        )
    return ds_id


def delete_dataset(ds_id: str) -> None:
    """Remove a dataset row whose object never made it into storage.

    Used only by the ingest path to clean up after a mid-stream refusal --
    an upload refused once its true size is known has no object behind it, so
    nothing but the row is left to remove. Deleting a dataset that reached a
    report is not this function's contract.
    """
    with connect() as c:
        c.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def finish_dataset(ds_id: str, report: dict) -> None:
    with connect() as c:
        c.execute(
            "UPDATE datasets SET status=?, row_count=?, schema_type=?, "
            "enable_thinking=?, report_json=?, progress_json=NULL WHERE id=?",
            (
                "valid" if report.get("valid") else "invalid",
                report.get("row_count"),
                report.get("schema_type"),
                None
                if report.get("enable_thinking") is None
                else int(report["enable_thinking"]),
                json.dumps(report),
                ds_id,
            ),
        )


def set_dataset_progress(ds_id: str, progress: dict) -> None:
    """Record where validation has got to, for the page watching it run."""
    with connect() as c:
        c.execute(
            "UPDATE datasets SET progress_json=? WHERE id=?",
            (json.dumps(progress), ds_id),
        )


def get_dataset(ds_id: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM datasets WHERE id=?", (ds_id,)).fetchone()
    return _row(r, {"report_json": "report", "progress_json": "progress"})


def require_dataset(ds_id: str) -> dict:
    """The dataset row, or a coded error -- never None.

    For callers that cannot proceed without the row: the orchestrator reads
    the dataset on the path that spends money, and indexing a None there
    must not be the way a missing row is discovered.
    """
    ds = get_dataset(ds_id)
    if ds is None:
        raise OrchestratorError(
            "dataset_not_found", f"No dataset with id '{ds_id}'."
        )
    return ds


def list_datasets(limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM datasets ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        _present(
            _row(r, {"report_json": "report", "progress_json": "progress"})
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


def create_job(
    dataset_id: str,
    base_model: str,
    hyperparams: dict,
    warnings: list | None = None,
    base_revision: str | None = None,
    quote: dict | None = None,
    overrides: list | None = None,
) -> str:
    job_id = new_id("job")
    with connect() as c:
        c.execute(
            "INSERT INTO jobs (id, dataset_id, base_model, base_revision, "
            "hyperparams_json, status, warnings_json, quote_json, "
            "overrides_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                dataset_id,
                base_model,
                base_revision,
                json.dumps(hyperparams),
                "queued",
                json.dumps(warnings) if warnings else None,
                json.dumps(quote) if quote else None,
                json.dumps(overrides) if overrides else None,
                time.time(),
            ),
        )
        _append_event(c, job_id, "state", "queued")
    return job_id


def set_state(
    job_id: str, state: str, message: str | None = None, **fields
) -> None:
    """Move a job to a new state and record why, atomically.

    The event is written in the same transaction as the status change on
    purpose: a status that moved with no event is a run you cannot account for.
    """
    if state not in JOB_STATES:
        raise ValueError(f"unknown state {state!r}")
    cols, vals = ["status=?"], [state]
    if state == "training" and "started_at" not in fields:
        fields["started_at"] = time.time()
    if state in TERMINAL_STATES:
        fields.setdefault("finished_at", time.time())
    for k, v in fields.items():
        cols.append(f"{k}=?")
        vals.append(json.dumps(v) if k.endswith("_json") else v)
    vals.append(job_id)
    with connect() as c:
        c.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id=?", vals)
        _append_event(c, job_id, "state", message or state)


def request_cancel(job_id: str, note: str = "Cancellation requested") -> str:
    """Ask a job to stop. Says what it found, and never raises for it.

    Returns one of `accepted`, `already_cancelling`, `terminal` or `missing`.
    Strings rather than exceptions because none of the four is exceptional —
    they are the four honest answers to the request, and each maps to a
    different thing to tell the user.

    The read and the write are one transaction, so a job that reaches a
    terminal state between them cannot be flagged after the fact: the
    orchestrator would have stopped looking at the flag by then, and the row
    would claim a cancellation that will never happen.

    `note` is the event recorded against the job. It is the caller's words on
    purpose: the same sentence is what the request answers with, and one string
    said in both places is what stops the button and the run's own history from
    drifting apart.

    The event is written only on the transition. Cancelling is idempotent, and
    a second request that appended a second "cancellation requested" would make
    a double-clicked button look like two decisions in the job's own history.
    """
    with connect() as c:
        row = c.execute(
            "SELECT status, cancel_requested FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None:
            return "missing"
        if row["status"] in TERMINAL_STATES:
            return "terminal"
        if row["cancel_requested"]:
            return "already_cancelling"
        c.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
        _append_event(c, job_id, "log", note)
    return "accepted"


def cancel_requested(job_id: str) -> bool:
    """Whether the user has asked this job to stop.

    Read from the row on every check rather than cached: the request arrives on
    a different thread than the one running the job, and a value read once at
    the start is a value that can never say yes.
    """
    with connect() as c:
        row = c.execute(
            "SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
    return bool(row and row["cancel_requested"])


def add_event(
    job_id: str, kind: str, message: str, data: dict | None = None
) -> None:
    with connect() as c:
        _append_event(c, job_id, kind, message, data)


def _append_event(conn, job_id, kind, message, data=None) -> None:
    conn.execute(
        "INSERT INTO events (job_id, ts, kind, message, data_json) VALUES (?,?,?,?,?)",
        (
            job_id,
            time.time(),
            kind,
            message,
            json.dumps(data) if data is not None else None,
        ),
    )


def _with_warnings(job: dict | None) -> dict | None:
    if job is not None and job["warnings"] is None:
        # An absent list reads as empty, so no client special-cases null.
        job["warnings"] = []
    return job


def get_job(job_id: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _with_warnings(
        _row(
            r,
            {
                "hyperparams_json": "hyperparameters",
                "result_json": "result",
                "warnings_json": "warnings",
                "quote_json": "quote",
                "overrides_json": "overrides",
            },
            defaults={"overrides": []},
        )
    )


def require_job(job_id: str) -> dict:
    """The job row, or a coded error -- never None.

    The raising companion to `get_job`, for the orchestration path: a job
    row that goes missing between launch and a re-read must surface as a
    named failure on the record, not as a TypeError mid-provisioning.
    """
    job = get_job(job_id)
    if job is None:
        raise OrchestratorError("job_not_found", f"No job with id '{job_id}'.")
    return job


def list_jobs(limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        _present(
            _with_warnings(
                _row(
                    r,
                    {
                        "hyperparams_json": "hyperparameters",
                        "result_json": "result",
                        "warnings_json": "warnings",
                        "quote_json": "quote",
                        "overrides_json": "overrides",
                    },
                    defaults={"overrides": []},
                )
            )
        )
        for r in rows
    ]


def get_events(job_id: str, after_id: int = 0, limit: int = 500) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id LIMIT ?",
            (job_id, after_id, limit),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["data"] = (
            json.loads(d.pop("data_json")) if d.get("data_json") else None
        )
        out.append(d)
    return out


def active_jobs() -> list[dict]:
    """Non-terminal jobs. Used at startup to spot runs orphaned by a restart."""
    with connect() as c:
        q = ",".join("?" * len(TERMINAL_STATES))
        rows = c.execute(
            f"SELECT * FROM jobs WHERE status NOT IN ({q})",
            tuple(TERMINAL_STATES),
        ).fetchall()
    return [
        _present(
            _with_warnings(
                _row(
                    r,
                    {
                        "hyperparams_json": "hyperparameters",
                        "quote_json": "quote",
                        "overrides_json": "overrides",
                    },
                    defaults={"overrides": []},
                )
            )
        )
        for r in rows
    ]


def _row(
    r, json_fields: dict[str, str], defaults: dict[str, Any] | None = None
) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for col, name in json_fields.items():
        raw = d.pop(col, None)
        if raw:
            d[name] = json.loads(raw)
        elif defaults and name in defaults:
            d[name] = defaults[name]
        else:
            d[name] = None
    return d


def _present(row: dict | None) -> dict:
    """Narrow a listed row that cannot actually be absent.

    `fetchall()` returns one row object per matched record, so the None
    case of `_row` is unreachable in a list comprehension over its result
    -- but the Optional still fails the `list[dict]` return type, and
    filtering the Nones out would hide a defect rather than surface it.
    """
    if row is None:
        raise ValueError("a listed row came back missing")
    return row
