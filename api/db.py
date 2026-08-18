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
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "temper.db"

# The full lifecycle. `preparing` covers image build and model download --
# separated from `training` because they fail for completely different reasons
# and a user staring at a progress bar deserves to know which one they are in.
JOB_STATES = (
    "queued", "provisioning", "preparing", "training",
    "packaging", "complete", "failed", "cancelled",
)
TERMINAL_STATES = {"complete", "failed", "cancelled"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    path          TEXT NOT NULL,
    created_at    REAL NOT NULL,
    row_count     INTEGER,
    schema_type   TEXT,
    enable_thinking INTEGER,      -- detected, not chosen; NULL until validated
    status        TEXT NOT NULL,  -- validating | valid | invalid
    report_json   TEXT            -- the full validation report, errors included
);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    dataset_id    TEXT NOT NULL REFERENCES datasets(id),
    base_model    TEXT NOT NULL,
    -- Frozen at creation. The run spec is immutable after launch.
    hyperparams_json TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    machine_id    INTEGER,
    gpu_type      TEXT,
    price_per_hour REAL,
    currency      TEXT,
    error_code    TEXT,
    error_message TEXT,
    result_json   TEXT,
    adapter_path  TEXT
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


def init() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

def create_dataset(filename: str, path: Path) -> str:
    ds_id = new_id("ds")
    with connect() as c:
        c.execute(
            "INSERT INTO datasets (id, filename, path, created_at, status) "
            "VALUES (?,?,?,?,?)",
            (ds_id, filename, str(path), time.time(), "validating"))
    return ds_id


def finish_dataset(ds_id: str, report: dict) -> None:
    with connect() as c:
        c.execute(
            "UPDATE datasets SET status=?, row_count=?, schema_type=?, "
            "enable_thinking=?, report_json=? WHERE id=?",
            ("valid" if report.get("valid") else "invalid",
             report.get("row_count"), report.get("schema_type"),
             None if report.get("enable_thinking") is None
             else int(report["enable_thinking"]),
             json.dumps(report), ds_id))


def get_dataset(ds_id: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM datasets WHERE id=?", (ds_id,)).fetchone()
    return _row(r, {"report_json": "report"})


def list_datasets(limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM datasets ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
    return [_row(r, {"report_json": "report"}) for r in rows]


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

def create_job(dataset_id: str, base_model: str, hyperparams: dict) -> str:
    job_id = new_id("job")
    with connect() as c:
        c.execute(
            "INSERT INTO jobs (id, dataset_id, base_model, hyperparams_json, "
            "status, created_at) VALUES (?,?,?,?,?,?)",
            (job_id, dataset_id, base_model, json.dumps(hyperparams),
             "queued", time.time()))
        _append_event(c, job_id, "state", "queued")
    return job_id


def set_state(job_id: str, state: str, message: str | None = None, **fields) -> None:
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


def add_event(job_id: str, kind: str, message: str, data: dict | None = None) -> None:
    with connect() as c:
        _append_event(c, job_id, kind, message, data)


def _append_event(conn, job_id, kind, message, data=None) -> None:
    conn.execute(
        "INSERT INTO events (job_id, ts, kind, message, data_json) VALUES (?,?,?,?,?)",
        (job_id, time.time(), kind, message,
         json.dumps(data) if data is not None else None))


def get_job(job_id: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row(r, {"hyperparams_json": "hyperparameters", "result_json": "result"})


def list_jobs(limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r, {"hyperparams_json": "hyperparameters", "result_json": "result"})
            for r in rows]


def get_events(job_id: str, after_id: int = 0, limit: int = 500) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id LIMIT ?",
            (job_id, after_id, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["data"] = json.loads(d.pop("data_json")) if d.get("data_json") else None
        out.append(d)
    return out


def active_jobs() -> list[dict]:
    """Non-terminal jobs. Used at startup to spot runs orphaned by a restart."""
    with connect() as c:
        q = ",".join("?" * len(TERMINAL_STATES))
        rows = c.execute(
            f"SELECT * FROM jobs WHERE status NOT IN ({q})",
            tuple(TERMINAL_STATES)).fetchall()
    return [_row(r, {"hyperparams_json": "hyperparameters"}) for r in rows]


def _row(r, json_fields: dict[str, str]) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for col, name in json_fields.items():
        raw = d.pop(col, None)
        d[name] = json.loads(raw) if raw else None
    return d
