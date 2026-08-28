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
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any

from temper_core import artifacts
from temper_core.errors import OrchestratorError

from . import config, storage

# Via config, not by counting directories up from this file. The counted form
# meant the repo root at `api/db.py` and `apps/control-plane/src/` after the
# move, which put the database outside the anchored `/data/` gitignore rule.
# The e2e journeys point it at a database of their own (`TEMPER_DB_PATH`), so
# a journey never writes to the developer's data/ nor inherits its orphans.
DB_PATH = config.DB_PATH

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
    -- The token count and its bounded distribution (issue #42), produced by
    -- the counting phase that runs after validation and recorded with the
    -- dataset version so the quote reads it without recomputation. The status
    -- is the counting phase's own state: NULL (never started / not applicable),
    -- 'counting', 'done' or 'failed'.
    token_count_status TEXT,
    token_count       INTEGER,
    token_stats_json  TEXT,
    counting_progress_json TEXT
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
    -- The measured actuals, frozen at terminal (issue #77): what the run
    -- actually took and cost, beside the quote it was predicted against.
    -- Written once, like the quote -- the mirror image of it.
    actuals_json   TEXT,
    result_json   TEXT,
    -- The storage seam's address for this job's artifact weights.
    artifact_key  TEXT,
    -- The artifact record (issue #32): the members the artifact consists of
    -- -- their storage keys -- plus the verification's bytes and checksum.
    -- Assembled by the orchestrator at packaging time; `kind` is deliberately
    -- NOT stored here but derived from `method`, so a row can never record a
    -- kind its method did not produce and pre-existing rows read correctly
    -- with no migration.
    artifact_json TEXT,
    -- The checkpoints the control plane has verified in storage (issue #37):
    -- one record per slot, each carrying its step, its loss where one exists,
    -- and the key its bytes were verified at. Only verified checkpoints are
    -- recorded here; a partially written one is never presented as complete.
    checkpoints_json TEXT,
    -- The recorded choice of result checkpoint (issue #62): the step with
    -- the best held-out loss and the reason, frozen once at terminal time.
    -- Stored rather than re-derived, so a run's answer cannot change when
    -- retention evicts a checkpoint or the selection rule is edited.
    best_checkpoint_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id   TEXT NOT NULL REFERENCES jobs(id),
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,   -- state | metric | log | error
    message  TEXT,
    data_json TEXT
);

CREATE TABLE IF NOT EXISTS admitted_models (
    id          TEXT PRIMARY KEY,
    repo        TEXT NOT NULL,
    revision    TEXT NOT NULL,  -- pinned; never a branch name (issue #58)
    -- The probe result, frozen at admission (issue #58): a blocked probe is
    -- persisted too, so the user is shown why rather than retrying blindly.
    probe_json  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    UNIQUE(repo, revision)
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
    ("jobs", "checkpoints_json", "TEXT"),
    ("jobs", "best_checkpoint_json", "TEXT"),
    ("jobs", "artifact_json", "TEXT"),
    ("datasets", "progress_json", "TEXT"),
    ("jobs", "actuals_json", "TEXT"),
    ("datasets", "token_count_status", "TEXT"),
    ("datasets", "token_count", "INTEGER"),
    ("datasets", "token_stats_json", "TEXT"),
    ("datasets", "counting_progress_json", "TEXT"),
)


def init() -> None:
    if config.DB_RESET:
        # A clean slate, once, at startup: remove the database (and any WAL
        # side files) so the process boots against a fresh schema. The e2e
        # journeys ask for this so an interrupted run cannot leave an orphaned
        # non-terminal job behind that crashes the next run's startup.
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(str(DB_PATH) + suffix)
            except FileNotFoundError:
                pass
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


# --------------------------------------------------------------------------
# the counting phase (issue #42): its own state, recorded with the version
# --------------------------------------------------------------------------
# Counting runs as a separate phase after validation -- the measurement put the
# ticket on that branch -- so the count has its own state on the row rather
# than living inside the validation report. The report stands the moment
# validation finishes; the count lands later and is merged into the report the
# API publishes, which is the seam the quote already reads.


# The counting phase's own states (issue #42). Defined once here and read
# everywhere -- the phase's transitions write them, the tests assert against
# them -- so the vocabulary cannot drift between the two halves.
COUNT_PHASE_COUNTING = "counting"
COUNT_PHASE_DONE = "done"
COUNT_PHASE_FAILED = "failed"


def begin_token_count(ds_id: str) -> None:
    """Mark the counting phase as in progress, before it starts.

    The dataset stays `valid` -- it is launchable the whole time, and the
    quote renders the count as absent until it lands (ADR-0031's documented
    posture for a count that does not exist yet)."""
    with connect() as c:
        c.execute(
            "UPDATE datasets SET token_count_status=?, "
            "token_count=NULL, token_stats_json=NULL, "
            "counting_progress_json=NULL WHERE id=?",
            (COUNT_PHASE_COUNTING, ds_id),
        )


def set_counting_progress(ds_id: str, progress: dict) -> None:
    """Where the counting pass has got to, for the report page watching it."""
    with connect() as c:
        c.execute(
            "UPDATE datasets SET counting_progress_json=? WHERE id=?",
            (json.dumps(progress), ds_id),
        )


def finish_token_count(ds_id: str, counts: dict) -> None:
    """Record the completed count with the dataset version: the total, and the
    bounded distribution. A finished phase has no progress to show."""
    with connect() as c:
        c.execute(
            "UPDATE datasets SET token_count_status=?, token_count=?, "
            "token_stats_json=?, counting_progress_json=NULL WHERE id=?",
            (
                COUNT_PHASE_DONE,
                counts.get("total_tokens"),
                json.dumps(counts),
                ds_id,
            ),
        )


def fail_token_count(ds_id: str) -> None:
    """The count could not be produced. The dataset stays valid and launchable
    -- the quote renders the count absent -- and the phase's state records why
    rather than leaving the report page to wonder."""
    with connect() as c:
        c.execute(
            "UPDATE datasets SET token_count_status=?, "
            "counting_progress_json=NULL WHERE id=?",
            (COUNT_PHASE_FAILED, ds_id),
        )


def get_dataset(ds_id: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM datasets WHERE id=?", (ds_id,)).fetchone()
    return _dataset_row(r)


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
    return [_present(_dataset_row(r)) for r in rows]


# --------------------------------------------------------------------------
# admitted models (issue #58): the probe result is persisted and shown, and a
# job can be created against a model only after its probe is on record
# --------------------------------------------------------------------------


def create_admitted_model(repo: str, revision: str, probe: dict) -> str:
    """Persist one admission. The reference is unique, so re-probing the same
    pinned model returns its existing row rather than a second one."""
    model_id = new_id("m")
    with connect() as c:
        c.execute(
            "INSERT OR IGNORE INTO admitted_models "
            "(id, repo, revision, probe_json, created_at) VALUES (?,?,?,?,?)",
            (model_id, repo, revision, json.dumps(probe), time.time()),
        )
        row = c.execute(
            "SELECT id FROM admitted_models WHERE repo=? AND revision=?",
            (repo, revision),
        ).fetchone()
    return row["id"]


def get_admitted_model(model_id: str) -> dict | None:
    """One admitted model as the API publishes it: its probe result parsed,
    never the raw JSON string."""
    with connect() as c:
        r = c.execute(
            "SELECT * FROM admitted_models WHERE id=?", (model_id,)
        ).fetchone()
    return _admitted_row(r)


def list_admitted_models(limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM admitted_models ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_present(_admitted_row(r)) for r in rows]


def _admitted_row(r) -> dict | None:
    row = _row(r, {"probe_json": "probe"})
    return row


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
        # The event carries the state it entered as data, not only as a
        # message that may be the human narration ("Selecting hardware")
        # rather than the state's name: issue #77's stage durations are
        # measured from these events, and a state event that does not say
        # which state it entered cannot be measured.
        _append_event(c, job_id, "state", message or state, {"state": state})


def set_checkpoints(job_id: str, checkpoints: list[dict]) -> None:
    """Record the checkpoints the control plane has verified in storage.

    Written without an event: recording a verified checkpoint is bookkeeping,
    and the verification itself already appends the event a reader would want
    to see. The column is replaced whole each time -- the machine reports the
    run's checkpoints once, in result.json, and the control plane records its
    verdict on that set.
    """
    with connect() as c:
        c.execute(
            "UPDATE jobs SET checkpoints_json=? WHERE id=?",
            (json.dumps(checkpoints), job_id),
        )


def set_best_checkpoint(job_id: str, selection: dict) -> None:
    """Record the run's chosen result checkpoint (issue #62).

    Written once, at the same moment the verified checkpoints are recorded,
    and never recomputed: the whole point of recording the choice is that a
    later reader sees the same answer even if retention evicts a checkpoint
    or the selection rule is edited. `selection` is the stored shape of
    `temper_core.checkpoint.CheckpointSelection` -- the step, the basis and
    the reason -- so the record carries its own why.
    """
    with connect() as c:
        c.execute(
            "UPDATE jobs SET best_checkpoint_json=? WHERE id=?",
            (json.dumps(selection), job_id),
        )


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


def record_actuals(job_id: str, actuals: dict) -> None:
    """Freeze the measured figures onto a terminal job (issue #77).

    Written once, like the quote -- the mirror image of it: the quote says
    what was predicted before launch, the actuals say what was measured
    after. Called by the orchestrator the moment a run reaches a terminal
    state, so recording starts with the first run rather than the last.
    """
    with connect() as c:
        c.execute(
            "UPDATE jobs SET actuals_json=? WHERE id=?",
            (json.dumps(actuals), job_id),
        )


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


def artifact_members(job: dict) -> list[tuple[str, str]]:
    """The `(arcname, storage key)` pairs one job's artifact consists of.

    The one place a job's artifact is resolved to its stored objects (issue
    #32). The members come from the record the orchestrator assembled at
    packaging time -- whatever kind it is -- and, for a row written before the
    record existed, from the canonical adapter pair. The download path and the
    teardown path both read this, so the two cannot disagree about what the
    artifact is or drift apart in spelling (a value two components must agree
    on is defined once and read, never retyped).
    """
    record = job.get("artifact_record")
    if record and record.get("members"):
        return [
            (m["name"], m["key"])
            for m in record["members"]
            if m.get("name") and m.get("key")
        ]
    artifact_key = job.get("artifact_key")
    if artifact_key:
        # The legacy shape: a row written before the record existed named one
        # weights key, and the adapter's member names are the canonical pair,
        # each addressed by the seam's key helper.
        return [
            (name, storage.artifact_key(job["id"], name))
            for name in artifacts.ADAPTER_MEMBER_NAMES
        ]
    return []


def _with_artifact(job: dict | None) -> dict | None:
    """Publish the artifact record on a job row (issue #32).

    The artifact's kind is **derived from the job's method, never stored**: a
    QLoRA or LoRA run produces an adapter, a full fine-tune produces a fully
    trained model, and the derivation is what lets pre-existing rows read
    correctly with no migration. The members come from the same resolution the
    download path and teardown use (`artifact_members`).

    An artifact is available exactly when the job is complete: a failed or
    cancelled job that retains keys has had its objects deleted, so publishing
    one there would offer a download that cannot succeed. The published record
    is the safe subset: member *names* (arcnames) and the load path, never the
    storage keys -- where an object lives is the seam's business, not the
    browser's.
    """
    if job is None:
        return None
    members = artifact_members(job)
    if not members or job.get("status") != "complete":
        job["artifact"] = None
        return job
    kind = artifacts.kind_for(job.get("method"))
    record = job.get("artifact_record")
    job["artifact"] = {
        "kind": kind,
        "members": [name for name, _ in members],
        "bytes": record.get("bytes") if record else None,
        "loading": artifacts.loading_instructions(kind),
    }
    return job


def _with_best_checkpoint(job: dict | None) -> dict | None:
    """Publish the run's recorded choice, flagging the chosen checkpoint.

    The choice itself is stored once at terminal time (issue #62); this only
    decorates the published record so the interface can highlight which
    checkpoint the run stands by without re-selecting. Marking `selected`
    from the stored `best_checkpoint.step` is presentation, never a
    re-derivation of the choice: if the two ever disagreed, the stored choice
    is the one the run answers with, and the flag follows it.
    """
    if job is None:
        return None
    if job.get("checkpoints") is None:
        job["checkpoints"] = []
    if job.get("best_checkpoint") is None:
        return job
    best = job["best_checkpoint"]
    step = best.get("step") if isinstance(best, dict) else None
    if step is not None:
        for ckpt in job["checkpoints"]:
            if isinstance(ckpt, dict) and ckpt.get("step") == step:
                ckpt["selected"] = True
    return job


def get_job(job_id: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _with_best_checkpoint(
        _with_artifact(
            _with_warnings(
                _row(
                    r,
                    {
                        "hyperparams_json": "hyperparameters",
                        "result_json": "result",
                        "warnings_json": "warnings",
                        "quote_json": "quote",
                        "overrides_json": "overrides",
                        "actuals_json": "actuals",
                        "checkpoints_json": "checkpoints",
                        "best_checkpoint_json": "best_checkpoint",
                        "artifact_json": "artifact_record",
                    },
                    defaults={
                        "overrides": [],
                        "checkpoints": [],
                        "best_checkpoint": None,
                    },
                )
            )
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


def list_jobs(limit: int | None = 50) -> list[dict]:
    """The job list, newest first. `limit=None` reads every row: the job list
    endpoint caps at 50, but an aggregate that says "across N runs" must not
    silently drop the runs older than its reader's page."""
    with connect() as c:
        if limit is None:
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [
        _present(
            _with_best_checkpoint(
                _with_artifact(
                    _with_warnings(
                        _row(
                            r,
                            {
                                "hyperparams_json": "hyperparameters",
                                "result_json": "result",
                                "warnings_json": "warnings",
                                "quote_json": "quote",
                                "overrides_json": "overrides",
                                "actuals_json": "actuals",
                                "checkpoints_json": "checkpoints",
                                "best_checkpoint_json": "best_checkpoint",
                                "artifact_json": "artifact_record",
                            },
                            defaults={
                                "overrides": [],
                                "checkpoints": [],
                                "best_checkpoint": None,
                            },
                        )
                    )
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
            _with_artifact(
                _with_warnings(
                    _row(
                        r,
                        {
                            "hyperparams_json": "hyperparameters",
                            "quote_json": "quote",
                            "overrides_json": "overrides",
                            "actuals_json": "actuals",
                        },
                        defaults={"overrides": []},
                    )
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


def _dataset_row(r) -> dict | None:
    """A dataset row as the API publishes it.

    The token fields (issue #42) are produced by the phase that runs after
    validation, so they are stored separately and merged into the report here:
    `report.token_count` is the seam the quote reads, and the distribution
    travels beside it for the report page. The counting phase's own state and
    progress ride on the record too.
    """
    ds = _row(
        r,
        {
            "report_json": "report",
            "progress_json": "progress",
            "token_stats_json": "token_distribution",
            "counting_progress_json": "counting_progress",
        },
    )
    if ds is None:
        return None
    token_count = ds.pop("token_count", None)
    token_distribution = ds.pop("token_distribution", None)
    if ds.get("report") is not None:
        ds["report"]["token_count"] = token_count
        ds["report"]["token_distribution"] = token_distribution
    return ds


def _present(row: dict | None) -> dict:
    """Narrow a listed row that cannot actually be absent.

    `fetchall()` returns one row object per matched record, so the None
    case of `_row` is unreachable in a list comprehension over its output
    -- but the Optional still fails the `list[dict]` return type, and
    filtering the Nones out would hide a defect rather than surface it.
    """
    if row is None:
        raise ValueError("a listed row came back missing")
    return row
