"""Temper control plane.

One FastAPI process. Upload a dataset, launch a job, watch it, take the adapter.

Deliberately not here: auth, billing, a separate gateway process, a message
broker. The brief sanctions cutting auth and billing; the rest are scaling
answers to a problem this does not have. What *is* here is the full job
lifecycle with every transition recorded, because explaining a run is the
product.
"""

from __future__ import annotations

import io
import sys
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import catalog, config, db, feasibility, orchestrator, validation  # noqa: E402

UPLOADS = Path(__file__).parent.parent / "data" / "uploads"


def _fmt_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024:.1f} KB"


def _too_large(actual: int, limit: int) -> HTTPException:
    return HTTPException(413, {
        "code": "dataset_too_large",
        "message": (
            f"This dataset is {_fmt_size(actual)} ({actual:,} bytes); "
            f"the current upload limit is {_fmt_size(limit)} "
            f"({limit:,} bytes). The limit exists because validation "
            f"holds the whole dataset in memory (about 4.8x its size), "
            f"so it is a limit of the current in-memory validation path, "
            f"not a product rule -- streaming validation will remove it."),
        "limit_bytes": limit,
        "actual_bytes": actual,
    })


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    orchestrator.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # Fail loudly at boot rather than four seconds into someone's first job.
    if not config.provider_credentials_present():
        print("WARNING: no provider credentials (JL_API_KEY unset, no jl config "
              "file). Datasets validate fine; jobs will fail at provisioning.",
              file=sys.stderr)
    # A job left non-terminal by a restart is not silently resumed -- it is
    # surfaced. Pretending it is still running would be a lie the UI repeats,
    # and the VM it created may still be billing.
    for job in db.active_jobs():
        db.set_state(job["id"], "failed",
                     "Control plane restarted while this job was in flight",
                     error_code="orphaned_by_restart",
                     error_message="The process restarted mid-run. Any VM it "
                                   "created may still exist -- check the "
                                   "provider console.")
    yield


app = FastAPI(
    title="Temper",
    description="Fine-tuning platform. Dataset in, adapter out.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

@app.get("/v1/models", tags=["catalog"])
def list_models():
    """The curated base-model catalog.

    An allow-list, not a limitation: detection of an arbitrary architecture is
    easy, but *support* means testing its chat template, tokenizer quirks and
    packing compatibility. This list is a promise about what has been tested.
    """
    return {"models": catalog.listing(), "default": catalog.DEFAULT_MODEL}


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

@app.post("/v1/datasets", tags=["datasets"], status_code=201)
def upload_dataset(request: Request, file: UploadFile = File(...)):
    """Upload and validate a JSONL dataset.

    **Deliberately a sync handler.** It was `async def`, which ran the
    CPU-bound validation on the event loop and froze every other request for
    as long as it ran -- measured at roughly 80 ms per megabyte, so a 200 MB
    upload blocked the whole server for ~16 seconds. Declared `def` instead,
    FastAPI dispatches it to its worker pool and a long validation blocks only
    its own request. Regression-tested by
    `test_upload_limits.test_large_upload_does_not_block_concurrent_requests`.

    Validation is synchronous and always returns a report. A dataset that fails
    is still stored with its report attached, so the user can see exactly which
    lines to fix rather than re-uploading blind.

    Datasets over `config.MAX_DATASET_BYTES` are refused before validation --
    an unbounded upload fails as an out-of-memory crash rather than a typed
    error, which is worse for the user and for the process.
    """
    if not file.filename.endswith((".jsonl", ".json")):
        raise HTTPException(400, "Only .jsonl files are accepted.")

    limit = config.MAX_DATASET_BYTES

    # Refuse on the declared body size before reading: reading first would
    # load an arbitrarily large file into memory, which is the failure the
    # limit exists to prevent. The declared length is the multipart body, so
    # it slightly overstates the file itself -- close enough to refuse on,
    # and it never understates it. The header can be absent or lying, which
    # is why the authoritative check after the read remains.
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise _too_large(int(declared), limit)

    data = file.file.read()
    if len(data) > limit:
        raise _too_large(len(data), limit)

    ds_id = db.new_id("ds")
    path = UPLOADS / f"{ds_id}.jsonl"
    path.write_bytes(data)

    with db.connect() as c:
        import time as _t
        c.execute("INSERT INTO datasets (id, filename, path, created_at, status) "
                  "VALUES (?,?,?,?,?)",
                  (ds_id, file.filename, str(path), _t.time(), "validating"))

    report = validation.validate(path).to_dict()
    db.finish_dataset(ds_id, report)
    return {"id": ds_id, "filename": file.filename, **report}


@app.get("/v1/datasets", tags=["datasets"])
def list_datasets():
    return {"datasets": db.list_datasets()}


@app.get("/v1/datasets/{dataset_id}", tags=["datasets"])
def get_dataset(dataset_id: str):
    ds = db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "No such dataset.")
    return ds


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    dataset_id: str
    base_model: str = Field(default=catalog.DEFAULT_MODEL)
    hyperparameters: dict = Field(default_factory=dict)


@app.post("/v1/jobs", tags=["jobs"], status_code=201)
def create_job(req: JobRequest):
    """Launch a fine-tuning job.

    Refuses an invalid dataset rather than discovering it on a GPU four minutes
    later. The hyperparameters are frozen into the job row here: a run's spec is
    immutable once launched, so a later change to a default cannot retroactively
    alter what a finished run claims.

    A dataset that plainly cannot finish inside the maximum duration gets a
    **warning attached, not a refusal** (spec 002): the estimate is crude --
    measured throughput on one real run -- and a wrong block is worse than a
    wrong warning. The warning is frozen onto the job row like the
    hyperparameters, so what the user was told before launching stays part of
    the run's record.
    """
    ds = db.get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(404, "No such dataset.")
    if ds["status"] != "valid":
        raise HTTPException(400, {
            "code": "dataset_invalid",
            "message": "This dataset failed validation and cannot be trained on.",
            "errors": (ds.get("report") or {}).get("errors", []),
        })
    if not catalog.get(req.base_model):
        raise HTTPException(400, {
            "code": "unknown_model",
            "message": f"'{req.base_model}' is not in the catalog.",
            "available": [m["id"] for m in catalog.listing()],
        })

    # The ceiling is read at request time, not import: an operator changing
    # TEMPER_MAX_JOB_DURATION_S should not need a restart for the warning to
    # reflect it.
    warn = feasibility.warning(feasibility.usable_rows(ds),
                               req.hyperparameters, config.MAX_JOB_DURATION_S)
    warnings = [warn] if warn else []

    job_id = db.create_job(req.dataset_id, req.base_model, req.hyperparameters,
                           warnings=warnings)
    if warn:
        db.add_event(job_id, "log", warn["message"], {
            k: v for k, v in warn.items() if k != "message"})
    orchestrator.launch(job_id)
    return db.get_job(job_id)


@app.get("/v1/jobs", tags=["jobs"])
def list_jobs():
    return {"jobs": db.list_jobs()}


@app.get("/v1/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "No such job.")
    return job


@app.post("/v1/jobs/{job_id}/cancel", tags=["jobs"])
def cancel_job(job_id: str):
    """Ask a running job to stop. Destructive, and says so.

    The request records the user's decision; the thread running the job sees it
    within a poll interval, destroys the machine and ends the job `cancelled`.
    So this returns the state the job is in *now*, which is usually still a
    working one — reporting `cancelled` here would claim a teardown that has
    not happened yet, and teardown is the whole point of cancelling.

    Repeating it succeeds quietly: a double-clicked button is not an error.
    Cancelling a job that has already ended is refused with a stable code,
    because nothing was undone and the user should not think otherwise.
    """
    outcome = db.request_cancel(job_id, note=orchestrator.CANCEL_ACK)
    if outcome == "missing":
        raise HTTPException(404, "No such job.")
    if outcome == "terminal":
        job = db.get_job(job_id)
        raise HTTPException(409, {
            "code": "job_already_terminal",
            "message": f"Job is already '{job['status']}'; there is nothing to "
                       f"cancel and nothing has been undone.",
            "status": job["status"],
        })
    job = db.get_job(job_id)
    return {"id": job_id, "status": job["status"], "cancel_requested": True,
            "message": orchestrator.CANCEL_ACK}


@app.get("/v1/jobs/{job_id}/events", tags=["jobs"])
def get_events(job_id: str, after: int = 0):
    """Durable event log. `after` is the last event id the client holds.

    Polling against a monotonic id rather than streaming: a reconnecting client
    catches up from the database instead of losing whatever happened while it
    was away.
    """
    if not db.get_job(job_id):
        raise HTTPException(404, "No such job.")
    events = db.get_events(job_id, after_id=after)
    return {"events": events, "last_id": events[-1]["id"] if events else after}


@app.get("/v1/jobs/{job_id}/adapter", tags=["jobs"])
def download_adapter(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "No such job.")
    if not job.get("adapter_path"):
        raise HTTPException(409, {
            "code": "no_artifact",
            "message": f"Job is '{job['status']}'; no adapter is available yet.",
        })
    # Zipped, because an adapter is a directory: the weights plus the
    # adapter_config.json that makes them loadable. Built per request rather
    # than cached -- it is a few MB, and a stale zip beside fresh weights is a
    # worse failure than rebuilding it.
    directory = Path(job["adapter_path"]).parent
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(directory.iterdir()):
            if f.is_file():
                z.write(f, arcname=f.name)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="{job_id}-adapter.zip"'})


@app.get("/health", tags=["ops"])
def health():
    return {"ok": True}
