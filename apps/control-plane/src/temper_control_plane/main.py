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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from temper_control_plane import (
    config,
    datasets,
    db,
    jobs,
    orchestrator,
    storage,
)
from temper_control_plane.contracts_models import (
    DatasetList,
    DatasetRecord,
    DatasetUploaded,
    JobList,
    JobRecord,
    JobSpecPreview,
    ModelCatalog,
)
from temper_control_plane.storage import ObjectNotFound
from temper_control_plane.web import router as web_router
from temper_core import catalog, feasibility, hyperparams


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init()
    storage.STORE.ensure_ready()
    # Fail loudly at boot rather than four seconds into someone's first job.
    if not config.provider_credentials_present():
        print(
            "WARNING: no provider credentials (JL_API_KEY unset, no jl config "
            "file). Datasets validate fine; jobs will fail at provisioning.",
            file=sys.stderr,
        )
    # A job left non-terminal by a restart is not silently resumed -- it is
    # surfaced. Pretending it is still running would be a lie the UI repeats,
    # and the VM it created may still be billing.
    for job in db.active_jobs():
        db.set_state(
            job["id"],
            "failed",
            "Control plane restarted while this job was in flight",
            error_code="orphaned_by_restart",
            error_message="The process restarted mid-run. Any VM it "
            "created may still exist -- check the "
            "provider console.",
        )
    yield


app = FastAPI(
    title="Temper",
    description="Fine-tuning platform. Dataset in, adapter out.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(web_router)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@app.get("/v1/models", tags=["catalog"], response_model=ModelCatalog)
def list_models():
    """The curated base-model catalog.

    An allow-list, not a limitation: detection of an arbitrary architecture is
    easy, but *support* means testing its chat template, tokenizer quirks and
    packing compatibility. This list is a promise about what has been tested.
    """
    return {
        "models": catalog.listing(),
        "default": catalog.DEFAULT_MODEL,
    }


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


@app.post(
    "/v1/datasets",
    tags=["datasets"],
    status_code=201,
    response_model=DatasetUploaded,
)
def upload_dataset(request: Request, file: UploadFile = File(...)):
    """Upload and validate a JSONL dataset.

    **Deliberately a sync handler.** It was `async def`, which ran the
    CPU-bound validation on the event loop and froze every other request for
    as long as it ran -- measured at roughly 80 ms per megabyte, so a 200 MB
    upload blocked the whole server for ~16 seconds. Declared `def` instead,
    FastAPI dispatches it to its worker pool and a long validation blocks only
    its own request. Regression-tested by
    `test_upload_limits.test_large_upload_does_not_block_concurrent_requests`.

    Storage and validation live in `datasets.store_and_validate`, shared with
    the browser's upload form, so the two surfaces cannot drift apart.

    Datasets over `config.MAX_DATASET_BYTES` are refused before validation --
    an unbounded upload fails as an out-of-memory crash rather than a typed
    error, which is worse for the user and for the process.
    """
    datasets.refuse_before_read(request.headers.get("content-length"))
    data = file.file.read()
    ds_id, report = datasets.store_and_validate(file.filename, data)
    return {"id": ds_id, "filename": file.filename, **report}


@app.get(
    "/v1/datasets",
    tags=["datasets"],
    response_model=DatasetList,
)
def list_datasets():
    return {"datasets": db.list_datasets()}


@app.get(
    "/v1/datasets/{dataset_id}",
    tags=["datasets"],
    response_model=DatasetRecord,
)
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


@app.post("/v1/jobs", tags=["jobs"], status_code=201, response_model=JobRecord)
def create_job(req: JobRequest):
    """Launch a fine-tuning job.

    The whole creation path lives in `api.jobs.create`, shared with the
    browser's launch form so the two surfaces cannot drift apart. The
    hyperparameters are frozen into the job row there: a run's spec is
    immutable once launched, so a later change to a default cannot
    retroactively alter what a finished run claims.
    """
    job_id = jobs.create(req.dataset_id, req.base_model, req.hyperparameters)
    return db.get_job(job_id)


@app.get("/v1/jobs", tags=["jobs"], response_model=JobList)
def list_jobs():
    return {"jobs": db.list_jobs()}


@app.get("/v1/jobs/spec", tags=["jobs"], response_model=JobSpecPreview)
def get_job_spec_preview(dataset_id: str):
    """What a launch would train with, before anything is launched.

    Consumed by the shell's model-choice screen (#38), which must show the
    dataset, the effective specification and any feasibility warning while
    the user can still act on them. Declared **before** `/v1/jobs/{job_id}`:
    routes match in declaration order, and "spec" would otherwise be
    captured as a job id.

    Refusals come through `jobs.usable_dataset`, the same path the launch
    itself applies -- a dataset that cannot start a job is refused here with
    its stable code rather than at the moment of commitment.
    """
    ds = jobs.usable_dataset(dataset_id)
    # Estimated against the defaults this screen launches with: the estimate
    # must describe the job the button will start.
    warn = feasibility.warning(
        feasibility.usable_rows(ds), {}, config.MAX_JOB_DURATION_S
    )
    return {
        "dataset": ds,
        "hyperparameters": hyperparams.effective({}),
        "warning": warn,
    }


@app.get("/v1/jobs/{job_id}", tags=["jobs"], response_model=JobRecord)
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
        raise HTTPException(
            409,
            {
                "code": "job_already_terminal",
                "message": f"Job is already '{job['status']}'; there is nothing to "
                f"cancel and nothing has been undone.",
                "status": job["status"],
            },
        )
    job = db.get_job(job_id)
    return {
        "id": job_id,
        "status": job["status"],
        "cancel_requested": True,
        "message": orchestrator.CANCEL_ACK,
    }


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
    artifact_key = job.get("artifact_key")
    if not artifact_key:
        raise HTTPException(
            409,
            {
                "code": "no_artifact",
                "message": f"Job is '{job['status']}'; no adapter is available yet.",
            },
        )
    # Zipped, because an artifact is a set of objects: the weights plus the
    # adapter_config.json that makes them loadable. Built per request rather
    # than cached -- it is a few MB, and a stale zip beside fresh weights is a
    # worse failure than rebuilding it.
    #
    # The weights are read from the key the job row records, so the address
    # written at packaging time is the one read at download time. A missing
    # weights object refuses loudly rather than downloading an empty archive:
    # a download that "succeeds" with nothing in it looks like the deliverable
    # and is not one -- the same failure shape as shipping a bare .safetensors.
    # A missing config alone still zips the weights, because a run that
    # produced none already reported that as an error event when it ended.
    try:
        weights = storage.STORE.get(artifact_key)
    except ObjectNotFound:
        raise HTTPException(
            409,
            {
                "code": "artifact_missing",
                "message": "This job records an artifact, but its stored "
                "object is gone. The job's event log says what happened "
                "to it.",
            },
        ) from None

    zip_members = [("adapter_model.safetensors", weights)]
    try:
        zip_members.append(
            (
                "adapter_config.json",
                storage.STORE.get(storage.artifact_config_key(artifact_key)),
            )
        )
    except ObjectNotFound:
        pass

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, data in zip_members:
            z.writestr(arcname, data)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}-adapter.zip"'
        },
    )


@app.get("/health", tags=["ops"])
def health():
    # Which provider implementation a launch would use. The browser journeys
    # boot this process with TEMPER_FAKE_PROVIDER and refuse to drive a
    # launch unless this field confirms the switch took effect -- a launch
    # that reaches for the billing account from a test is the one mistake
    # this codebase refuses to make cheap.
    return {
        "ok": True,
        "provider": "fake" if config.FAKE_PROVIDER else "real",
    }
