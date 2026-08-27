"""Temper control plane.

One FastAPI process. Upload a dataset, launch a job, watch it, take the adapter.

Deliberately not here: auth, billing, a separate gateway process, a message
broker. The brief sanctions cutting auth and billing; the rest are scaling
answers to a problem this does not have. What *is* here is the full job
lifecycle with every transition recorded, because explaining a run is the
product.
"""

from __future__ import annotations

import itertools
import sys
import zipfile
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import BinaryIO

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from temper_control_plane import (
    chunks,
    config,
    datasets,
    db,
    jobs,
    models,
    orchestrator,
    quote,
    storage,
)
from temper_control_plane.contracts_models import (
    Calibration,
    DatasetAccepted,
    DatasetList,
    DatasetRecord,
    DecisionOverride,
    EventPage,
    JobList,
    JobRecord,
    JobSpecPreview,
    ModelCatalog,
    Quote,
    QuoteRequest,
)
from temper_control_plane.storage import ObjectNotFound
from temper_control_plane.web import router as web_router
from temper_core import (
    calibration,
    catalog,
    feasibility,
    gpus,
    hyperparams,
    memory,
    overrides,
)

# The default resolver, built once at import: `HuggingFaceModels()` performs
# no I/O until `.resolve()` is called, so this is as safe at import time as
# constructing any other stateless client. Tests replace this attribute
# outright with `fake_models.catalog_models()` -- the models-seam equivalent
# of `conftest.no_real_provider` -- so the suite never depends on network
# reachability.
MODELS: models.Models = models.new_models()


def _quote_for(
    ds: dict,
    m: catalog.BaseModel,
    hyperparameters: dict,
    overrides_list: list[overrides.Override] | None = None,
) -> dict | None:
    """The quote for one (dataset, model, hyperparameters, overrides)
    configuration, or None when it cannot be priced. Never raises for an
    absent estimate: the estimate warns, it does not block (spec 005). An
    override that cannot be honoured raises `QuoteRefused`/`OverrideError`
    for the handler to turn into a coded refusal."""
    return quote.for_config(ds, m, hyperparameters, overrides_list)


def _catalog_entry(m: catalog.BaseModel) -> dict:
    """One catalog model, with its peak-memory prediction computed fresh.

    Facts are resolved through the seam rather than stored on the catalog
    entry -- the whole point of spec 005's `models` seam is that a stored
    number cannot describe a model the catalog has never seen, so even a
    catalog model's peak is computed the same way an imported one's would
    be. `functools.lru_cache` would be the obvious next step once resolving
    costs a real network round trip on every request; not added yet because
    nothing has measured that it matters.
    """
    facts = MODELS.resolve(m.repo, m.revision)
    peak = memory.predict_peak(
        facts,
        method="qlora",
        lora_r=hyperparams.DEFAULTS["lora_r"],
        sequence_len=hyperparams.DEFAULTS["sequence_len"],
        micro_batch_size=hyperparams.DEFAULTS["micro_batch_size"],
    )
    capacity = gpus.CAPACITY_GB[m.min_gpu_type]
    return {
        **m.to_dict(),
        "peak_memory": {
            "weights_gb": peak.weights_gb,
            "gradients_gb": peak.gradients_gb,
            "optimizer_gb": peak.optimizer_gb,
            "activations_gb": peak.activations_gb,
            "overhead_gb": peak.overhead_gb,
            "total_gb": peak.total_gb,
            "trainable_params": peak.trainable_params,
            "tolerance": memory.PEAK_TOLERANCE,
            "gpu_type": m.min_gpu_type,
            "gpu_capacity_gb": capacity,
            "headroom_gb": memory.headroom_gb(peak, capacity),
        },
    }


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

    Each entry's `peak_memory` is computed fresh through the `models` seam
    (spec 005) rather than read from a stored figure -- see
    `_catalog_entry`.
    """
    return {
        "models": [_catalog_entry(m) for m in catalog.CATALOG.values()],
        "default": catalog.DEFAULT_MODEL,
    }


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


@app.post(
    "/v1/datasets",
    tags=["datasets"],
    status_code=202,
    response_model=DatasetAccepted,
)
def upload_dataset(request: Request, file: UploadFile = File(...)):
    """Upload, store and validate a JSONL dataset.

    **Returns before validation finishes.** The request refuses an oversized
    body on the declared Content-Length before reading anything, stores the
    bytes as they arrive (so the control plane never holds the file whole),
    and returns the dataset's id while validation runs in the background.
    Progress and the report land on `GET /v1/datasets/{id}` -- that is what
    lets a large upload show a proportion-complete bar instead of a frozen
    page, and it is why this handler is deliberately a `def` worker-pool
    handler (ADR-0006: CPU-bound work stays off the event loop).

    Storage and validation live in `datasets.ingest`, shared with the browser's
    upload form, so the two surfaces cannot drift apart.
    """
    ds_id, status = datasets.ingest(
        request.headers.get("content-length"), file.filename or "", file.file
    )
    return {"id": ds_id, "filename": file.filename or "", "status": status}


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
    overrides: list[DecisionOverride] = Field(default_factory=list)


def _override_list(
    req_overrides: list[DecisionOverride],
) -> list[overrides.Override]:
    """The API's `{decision, value}` pairs as the core override records."""
    return [overrides.Override(o.decision, o.value) for o in req_overrides]


def _refuse(e: quote.QuoteRefused | overrides.OverrideError) -> HTTPException:
    """An override that cannot be honoured, as a coded 400. Both refusal
    kinds carry the same stable-code payload shape."""
    payload = (
        e.to_payload() if isinstance(e, quote.QuoteRefused) else e.to_dict()
    )
    return HTTPException(400, payload)


@app.post("/v1/jobs", tags=["jobs"], status_code=201, response_model=JobRecord)
def create_job(req: JobRequest):
    """Launch a fine-tuning job.

    The whole creation path lives in `api.jobs.create`, shared with the
    browser's launch form so the two surfaces cannot drift apart. The
    hyperparameters are frozen into the job row there: a run's spec is
    immutable once launched, so a later change to a default cannot
    retroactively alter what a finished run claims.

    `req.overrides` (issue #79) are the decisions the user pinned instead of
    the predictor's. The launch is refused -- with a stable code and, where
    the refusal is on memory grounds, the same arithmetic the predictor used
    -- when an override cannot be honoured, and the overrides themselves are
    frozen into the job spec so the run says what it actually used.

    The quote the launch was shown is computed here and frozen onto the job
    with the rest of the spec. It never blocks: if the provider is unreachable
    or nothing fits, the job still launches -- an estimate warns, it does not
    refuse (spec 005). The only exceptions are refusals of the user's own
    overrides, which are not absent estimates but demands that cannot be met.
    """
    override_list = _override_list(req.overrides)
    try:
        job_id = jobs.create(
            req.dataset_id,
            req.base_model,
            req.hyperparameters,
            quote=quote.quote_for_launch(
                req.dataset_id,
                req.base_model,
                req.hyperparameters,
                override_list,
            ),
            overrides_list=override_list,
        )
    except (quote.QuoteRefused, overrides.OverrideError) as e:
        raise _refuse(e) from e
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

    Deliberately quote-free: the quote is fetched separately
    (`GET /v1/quotes`) once a model is selected, so this page -- which
    renders before anything is chosen -- never waits on live provider data.
    An estimate never blocks the surface it appears on (spec 005).

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


@app.get("/v1/quotes", tags=["jobs"], response_model=Quote | None)
def get_quote(dataset_id: str, base_model: str = catalog.DEFAULT_MODEL):
    """The quote for one (dataset, model) configuration, or null.

    Fetched by the plan screen once a model is selected, so the page renders
    before the estimate does and the estimate never blocks it. A
    configuration that cannot be priced (provider unreachable, nothing fits)
    is null rather than an error -- the estimate warns, it does not refuse
    (spec 005).
    """
    ds = jobs.usable_dataset(dataset_id)
    m = catalog.get(base_model)
    if m is None:
        raise HTTPException(
            400,
            {
                "code": "unknown_model",
                "message": f"'{base_model}' is not in the catalog.",
                "available": [m["id"] for m in catalog.listing()],
            },
        )
    return _quote_for(ds, m, {})


@app.post("/v1/quotes", tags=["jobs"], response_model=Quote | None)
def recompute_quote(req: QuoteRequest):
    """The plan recomputed around the user's overrides, or null.

    Issue #79: changing one decision re-requests the plan rather than mutating
    it locally, so the recomputation rules live in one place -- this endpoint
    and the launch path both read `quote.build_quote`. An override that cannot
    be honoured is refused with a stable code and, on memory grounds, the same
    arithmetic the predictor used (`configuration_does_not_fit`): the
    plan-with-overrides is a demand, not an estimate, and a demand that cannot
    be met must say so.

    Returns null (never a refusal) only when the estimate itself cannot be
    priced -- provider unreachable, model unresolvable -- because an estimate
    warns, it does not block (spec 005).
    """
    ds = jobs.usable_dataset(req.dataset_id)
    m = catalog.get(req.base_model)
    if m is None:
        raise HTTPException(
            400,
            {
                "code": "unknown_model",
                "message": f"'{req.base_model}' is not in the catalog.",
                "available": [m["id"] for m in catalog.listing()],
            },
        )
    try:
        return _quote_for(
            ds, m, {}, overrides_list=_override_list(req.overrides)
        )
    except (quote.QuoteRefused, overrides.OverrideError) as e:
        raise _refuse(e) from e


@app.get("/v1/calibration", tags=["jobs"], response_model=Calibration)
def calibration_summary():
    """Predictions against measurements across terminal jobs (issue #77).

    The aggregate the calibration view renders: per-metric and per-phase
    rolls of what was predicted against what happened, so a systematically
    wrong estimate is visible rather than absorbed into a better-looking
    average. Only jobs that reached a terminal state and carry both a frozen
    quote and frozen actuals contribute -- an estimate with no run to measure
    it against is not calibration, it is a guess. The runs themselves ride
    along so an outlier can be named rather than pointed at.
    """
    runs = []
    for job in db.list_jobs(limit=None):
        if job.get("status") not in db.TERMINAL_STATES:
            continue
        if not job.get("quote") or not job.get("actuals"):
            continue
        runs.append(
            calibration.Run(
                job_id=job["id"],
                base_model=job["base_model"],
                status=job["status"],
                created_at=job["created_at"],
                quote=job["quote"],
                actuals=job["actuals"],
            )
        )
    return calibration.aggregate(runs)


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
        if job is None:
            # The row existed when request_cancel answered; a re-read that
            # comes back empty gets the same named 404, not a TypeError.
            raise HTTPException(404, "No such job.")
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
    if job is None:
        raise HTTPException(404, "No such job.")
    return {
        "id": job_id,
        "status": job["status"],
        "cancel_requested": True,
        "message": orchestrator.CANCEL_ACK,
    }


@app.get("/v1/jobs/{job_id}/events", tags=["jobs"], response_model=EventPage)
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


class _CountingSink:
    """A write-only file object whose tell() counts rather than seeks.

    The zip format wants offsets, and a pipe has none; answering tell() from
    bytes written is what lets the archive be produced through one without
    ever being held whole.
    """

    def __init__(self, inner: BinaryIO) -> None:
        self._inner = inner
        self._pos = 0

    def write(self, data) -> int:
        n = self._inner.write(data)
        self._pos += n
        return n

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        # Declared false on purpose: a pipe cannot seek, and telling zipfile
        # so up front makes it write data descriptors instead of patching
        # headers after the fact.
        return False

    def flush(self) -> None:
        self._inner.flush()

    def close(self) -> None:
        # Present because the zip-writable protocol expects it; zipfile never
        # calls it on a file object that was passed in, and closing once is
        # the chunk helper's job anyway.
        self._inner.close()


def _open_member(key: str) -> Iterator[bytes] | None:
    """The object's chunks with the first already flowing, or None if absent.

    The peek is what lets a missing object refuse as a 409: once the
    response starts, the status code has already gone out and a failure
    could only be a broken download. Holding one bounded chunk is exactly
    what streaming permits; `get_stream` raises before that first chunk for
    an absent object -- which is why the call sits inside the try -- and an
    object that is legitimately empty comes back as an exhausted iterator
    rather than an absence.
    """
    try:
        stream = storage.STORE.get_stream(key)
        first = next(stream)
    except StopIteration:
        return iter(())
    except ObjectNotFound:
        return None
    return itertools.chain([first], stream)


def _zip_chunks(members: list[tuple[str, Iterator[bytes]]]) -> Iterator[bytes]:
    """The zip of stored objects as bounded chunks.

    Built per request rather than cached -- a stale zip beside fresh adapter
    files is a worse failure than rebuilding it. Streamed rather than
    buffered: the artifact is the payload that grows without bound, and
    zipping into an in-memory buffer would hold every byte of a full
    fine-tune per request. Members arrive as chunk streams from the storage
    seam and are written through one at a time.

    Zip entries get fixed timestamps: the archive is rebuilt per request, so
    a clock in the headers would be noise, and deterministic bytes make two
    downloads of one artifact comparable.
    """

    def pour(sink: BinaryIO) -> None:
        with zipfile.ZipFile(
            _CountingSink(sink), "w", zipfile.ZIP_DEFLATED
        ) as z:
            for arcname, member_chunks in members:
                info = zipfile.ZipInfo(arcname)
                with z.open(info, "w") as dst:
                    for chunk in member_chunks:
                        dst.write(chunk)

    return chunks.piped_chunks(pour)


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
    # adapter_config.json that makes them loadable.
    #
    # Both objects are read from the keys the job row records -- the address
    # written at packaging time is the one read at download time -- through
    # streaming reads, so the payload is never held whole. A missing weights
    # object refuses loudly rather than downloading an empty archive: a
    # download that "succeeds" with nothing in it looks like the deliverable
    # and is not one -- the same failure shape as shipping a bare .safetensors.
    # A missing config alone still zips the weights, because a run that
    # produced none already reported that as an error event when it ended.
    weights = _open_member(artifact_key)
    if weights is None:
        raise HTTPException(
            409,
            {
                "code": "artifact_missing",
                "message": "This job records an artifact, but its stored "
                "object is gone. The job's event log says what happened "
                "to it.",
            },
        )

    members: list[tuple[str, Iterator[bytes]]] = [
        (storage.ADAPTER_WEIGHTS_NAME, weights)
    ]
    config_stream = _open_member(storage.artifact_config_key(artifact_key))
    if config_stream is not None:
        members.append((storage.ADAPTER_CONFIG_NAME, config_stream))

    return StreamingResponse(
        _zip_chunks(members),
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
