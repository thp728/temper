"""Temper control plane.

One FastAPI process. Upload a dataset, launch a job, watch it, take the artifact.

Deliberately not here: auth, billing, a separate gateway process, a message
broker. The brief sanctions cutting auth and billing; the rest are scaling
answers to a problem this does not have. What *is* here is the full job
lifecycle with every transition recorded, because explaining a run is the
product.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
import zipfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import BinaryIO

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from temper_control_plane import (
    admission,
    channel,
    chunks,
    config,
    datasets,
    db,
    jobs,
    models,
    orchestrator,
    quote,
    remote_datasets,
    seed_demo,
    storage,
    trainer_build,
)
from temper_control_plane.contracts_models import (
    AdmittedModel,
    AdvancedSurface,
    Calibration,
    DatasetAccepted,
    DatasetImportRequest,
    DatasetList,
    DatasetRecord,
    DatasetRenameRequest,
    DecisionOverride,
    EndpointCreated,
    EndpointPreview,
    EndpointRecord,
    EventPage,
    InferRequest,
    InferResponse,
    JobList,
    JobRecord,
    JobSpecPreview,
    ModelCatalog,
    Quote,
    QuoteRequest,
)
from temper_control_plane.correlation import (
    CORRELATION_HEADER,
    REQUEST_ID_HEADER,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
)
from temper_control_plane.logging import configure_logging, get_logger
from temper_control_plane.sentry import init_sentry
from temper_control_plane.storage import ObjectNotFound
from temper_core import (
    artifacts,
    calibration,
    catalog,
    delivery,
    feasibility,
    gpus,
    hyperparams,
    memory,
    overrides,
    surface,
)
from temper_core import manifest as provenance_manifest

# Configure structured JSON logging for every process that imports this
# module -- the control plane's request handler and the worker's
# orchestrator both import ``main`` transitively (``jobs.create`` etc),
# so configuring at import makes every entry point emit one JSON line per
# event, machine-readable, rather than two shapes. ``configure_logging``
# is idempotent, so calling it again in ``lifespan`` or ``worker.main``
# is harmless.
configure_logging()
logger = get_logger(__name__)

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
    init_sentry()
    db.init()
    storage.STORE.ensure_ready()
    # Spec 012's second front door, promoted: when the zero-cost tier boots
    # against an empty database, plant the sample dataset and one completed
    # run so the reviewer opens the stack to something to look at, not a
    # blank page. A no-op in the real tier and on any non-empty database
    # (seed_demo.py), so it can neither spend nor collide with a human's
    # history.
    seed_demo.maybe_seed()
    # Fail loudly at boot rather than four seconds into someone's first job.
    # The zero-cost tier (TEMPER_FAKE_PROVIDER) never needs credentials -- it
    # cannot reach the account by construction -- so the warning is only for
    # the real tier, where a missing key is a jobs-will-fail-at-provisioning
    # surprise rather than a demo detail.
    if not config.FAKE_PROVIDER and not config.provider_credentials_present():
        logger.warning(
            "missing provider credentials, jobs will fail at provisioning",
            reason="JL_API_KEY unset and no jl config file; datasets validate fine",
        )
    # Issue #51: orchestration now lives in the worker, not in this process.
    # A restart of the control plane no longer loses the thread driving a
    # job, because there is no thread in this process to lose. Jobs that
    # are still non-terminal after a restart are left for the worker to
    # claim (``SELECT ... FOR UPDATE SKIP LOCKED``) rather than being
    # failed as ``orphaned_by_restart`` here -- that marking was the
    # pre-worker behaviour where orphaning meant "the only thread that
    # could have finished this job died with this process". The worker now
    # watches spend and teardown (ADR-0057, ADR-0063) directly, while
    # serving endpoints' idle/max timers (ADR-0065) stay in this process
    # -- they are what stop a serving machine from billing forever, and a
    # restart that lost them would leave a billed GPU warm until someone
    # notices.
    # Endpoints that outlive their usefulness are the top complaint against
    # the commercial baseline -- the training is cheap and the forgotten warm
    # machine is the bill -- so stopping itself is the feature (ADR-0065).
    # An endpoint that survives a restart without a timer is an endpoint
    # that never stops, so re-arm timers for any still-running endpoints,
    # or stop those already past their deadline. The sweep thread then
    # keeps the expiry visible without anyone asking.
    try:
        from temper_control_plane import serving as serving_mod

        serving_mod.rearm_after_restart()
        serving_mod.start_sweep_thread()
    except Exception:  # noqa: S110
        pass
    try:
        yield
    finally:
        try:
            from temper_control_plane import serving as serving_mod2

            serving_mod2.stop_sweep_thread()
        except Exception:  # noqa: S110
            pass


app = FastAPI(
    title="Temper",
    description="Fine-tuning platform. Dataset in, adapter out.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# correlation middleware -- one identifier per request, on every log line
# ---------------------------------------------------------------------------


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    """Bind one correlation identifier for this request.

    The header is honoured when the caller set it (so a retry can carry the
    same story), otherwise a fresh ``req_`` identifier is generated. The
    identifier is bound to ``contextvars`` so every structured log line this
    request or its resulting job emits carries it (the worker re-hydrates
    from the job row), and it is echoed on the response headers and on
    user-visible errors so a report can be traced. Structured request
    start/finish lines are emitted so a failure spanning a request, a job and
    a machine can be reassembled from logs alone (spec 008).
    """
    start = time.time()
    # Honour a caller-provided identifier (either header spelling) so a
    # retry or an external caller can thread its own retry into the same
    # trace; generate when absent so every request has one even without a
    # header.
    incoming = request.headers.get(CORRELATION_HEADER) or request.headers.get(
        REQUEST_ID_HEADER
    )
    cid = (
        incoming.strip()
        if incoming and incoming.strip()
        else generate_correlation_id()
    )
    set_correlation_id(cid)
    # Bind into structlog's contextvars as well, so ``merge_contextvars``
    # sees it even when the logger was obtained before the middleware bound
    # it. The two bindings are the same value, the same identifier, read
    # from one definition.
    try:
        from structlog import contextvars as structlog_contextvars

        structlog_contextvars.bind_contextvars(correlation_id=cid)
    except Exception:  # noqa: S110
        pass
    logger.info(
        "request started",
        method=request.method,
        path=request.url.path,
        correlation_id=cid,
    )
    try:
        response = await call_next(request)
    except Exception as e:
        # An unhandled exception is the one case the per-route handlers did
        # not already surface with a correlation-bearing payload; report it
        # once, with the identifier, and surface it on the response as well
        # so a report can be traced. The sentry integration is inert when no
        # DSN is configured, so reporting here is present in shape even when
        # it does nothing locally.
        duration_ms = round((time.time() - start) * 1000, 1)
        logger.exception(
            "request failed: unhandled exception",
            method=request.method,
            path=request.url.path,
            correlation_id=cid,
            duration_ms=duration_ms,
            exc_info=e,
        )
        try:
            import sentry_sdk

            from temper_control_plane.sentry import (
                before_send as _scrub,  # noqa: F401
            )

            sentry_sdk.capture_exception(e)
        except Exception:  # noqa: S110
            pass
        # Surface the correlation on the error payload so a report can be
        # traced (acceptance: "The identifier is surfaced on user-visible
        # errors"). Raising an HTTPException here would be another handler;
        # returning a JSONResponse keeps the middleware as the single place
        # this surfacing lives.
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                },
                "correlation_id": cid,
            },
            headers={
                CORRELATION_HEADER: cid,
                REQUEST_ID_HEADER: cid,
            },
        )
    duration_ms = round((time.time() - start) * 1000, 1)
    # Structured completion line, machine-readable, carrying the same
    # correlation so a request, its job and its machine read as one story.
    logger.info(
        "request finished",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        correlation_id=cid,
        duration_ms=duration_ms,
    )
    response.headers[CORRELATION_HEADER] = cid
    response.headers[REQUEST_ID_HEADER] = cid
    # Also bind into structlog's contextvars for any logger that reads from
    # there; cleared on the next request's bind.
    try:
        from structlog import contextvars as structlog_contextvars

        structlog_contextvars.bind_contextvars(correlation_id=cid)
    except Exception:  # noqa: S110
        pass
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler_with_correlation(
    request: Request, exc: HTTPException
):
    """Surface the correlation identifier on every user-visible error.

    FastAPI's default handler returns ``{"detail": ...}``; this handler keeps
    that shape but ensures the correlation identifier is both in the response
    headers and, when the payload is a dict, inside the payload so a report
    can be traced without hunting headers. Structured log line is emitted
    at the same time, so the error's own log and the response share the
    same identifier.
    """
    cid = get_correlation_id() or generate_correlation_id()
    # Log the error as structured JSON, scrubbed, with the correlation -- a
    # training platform that logs user data has a problem no access control
    # fixes, so the logger's redaction still applies, and no prompt ever
    # reaches this line because the handler never logs the request body.
    logger.warning(
        "request error",
        method=request.method,
        path=request.url.path,
        status_code=exc.status_code,
        correlation_id=cid,
        code=getattr(exc.detail, "get", lambda *_: None)("code")
        if isinstance(exc.detail, dict)
        else None,
    )
    # Report to Sentry when a DSN is configured; inert otherwise (the same
    # integration point the logging module describes -- present in shape,
    # inert in effect when no credential is set locally).
    if exc.status_code >= 500:
        try:
            import sentry_sdk

            sentry_sdk.capture_exception(exc)
        except Exception:  # noqa: S110
            pass
    detail = exc.detail
    if isinstance(detail, dict):
        # Do not mutate the original detail in place (it may be reused), and
        # ensure the correlation is surfaced on user-visible errors so a report
        # can be traced. Redact any secret that somehow reached the payload --
        # no secret ever appears in a response (acceptance). The wrapper is
        # ``{"detail": payload}`` so a client that expects
        # ``r.json()["detail"]["code"]`` keeps working, and ``correlation_id``
        # is surfaced both inside the payload and at the top level/header so
        # a report can be traced without hunting.
        payload = dict(detail)
        # Scrub secrets from the payload itself before it leaves the process:
        # a secret that reached an error message (e.g. a DSN in an exception
        # string) must not be surfaced to the user.
        try:
            from temper_control_plane.logging import (
                _SECRET_KEY_RE,
                _SECRET_VALUE_RE,
                _known_secrets,
            )

            redacted: dict = {}
            for k, v in payload.items():
                if isinstance(k, str) and _SECRET_KEY_RE.search(k):
                    redacted[k] = "[REDACTED]"
                elif isinstance(v, str) and _SECRET_VALUE_RE.search(v):
                    redacted[k] = _SECRET_VALUE_RE.sub("[REDACTED]", v)
                elif isinstance(v, str) and any(
                    s and s in v for s in _known_secrets
                ):
                    redacted[k] = "[REDACTED]"
                else:
                    redacted[k] = v
            payload = redacted
        except Exception:  # noqa: S110
            pass
        payload.setdefault("correlation_id", cid)
        content: dict = {"detail": payload, "correlation_id": cid}
    else:
        payload = {"detail": detail, "correlation_id": cid}
        content = payload
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers={
            CORRELATION_HEADER: cid,
            REQUEST_ID_HEADER: cid,
            **(exc.headers or {}),
        },
    )


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@app.get("/v1/models", tags=["catalog"], response_model=ModelCatalog)
def list_models():
    """The base-model choices: the curated catalog plus any admitted models.

    The catalog is an allow-list and a promise about what has been tested, not
    a limitation (issue #58): a model outside it becomes usable once it has
    passed a compatibility probe, and `admitted` carries each probed model
    with its result -- verdict and findings -- shown, not merely enforced. The
    curated entries stay the recommended, tested path.

    Each catalog entry's `peak_memory` is computed fresh through the `models`
    seam (spec 005) rather than read from a stored figure -- see
    `_catalog_entry`.
    """
    return {
        "models": [_catalog_entry(m) for m in catalog.CATALOG.values()],
        "admitted": admission.listing(),
        "default": catalog.DEFAULT_MODEL,
    }


class ModelProbeRequest(BaseModel):
    repo: str
    revision: str


@app.post(
    "/v1/models/probe",
    tags=["catalog"],
    status_code=201,
    response_model=AdmittedModel,
)
def probe_model(req: ModelProbeRequest):
    """Admit a model from outside the catalog by probing it (issue #58).

    Resolves the pinned reference through the same `models` seam the
    predictor reads, runs the compatibility probe over the resolved facts, and
    persists the result -- verdict and findings -- so the user is shown what
    they are taking on before a job can be created against it. Re-probing the
    same pinned reference is idempotent: the existing record is returned, not
    a second one.

    A reference that is not a pinned revision is refused up front with
    `unpinned_revision`. A pinned reference that fails the probe is still
    admitted -- as a blocked record, persisted and shown -- and the launch
    path refuses it with its blocking findings rather than launching into a
    model that cannot be formatted.
    """
    try:
        record = admission.probe_and_admit(req.repo, req.revision)
    except admission.AdmissionError as e:
        raise HTTPException(400, e.to_payload()) from e
    return record


@app.get("/v1/surface", tags=["surface"], response_model=AdvancedSurface)
def get_advanced_surface():
    """The generated advanced surface, published for the interface to render.

    Issue #80. The exposed set, each field's tier with its reason and failure
    mode, and the runtime-only validators are all derived from the pinned
    trainer's schema and its tier data (issue #33) -- the interface generates
    its advanced panel from this rather than hand-listing the vocabulary, so
    the panel cannot drift from what the pre-launch gate refuses. The document
    is a pure function of two checked-in data files, so it is deterministic
    for a given pinned image.
    """
    return surface.surface_document()


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


@app.post(
    "/v1/datasets/import",
    tags=["datasets"],
    status_code=202,
    response_model=DatasetAccepted,
)
def import_dataset(req: DatasetImportRequest):
    """Import a dataset by reference (issue #45): a public repository,
    optionally a configuration and a split, and Temper fetches it for you.

    The reference is resolved before anything is stored, and a reference that
    cannot be fetched -- repository missing, configuration unnamed, split
    absent, split empty -- is refused with its reason as a coded 400, never
    left to fail mid-fetch. That resolution is the only network call this
    handler waits on: the rows themselves are fetched from a third party that
    can be arbitrarily slow (issue: "dataset import from HF"), so streaming
    them into storage, enforcing the size ceiling, and validating all run in
    the background (status `importing`, then `validating`) -- the *same*
    background path an upload uses past its own fetch, same schema
    detection, same line-numbered errors, same thinking-mode detection.
    Nothing gets a shortcut for arriving over a network, and an imported
    dataset that fails to fetch, or fails validation, is stored with its
    report like any other.

    This answers 202 with the dataset's id once the reference resolves;
    progress and the report land on `GET /v1/datasets/{id}`. The handler is a
    worker-pool `def` because `resolve()` is synchronous and IO-bound
    (ADR-0006).
    """
    try:
        ds_id, filename, status = datasets.import_dataset(
            req.repo, req.config, req.split
        )
    except remote_datasets.RemoteDatasetError as e:
        raise HTTPException(400, e.to_payload()) from e
    return {"id": ds_id, "filename": filename, "status": status}


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


@app.patch(
    "/v1/datasets/{dataset_id}",
    tags=["datasets"],
    response_model=DatasetRecord,
)
def rename_dataset(dataset_id: str, req: DatasetRenameRequest):
    """Rename a dataset. The only field a dataset can be updated with --
    the stored object never moves, this changes what it is called.

    Refused (404) if it does not exist, and refused (400,
    `unsupported_extension`) the same way an upload's own name is: the
    product only ever ingests JSONL.
    """
    return datasets.rename_dataset(dataset_id, req.filename)


@app.delete(
    "/v1/datasets/{dataset_id}",
    tags=["datasets"],
    status_code=204,
)
def delete_dataset(dataset_id: str):
    """Delete a dataset: the stored object, then the row.

    Refused (404) if it does not exist, and refused (409, `dataset_in_use`)
    if any job was launched against it -- a completed job's record must keep
    being able to say what it trained on, so the data it points at does not
    disappear out from under it. There is no undo: the object is gone, not
    archived.
    """
    datasets.delete_dataset(dataset_id)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


class JobRequest(BaseModel):
    dataset_id: str
    base_model: str = Field(default=catalog.DEFAULT_MODEL)
    hyperparameters: dict = Field(default_factory=dict)
    overrides: list[DecisionOverride] = Field(default_factory=list)
    # Issue #74: the delivery formats this launch asks for (beyond the
    # canonical artifact): "merged" for a single-file serving model, "quantised"
    # for a local-inference format. Validated against the one delivery
    # vocabulary at creation.
    delivery: list[str] = Field(default_factory=list)


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
    with the rest of the spec. Time and cost never refuse: if the provider is
    unreachable or nothing is free, the job still launches -- an estimate
    warns, it does not refuse (spec 005). Memory is the exception that issue
    #54 makes structural: a configuration predicted not to fit any available
    card is refused at creation with `configuration_does_not_fit` and the
    peak-vs-capacity arithmetic, even when nothing was overridden, because an
    out-of-memory failure on a machine the user is paying for is not an
    estimate. The advanced-surface gate itself lives in `jobs.create`, the
    one shared creation path, so every entry point refuses identically (issue
    #80).
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
            delivery_request=req.delivery,
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
    # Issue #74: the delivery formats a launch may ask for, read from the one
    # domain vocabulary -- the launch screen offers each by what it is for,
    # exactly as the finished page does, so the two cannot drift about what a
    # format is.
    delivery_formats = [
        {
            "id": fmt.id,
            "what_for": fmt.what_for,
        }
        for fmt in delivery.FORMATS.values()
    ]
    # Whether a real launch could pull the image it would run. The
    # orchestrator refuses with `image_not_published` when the digest
    # contract carries none; reporting it here lets the launch screen say so
    # while the user can still act on it, rather than after they have
    # committed. The simulated provider pulls no image, so it is never
    # blocked by this.
    image_published = (
        config.FAKE_PROVIDER or trainer_build.published_reference() is not None
    )
    # And whether the machine could upload what it produces. The filesystem
    # backend's grants are tokens only this process redeems, so a real run on
    # it is billed in full and delivers nothing; the orchestrator refuses
    # with `artifact_undeliverable`, and saying so here means the launch
    # screen can refuse before the money rather than after it.
    artifact_deliverable = (
        config.FAKE_PROVIDER or storage.STORE.grants_are_remotely_redeemable
    )
    return {
        "dataset": ds,
        "hyperparameters": hyperparams.effective({}),
        "warning": warn,
        "delivery_formats": delivery_formats,
        "trainer_image_published": image_published,
        "artifact_deliverable": artifact_deliverable,
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
    m = admission.get(base_model)
    if m is None:
        raise HTTPException(
            400,
            {
                "code": "unknown_model",
                "message": f"'{base_model}' is not in the catalog and has not "
                "been admitted by a probe.",
                "available": admission.available_ids(),
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

    `req.hyperparameters` (issue #80) are the advanced-surface settings the
    user has overridden. They are run through the same pre-launch gate as a
    launch (`temper_core.surface.validate_overrides`), so a key unknown to the
    trainer is echoed back and a known-but-unsupported key refused with its
    reason here too, and they re-price the plan -- a `micro_batch_size` that
    no longer fits is refused before anything is provisioned.

    Returns null (never a refusal) only when the estimate itself cannot be
    priced -- provider unreachable, model unresolvable -- because an estimate
    warns, it does not block (spec 005).
    """
    refusals = surface.validate_overrides(req.hyperparameters)
    if refusals:
        raise HTTPException(400, refusals[0])
    ds = jobs.usable_dataset(req.dataset_id)
    m = admission.get(req.base_model)
    if m is None:
        raise HTTPException(
            400,
            {
                "code": "unknown_model",
                "message": f"'{req.base_model}' is not in the catalog and has "
                "not been admitted by a probe.",
                "available": admission.available_ids(),
            },
        )
    try:
        return _quote_for(
            ds,
            m,
            req.hyperparameters,
            overrides_list=_override_list(req.overrides),
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
    working one; reporting `cancelled` here would claim a teardown that has
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


@app.post(
    "/v1/jobs/{job_id}/retry",
    tags=["jobs"],
    response_model=JobRecord,
    status_code=201,
)
def retry_job(job_id: str):
    """Create a single retry at half the learning rate, offered as a choice.

    Issue #36: a diverging run usually means the data or the rate is wrong, and
    repeating it is rarely the answer. Repeating it automatically would be the
    easy thing to test; offering a single reduced-rate retry as a choice is the
    honest one. This endpoint is the choice: a diverged job (failed with
    ``training_diverged``) can be retried once at half its learning rate, and
    the new job carries ``retry_from`` so the history can name what came from
    what. A job that already has a retry child, or that did not diverge, is
    refused with a stable code rather than silently creating a second retry.
    """
    new_job_id = jobs.retry_diverged_job(job_id)
    job = db.get_job(new_job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    return job


@app.get("/v1/jobs/{job_id}/events", tags=["jobs"], response_model=EventPage)
def get_events(job_id: str, after: int = 0, limit: int = 500):
    """Durable event log. `after` is the last event id the client holds.

    Polling against a monotonic id rather than streaming: a reconnecting client
    catches up from the database instead of losing whatever happened while it
    was away.

    The page also carries the job's progress (issue #49): the current per-phase
    snapshot and the retained raw lines that were promoted into it. Progress
    supersedes per phase, so the snapshot is small by construction; the
    retained lines are the collapsed detail that keeps nothing discarded.

    `total` is the job's event count regardless of the paging window
    (issue #56): where a limit still applies the page says what it is showing
    and of how many, and a silent truncation becomes a stated one. `limit`
    caps the page at 500 (the stored cap that #56 keeps), so a job that
    genuinely produces very many events is paginated rather than silently cut,
    and the full record remains reachable by paging with `after`.
    """
    if limit < 1 or limit > 500:
        raise HTTPException(
            400,
            {
                "code": "invalid_limit",
                "message": "limit must be between 1 and 500",
            },
        )
    if not db.get_job(job_id):
        raise HTTPException(404, "No such job.")
    events = db.get_events(job_id, after_id=after, limit=limit)
    total = db.count_events(job_id)
    return {
        "events": events,
        "last_id": events[-1]["id"] if events else after,
        "total": total,
        "progress": db.get_progress(job_id),
        "output": db.get_output(job_id),
    }


# How long the live stream may stay silent before sending a keep-alive
# comment, and how long it waits on the job's channel for a publish before
# looping again. Both are tuning, not contract: the stream's shape -- events
# pushed as they are recorded, ending only at a terminal state -- is what the
# interface consumes. The channel is the wake; the wait timeout only bounds
# how quickly an idle stream notices a disconnect or sends its next
# heartbeat, and an idle stream makes no store reads (ADR-0070).
STREAM_WAIT_S = 0.25
STREAM_HEARTBEAT_S = 15.0


def _stream_cursor(request: Request, after: int) -> int:
    """Where a client's history ends: the `Last-Event-ID` a reconnecting
    EventSource replays, else the `after` it asked for on first connection.

    The two are the same idea in two moments. On first connection the browser
    knows only what the server-rendered page already showed, so it passes that
    as `after`; once the stream is open the browser replays the last event id
    it actually received, so nothing is re-sent and nothing is skipped.
    """
    header = request.headers.get("last-event-id")
    if header and header.strip():
        try:
            return int(header.strip())
        except ValueError:
            # A header we cannot parse is not a cursor: fall back to the query
            # parameter rather than inventing one the client never held.
            pass
    return after


def _sse_event(e: dict) -> str:
    """One job event as a server-sent event.

    `id` is what EventSource replays on reconnect; the payload is the same
    `JobEvent` the polling endpoint publishes, so the interface parses one
    shape wherever it reads a job's history.
    """
    payload = {
        "id": e["id"],
        "job_id": e["job_id"],
        "ts": e["ts"],
        "kind": e["kind"],
        "message": e.get("message"),
        "data": e.get("data"),
    }
    return f"id: {e['id']}\nevent: job\ndata: {json.dumps(payload)}\n\n"


def _sse_progress(row: dict) -> str:
    """One phase's progress snapshot as a server-sent event.

    Not replayable by `Last-Event-ID`: the page renders the current snapshot
    from the durable events endpoint on load, and the stream re-sends each
    snapshot only when it changes, so a client that reconnects never misses
    the latest per phase.
    """
    payload = {
        "phase": row["phase"],
        "done": row["done"],
        "total": row["total"],
        "rate": row["rate"],
        "eta_s": row["eta_s"],
        "ts": row["ts"],
        "message": row.get("message"),
    }
    return f"event: progress\ndata: {json.dumps(payload)}\n\n"


def _sse_output(row: dict) -> str:
    """One retained raw line as a server-sent event.

    These are the lines that were promoted into progress (issue #49) and so
    are not events; they ride the same connection so the collapsed detail on
    the running view stays live. The client deduplicates by id, because a
    reconnecting stream re-sends the retained record from the start.
    """
    payload = {"id": row["id"], "phase": row["phase"], "line": row["line"]}
    return f"event: output\ndata: {json.dumps(payload)}\n\n"


async def _job_event_stream(
    request: Request, job_id: str, cursor: int
) -> AsyncIterator[str]:
    """The job's events, oldest-first, as they are recorded, with the live
    per-phase progress snapshot riding the same connection (issue #49).

    The durable store is the source, never this connection: a visitor
    returning to the page or a stream that dropped is caught up from the
    database by the last identifier it saw, which is the same answer the
    polling endpoint gives -- live streaming and durable history are one log,
    not two. The stream *reads history from the store first, then follows the
    channel* (spec 008, ADR-0070): it subscribes to the job's channel before
    the first read (so nothing can be persisted-and-notified in a gap the
    read would miss), and every wake after that is a publish on the channel,
    which makes the channel the delivery mechanism rather than an ornament on
    a poll -- when nothing is written, the stream waits and makes no store
    reads at all. Each wake re-reads the store from its cursor and emits
    whatever is new: the store is the truth, the channel only says that new
    data exists. A channel that dropped while no connection was listening
    loses nothing, because a re-established LISTEN triggers the same store
    re-read (the replay a watcher's own reconnection performs).

    The stream ends only when the job is terminal -- every event, including
    the terminal transition itself, is delivered before it closes, which is
    the interface's cue that the record has stopped changing.

    Progress and the retained output lines ride the same connection as the
    events (no second channel; issue #49), and both publish to the same
    channel from `db.py`, so a progress or output write wakes the stream to
    re-read them like an event would. Progress supersedes rather than
    accumulates, so the snapshot is the whole of it -- the running view
    replaces its per-phase figures, never appends to them -- and the output
    lines are deduplicated by id on the client, because a reconnecting
    stream re-sends the retained record from the start.

    DB reads go through `asyncio.to_thread`: they are quick local reads, but a
    blocking read on the event loop would stall every other request for as long
    as a connection stays open, which is exactly the mistake the upload path
    records in its own docstring. The channel subscription runs on its own
    thread for the same reason (ADR-0070: Windows' event loop cannot run
    psycopg's async connection).
    """
    last_send = time.monotonic()
    last_progress: list[dict] | None = None
    output_cursor = 0
    first = True
    async with channel.EventSubscription(job_id) as sub:
        while True:
            if await request.is_disconnected():
                return
            # The first wake is the catch-up read itself: history from the
            # store. Every wake after it is a publish on the channel (or a
            # re-established LISTEN after a drop); a wait that times out means
            # nothing was written, so nothing is read and the loop only sends
            # a heartbeat if one is due.
            woke = first or await sub.wait(STREAM_WAIT_S)
            first = False
            if woke or sub.take_reconnected():
                job = await asyncio.to_thread(db.get_job, job_id)
                if job is None:
                    # The row cannot be absent (the endpoint refused a
                    # missing job before streaming), but a re-read that comes
                    # back empty ends the stream rather than looping on a
                    # ghost.
                    return
                for e in await asyncio.to_thread(
                    db.get_events, job_id, after_id=cursor
                ):
                    cursor = e["id"]
                    yield _sse_event(e)
                progress_rows = await asyncio.to_thread(
                    db.get_progress, job_id
                )
                if progress_rows != last_progress:
                    last_progress = progress_rows
                    for row in progress_rows:
                        yield _sse_progress(row)
                for row in await asyncio.to_thread(db.get_output, job_id):
                    if row["id"] > output_cursor:
                        output_cursor = row["id"]
                        yield _sse_output(row)
                if job["status"] in db.TERMINAL_STATES:
                    # The explicit end marker is the hand-back the interface
                    # waits for. Relying on the connection merely closing
                    # would not be enough: a browser's EventSource does not
                    # report a server-initiated close as CLOSED -- it goes
                    # CONNECTING and reconnects, and a page that only reacted
                    # to CLOSED would loop forever against a terminal job.
                    # The marker arrives right after the terminal transition,
                    # and the client reloads into the finished record on it.
                    yield "event: end\n\n"
                    return
            now = time.monotonic()
            if now - last_send >= STREAM_HEARTBEAT_S:
                yield ": heartbeat\n\n"
                last_send = now


@app.get("/v1/jobs/{job_id}/stream", tags=["jobs"])
async def stream_job(request: Request, job_id: str, after: int = 0):
    """Server-pushed events for a running job, replacing the watch page's poll.

    The interface's live channel (Spec 007): one connection, events delivered
    as they are recorded, no page refresh. `after` is where the client's
    history already ends -- the page renders the durable log first and streams
    from its last id, so leaving and returning shows continuous history rather
    than starting at the moment of return. A dropped connection reconnects on
    its own: EventSource replays `Last-Event-ID`, this endpoint honours it, and
    the stream ends only when the job reaches a terminal state.

    The interface consumes this with a browser `EventSource`; the generated
    fetch client cannot, which is why this endpoint (like the adapter
    download) is transport the contract documents rather than a function the
    client calls.
    """
    if not await asyncio.to_thread(db.get_job, job_id):
        raise HTTPException(404, "No such job.")
    return StreamingResponse(
        _job_event_stream(request, job_id, _stream_cursor(request, after)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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

    Built per request rather than cached -- a stale zip beside fresh artifact
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


@app.get("/v1/jobs/{job_id}/artifact", tags=["jobs"])
def download_artifact(job_id: str, format: str | None = None):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "No such job.")

    # Issue #74: the default download is the canonical artifact; a `format`
    # query selects one produced delivery format (merged, quantised). The
    # members resolve through the job row's own record in both cases -- the
    # address written at packaging time is the one read at download time -- so
    # this path serves any format without special-casing the bytes.
    delivery_format = None
    if format is not None:
        try:
            delivery.require_known(format)
        except delivery.UnknownDeliveryFormat:
            raise HTTPException(
                404,
                {
                    "code": "unknown_delivery_format",
                    "message": (
                        f"'{format}' is not a delivery format this platform "
                        "offers. Download the artifact without a format for "
                        "the canonical one, or name merged or quantised."
                    ),
                },
            ) from None
        delivery_format = format
        members = db.delivery_format_members(job, format)
        if not members:
            raise HTTPException(
                404,
                {
                    "code": "format_unavailable",
                    "message": (
                        f"This job did not produce a verified '{format}' "
                        "format (it was not requested, or its verification "
                        "failed). The job's record says which formats it "
                        "produced."
                    ),
                    "format": format,
                },
            )
    else:
        members = db.canonical_artifact_members(job)
        if not members:
            raise HTTPException(
                409,
                {
                    "code": "no_artifact",
                    "message": (
                        f"Job is '{job['status']}'; no artifact is available "
                        "yet."
                    ),
                },
            )

    # Zipped, because an artifact is a set of stored objects -- which objects,
    # and how many, is the artifact record's business, not this endpoint's: the
    # members were recorded at packaging time, and this path serves any kind
    # (adapter, fully trained model, a delivery format, ...) by streaming
    # whatever the record names, without special-casing. A row written before
    # the record existed is described by the canonical adapter pair, so a
    # legacy download behaves exactly as it always did.
    kind = artifacts.kind_for(job.get("method"))
    record = job.get("artifact_record")
    declared_bytes = record.get("bytes") if record else None

    # All members are read from the keys the job row records -- the address
    # written at packaging time is the one read at download time -- through
    # streaming reads, so the payload is never held whole. The first member is
    # required: a missing weights object refuses loudly rather than
    # downloading an empty archive, because a download that "succeeds" with
    # nothing in it looks like the deliverable and is not one. A later member
    # missing alone still serves the rest, because a run that produced none
    # already reported that as an error event when it ended.
    primary_name, primary_key = members[0]
    primary = _open_member(primary_key)
    if primary is None:
        raise HTTPException(
            409,
            {
                "code": "artifact_missing",
                "message": "This job records an artifact, but its stored "
                "object is gone. The job's event log says what happened "
                "to it.",
            },
        )

    member_streams: list[tuple[str, Iterator[bytes]]] = [
        (primary_name, primary)
    ]
    for name, key in members[1:]:
        stream = _open_member(key)
        if stream is not None:
            member_streams.append((name, stream))

    # The provenance manifest (issue #70): generated from the job record,
    # never hand-written, it records base model and pinned revision, dataset
    # fingerprint with counts, the full configuration including overrides,
    # the evaluation summary, the checkpoint the result came from, and the
    # licence obligations that propagate. It ships with the artifact rather
    # than beside it, and a missing required field fails generation rather
    # than producing a placeholder. Human-readable Markdown travels alongside
    # the machine-readable JSON.
    dataset_row = None
    try:
        dataset_row = db.get_dataset(job["dataset_id"])
    except Exception:  # noqa: S110
        dataset_row = None
    try:
        provenance = provenance_manifest.generate(
            job,
            dataset_row,
            generated_at=time.time(),
            delivery_format=delivery_format,
        )
        # The streamed members' names are the ground truth for what the zip
        # contains; the provenance's artifact.members is forced to match them
        # so the manifest cannot drift from the bytes it describes.
        streamed_names = [name for name, _ in member_streams]
        provenance["artifact"]["members"] = streamed_names
        provenance["members"] = streamed_names
        # Keep the declared bytes from the verification in sync with the
        # provenance's artifact bytes (the recorded one wins, but the zip's
        # members are the source of truth for names).
        if declared_bytes is not None:
            provenance["artifact"]["bytes"] = declared_bytes
            provenance["bytes"] = declared_bytes
        manifest_bytes = provenance_manifest.to_pretty_json(provenance).encode(
            "utf-8"
        )
        provenance_text = provenance_manifest.render_text(provenance)
        provenance_text_bytes = provenance_text.encode("utf-8")
    except provenance_manifest.MissingField as e:
        # A missing required field is a defect, not a placeholder: the
        # generation fails loudly. For artifact downloads this surfaces as a
        # coded 500 so the failure can be diagnosed without guessing which
        # field was absent. Legacy rows written before provenance existed will
        # hit this path; they are still downloadable via the minimal manifest
        # that issue #32 introduced, so the platform does not retroactively
        # break an artifact that predates the provenance requirement.
        # The minimal manifest is kept as a fallback only for those legacy
        # rows -- a provenance-capable run never reaches this branch.
        if delivery_format is not None:
            try:
                kind = delivery.kind_for(delivery_format)
            except delivery.UnknownDeliveryFormat:  # pragma: no cover
                kind = artifacts.kind_for(job.get("method"))
        else:
            kind = artifacts.kind_for(job.get("method"))
        fell_back = {
            "kind": kind,
            "base_model": job.get("base_model"),
            "base_revision": job.get("base_revision"),
            "members": [name for name, _ in member_streams],
            "bytes": declared_bytes,
            "loading": artifacts.loading_instructions(kind),
            "provenance_error": str(e),
            "provenance_missing_field": e.field,
        }
        manifest_bytes = json.dumps(fell_back, indent=2).encode("utf-8")
        provenance_text = (
            "# Provenance Manifest (incomplete)\n\n"
            f"This artifact's provenance could not be generated: {e}\n\n"
            f"Missing field: `{e.field}`\n\n"
            "The minimal manifest below is the legacy record (issue #32) "
            "so the artifact remains downloadable, but the full provenance "
            "required by issue #70 is not available for this run.\n\n"
            "```json\n" + json.dumps(fell_back, indent=2) + "\n```\n"
        )
        provenance_text_bytes = provenance_text.encode("utf-8")
    member_streams.append((artifacts.MANIFEST_NAME, iter((manifest_bytes,))))
    member_streams.append(("PROVENANCE.md", iter((provenance_text_bytes,))))

    return StreamingResponse(
        _zip_chunks(member_streams),
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{job_id}-{delivery_format or "artifact"}.zip"'
            )
        },
    )


@app.get("/v1/jobs/{job_id}/checkpoints/{step}", tags=["jobs"])
def download_checkpoint(job_id: str, step: int):
    """Download one retained checkpoint by its step.

    Issue #62: the best checkpoint is chosen and recorded on the run, but every
    other retained checkpoint stays downloadable -- a user is never locked out
    of their own run's history. The step, not the slot, is what the user sees
    and addresses; the storage key is the job row's own record, read here and
    never published to the browser.

    A checkpoint that is not retained (failed, superseded, or never verified)
    refuses with a stable code rather than a broken download: the bytes are
    not there, and a download that looks like it succeeded is worse than one
    that names its absence.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "No such job.")
    record = next(
        (
            c
            for c in job.get("checkpoints", [])
            if isinstance(c, dict) and c.get("step") == step
        ),
        None,
    )
    if (
        not record
        or record.get("verified") is not True
        or not record.get("key")
    ):
        raise HTTPException(
            404,
            {
                "code": "checkpoint_unavailable",
                "message": f"Checkpoint at step {step} is not retained for "
                "download (it was never verified, or retention has since "
                "superseded it).",
                "step": step,
            },
        )
    stream = _open_member(record["key"])
    if stream is None:
        raise HTTPException(
            409,
            {
                "code": "checkpoint_missing",
                "message": f"Checkpoint at step {step} is recorded but its "
                "stored object is gone. The job's event log says what "
                "happened to it.",
                "step": step,
            },
        )
    return StreamingResponse(
        stream,
        media_type="application/x-tar",
        headers={
            "Content-Disposition": (
                f'attachment; filename="checkpoint-{step}.tar"'
            )
        },
    )


# ---------------------------------------------------------------------------
# serving endpoints (issue #78): temporary authenticated endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/v1/jobs/{job_id}/endpoint/preview",
    tags=["jobs"],
    response_model=EndpointPreview,
)
def endpoint_preview(job_id: str):
    """What starting an endpoint would cost and when it would stop, before it starts.

    The hourly cost is the job's frozen price and the stop times are now +
    idle and now + max, both domain constants (temper_core.serving). The
    preview is what the interface shows before the user confirms the start,
    so there is no surprise about the bill or the lifetime (spec 011: "the
    hourly cost and the stop time are shown before it starts").
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(
            404, {"code": "job_not_found", "message": "No such job."}
        )
    if job.get("status") != "complete":
        raise HTTPException(
            409,
            {
                "code": "job_not_complete",
                "message": f"Job is '{job.get('status')}'; only a complete job can be served.",
            },
        )
    from temper_control_plane import serving as serving_mod

    return serving_mod.preview_for_job(job)


@app.post(
    "/v1/jobs/{job_id}/endpoint",
    tags=["jobs"],
    response_model=EndpointCreated,
    status_code=201,
)
def create_endpoint(job_id: str):
    """Start a temporary authenticated endpoint for a finished job.

    The endpoint requires a key (returned once, stored hashed), carries its
    own expiry from the moment it starts, extends on use, and stops itself
    via a timer -- the forgotten warm machine is the loudest complaint
    against the commercial baseline, so stopping itself is the feature.
    Reachability from outside is verified before the key is handed out: the
    platform's firewall does not filter published container ports the way it
    appears to, which is why the trainer publishes nothing and why anything
    that does publish is checked from outside (spec 011). The verification
    originates on the control plane (outside the machine), never on the
    machine itself, so a `curl localhost` on the machine cannot pass it.
    """
    from temper_control_plane import serving as serving_mod
    from temper_core.errors import OrchestratorError

    try:
        return serving_mod.start_endpoint(job_id)
    except OrchestratorError as e:
        status = (
            409
            if e.code
            in (
                "endpoint_not_complete",
                "endpoint_already_running",
                "endpoint_provision_failed",
                "endpoint_reachable",
                # The machine was provisioned and then destroyed again
                # because its model never loaded, so no key exists and the
                # start is refused -- the same shape as the two above it.
                "endpoint_model_not_ready",
            )
            else 400
        )
        if e.code == "job_not_found":
            raise HTTPException(
                404, {"code": "job_not_found", "message": "No such job."}
            ) from e
        raise HTTPException(status, {"code": e.code, "message": str(e)}) from e


@app.get(
    "/v1/jobs/{job_id}/endpoint", tags=["jobs"], response_model=EndpointRecord
)
def get_endpoint(job_id: str):
    """The job's active endpoint, if any. The key hash is never returned."""
    if not db.get_job(job_id):
        raise HTTPException(
            404, {"code": "job_not_found", "message": "No such job."}
        )
    from temper_control_plane import serving as serving_mod

    ep = serving_mod.get_endpoint(job_id)
    if ep is None:
        raise HTTPException(
            404,
            {
                "code": "endpoint_not_found",
                "message": f"No running endpoint for job {job_id}.",
            },
        )
    # Enrich with the domain constants so the interface can show the idle
    # and max windows without retyping them (ADR-0010).
    from temper_core import serving as core_serving

    ep["idle_timeout_s"] = float(core_serving.ENDPOINT_IDLE_TIMEOUT_S)
    ep["max_lifetime_s"] = float(core_serving.ENDPOINT_MAX_LIFETIME_S)
    return ep


@app.delete(
    "/v1/jobs/{job_id}/endpoint", tags=["jobs"], response_model=EndpointRecord
)
def delete_endpoint(job_id: str):
    """Stop the job's active endpoint immediately, via confirmed teardown."""
    if not db.get_job(job_id):
        raise HTTPException(
            404, {"code": "job_not_found", "message": "No such job."}
        )
    from temper_control_plane import serving as serving_mod
    from temper_core.errors import OrchestratorError

    try:
        ep = serving_mod.stop_endpoint(job_id)
        from temper_core import serving as core_serving

        ep["idle_timeout_s"] = float(core_serving.ENDPOINT_IDLE_TIMEOUT_S)
        ep["max_lifetime_s"] = float(core_serving.ENDPOINT_MAX_LIFETIME_S)
        return ep
    except OrchestratorError as e:
        if e.code == "endpoint_not_found":
            raise HTTPException(
                404, {"code": e.code, "message": str(e)}
            ) from e
        raise HTTPException(409, {"code": e.code, "message": str(e)}) from e


@app.post(
    "/v1/jobs/{job_id}/endpoint/infer",
    tags=["jobs"],
    response_model=InferResponse,
)
def infer_endpoint(
    job_id: str,
    body: InferRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
):
    """Answer a prompt via the job's active endpoint. Requires the endpoint's key.

    The key is verified against the stored hash (constant-time), the expiry
    is checked and extended on success (capped by the max), and a completion
    is returned. A busy endpoint that is kept alive by traffic still dies at
    the max, because the extension is capped. The generation is canned in the
    simulated tier; on real hardware it reaches the machine and runs the
    model there.
    """
    if not db.get_job(job_id):
        raise HTTPException(
            404, {"code": "job_not_found", "message": "No such job."}
        )
    # Accept either X-API-Key or Authorization: Bearer <key>
    key = x_api_key
    if not key and authorization:
        if authorization.lower().startswith("bearer "):
            key = authorization[7:].strip()
        else:
            key = authorization.strip()
    if not key:
        raise HTTPException(
            401,
            {
                "code": "endpoint_unauthorized",
                "message": (
                    "This endpoint requires a key. Pass it as X-API-Key or "
                    "Authorization: Bearer <key>."
                ),
            },
        )
    from temper_control_plane import serving as serving_mod
    from temper_core.errors import OrchestratorError

    try:
        return serving_mod.infer(job_id, key, body.prompt)
    except OrchestratorError as e:
        if e.code == "endpoint_not_found":
            raise HTTPException(
                404, {"code": e.code, "message": str(e)}
            ) from e
        if e.code == "endpoint_unauthorized":
            raise HTTPException(
                401, {"code": e.code, "message": str(e)}
            ) from e
        if e.code == "endpoint_expired":
            raise HTTPException(
                410, {"code": e.code, "message": str(e)}
            ) from e
        # The request was well formed and authorised; the machine behind it
        # could not answer. 400 would tell the caller to change their prompt,
        # which would not help.
        if e.code in (
            "endpoint_generation_failed",
            "endpoint_machine_unreachable",
        ):
            raise HTTPException(
                502, {"code": e.code, "message": str(e)}
            ) from e
        raise HTTPException(400, {"code": e.code, "message": str(e)}) from e


def _probe_dependency(report: dict[str, dict], name: str, probe) -> None:
    """Probe one dependency into `report`, naming it and its failure kind.

    The exception's class name (NotADirectoryError, OperationalError) is what
    a reader needs to tell one failure from another; the raw message can carry
    absolute paths and connection details, which a health endpoint that is
    reachable by anyone who can reach the product should not volunteer.
    """
    try:
        probe()
        report[name] = {"ok": True}
    except Exception as e:  # noqa: S110
        report[name] = {"ok": False, "error": type(e).__name__}


def _dependency_report() -> dict[str, dict]:
    """Each dependency's readiness, reported separately (issue #29).

    One endpoint returning a single ok is the failure the criterion names: a
    broken database must be distinguishable from a broken object store, and
    either from a broken application. So each dependency is probed on its own
    and reported with its own ok/error, and the endpoint keeps answering
    (with a non-200 status) while the process is alive -- a broken
    application is exactly what stops answering at all.
    """
    report: dict[str, dict] = {}
    _probe_dependency(report, "database", db.ping)
    _probe_dependency(report, "storage", storage.STORE.ensure_ready)
    return report


@app.get("/health", tags=["ops"])
def health():
    # Which provider implementation a launch would use. The browser journeys
    # boot this process with TEMPER_FAKE_PROVIDER and refuse to drive a
    # launch unless this field confirms the switch took effect -- a launch
    # that reaches for the billing account from a test is the one mistake
    # this codebase refuses to make cheap.
    dependencies = _dependency_report()
    ok = all(d["ok"] for d in dependencies.values())
    payload = {
        "ok": ok,
        "provider": "fake" if config.FAKE_PROVIDER else "real",
        "dependencies": dependencies,
    }
    if not ok:
        # A broken dependency is not a dead process: the app answers, and
        # says which dependency failed and why. 503 is what a compose
        # healthcheck (or a load balancer) treats as not-ready, which is what
        # a readiness endpoint is for.
        raise HTTPException(503, payload)
    return payload
