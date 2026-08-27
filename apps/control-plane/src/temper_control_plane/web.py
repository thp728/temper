"""Server-rendered pages: the browser surface over the existing control plane.

Spec 003. Deliberately disposable -- Phase B replaces this with a typed SPA --
so there is no framework, build step or component library here. Every journey
works by plain HTML forms and links; the one exception to "no JavaScript" is
the watch page's poller, and it is emitted only while there is something live
to follow: a terminal job's page carries no scripting at all. The interface
exists to answer the checkpoint's question -- *can a user go from dataset to
adapter through the product?* -- and its replacement is already specified.

The validation report is the point of the page set. The API already produces
line-numbered, coded, actionable errors; these pages exist so that a user who
cannot read raw JSON can still act on them.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from temper_control_plane import (
    config,
    datasets,
    db,
    jobs,
    orchestrator,
    quote,
)
from temper_core import catalog, feasibility, hyperparams

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


templates.env.filters["datetimeformat"] = _fmt_ts


def _dataset_label(ds: dict | None, dataset_id: str) -> str:
    """The name a dataset goes by on a page: its filename, or its id when the
    row has gone -- which can happen to nothing except a deleted file, but a
    page that renders an id beats a page that renders a traceback."""
    return ds["filename"] if ds else dataset_id


def _render_error(
    request: Request, status: int, code: str, message: str, title: str
):
    """Every refusal renders as a page carrying its stable code -- the same
    contract the API honours, never laundered into a generic 'error'."""
    return templates.TemplateResponse(
        request,
        "error.html",
        {"status": status, "code": code, "message": message, "title": title},
        status_code=status,
    )


def _error_from_exception(request: Request, exc: HTTPException, title: str):
    """Render an HTTPException as a page. A dict detail carries its own stable
    code; a bare-string detail gets the caller's fallback code rather than a
    fabricated generic one."""
    detail = (
        exc.detail
        if isinstance(exc.detail, dict)
        else {"message": str(exc.detail)}
    )
    return _render_error(
        request,
        exc.status_code,
        detail.get("code", "error"),
        detail.get("message", str(detail)),
        title,
    )


@router.get("/", name="upload_page")
def upload_page(request: Request):
    return templates.TemplateResponse(request, "upload.html", {})


@router.post("/upload", name="upload_form")
def upload_form(request: Request, file: UploadFile = File(...)):
    """The form twin of `POST /v1/datasets`, through the same ingest path.

    A rejection is not an exception to hide behind a JSON body here: it is the
    most useful thing the product says before a run starts, so it renders as a
    page with the same stable code and message the API returns.
    """
    try:
        # A part with no filename fails the extension check as a coded
        # refusal, the same answer the API gives.
        ds_id, _report = datasets.ingest(
            request.headers.get("content-length"),
            file.filename or "",
            file.file,
        )
    except HTTPException as exc:
        return _error_from_exception(request, exc, "The upload was refused")
    return RedirectResponse(f"/datasets/{ds_id}", status_code=303)


@router.get("/datasets/{ds_id}", name="dataset_report")
def dataset_report(request: Request, ds_id: str):
    ds = db.get_dataset(ds_id)
    if not ds:
        return _render_error(
            request,
            404,
            "not_found",
            f"No dataset with id '{ds_id}'.",
            "Not found",
        )
    return templates.TemplateResponse(
        request, "report.html", {"ds": ds, "report": ds.get("report") or {}}
    )


@router.get("/jobs/new", name="create_job_page")
def create_job_page(request: Request, dataset_id: str):
    """Everything the user is about to commit to, before they commit to it.

    The models with their licences and pinned revisions, the hyperparameters
    that will be frozen, any feasibility warning -- all rendered ahead of the
    one action that starts the spending. A dataset that is not valid is
    refused here with its stable code rather than at launch.
    """
    try:
        ds = jobs.usable_dataset(dataset_id)
    except HTTPException as exc:
        return _error_from_exception(request, exc, "Dataset not usable")
    # Estimated against the default hyperparameters, which is what this form
    # launches with: the estimate must describe the job the button will start.
    warn = feasibility.warning(
        feasibility.usable_rows(ds), {}, config.MAX_JOB_DURATION_S
    )
    return templates.TemplateResponse(
        request,
        "create_job.html",
        {
            "ds": ds,
            "models": catalog.listing(),
            "default_model": catalog.DEFAULT_MODEL,
            "spec": hyperparams.effective({}),
            "warning": warn,
        },
    )


@router.post("/jobs/new", name="create_job_form")
def create_job_form(
    request: Request, dataset_id: str = Form(...), base_model: str = Form(...)
):
    """The form twin of `POST /v1/jobs`, through the same creation path.

    A refusal renders as a page carrying its stable code, like every other
    refusal on these pages. Success redirects to the job's own page, which is
    where the run is watched and, later, collected.
    """
    try:
        job_id = jobs.create(
            dataset_id,
            base_model,
            {},
            quote=_quote_for_form(dataset_id, base_model),
        )
    except HTTPException as exc:
        return _error_from_exception(
            request, exc, "The job could not be launched"
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


def _quote_for_form(dataset_id: str, base_model: str) -> dict | None:
    """The quote the form freezes, or None when it cannot be priced.

    The same helper the JSON API uses, so the two creation surfaces freeze the
    same thing; like the API path, an unpricable configuration carries no
    quote rather than refusing the launch (spec 005: the estimate never
    blocks).
    """
    try:
        ds = jobs.usable_dataset(dataset_id)
    except HTTPException:
        return None
    m = catalog.get(base_model)
    if m is None:
        return None
    return quote.for_config(ds, m, {})


@router.get("/jobs", name="jobs_list_page")
def jobs_list_page(request: Request):
    """Every job, newest first, each with its outcome and a link to its record.

    The record is the watch page -- the same page during and after the run --
    so a user who closed the tab finds their adapter by following one link
    from here. The list itself is static HTML: nothing on it needs JavaScript.
    """
    listing = []
    for j in db.list_jobs():
        ds = db.get_dataset(j["dataset_id"])
        listing.append(
            {"job": j, "dataset_filename": _dataset_label(ds, j["dataset_id"])}
        )
    return templates.TemplateResponse(
        request, "jobs.html", {"listing": listing}
    )


@router.get("/jobs/{job_id}", name="watch_job_page")
def watch_job_page(request: Request, job_id: str):
    """Watch a running job -- and collect its result. One page, both jobs.

    A job's record is the same thing during and after the run, so there is no
    separate finished-job page to drift out of agreement with this one. The
    page renders the current state server-side on every load: everything but
    live log updates works without JavaScript.

    The polling script is emitted **only while the job is non-terminal**.
    Polling that stops once terminal is thus a property of the page itself,
    not a promise made in script a reader must trust; and a finished job's
    page carries no scripting at all.
    """
    job = db.get_job(job_id)
    if not job:
        return _render_error(
            request,
            404,
            "not_found",
            f"No job with id '{job_id}'.",
            "Not found",
        )
    events = db.get_events(job_id)
    loss = None
    for e in events:
        data = e.get("data") or {}
        if e["kind"] == "metric" and "loss" in data:
            loss = {"value": data["loss"], "step": data.get("step")}

    start = job["started_at"] or job["created_at"]
    end = job["finished_at"] or time.time()
    ds = db.get_dataset(job["dataset_id"])
    return templates.TemplateResponse(
        request,
        "watch.html",
        {
            "job": job,
            "dataset_filename": _dataset_label(ds, job["dataset_id"]),
            "events": events,
            "last_event_id": events[-1]["id"] if events else 0,
            "loss": loss,
            "elapsed": _fmt_duration(max(end - start, 0)),
            "started_at": start,
            "terminal": job["status"] in db.TERMINAL_STATES,
        },
    )


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


@router.post("/jobs/{job_id}/cancel", name="cancel_job_form")
def cancel_job_form(request: Request, job_id: str):
    """The form twin of `POST /v1/jobs/{id}/cancel`, through the same path.

    The consequence -- no adapter will be produced -- is stated on the page
    beside the button, before the request exists. Every outcome lands back on
    the watch page, which shows the job as it actually is rather than as the
    click wished it.
    """
    outcome = db.request_cancel(job_id, note=orchestrator.CANCEL_ACK)
    if outcome == "missing":
        return _render_error(
            request,
            404,
            "not_found",
            f"No job with id '{job_id}'.",
            "Not found",
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.get("/static/styles.css", name="stylesheet")
def stylesheet():
    return FileResponse(
        Path(__file__).parent / "static" / "styles.css", media_type="text/css"
    )
