"""Server-rendered pages: the browser surface over the existing control plane.

Spec 003. Deliberately disposable -- Phase B replaces this with a typed SPA --
so there is no JavaScript at all on these pages: every journey works by plain
HTML forms and links, and live updates (the watch page) will poll rather than
stream until Phase B. The interface exists to answer the checkpoint's question
-- *can a user go from dataset to adapter through the product?* -- and its
replacement is already specified.

The validation report is the point of the page set. The API already produces
line-numbered, coded, actionable errors; these pages exist so that a user who
cannot read raw JSON can still act on them.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from api import db
from api import datasets

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _render_error(request: Request, status: int, code: str, message: str,
                  title: str):
    """Every refusal renders as a page carrying its stable code -- the same
    contract the API honours, never laundered into a generic 'error'."""
    return templates.TemplateResponse(
        request, "error.html",
        {"status": status, "code": code, "message": message, "title": title},
        status_code=status)


def _error_from_exception(request: Request, exc: HTTPException, title: str):
    """Render an HTTPException as a page. A dict detail carries its own stable
    code; a bare-string detail gets the caller's fallback code rather than a
    fabricated generic one."""
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    return _render_error(request, exc.status_code,
                         detail.get("code", "error"),
                         detail.get("message", str(detail)), title)


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
        ds_id, _report = datasets.ingest(request.headers.get("content-length"),
                                         file.filename, file.file)
    except HTTPException as exc:
        return _error_from_exception(request, exc, "The upload was refused")
    return RedirectResponse(f"/datasets/{ds_id}", status_code=303)


@router.get("/datasets/{ds_id}", name="dataset_report")
def dataset_report(request: Request, ds_id: str):
    ds = db.get_dataset(ds_id)
    if not ds:
        return _render_error(request, 404, "not_found",
                             f"No dataset with id '{ds_id}'.", "Not found")
    return templates.TemplateResponse(
        request, "report.html",
        {"ds": ds, "report": ds.get("report") or {}})


@router.get("/static/styles.css", name="stylesheet")
def stylesheet():
    return FileResponse(Path(__file__).parent / "static" / "styles.css",
                        media_type="text/css")
