"""The one dataset-ingest path, shared by the JSON API and the browser pages.

Two entry points that ingest differently would eventually report differently --
and the report is the product's best work before a run starts. Both the API
handler and the upload form call `ingest`, so a change to validation lands
everywhere at once.

Ingest is **asynchronous**. The request refuses an oversized body, stores the
bytes, and returns while validation runs in the background -- validation of a
large file takes long enough that a synchronous upload would read as a frozen
page. The dataset row is created first (status "validating") so the pages have
something to watch: they read `GET /v1/datasets/{id}`, whose `progress` field
the validating thread writes as it goes, and the report lands on the same row
when it finishes.

An import (`import_dataset`) has one more phase ahead of that: its bytes are
not in hand yet, and fetching them from a third party can be slow or flaky in
ways a client's own upload never is (issue: "dataset import from HF"). The
row is created with status "importing" and the fetch itself is also
backgrounded, so the request only ever waits on `resolve()` -- a few small,
bounded API calls that confirm the reference is real. `begin_validating`
moves the row into the same "validating" phase an upload's row was already
in, once the fetch finishes.
"""

from __future__ import annotations

import threading

from fastapi import HTTPException

from temper_control_plane import config as cfg
from temper_control_plane import db, remote_datasets, storage
from temper_control_plane.correlation import (
    get_correlation_id,
    set_correlation_id,
)
from temper_core import counting, hyperparams, validation

from . import tokenize


def _fmt_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024:.1f} KB"


def too_large(actual: int, limit: int) -> HTTPException:
    return HTTPException(
        413,
        {
            "code": "dataset_too_large",
            "message": (
                f"This dataset is {_fmt_size(actual)} ({actual:,} bytes); "
                f"the current dataset size limit is {_fmt_size(limit)} "
                f"({limit:,} bytes). The limit is a product decision derived "
                f"from the measured validation throughput, so that validation "
                f"completes within a tolerable wait -- see the decision record "
                f"that replaced the old memory-derived ceiling."
            ),
            "limit_bytes": limit,
            "actual_bytes": actual,
        },
    )


def refuse_before_read(declared: str | None) -> None:
    """Refuse on the declared body size before reading anything.

    Reading first would load an arbitrarily large file into memory, which is
    the failure the limit exists to prevent. The declared length is the
    multipart body, so it slightly overstates the file itself -- close enough
    to refuse on, and it never understates it. The header can be absent or
    lying, which is why `ingest` also refuses mid-stream once the true size is
    known.
    """
    limit = cfg.MAX_DATASET_BYTES
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise too_large(int(declared), limit)


def _check_extension(filename: str) -> None:
    """Refuse a filename that doesn't say .jsonl -- shared by `ingest` (an
    upload's own name) and `rename_dataset` (a name a user chose): the
    product only ever ingests JSONL, and a display name that doesn't say so
    is a lie the file doesn't back up."""
    if not filename.endswith((".jsonl", ".json")):
        raise HTTPException(
            400,
            {
                "code": "unsupported_extension",
                "message": f"'{filename}' is not a .jsonl file. Temper accepts "
                f"chat-format JSONL.",
                "filename": filename,
            },
        )


def _chunks_of(fh):
    """Read a file object in bounded chunks. Holds at most one chunk."""
    while chunk := fh.read(storage.STREAM_CHUNK_BYTES):
        yield chunk


def _synthetic_invalid_report(code: str, message: str) -> dict:
    """The shape every "this dataset could not be produced" refusal wears,
    whether validation crashed or the import's fetch failed. A page that
    never resolves is worse than one that reports a refusal -- so both
    failure modes end at the same stored report, not a row stuck mid-phase
    forever."""
    return {
        "valid": False,
        "row_count": 0,
        "usable_rows": 0,
        "schema_type": None,
        "enable_thinking": None,
        "errors": [{"line": None, "code": code, "message": message}],
        "warnings": [],
        "preview": [],
        "error_count": 1,
        "warning_count": 0,
        "errors_suppressed": 0,
        "warnings_suppressed": 0,
        "error_code_counts": {code: 1},
        "warning_code_counts": {},
    }


def _run_validation(ds_id: str, key: str, total_bytes: int) -> None:
    """Validate the stored object, writing progress and the final report to
    the dataset row. The caller decides whether this runs on its own thread
    (`_validate_in_background`, an upload's bytes are already in hand) or on
    a thread already running in the background (`_import_in_background`,
    which validates right after its own fetch finishes -- no reason to hop
    threads twice)."""
    try:

        def on_progress(p) -> None:
            db.set_dataset_progress(ds_id, p.to_dict())

        report = validation.validate_chunks(
            storage.STORE.get_stream(key),
            total_bytes=total_bytes,
            on_progress=on_progress,
        ).to_dict()
        db.finish_dataset(ds_id, report)
        if report.get("valid"):
            # Only a usable dataset is counted: an invalid one cannot
            # launch, and the counting pass is the expensive half of
            # validation (issue #42), so spending it on a file nobody can
            # quote is the one thing the phase exists to avoid.
            _count_tokens_in_background(ds_id, key, total_bytes)
    except Exception:
        # A validation that dies must not strand the dataset at
        # "validating" -- the synthetic report is a coded failure, the same
        # shape every refusal wears.
        db.finish_dataset(
            ds_id,
            _synthetic_invalid_report(
                "validation_failed",
                "Validation stopped unexpectedly. Upload the dataset again.",
            ),
        )


def _validate_in_background(ds_id: str, key: str, total_bytes: int) -> None:
    """Validate the stored object on its own thread.

    Runs off the request thread because validation is CPU-bound and a large
    file takes tens of seconds: the row is created first so the pages watching
    it can render progress while this runs.
    """

    # The request's correlation identifier is captured here so the background
    # thread's structured logs still carry the same story when they emit.
    _cid = get_correlation_id()

    def run() -> None:
        # Re-bind the request's correlation before doing work that may log.
        if _cid:
            set_correlation_id(_cid)
        _run_validation(ds_id, key, total_bytes)

    threading.Thread(target=run, daemon=True, name=f"validate-{ds_id}").start()


def _count_tokens_in_background(
    ds_id: str, key: str, total_bytes: int
) -> None:
    """Count the dataset's tokens on its own thread, writing the counting
    phase's state and progress to the dataset row.

    Counting is the expensive half of validation -- re-measured at ~7x the
    validate-only pass (issue #42) -- which is the measurement that put it in
    its own phase with its own state rather than blocking the report. It runs
    only for datasets that passed validation; see `_validate_in_background`.
    """

    _cid = get_correlation_id()

    def run() -> None:
        if _cid:
            set_correlation_id(_cid)
        try:
            db.begin_token_count(ds_id)

            def on_progress(p) -> None:
                db.set_counting_progress(ds_id, p.to_dict())

            counts = counting.count_tokens_chunks(
                storage.STORE.get_stream(key),
                tokenize.count_row_for(),
                # The length truncation is measured against is the trainer's
                # default, read once from the same table the resolver reads.
                sequence_len=hyperparams.DEFAULTS["sequence_len"],
                total_bytes=total_bytes,
                on_progress=on_progress,
            )
            db.finish_token_count(ds_id, counts.to_dict())
        except Exception:
            # A count that cannot be produced is an absent estimate, not a
            # broken dataset: the report stands, the launch proceeds, and the
            # quote renders the count absent (ADR-0031). The phase records the
            # failure so the page can say what happened rather than wonder --
            # and that recording must not itself escape: the dataset row can
            # be gone (deleted while the pass ran), and a thread that dies
            # while reporting a failure is noise, not a signal.
            try:
                db.fail_token_count(ds_id)
            except Exception:  # noqa: S110 - a vanished row is not a signal
                pass

    threading.Thread(target=run, daemon=True, name=f"count-{ds_id}").start()


def _store_counted(
    ds_id: str,
    key: str,
    chunks,
    *,
    delete_on_failure: bool = True,
    on_progress=None,
) -> int:
    """Store `chunks` with the size ceiling enforced as the true size becomes
    known, returning how many bytes were stored.

    The one place either ingest path turns a chunk source into a stored
    object, so the ceiling is identical for a file upload and a remote
    import. `put_stream` publishes whole and cleans up its own partial write
    on failure. Refusing once the true size is known is the only enforcement
    a remote reference earns -- it has no declared size to refuse on ahead of
    time -- and it is the same refusal a lying Content-Length earns on an
    upload.

    `delete_on_failure` (default True) removes the row created to be
    watched, correct for an upload: the request has not answered yet, so
    nobody has been told this id exists. An import backgrounds this call
    (`_import_in_background`) after already answering with the id, so it
    passes False -- the row must survive to carry the failure's report,
    exactly like a validation crash already does.

    `on_progress`, when given, is called with the running byte total as
    chunks are stored -- the import path's fetch phase watches it the same
    way validation's own `on_progress` is watched (issue: "dataset import
    from HF"); an upload has no need of it, since its own bytes are already
    in hand and the phase is fast.
    """
    total = 0

    def counted_chunks():
        nonlocal total
        for chunk in chunks:
            total += len(chunk)
            if total > cfg.MAX_DATASET_BYTES:
                raise too_large(total, cfg.MAX_DATASET_BYTES)
            if on_progress is not None:
                on_progress(total)
            yield chunk

    try:
        storage.STORE.put_stream(key, counted_chunks())
    except Exception:
        if delete_on_failure:
            db.delete_dataset(ds_id)
        raise
    return total


def ingest(
    declared_length: str | None, filename: str, fileobj
) -> tuple[str, str]:
    """Refuse on the declared size, store the body, start validating in the
    background. Returns (dataset id, status).

    A dataset that fails is still stored with its report attached, so the user
    can see exactly which lines to fix rather than re-uploading blind.
    Raises HTTPException with a stable code for a wrong extension or an
    oversized body -- the same codes whichever surface the upload came in on.
    """
    _check_extension(filename)
    refuse_before_read(declared_length)

    ds_id = db.new_id("ds")
    key = storage.dataset_key(ds_id)
    db.create_dataset(filename, key, ds_id=ds_id)

    total = _store_counted(ds_id, key, _chunks_of(fileobj))

    _validate_in_background(ds_id, key, total)
    return ds_id, "validating"


def _import_in_background(ds_id: str, key: str, source) -> None:
    """Fetch a remote source's rows and store them, then validate -- all on
    one background thread. Returns nothing; every outcome is written to the
    dataset row, because the request that started this already answered with
    `ds_id` and has nobody left to hand a return value to.

    Backgrounded because the fetch reads from a third party (issue: "dataset
    import from HF"): `resolve()` stays synchronous -- it is a few small,
    now-bounded API calls, and its refusal (bad repository, bad config,
    empty split) is worth answering before the caller is told an id exists --
    but the rows themselves can be arbitrarily slow to arrive, and a request
    is not the place to find that out. A fetch that fails, or a split that
    turns out larger than the size ceiling, is caught here and stored as an
    invalid report exactly the way a validation crash already is: the
    request has already told the caller this id exists, so there is no
    synchronous response left to refuse with.
    """
    _cid = get_correlation_id()

    def run() -> None:
        if _cid:
            set_correlation_id(_cid)

        def on_progress(total: int) -> None:
            db.set_dataset_progress(
                ds_id, {"bytes_read": total, "bytes_total": None, "rows": 0}
            )

        try:
            total = _store_counted(
                ds_id,
                key,
                source.stream(),
                delete_on_failure=False,
                on_progress=on_progress,
            )
        except remote_datasets.RemoteDatasetError as e:
            db.finish_dataset(
                ds_id, _synthetic_invalid_report(e.code, e.message)
            )
            return
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, dict) else {}
            db.finish_dataset(
                ds_id,
                _synthetic_invalid_report(
                    detail.get("code", "import_failed"),
                    detail.get(
                        "message", "The import could not be completed."
                    ),
                ),
            )
            return
        except Exception:
            db.finish_dataset(
                ds_id,
                _synthetic_invalid_report(
                    "import_failed",
                    "The import stopped unexpectedly. Try again.",
                ),
            )
            return

        db.begin_validating(ds_id)
        _run_validation(ds_id, key, total)

    threading.Thread(target=run, daemon=True, name=f"import-{ds_id}").start()


def import_dataset(
    repo: str,
    config: str | None = None,
    split: str | None = None,
) -> tuple[str, str, str]:
    """Import a public dataset by reference. Returns (dataset id, display
    name, status).

    The reference is resolved first -- a repository that cannot be fetched,
    a configuration that must be named, or a split that resolves to nothing
    is refused with its reason before anything is stored. Once resolved, the
    fetch, the store, the validation and the token count all run in the
    background (`_import_in_background`) through the very same functions an
    upload uses past its own fetch, so an imported dataset that fails
    validation is stored with its report like any other. There is one
    validator; the only difference is where the bytes came from, and now
    also how long fetching them is allowed to take before the caller stops
    waiting on it.
    """
    source = remote_datasets.RESOLVER.resolve(repo, config, split)
    filename = source.display_name()

    ds_id = db.new_id("ds")
    key = storage.dataset_key(ds_id)
    db.create_dataset(filename, key, ds_id=ds_id, status="importing")

    _import_in_background(ds_id, key, source)
    return ds_id, filename, "importing"


def rename_dataset(ds_id: str, filename: str) -> dict:
    """Rename a dataset, returning its updated record. Raises HTTPException
    with a stable code if it cannot proceed.

    Refused (404) if it does not exist, and refused (400,
    `unsupported_extension`) the same way an upload's own name is -- the
    product only ever ingests JSONL, and a display name is supposed to say
    what it is. Allowed regardless of status: renaming touches only the
    `filename` column, never the stored object, so there is no race with a
    background thread still fetching or validating it (contrast
    `delete_dataset`, which removes the object and must wait for that).
    """
    filename = filename.strip()
    _check_extension(filename)
    ds = db.get_dataset(ds_id)
    if ds is None:
        raise HTTPException(
            404, {"code": "not_found", "message": "No such dataset."}
        )
    db.rename_dataset(ds_id, filename)
    return db.get_dataset(ds_id)


def delete_dataset(ds_id: str) -> None:
    """Delete a dataset: the stored object, then the row. Returns nothing;
    raises HTTPException with a stable code if it cannot proceed.

    Refused (404) if the dataset does not exist; refused (409,
    `dataset_not_ready`) while it is `importing` or `validating`, because a
    background thread still holds the object open for reading or writing --
    deleting under it is a race with no defined outcome across storage
    backends (a hard error on a local filesystem, silently undefined on
    S3-compatible stores), and a resource still being ingested is not yet a
    settled thing to delete; and refused (409, `dataset_in_use`) if any job
    was launched against it -- `jobs.dataset_id` is `NOT NULL REFERENCES
    datasets(id)` with no cascade, so the database would refuse the row
    delete anyway; checking first turns that into an explained refusal
    rather than a raw constraint violation, and it is the right call
    regardless: a completed job's record should keep being able to say what
    it trained on.

    The object is removed before the row, not after -- a row that outlives
    its object is a 404 waiting to happen the next time it is read; an
    object that outlives a deleted row is merely unreferenced, or picked up
    the next time a mid-stream refusal reuses the same key, so this ordering
    is the one that fails safe.
    """
    ds = db.get_dataset(ds_id)
    if ds is None:
        raise HTTPException(
            404, {"code": "not_found", "message": "No such dataset."}
        )
    if ds["status"] in ("importing", "validating"):
        raise HTTPException(
            409,
            {
                "code": "dataset_not_ready",
                "message": (
                    f"This dataset is still '{ds['status']}'; wait for it "
                    f"to finish before deleting it."
                ),
            },
        )
    job_count = db.count_jobs_for_dataset(ds_id)
    if job_count > 0:
        raise HTTPException(
            409,
            {
                "code": "dataset_in_use",
                "message": (
                    f"This dataset was used by {job_count} job"
                    f"{'' if job_count == 1 else 's'} and cannot be "
                    f"deleted. A job's record must keep pointing at the "
                    f"data it trained on."
                ),
                "job_count": job_count,
            },
        )
    storage.STORE.delete(ds["object_key"])
    db.delete_dataset(ds_id)
