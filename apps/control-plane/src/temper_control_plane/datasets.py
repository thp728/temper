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


def _chunks_of(fh):
    """Read a file object in bounded chunks. Holds at most one chunk."""
    while chunk := fh.read(storage.STREAM_CHUNK_BYTES):
        yield chunk


def _validate_in_background(ds_id: str, key: str, total_bytes: int) -> None:
    """Validate the stored object on its own thread, writing progress and the
    final report to the dataset row.

    Runs off the request thread because validation is CPU-bound and a large
    file takes tens of seconds: the row is created first so the pages watching
    it can render progress while this runs. A validation that raises records a
    coded failure rather than leaving the row stuck at "validating" forever.
    """

    # The request's correlation identifier is captured here so the background
    # thread's structured logs still carry the same story when they emit.
    _cid = get_correlation_id()

    def run() -> None:
        # Re-bind the request's correlation before doing work that may log.
        if _cid:
            set_correlation_id(_cid)
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
            # "validating" -- a page that never resolves is worse than one
            # that reports a refusal. The synthetic report is a coded failure,
            # the same shape every refusal wears.
            db.finish_dataset(
                ds_id,
                {
                    "valid": False,
                    "row_count": 0,
                    "usable_rows": 0,
                    "schema_type": None,
                    "enable_thinking": None,
                    "errors": [
                        {
                            "line": None,
                            "code": "validation_failed",
                            "message": "Validation stopped unexpectedly. "
                            "Upload the dataset again.",
                        }
                    ],
                    "warnings": [],
                    "preview": [],
                    "error_count": 1,
                    "warning_count": 0,
                    "errors_suppressed": 0,
                    "warnings_suppressed": 0,
                },
            )

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


def _store_counted(ds_id: str, key: str, chunks) -> int:
    """Store `chunks` with the size ceiling enforced as the true size becomes
    known, returning how many bytes were stored.

    The one place either ingest path turns a chunk source into a stored
    object, so the ceiling and the clean-up are identical for a file upload
    and a remote import. `put_stream` publishes whole and cleans up its own
    partial write on failure; the row created to be watched is deleted here.
    Refusing once the true size is known is the only enforcement a remote
    reference earns -- it has no declared size to refuse on ahead of time --
    and it is the same refusal a lying Content-Length earns on an upload.
    """
    total = 0

    def counted_chunks():
        nonlocal total
        for chunk in chunks:
            total += len(chunk)
            if total > cfg.MAX_DATASET_BYTES:
                raise too_large(total, cfg.MAX_DATASET_BYTES)
            yield chunk

    try:
        storage.STORE.put_stream(key, counted_chunks())
    except Exception:
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
    refuse_before_read(declared_length)

    ds_id = db.new_id("ds")
    key = storage.dataset_key(ds_id)
    db.create_dataset(filename, key, ds_id=ds_id)

    total = _store_counted(ds_id, key, _chunks_of(fileobj))

    _validate_in_background(ds_id, key, total)
    return ds_id, "validating"


def import_dataset(
    repo: str,
    config: str | None = None,
    split: str | None = None,
) -> tuple[str, str, str]:
    """Import a public dataset by reference, through the same ingest path as
    an upload. Returns (dataset id, display name, status).

    The reference is resolved first, and a reference that cannot be fetched --
    repository missing, configuration unnamed, split absent, split empty -- is
    refused with its reason before anything is stored. The rows are then
    streamed straight from the remote source into storage, with the size
    ceiling enforced mid-stream exactly as it is for a lying Content-Length on
    an upload: `put_stream` publishes whole and cleans up its own partial
    write, and the row created to be watched is deleted on refusal.

    Validation and token counting then run in the background through the very
    same functions an upload uses (`_validate_in_background` reads the stored
    object), so an imported dataset that fails validation is stored with its
    report like any other. There is one validator; the only difference is
    where the bytes came from.
    """
    source = remote_datasets.RESOLVER.resolve(repo, config, split)
    filename = source.display_name()

    ds_id = db.new_id("ds")
    key = storage.dataset_key(ds_id)
    db.create_dataset(filename, key, ds_id=ds_id)

    total = _store_counted(ds_id, key, source.stream())

    _validate_in_background(ds_id, key, total)
    return ds_id, filename, "validating"
