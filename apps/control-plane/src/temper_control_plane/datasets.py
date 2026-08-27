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

from temper_control_plane import config, db, storage
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
                f"the current upload limit is {_fmt_size(limit)} "
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
    limit = config.MAX_DATASET_BYTES
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

    def run() -> None:
        try:

            def on_progress(p) -> None:
                db.set_dataset_progress(
                    ds_id,
                    {
                        "bytes_read": p.bytes_read,
                        "bytes_total": p.bytes_total,
                        "rows": p.rows,
                    },
                )

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

    def run() -> None:
        try:
            db.begin_token_count(ds_id)

            def on_progress(p) -> None:
                db.set_counting_progress(
                    ds_id,
                    {
                        "bytes_read": p.bytes_read,
                        "bytes_total": p.bytes_total,
                        "rows": p.rows,
                    },
                )

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

    total = 0

    def counted_chunks():
        nonlocal total
        for chunk in _chunks_of(fileobj):
            total += len(chunk)
            if total > config.MAX_DATASET_BYTES:
                # A lying or absent Content-Length must not defeat the
                # ceiling: refuse once the true size is known. put_stream
                # cleans up its own partial write; the row we created to be
                # watched is deleted here.
                raise too_large(total, config.MAX_DATASET_BYTES)
            yield chunk

    try:
        storage.STORE.put_stream(key, counted_chunks())
    except Exception:
        db.delete_dataset(ds_id)
        raise

    _validate_in_background(ds_id, key, total)
    return ds_id, "validating"
