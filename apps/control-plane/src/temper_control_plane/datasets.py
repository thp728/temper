"""The one dataset-ingest path, shared by the JSON API and the browser pages.

Two entry points that store and validate differently would eventually report
differently -- and the report is the product's best work before a run starts.
Both the API handler and the upload form call `store_and_validate`, so a fix
to validation lands everywhere at once.
"""

from __future__ import annotations

from fastapi import HTTPException

from temper_control_plane import config, db
from temper_core import validation

# Via config, for the reason given on db.DB_PATH: a counted path silently
# meant somewhere else once this module moved, and uploaded datasets are the
# worst possible thing to relocate into a directory git is willing to commit.
UPLOADS = config.REPO_ROOT / "data" / "uploads"


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
                f"({limit:,} bytes). The limit exists because validation "
                f"holds the whole dataset in memory (about 4.8x its size), "
                f"so it is a limit of the current in-memory validation path, "
                f"not a product rule -- streaming validation will remove it."
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
    lying, which is why the authoritative check after the read remains.
    """
    limit = config.MAX_DATASET_BYTES
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise too_large(int(declared), limit)


def store_and_validate(filename: str, data: bytes) -> tuple[str, dict]:
    """Persist an upload and validate it. Returns (dataset id, report).

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

    limit = config.MAX_DATASET_BYTES
    if len(data) > limit:
        raise too_large(len(data), limit)

    ds_id = db.new_id("ds")
    path = UPLOADS / f"{ds_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    db.create_dataset(filename, path, ds_id=ds_id)

    report = validation.validate(path).to_dict()
    db.finish_dataset(ds_id, report)
    return ds_id, report


def ingest(
    declared_length: str | None, filename: str, fileobj
) -> tuple[str, dict]:
    """One ingest sequence for both surfaces: refuse on the declared size,
    read, store, validate. Raises HTTPException with a stable code."""
    refuse_before_read(declared_length)
    return store_and_validate(filename, fileobj.read())
