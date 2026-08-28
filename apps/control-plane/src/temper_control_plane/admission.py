"""Admitting a model from outside the catalog (Spec 009 / issue #58).

The catalog is a curated, tested path and a promise about what has been run --
a default, not a boundary. This module is the boundary coming down: any model
repository at a pinned revision may be admitted once it has passed a
compatibility probe, and the probe's result is persisted and shown before a
job can be created against that model.

The probe reads its facts through **the same seam the predictor reads** --
`quote.models_for_quote()` -- so there is exactly one implementation of *how
many parameters does this model have*, and the probe and the predictor cannot
drift (ADR-0010; spec 009's "the `models` seam gains the probe"). The memory
line is `temper_core.probe`'s own `memory.predict_peak` call, the same
arithmetic the creation-time refusal (#54) is built on.

A reference that is not a pinned revision is refused up front with the stable
code `unpinned_revision` (the catalog's pinning rule, applied to every path).
A pinned reference whose revision does not resolve is still admitted -- as a
*blocked* probe, persisted and shown, so the user learns the reason rather
than retrying the same thing (spec 009's user story 3). A blocked probe is
refused at job creation with its blocking findings; a probe that passes, with
or without warnings, makes the model selectable and launchable.
"""

from __future__ import annotations

from temper_core import catalog, gpus, probe
from temper_core.models import Models

from . import db

# The stable code for a reference that is not a pinned revision. A branch name
# or short SHA can change under a completed run, which breaks the
# reproducibility claim the catalog's own pinning exists to support.
UNPINNED_REVISION = "unpinned_revision"


class AdmissionError(Exception):
    """A refusal to admit a reference, with the stable code and message every
    API error carries. The one up-front refusal this module raises; everything
    after the pinned-revision check is a probe verdict, not an exception."""

    def __init__(self, code: str, message: str, **fields: object) -> None:
        self.code = code
        self.message = message
        self.fields = fields
        super().__init__(message)

    def to_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, **self.fields}


def _resolver() -> Models:
    """The model-facts resolver, read through the predictor's own seam.

    `quote.models_for_quote()` is the exact seam the quote (and therefore the
    hardware search and the creation-time refusal) reads facts through. Using
    it here, rather than a second resolver, is what makes "the probe reads
    model facts through the same seam the predictor uses" literally true: in
    production both are `new_models()`, and under test both are whatever the
    quote seam has been pointed at.
    """
    from . import quote

    return quote.models_for_quote()


def probe_and_admit(repo: str, revision: str) -> dict:
    """Resolve, probe and persist one pinned model reference.

    Returns the stored admitted-model record, whose `probe` carries the
    verdict and findings. Raises `AdmissionError` only for a reference that is
    not a pinned revision; every other outcome -- including a revision that
    does not resolve -- is a persisted probe verdict, because the whole point
    of showing the result is that the user learns why rather than being told
    nothing.
    """
    if not catalog.is_pinned_revision(revision):
        raise AdmissionError(
            UNPINNED_REVISION,
            f"'{revision}' is not a pinned revision. A model must be pinned "
            "to a 40-character commit SHA so a completed run describes a "
            "model that cannot change afterwards.",
            revision=revision,
        )
    try:
        facts = _resolver().resolve(repo, revision)
    except Exception as e:  # noqa: BLE001 - any failure to resolve is the same
        # finding: the revision does not resolve, so the model is blocked.
        result = probe.unresolvable(repo, revision, str(e))
    else:
        result = probe.probe(facts, repo=repo, revision=revision)
    record_id = db.create_admitted_model(repo, revision, result.as_dict())
    record = db.get_admitted_model(record_id)
    if record is None:
        # The row was written a line above; a read that comes back empty is a
        # defect, not a missing model.
        raise AssertionError("admitted model vanished immediately after write")
    return record


def resolve(base_model: str) -> catalog.BaseModel | None:
    """The model a launch refers to, as a `catalog.BaseModel` -- whether it is
    a catalog entry or an admitted one.

    Every downstream consumer (quote, jobs) reads one shape, so admitting a
    model costs them nothing. An admitted record is materialised from its
    stored probe: the facts were resolved and snapshot at admission, and a
    pinned revision cannot change, so the materialisation is exact.
    """
    m = catalog.get(base_model)
    if m is not None:
        return m
    return _as_catalog_model(db.get_admitted_model(base_model))


def _as_catalog_model(record: dict | None) -> catalog.BaseModel | None:
    """One admitted record as the `catalog.BaseModel` shape.

    `min_gpu_type` is the smallest card the probe's memory line named; the
    peak arithmetic that chose it is the same the launch-time search reads.
    """
    if record is None:
        return None
    p = record.get("probe") or {}
    card = (p.get("memory") or {}).get("card")
    if card is None or card not in gpus.CAPACITY_GB:
        card = max(gpus.CAPACITY_GB, key=lambda c: gpus.CAPACITY_GB[c])
    capacity = gpus.CAPACITY_GB[card]
    return catalog.BaseModel(
        id=record["id"],
        repo=record["repo"],
        revision=record["revision"],
        params_b=p.get("params_b", 0.0),
        license=p.get("license") or "Unknown",
        license_url=f"https://huggingface.co/{record['repo']}",
        context_length=p.get("context_length", 0),
        good_for=(
            "Imported from outside the catalog; its compatibility probe "
            "result is shown with it."
        ),
        min_gpu=f"{card} ({capacity:.0f} GB)",
        min_gpu_type=card,
    )


def listing() -> list[dict]:
    """Every admitted model, newest first, with its persisted probe."""
    return db.list_admitted_models()


def available_ids() -> list[str]:
    """The ids a launch could name: catalog entries plus admitted models."""
    return [m["id"] for m in catalog.listing()] + [
        r["id"] for r in db.list_admitted_models()
    ]
