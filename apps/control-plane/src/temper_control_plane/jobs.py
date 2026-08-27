"""The one job-creation path, shared by the JSON API and the browser form.

Two entry points that created jobs differently would eventually refuse
differently -- and a refusal the browser shows but the API does not honour (or
the reverse) means one of the two surfaces is lying about what launches a job.
Both call `create`, so a change to validation or to the frozen-warnings
behaviour lands everywhere at once. The same reasoning put the dataset ingest
in `datasets.py`.
"""

from __future__ import annotations

from fastapi import HTTPException

from temper_control_plane import config, db, orchestrator
from temper_core import catalog, feasibility, hyperparams


def usable_dataset(dataset_id: str) -> dict:
    """The dataset row a launch would train on, or a coded refusal.

    Shared by the JSON API, the browser's create-job page and the browser's
    launch form -- the three places that answer "can this dataset start a
    job?" -- so all three refuse identically.
    """
    ds = db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(
            404,
            {
                "code": "not_found",
                "message": f"No dataset with id '{dataset_id}'.",
            },
        )
    if ds["status"] != "valid":
        raise HTTPException(
            400,
            {
                "code": "dataset_invalid",
                "message": "This dataset failed validation and cannot be trained "
                "on. Fix the lines named in its report and upload it "
                "again.",
                "errors": (ds.get("report") or {}).get("errors", []),
            },
        )
    return ds


def create(
    dataset_id: str,
    base_model: str,
    hyperparameters: dict,
    quote: dict | None = None,
) -> str:
    """Validate the request, freeze the spec and launch. Returns the job id.

    Refuses an invalid dataset rather than discovering it on a GPU four minutes
    later. A dataset that plainly cannot finish inside the maximum duration
    gets a **warning attached, not a refusal** (spec 002): the estimate is
    crude -- measured throughput on one real run -- and a wrong block is worse
    than a wrong warning. The warning is frozen onto the job row like the
    hyperparameters, so what the user was told before launching stays part of
    the run's record.

    `quote` is the prediction the launch was shown, frozen alongside the
    warning (issue #72). The caller computes it -- the seams that feed it live
    beside the HTTP handlers -- and the job row carries it exactly as it was,
    never updated: a completed job can say what it was predicted to cost.

    Raises HTTPException with a stable code for every refusal, whichever
    surface it arrived on.
    """
    ds = usable_dataset(dataset_id)
    model = catalog.get(base_model)
    if not model:
        raise HTTPException(
            400,
            {
                "code": "unknown_model",
                "message": f"'{base_model}' is not in the catalog.",
                "available": [m["id"] for m in catalog.listing()],
            },
        )

    # Refused here rather than left for the trainer's guard: resolution now
    # happens before launch (#83), so an unknown key would be dropped by the
    # resolver without ever reaching the machine -- and a key the caller
    # believes is in effect but isn't is worse than a refusal.
    unknown = sorted(
        k for k in hyperparameters if k not in hyperparams.ALLOWED_OVERRIDES
    )
    if unknown:
        raise HTTPException(
            400,
            {
                "code": "unknown_hyperparameter",
                "message": "Unknown hyperparameter keys are refused: "
                f"{unknown}. Nothing was launched.",
                "unknown": unknown,
            },
        )

    # The ceiling is read at request time, not import: an operator changing
    # TEMPER_MAX_JOB_DURATION_S should not need a restart for the warning to
    # reflect it.
    warn = feasibility.warning(
        feasibility.usable_rows(ds), hyperparameters, config.MAX_JOB_DURATION_S
    )
    warnings = [warn] if warn else []

    job_id = db.create_job(
        dataset_id,
        base_model,
        hyperparameters,
        warnings=warnings,
        base_revision=model.revision,
        quote=quote,
    )
    if warn:
        db.add_event(
            job_id,
            "log",
            warn["message"],
            {k: v for k, v in warn.items() if k != "message"},
        )
    orchestrator.launch(job_id)
    return job_id
