"""The one job-creation path, shared by the JSON API and the browser form.

Two entry points that created jobs differently would eventually refuse
differently -- and a refusal the browser shows but the API does not honour (or
the reverse) means one of the two surfaces is lying about what launches a job.
Both call `create`, so a change to validation or to the frozen-warnings
behaviour lands everywhere at once. The same reasoning put the dataset ingest
in `datasets.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import HTTPException

from temper_control_plane import config, db, orchestrator
from temper_core import catalog, feasibility, hyperparams, overrides


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
    overrides_list: Sequence[overrides.Override] | None = None,
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

    `overrides_list` (issue #79) are the decisions the user pinned instead of
    the predictor's. They are refused when they describe no real
    configuration (`overrides.OverrideError`, e.g. an unknown decision or an
    inconsistent precision/method pair), refused when the trainer cannot run
    them (`not_executable`, spec 009's territory), and frozen into the job
    row beside the hyperparameters -- a run says what it actually used, and
    the orchestrator re-provisions against them rather than silently falling
    back to the predictor's pick.

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

    base_hp = hyperparams.effective(hyperparameters)
    frozen_hp = dict(hyperparameters)
    frozen_overrides: list[dict] = []
    if overrides_list:
        # The vocabulary and coupling are refused before anything is priced or
        # launched: a decision the user believes is in effect but is not is
        # the same silent lie as an unknown hyperparameter.
        try:
            resolved = overrides.resolve(base_hp, overrides_list)
        except overrides.OverrideError as e:
            raise HTTPException(400, e.to_dict()) from e
        # Only the sequence-length override changes what the trainer runs, so
        # only it is folded into the frozen hyperparameters; the rest are the
        # hardware/disk decisions the orchestrator re-provisions against.
        if "sequence length" in resolved.overridden:
            frozen_hp["sequence_len"] = resolved.hyperparameters[
                "sequence_len"
            ]
        frozen_overrides = [overrides.to_dict(o) for o in overrides_list]
        # Executability is separate from feasibility: an override that cannot
        # be *run* today (spec 009) is refused here, because the trainer
        # would silently run something else -- a run that lies about what it
        # did is the one failure this product refuses to make cheap.
        eff_method = resolved.method or overrides.executable_default()
        eff_count = resolved.device_count or 1
        if not overrides.executable(eff_method, eff_count):
            raise HTTPException(
                400,
                {
                    "code": "not_executable",
                    "message": (
                        f"{eff_method} on {eff_count} device(s) is described "
                        "but not runnable yet: the trainer executes only "
                        f"{overrides.executable_names()} on a single device "
                        "today. Spec 009 teaches it the rest; until then this "
                        "configuration can be planned and refused, not "
                        "launched."
                    ),
                    "method": eff_method,
                    "device_count": eff_count,
                },
            )

    # The ceiling is read at request time, not import: an operator changing
    # TEMPER_MAX_JOB_DURATION_S should not need a restart for the warning to
    # reflect it.
    warn = feasibility.warning(
        feasibility.usable_rows(ds), frozen_hp, config.MAX_JOB_DURATION_S
    )
    warnings = [warn] if warn else []

    job_id = db.create_job(
        dataset_id,
        base_model,
        frozen_hp,
        warnings=warnings,
        base_revision=model.revision,
        quote=quote,
        overrides=frozen_overrides,
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
