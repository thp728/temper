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

from temper_control_plane import admission, config, db, orchestrator
from temper_core import (
    faults,
    feasibility,
    hyperparams,
    overrides,
    surface,
)


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
    model, admitted_probe = admission.lookup(base_model)
    if not model:
        raise HTTPException(
            400,
            {
                "code": "unknown_model",
                "message": f"'{base_model}' is not in the catalog and has not "
                "been admitted by a probe.",
                "available": admission.available_ids(),
            },
        )

    # Issue #58: a model admitted from outside the catalog may be launched
    # only after its probe is on record, and a probe with blocking findings
    # refuses the launch with those findings shown. The probe result was
    # persisted at admission; this gate is what "before a job can be created"
    # means -- a blocked model cannot be launched into, because the probe
    # exists to do the testing in front of the user rather than on a paid
    # machine. A catalog model has no probe (it is the tested default) and
    # passes by construction. `admitted_probe` came from the same lookup that
    # materialised the model, so the gate does not re-read the row.
    if admitted_probe is not None and not admitted_probe.get("ok", False):
        blocks = [
            f
            for f in admitted_probe.get("findings", [])
            if f.get("severity") == "block"
        ]
        raise HTTPException(
            400,
            {
                "code": "model_probe_blocked",
                "message": (
                    f"'{model.repo}' at revision '{model.revision}' is blocked "
                    "by its compatibility probe; it cannot be trained on here. "
                    "The probe's findings say why."
                ),
                "findings": blocks,
            },
        )

    # Refused here rather than left for the trainer's guard: resolution now
    # happens before launch (#83), so an unknown key would be dropped by the
    # resolver without ever reaching the machine -- and a key the caller
    # believes is in effect but isn't is worse than a refusal. The gate is the
    # generated surface (issue #33): keys unknown to the trainer are refused
    # and echoed back, keys the trainer knows but the platform does not expose
    # are refused with their reason, and exposed values that violate the
    # schema's expressed constraints are refused before anything is priced.
    refusals = surface.validate_overrides(hyperparameters)
    if refusals:
        raise HTTPException(400, refusals[0])

    # Issue #24: the fault surface is off by default, and "off" is enforced
    # here at the single creation path, not hoped for downstream. A fault spec
    # -- the dict form of `simulated_failure_code` -- can only be created when
    # the deployment has deliberately switched the surface on, and a spec that
    # names an unknown fault or carries a parameter that fault does not take
    # is refused before anything is priced: a fault spec the caller believes
    # is in effect but is not is worse than a refusal. The policy (which
    # switches permit which faults) is defined once in `config`.
    fault_spec = faults.from_hyperparameters(hyperparameters)
    if fault_spec is not None:
        name = fault_spec.get("name")
        if not isinstance(name, str) or not faults.is_known(name):
            raise HTTPException(
                400,
                {
                    "code": "fault_unknown",
                    "message": (
                        f"'{name}' is not a fault the surface knows; the job "
                        "was refused rather than run under a fault nobody can "
                        "explain."
                    ),
                },
            )
        problem = faults.spec_error(fault_spec)
        if problem is not None:
            raise HTTPException(
                400,
                {
                    "code": "fault_invalid",
                    "message": problem,
                },
            )
        refusal = config.fault_surface_refusal(name)
        if refusal is not None:
            raise HTTPException(400, refusal)

    base_hp = hyperparams.effective(hyperparameters)
    # The frozen record is the user's request, coerced to the schema's type
    # (issue #80): a launch typed "16" into a number field, and the record says
    # 16 -- the same typed value the resolver derives and the trainer reads.
    frozen_hp = {
        k: surface.coerce_value(k, v)
        for k, v in (hyperparameters or {}).items()
    }
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
