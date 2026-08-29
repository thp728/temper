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

from temper_control_plane import admission, config, db
from temper_control_plane import orchestrator as _orch
from temper_core import (
    delivery,
    divergence,
    faults,
    feasibility,
    hyperparams,
    overrides,
    surface,
)

# The original thread starter, captured so tests that monkeypatch ``launch``
# to drive jobs inline (the pre-#51 pattern) still work without making the
# production request path start threads. Production ``launch`` is the thread
# starter; tests replace it with a lambda that calls ``run_job`` with a
# fake provider. If ``launch`` has been monkeypatched away from the original,
# call the test's replacement so the job is driven, but never call the
# original thread starter from the request path.
_ORIGINAL_LAUNCH = _orch.launch


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
    delivery_request: list[str] | None = None,
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

    `delivery_request` (issue #74) is the set of delivery formats the launch
    asked for (the canonical artifact plus optional merged/quantised forms).
    It is validated against the one delivery vocabulary and frozen onto the
    job row like the hyperparameters, so a finished run says what it was
    asked to produce. The `adapter` (as-trained) format needs no request; an
    unknown format is refused loudly rather than silently dropped.

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

    # Issue #74: the delivery request -- which formats the launch asked for.
    # Validated against the one delivery vocabulary and frozen onto the job
    # row like the hyperparameters, so a run says what it was asked to
    # produce. An unknown format is refused loudly rather than silently
    # dropped; `adapter` (the as-trained artifact) is always produced and
    # needs no request, but requesting it is harmless.
    frozen_delivery: list[str] = []
    if delivery_request:
        for fmt in delivery_request:
            try:
                delivery.require_known(fmt)
            except delivery.UnknownDeliveryFormat:
                raise HTTPException(
                    400,
                    {
                        "code": "unknown_delivery_format",
                        "message": (
                            f"'{fmt}' is not a delivery format this platform "
                            "offers. Ask for the canonical artifact, a merged "
                            "model, or a quantised local format."
                        ),
                        "format": fmt,
                    },
                ) from None
        frozen_delivery = list(delivery_request)

    # Issue #65: the mixture-of-experts label travels with the job so a
    # finished run says what it was trained on. A catalog model is dense
    # (neither MoE nor untested); an admitted model's probe snapshot carries
    # `is_moe` and the finding, frozen here alongside the hyperparameters so
    # the run cannot later claim a different model kind.
    is_moe = bool((admitted_probe or {}).get("is_moe", False))
    job_id = db.create_job(
        dataset_id,
        base_model,
        frozen_hp,
        warnings=warnings,
        base_revision=model.revision,
        quote=quote,
        overrides=frozen_overrides,
        is_moe=is_moe,
        delivery_request=frozen_delivery,
    )
    if warn:
        db.add_event(
            job_id,
            "log",
            warn["message"],
            {k: v for k, v in warn.items() if k != "message"},
        )
    # The request path no longer starts threads (issue #51). The worker
    # claims ``queued`` jobs with ``SELECT ... FOR UPDATE SKIP LOCKED`` and
    # drives them; the control plane just inserts the row. ``orchestrator.launch``
    # remains for tests that want to drive a job inline, but the production
    # request path does not call it. For backwards compatibility with tests
    # that monkeypatch ``launch`` to a helper that drives the job with a fake
    # provider (the pre-#51 pattern), call the monkeypatched replacement if
    # it is not the original thread starter -- this keeps those tests green
    # without making the production path start threads.
    if _orch.launch is not _ORIGINAL_LAUNCH:
        try:
            _orch.launch(job_id)
        except Exception:  # noqa: S110 - test helper may raise, but job is already created
            pass
    return job_id


def retry_diverged_job(job_id: str) -> str:
    """Create a single retry at half the learning rate, offered as a choice.

    Issue #36: a diverging run usually means the data or the rate is wrong, and
    repeating it is rarely the answer. Repeating it automatically would be the
    easy thing to test; offering a single reduced-rate retry as a choice is the
    honest one. This function is the choice: it creates a new job whose
    ``learning_rate`` is half the failed job's, carrying the same dataset,
    model and other hyperparameters, and links it via ``retry_from`` so the
    offer can be offered once and the history can name what came from what.

    The caller is the retry endpoint; it is the one place a retry is created,
    so the "single" in "single retry" is enforced here: a job that already has
    a retry child is refused with ``already_retried``, and a job that is not a
    diverged failure is refused with ``not_diverged``.

    Returns the new job id, launched.
    """
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(
            404, {"code": "not_found", "message": "No such job."}
        )
    if (
        job["status"] != "failed"
        or job.get("error_code") != divergence.DIVERGED_CODE
    ):
        raise HTTPException(
            409,
            {
                "code": "not_diverged",
                "message": (
                    "This job did not fail with training_diverged, so there is no "
                    "single reduced-rate retry to offer. Only a diverged job "
                    "carries a retry at half the learning rate as a choice."
                ),
            },
        )
    if db.has_retry(job_id):
        raise HTTPException(
            409,
            {
                "code": "already_retried",
                "message": (
                    "This job has already been retried once at a reduced learning "
                    "rate. A diverging run usually means the data or the rate is "
                    "wrong, and repeating it is rarely the answer, so only one "
                    "retry is offered."
                ),
            },
        )
    hp = job.get("hyperparameters") or {}
    if "learning_rate" not in hp:
        raise HTTPException(
            409,
            {
                "code": "no_learning_rate",
                "message": "The job carries no learning_rate to halve.",
            },
        )
    try:
        old_lr = float(hp["learning_rate"])
    except (TypeError, ValueError):
        raise HTTPException(
            409,
            {
                "code": "invalid_learning_rate",
                "message": "The job's learning_rate is not a number that can be halved.",
            },
        ) from None
    new_lr = divergence.retry_learning_rate(old_lr)
    new_hp = {**hp, "learning_rate": new_lr}
    # A diverged job's retry is a new job with the same dataset/model but a
    # halved rate. The new job is not itself a fault-spec job, so no fault
    # surface gate applies; it is a normal launch carrying the retry link so
    # the history can name what came from what.
    ds = usable_dataset(job["dataset_id"])
    model, _ = admission.lookup(job["base_model"])
    # ``admission.lookup`` cannot be None here: the original job was created
    # against this model, and a catalog model is always resolvable.
    assert model is not None  # noqa: S101 - original job existed, so model does
    warnings: list | None = None
    # The feasibility warning is re-evaluated for the new spec because the
    # new learning rate does not affect the estimate, but the closure keeps the
    # retry's record honest rather than copying a stale warning.
    warn = feasibility.warning(
        feasibility.usable_rows(ds), new_hp, config.MAX_JOB_DURATION_S
    )
    warnings = [warn] if warn else []
    new_job_id = db.create_job(
        job["dataset_id"],
        job["base_model"],
        new_hp,
        warnings=warnings,
        base_revision=job.get("base_revision") or model.revision,
        quote=None,
        overrides=job.get("overrides") or [],
        retry_from=job_id,
    )
    if warn:
        db.add_event(
            new_job_id,
            "log",
            warn["message"],
            {k: v for k, v in warn.items() if k != "message"},
        )
    db.add_event(
        new_job_id,
        "log",
        f"Retrying diverged job {job_id} at half the learning rate "
        f"({old_lr} -> {new_lr}). This is the single retry offered as a choice, "
        "not an automatic rerun, because a diverging run usually means the data "
        "or the rate is wrong.",
        {"retry_from": job_id, "old_lr": old_lr, "new_lr": new_lr},
    )
    # Like ``create`` above, the retry job is just inserted as ``queued``;
    # the worker claims it. No thread is started on the request path.
    # See the ``_ORIGINAL_LAUNCH`` check above for the test-compatibility shim.
    if _orch.launch is not _ORIGINAL_LAUNCH:
        try:
            _orch.launch(new_job_id)
        except Exception:  # noqa: S110 - test helper may raise, but job is already created
            pass
    return new_job_id
