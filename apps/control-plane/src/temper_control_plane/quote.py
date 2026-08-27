"""The quote, assembled from the seams that feed it.

`temper_core.quote.estimate` is the pure arithmetic: it turns a configuration
into a per-phase duration-and-cost prediction, and it must run without a
network or a GPU to be tested at all (spec 005's testing decisions). What it
cannot do is supply its own inputs -- those cross seams this package owns:

* **Model facts** come from the `models` seam (the same one `memory` and
  `selection` read, so the quote cannot drift from what actually launches).
* **Price and availability** come from the provider seam, read live -- the
  account's currency on every quote, never assumed.
* **Disk** is `temper_core.disk.required_disk`, and its storage line stays a
  separate USD figure per ADR-0030, never silently converted into the
  account's currency.

Everything here is an estimate, and on time and cost grounds nothing here
blocks a launch. If the provider has nothing free, or the disk cannot be
sized, or the model cannot be resolved, the answer is `None` -- no quote --
never a refusal, because time and cost are the half of the predictor that
warns (spec 005). Memory is the other half, and it blocks: on the
job-creation path (`quote_for_launch`, issue #54) a configuration the
`selection` search predicts will not fit any available card is refused with
the same arithmetic the search refused with (`configuration_does_not_fit`),
because being wrong about memory means an out-of-memory failure on a machine
the user is paying for. The estimate surfaces (GET/POST `/v1/quotes`) keep the
warn posture: they return None, they do not refuse.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from temper_core import (
    catalog,
    decisions,
    disk,
    feasibility,
    hyperparams,
    overrides,
    quote,
    selection,
)
from temper_core.models import Models

from . import config
from .provider import Provider

# The provider seam the quote is priced against. Defaults to the real one,
# built per call; tests replace this attribute with a `FakeProvider`, exactly
# as the model seam is replaced with `fake_models`, so the plan screen and job
# creation price quotes against a fake and never reach the account.
QUOTE_PROVIDER: Provider | None = None

# The model-facts seam the quote resolves through. Defaults to the real
# resolver; tests replace it with `fake_models.catalog_models()` so the quote
# path never reaches the network, mirroring how `main.MODELS` is replaced for
# the rest of the application.
QUOTE_MODELS: Models | None = None


class QuoteRefused(Exception):
    """A configuration *with overrides* cannot be honoured -- a refusal, not
    an absent estimate.

    The distinction is load-bearing (issue #79): an estimate that cannot be
    priced warns and never blocks (spec 005), but an override is the user
    asking for something specific, and a plan that could not honour it must
    say so with the arithmetic that refused it rather than silently showing
    nothing. `to_payload` is the detail a 400 returns: a stable code, a
    message, and the structured arithmetic where one applies."""

    def __init__(self, code: str, message: str, **fields: object) -> None:
        self.code = code
        self.message = message
        self.fields = fields
        super().__init__(message)

    def to_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, **self.fields}


def provider_for_quote() -> Provider | None:
    """The provider a quote is priced against, or None when none is available.

    A quote is an estimate that never blocks a launch, so an unreachable
    provider produces no quote -- never an error -- and the surfaces that show
    quotes render them absent instead.
    """
    if QUOTE_PROVIDER is not None:
        return QUOTE_PROVIDER
    try:
        from .provider import new_provider

        return new_provider()
    except Exception:  # noqa: BLE001 - no quote, never a broken page
        return None


def models_for_quote() -> Models:
    """The model-facts resolver the quote is computed against."""
    if QUOTE_MODELS is not None:
        return QUOTE_MODELS
    from .models import new_models

    return new_models()


def for_config(
    ds: dict,
    model: catalog.BaseModel,
    hyperparameters: dict,
    overrides_list: Sequence[overrides.Override] | None = None,
    *,
    refuse_unfittable: bool = False,
) -> dict[str, Any] | None:
    """The quote for one (dataset, model, hyperparameters, overrides)
    configuration, or None when it cannot be priced.

    With no overrides this never raises: the estimate warns, it does not block
    (spec 005). With overrides (issue #79) an override that cannot be
    honoured -- unknown decision or value, a configuration that does not fit,
    a disk below the need -- raises `QuoteRefused` (or the core
    `overrides.OverrideError`), because a plan that cannot honour what the
    user asked for must refuse rather than silently show nothing.

    `refuse_unfittable` is the memory half's refusal at job creation (issue
    #54): the caller is committing to a machine, and a configuration that
    fits no available card on memory grounds is refused with the arithmetic
    even when nothing was overridden -- being wrong about memory costs an
    out-of-memory failure on a machine the user is paying for, so memory
    blocks where time and cost only warn. The estimate surfaces (GET/POST
    `/v1/quotes`) leave it False: they warn, they do not refuse (spec 005).
    """
    p = provider_for_quote()
    if p is None:
        return None
    try:
        return build_quote(
            ds,
            model,
            hyperparameters,
            models=models_for_quote(),
            provider=p,
            now=time.time(),
            ttl_s=config.QUOTE_TTL_S,
            overrides_list=overrides_list,
            refuse_unfittable=refuse_unfittable,
        )
    except (QuoteRefused, overrides.OverrideError):
        raise
    except Exception:  # noqa: BLE001 - no quote, never a broken page
        return None


def quote_for_launch(
    dataset_id: str,
    base_model: str,
    hyperparameters: dict,
    overrides_list: Sequence[overrides.Override] | None = None,
) -> dict[str, Any] | None:
    """The quote for the configuration a launch commits to, or None.

    Shared by the JSON API and the browser's launch form so the two creation
    surfaces freeze the same thing (the repo's one-creation-path rule). It
    resolves the dataset and model through the same seams the launch itself
    applies, then prices them. Refusals inside `jobs.usable_dataset` and
    `catalog.get` are the caller's; this helper only prices, and a
    configuration that cannot be priced produces no quote rather than a
    broken launch. An override that cannot be honoured raises, for the caller
    to refuse the launch.

    This is the job-creation path, so the memory half blocks here (issue
    #54): a configuration predicted not to fit any available card is refused
    with `configuration_does_not_fit` and the arithmetic that refused it even
    when nothing was overridden -- the launch refuses rather than discovering
    an out-of-memory failure on a machine the user is paying for. A
    configuration that *fits* but is not currently free still produces no
    quote, never a refusal: that is a fact about the moment, and refusing it
    would block work that could run once hardware frees.
    """
    from . import jobs

    try:
        ds = jobs.usable_dataset(dataset_id)
    except Exception:  # noqa: BLE001 - no quote, never a broken launch
        return None
    m = catalog.get(base_model)
    if m is None:
        return None
    return for_config(
        ds, m, hyperparameters, overrides_list, refuse_unfittable=True
    )


def _no_fit_refusal(err: selection.NoFittingHardwareError) -> QuoteRefused:
    """The refusal for hardware that cannot be honoured, with the arithmetic
    that refused it where the refusal was on memory grounds.

    `configuration_does_not_fit` is the memory half refusing with the same
    `memory.headroom_gb` numbers that did the refusing (spec 005: memory
    blocks, and an override must not lose that safety property by taking
    control). A configuration that would fit but is simply not free is the
    availability half, `provider_capacity_unavailable`, matching the code the
    orchestrator already uses for it at provisioning.

    The arithmetic itself rides on `selection.NoFittingHardwareError` -- the
    search owns its numbers, for a pinned configuration (#79) and for an
    unpinned one with a card free that nothing fits (issue #80) -- so this
    only renames it for the HTTP boundary.

    On a memory refusal the message names what the user can change (spec
    005's user story: "a smaller model, a shorter sequence, or a different
    method"). The lighter-method lever is dropped when the refused method is
    already the lightest that trains (qlora): recommending a heavier one
    would mislead. Whether each lever helps is the refusal's own arithmetic
    (fewer params shrink every pool, a shorter sequence shrinks activations,
    a lighter method shrinks weights and the trainable set) -- named, not
    hidden, and the same three levers the issue itself names.
    """
    arithmetic: dict[str, object] | None = None
    if err.peak_gb is not None:
        arithmetic = {
            "method": err.method,
            "gpu_type": err.gpu_type,
            "device_count": err.device_count,
            "peak_gb": err.peak_gb,
            "capacity_gb": err.capacity_gb,
            "shortfall_gb": err.shortfall_gb,
        }
        code = (
            "configuration_does_not_fit"
            if err.is_memory_refusal
            else "provider_capacity_unavailable"
        )
    else:
        code = "provider_capacity_unavailable"
    if code == "configuration_does_not_fit":
        if err.method == selection.EXECUTABLE_METHODS[0]:
            guidance = (
                " To make it fit, choose a smaller model or a shorter "
                "sequence."
            )
        else:
            guidance = (
                " To make it fit, choose a smaller model, a shorter "
                "sequence, or a lighter training method such as qlora."
            )
    else:
        guidance = ""
    return QuoteRefused(code, str(err) + guidance, arithmetic=arithmetic)


def build_quote(
    dataset: dict,
    model: catalog.BaseModel,
    hyperparameters: dict,
    *,
    models: Models,
    provider: Provider,
    now: float,
    ttl_s: float,
    overrides_list: Sequence[overrides.Override] | None = None,
    refuse_unfittable: bool = False,
) -> dict[str, Any] | None:
    """The quote for one (dataset, model, hyperparameters, overrides)
    configuration.

    Returns a dict matching `temper_control_plane.contracts_models.Quote`, or
    None when the configuration cannot be priced -- nothing fits available
    hardware, the disk exceeds the ceiling, or the model cannot be resolved.
    None never blocks: the caller decides what a missing quote means for the
    surface it is rendering, and a launch is never refused for one.

    `overrides` (issue #79) pin one or more of the predictor's decisions; the
    rest recompute around them in this one function -- the same seams the
    unpinned plan reads (`selection` for hardware, `disk` for disk, `memory`
    for the fit) -- so a change to one decision can never leave a stale value
    beside it. An override that cannot be honoured raises `QuoteRefused`
    (hardware/disk arithmetic) or the core `overrides.OverrideError`
    (vocabulary, the precision-method coupling): the launch is refused with
    the same arithmetic, never silently fallen back to the predictor's pick.

    `refuse_unfittable` (issue #54) makes the memory half refuse even when
    nothing was demanded: the caller is committing to a machine, and a
    configuration predicted not to fit any available card on memory grounds
    (`configuration_does_not_fit`) is refused with the arithmetic rather than
    silently unpriced. Availability alone (it fits, but nothing is free) still
    returns None -- a refusal there would block work that could run once
    hardware frees, which is the over-eager refusal this issue warns against.
    """
    try:
        facts = models.resolve(model.repo, model.revision)
    except Exception:
        # A model that cannot be resolved today is one that cannot be quoted
        # today. The launch path resolves it again with the same outcome and
        # its own error handling; here the honest answer is no quote.
        return None

    base_hp = hyperparams.effective(hyperparameters)
    resolved = (
        overrides.resolve(base_hp, overrides_list) if overrides_list else None
    )
    hp = resolved.hyperparameters if resolved is not None else base_hp
    # A configuration is a *demand* -- and an infeasible one is refused rather
    # than silently unpriced -- when the caller pinned a plan decision (#79) or
    # an advanced hyperparameter (issue #80). Taking control of the surface
    # must not lose the safety property by falling back to a null estimate: the
    # user asked for something specific, and a plan that cannot honour it says
    # so with the arithmetic that refused it.
    demanded = bool(overrides_list) or bool(hyperparameters)

    availability = provider.gpu_availability()
    try:
        plan = selection.select_hardware(
            facts,
            lora_r=hp["lora_r"],
            sequence_len=hp["sequence_len"],
            micro_batch_size=hp["micro_batch_size"],
            availability=availability,
            currency=provider.currency(),
            method=resolved.method if resolved is not None else None,
            gpu_type=resolved.gpu_type if resolved is not None else None,
            device_count=resolved.device_count
            if resolved is not None
            else None,
        )
    except selection.NoFittingHardwareError as e:
        # Memory blocks whether or not the user demanded the configuration
        # (issue #54): an unpinned job that fits no card is refused at creation
        # with the same arithmetic the search refused with, not launched to
        # discover the OOM on a machine it is paying for. Availability alone --
        # it fits, but nothing is free -- keeps the warn posture: that is a
        # fact about the moment, not about the configuration.
        if demanded or (refuse_unfittable and e.is_memory_refusal):
            raise _no_fit_refusal(e) from e
        return None
    try:
        disk_plan = disk.required_disk(
            facts,
            method=plan.method,
            lora_r=hp["lora_r"],
            retained_checkpoints=hp["save_total_limit"],
            provisioned_gb=resolved.disk_gb if resolved is not None else None,
        )
    except disk.DiskBelowNeedError as e:
        raise QuoteRefused(
            "disk_below_need",
            str(e),
            required_gb=e.required_gb,
            requested_gb=e.requested_gb,
        ) from e
    except disk.DiskBelowMinimumError as e:
        raise QuoteRefused(
            "disk_below_minimum",
            str(e),
            requested_gb=e.requested_gb,
            minimum_gb=e.minimum_gb,
        ) from e
    except disk.DiskExceedsCeilingError as e:
        if demanded:
            raise QuoteRefused(
                "disk_exceeds_ceiling",
                str(e),
                required_gb=e.required_gb,
                ceiling_gb=e.ceiling_gb,
                shortfall_gb=e.shortfall_gb,
            ) from e
        return None

    token_count = (dataset.get("report") or {}).get("token_count")
    q = quote.estimate(
        facts,
        usable_rows=feasibility.usable_rows(dataset),
        token_count=token_count if isinstance(token_count, int) else None,
        hyperparameters=hp,
        price_per_hour=plan.price_per_hour,
        currency=plan.currency,
        storage_cost_usd_per_hour=disk_plan.storage_cost_usd_per_hour,
        dataset_id=dataset["id"],
        dataset_created_at=dataset["created_at"],
        base_revision=model.revision,
        expires_at=now + ttl_s,
        # The reasons ride with the prediction (issue #76): computed from the
        # same seams the quote reads, frozen with it, never regenerated on
        # read -- a finished job explains itself like a planned one. The
        # `overridden` marks (issue #79) travel on the same records.
        decisions=decisions.decide(
            facts,
            hyperparameters=hp,
            plan=plan,
            disk_plan=disk_plan,
            overridden=(
                resolved.overridden if resolved is not None else frozenset()
            ),
        ),
    )
    return {
        "currency": q.currency,
        "minor_unit": q.minor_unit,
        "dataset_id": q.dataset_id,
        "dataset_created_at": q.dataset_created_at,
        "base_revision": q.base_revision,
        "token_count": q.token_count,
        "expires_at": q.expires_at,
        "phases": [
            {
                "name": p.name,
                "duration_low_s": p.duration_low_s,
                "duration_high_s": p.duration_high_s,
                "cost_low_minor": p.cost_low_minor,
                "cost_high_minor": p.cost_high_minor,
            }
            for p in q.phases
        ],
        "duration_low_s": q.duration_low_s,
        "duration_high_s": q.duration_high_s,
        "cost_low_minor": q.cost_low_minor,
        "cost_high_minor": q.cost_high_minor,
        "storage_cost_usd_per_hour": q.storage_cost_usd_per_hour,
        "storage_cost_usd_total_low_minor": q.storage_cost_usd_total_low_minor,
        "storage_cost_usd_total_high_minor": q.storage_cost_usd_total_high_minor,
        "is_estimate": q.is_estimate,
        # The predicted peak VRAM, from the same selection that chose the
        # card (issue #77 records it against the measured figure). A point,
        # not a range: memory is the half of the predictor that blocks.
        "peak_memory_gb": plan.peak.total_gb,
        "decisions": [decisions.to_dict(d) for d in q.decisions],
        # The legal values each decision's control can offer (issue #79): the
        # interface generates its controls from this rather than hand-listing
        # the vocabulary, so a value the server accepts is a value the plan
        # offers and vice versa.
        "override_options": {
            d: options
            for d in overrides.DECISIONS
            if (options := overrides.options_for(d)) is not None
        },
    }
