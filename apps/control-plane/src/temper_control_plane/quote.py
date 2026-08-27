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

Everything here is an estimate, and nothing here blocks a launch. If the
provider has nothing free, or the disk cannot be sized, or the model cannot be
resolved, the answer is `None` -- no quote -- never a refusal, because time and
cost are the half of the predictor that warns (spec 005).
"""

from __future__ import annotations

import time
from typing import Any

from temper_core import (
    catalog,
    disk,
    feasibility,
    hyperparams,
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
) -> dict[str, Any] | None:
    """The quote for one (dataset, model, hyperparameters) configuration, or
    None when it cannot be priced. Never raises: the estimate warns, it does
    not block (spec 005)."""
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
        )
    except Exception:  # noqa: BLE001 - no quote, never a broken page
        return None


def quote_for_launch(
    dataset_id: str, base_model: str, hyperparameters: dict
) -> dict[str, Any] | None:
    """The quote for the configuration a launch commits to, or None.

    Shared by the JSON API and the browser's launch form so the two creation
    surfaces freeze the same thing (the repo's one-creation-path rule). It
    resolves the dataset and model through the same seams the launch itself
    applies, then prices them. Refusals inside `jobs.usable_dataset` and
    `catalog.get` are the caller's; this helper only prices, and a
    configuration that cannot be priced produces no quote rather than a
    broken launch.
    """
    from . import jobs

    try:
        ds = jobs.usable_dataset(dataset_id)
    except Exception:  # noqa: BLE001 - no quote, never a broken launch
        return None
    m = catalog.get(base_model)
    if m is None:
        return None
    return for_config(ds, m, hyperparameters)


def build_quote(
    dataset: dict,
    model: catalog.BaseModel,
    hyperparameters: dict,
    *,
    models: Models,
    provider: Provider,
    now: float,
    ttl_s: float,
) -> dict[str, Any] | None:
    """The quote for one (dataset, model, hyperparameters) configuration.

    Returns a dict matching `temper_control_plane.contracts_models.Quote`, or
    None when the configuration cannot be priced -- nothing fits available
    hardware, the disk exceeds the ceiling, or the model cannot be resolved.
    None never blocks: the caller decides what a missing quote means for the
    surface it is rendering, and a launch is never refused for one.
    """
    try:
        facts = models.resolve(model.repo, model.revision)
    except Exception:
        # A model that cannot be resolved today is one that cannot be quoted
        # today. The launch path resolves it again with the same outcome and
        # its own error handling; here the honest answer is no quote.
        return None

    hp = hyperparams.effective(hyperparameters)
    try:
        plan = selection.select_hardware(
            facts,
            lora_r=hp["lora_r"],
            sequence_len=hp["sequence_len"],
            micro_batch_size=hp["micro_batch_size"],
            availability=provider.gpu_availability(),
            currency=provider.currency(),
        )
    except selection.NoFittingHardwareError:
        return None
    try:
        disk_plan = disk.required_disk(
            facts,
            method=plan.method,
            lora_r=hp["lora_r"],
            retained_checkpoints=hp["save_total_limit"],
        )
    except disk.DiskExceedsCeilingError:
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
    }
