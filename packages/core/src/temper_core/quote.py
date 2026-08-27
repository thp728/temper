"""The quote: what a job is predicted to cost and how long it should take.

Spec 005's predictor, reduced to the half that warns rather than blocks. The
product has exactly one measured anchor per uncertain number (see each constant
below), so every figure here is a **range**, never a point, and the whole
object is labelled an estimate wherever it appears. Nothing here ever refuses
a launch: memory is the half that blocks (spec 005's "memory blocks, time and
cost warn"), and that arithmetic lives in `temper_core.memory` + the caller's
selection. This module prices time.

The phases are spec 005's list, in the order a job passes through them:
provisioning, readiness, image pull, model download, training, teardown. Cost
is composed **per phase** rather than as one blended rate, because the first
four are largely independent of the dataset: a cold start that is four minutes
on an 8 GB model and forty on a 140 GB one is the difference between a rounding
error and half the bill, and one blended rate hides exactly the number a user
most wants.

Two deliberate simplifications, both recorded rather than hidden:

* **Training duration is anchored to the measured row-pass throughput, not to
  tokens.** No tokens/second was ever measured (spike 4's "calibrate MFU" is
  still open); the one real anchor is 1.19 row-passes/s on 2026-08-19
  (`temper_core.feasibility`). The token count is carried on the quote -- cost
  is quoted per training token -- but producing it is Spec 006's job (#42), so
  the field is nullable and the training *duration* does not depend on it.
* **Storage is a separate USD line, not folded into the account-currency
  phases.** ADR-0030: storage bills separately from the GPU-hour and no live
  per-account storage price exists to convert the documented USD figure
  against, so it is exposed labelled rather than silently presented as though
  it were the account's currency.

Currency travels with the amount as a unit: costs are integers in the
currency's smallest unit (paisa for INR, cent for USD) with the currency code
alongside, never a formatting choice applied after the fact. An unknown
currency is refused, not defaulted -- the config.py rule that a value which
cannot be honoured must not be silently assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from temper_core import disk, feasibility
from temper_core.models import ModelFacts

# --- the phases, in the order a job passes through them ----------------------
# Names are spec 005's. `model_download` is the phase the issue singles out:
# it must use the measured rate so a large model's cold start is visible
# rather than buried in one blended number.
PHASES: tuple[str, ...] = (
    "provisioning",
    "readiness",
    "image_pull",
    "model_download",
    "training",
    "teardown",
)

# --- measured phase durations (seconds), each a (low, high) range -----------
# provisioning: spike 1 measured "VM reaches Running in 15-17s"; spike 5's
# create_seconds was 16.2; spike 2's cold-start split used 13s create. The
# range brackets those.
PROVISIONING_S: tuple[float, float] = (13.0, 17.0)

# readiness: spike 1 measured SSH refusing for a further ~42s after Running
# (usable at roughly T+60s); spike 5's ssh_ready_seconds was 68. Both ends are
# measured, so both are quoted.
READINESS_S: tuple[float, float] = (40.0, 68.0)

# image pull / build: spike 4 measured both ends of the spread -- 183s on
# 2026-08-18 and 87s on 2026-08-19 -- "the spread is pull throughput, measured
# at both ends". The technical-architecture calls the cold start "two to four
# minutes ... 87-183s image build/pull", and the quote must carry that range.
IMAGE_PULL_S: tuple[float, float] = (87.0, 183.0)

# teardown: spike 5's teardown trace confirmed absence at 1s, 21s and 42s
# (destroy + independent confirmation). A few seconds to ~20s brackets what
# was observed. Not as precisely measured as the phases above; labelled a
# range rather than a point for exactly that reason.
TEARDOWN_S: tuple[float, float] = (5.0, 20.0)

# --- model download ----------------------------------------------------------
# spike 5 measured Qwen3-8B: 16.4 GB in 45s = 364 MB/s mean (2.7s per GB),
# bursty -- window rates 0-730 MB/s, coefficient of variation 0.59. A point
# estimate from a bursty mean would oscillate; the quote derives a range from
# the measured mean and its measured spread instead.
DOWNLOAD_RATE_MBPS_MEAN = 364.0
DOWNLOAD_RATE_COV = 0.59  # measured, spike 5

# --- training ---------------------------------------------------------------
# The softest number in the model, quoted as a wide range. feasibility's
# ROWS_PER_SECOND is the one measured anchor (1.19 row-passes/s, 2026-08-19).
# The spec's risk note says the published range for the underlying efficiency
# figure "spans nearly an order of magnitude across workloads" -- so the
# duration range is widened by sqrt(10) in each direction, a full order of
# magnitude, DERIVED from that stated published spread rather than invented.
# Tightens as calibration (#77) records real runs.
TRAINING_DURATION_LOW_FACTOR = 1.0 / math.sqrt(10.0)
TRAINING_DURATION_HIGH_FACTOR = math.sqrt(10.0)

# --- currency -----------------------------------------------------------------
# The smallest unit per currency, as an integer: paisa for INR, cent for USD.
# A currency the quote cannot price in is refused (see the module docstring).
MINOR_UNITS: dict[str, int] = {"INR": 100, "USD": 100}

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class PhaseEstimate:
    """One phase's predicted duration and GPU cost, both as ranges."""

    name: str
    duration_low_s: float | None
    duration_high_s: float | None
    cost_low_minor: int | None
    cost_high_minor: int | None


@dataclass(frozen=True)
class Quote:
    """The whole prediction, labelled an estimate.

    Duration is a range, never a point. Cost is an integer in the currency's
    smallest unit with the currency alongside. The quote pins what it was
    computed against (`dataset_id`/`dataset_created_at`, `base_revision`), so
    it cannot outlive its inputs, and `expires_at` says how long the prices
    and availability it reflects are honoured for.
    """

    currency: str
    minor_unit: int
    dataset_id: str
    dataset_created_at: float
    base_revision: str
    token_count: int | None
    expires_at: float
    phases: tuple[PhaseEstimate, ...]
    duration_low_s: float
    duration_high_s: float
    cost_low_minor: int
    cost_high_minor: int
    storage_cost_usd_per_hour: float
    storage_cost_usd_total_low_minor: int
    storage_cost_usd_total_high_minor: int
    is_estimate: bool = True


def minor_unit_for(currency: str) -> int:
    """The smallest unit `currency` is priced in, or a refusal.

    An unknown currency is never silently assumed to price like a known one --
    misreporting every figure by a made-up factor is worse than asking.
    """
    try:
        return MINOR_UNITS[currency]
    except KeyError:
        raise ValueError(
            f"Temper cannot quote in '{currency}'; it prices in one of "
            f"{sorted(MINOR_UNITS)}. Read the account's currency, do not "
            f"assume one."
        ) from None


def _download_duration_s(facts: ModelFacts) -> tuple[float, float]:
    """The model-download phase as a duration range, from the measured rate.

    The payload is the weights at their download precision (always bf16,
    `disk.WEIGHTS_DOWNLOAD_BYTES_PER_PARAM`) -- the same bytes that cross the
    wire, priced by the measured rate. The rate range is the measured mean
    plus/minus one measured coefficient of variation; duration is size over
    rate, so the fast end of the range comes from the fast end of the rate.
    Sizes and the rate are both decimal (1e6 bytes per MB), matching spike 5's
    own `16397486450 bytes / 45.1s = 364 MB/s`.
    """
    size_mb = facts.params * disk.WEIGHTS_DOWNLOAD_BYTES_PER_PARAM / 1e6
    rate_low = DOWNLOAD_RATE_MBPS_MEAN * (1.0 - DOWNLOAD_RATE_COV)
    rate_high = DOWNLOAD_RATE_MBPS_MEAN * (1.0 + DOWNLOAD_RATE_COV)
    return (size_mb / rate_high, size_mb / rate_low)


def _cost_minor(
    duration_low_s: float,
    duration_high_s: float,
    price_per_hour: float,
    minor_unit: int,
) -> tuple[int, int]:
    """A phase's GPU cost range, as integers in the currency's smallest unit."""
    low = round(
        duration_low_s / SECONDS_PER_HOUR * price_per_hour * minor_unit
    )
    high = round(
        duration_high_s / SECONDS_PER_HOUR * price_per_hour * minor_unit
    )
    return (low, high)


def estimate(
    facts: ModelFacts,
    *,
    usable_rows: int,
    token_count: int | None,
    hyperparameters: dict[str, Any],
    price_per_hour: float,
    currency: str,
    storage_cost_usd_per_hour: float,
    dataset_id: str,
    dataset_created_at: float,
    base_revision: str,
    expires_at: float,
    training_duration_s: float | None = None,
) -> Quote:
    """The quote for one configuration, computed from its inputs alone.

    `price_per_hour` is the selected hardware's total hourly rate in the
    account's currency (the sum across devices), `storage_cost_usd_per_hour`
    is the disk's separate USD line (ADR-0030). `training_duration_s` lets a
    caller reuse `temper_core.feasibility.estimated_duration_s`; when None it
    is derived here from `usable_rows` and the hyperparameters, and when the
    volume cannot bound the run (`max_steps` set) the training phase is
    reported as not estimable rather than guessed.

    Pure arithmetic, no I/O, no clock -- `expires_at` and `dataset_created_at`
    are supplied so the arithmetic can be tested without a wall clock.
    """
    minor = minor_unit_for(currency)

    provisioning_low, provisioning_high = PROVISIONING_S
    readiness_low, readiness_high = READINESS_S
    image_pull_low, image_pull_high = IMAGE_PULL_S
    download_low, download_high = _download_duration_s(facts)
    teardown_low, teardown_high = TEARDOWN_S

    if training_duration_s is None:
        training_duration_s = feasibility.estimated_duration_s(
            usable_rows, hyperparameters
        )
    training_low: float | None
    training_high: float | None
    if training_duration_s is not None:
        training_low = training_duration_s * TRAINING_DURATION_LOW_FACTOR
        training_high = training_duration_s * TRAINING_DURATION_HIGH_FACTOR
    else:
        training_low = training_high = None

    phase_durations: list[tuple[str, float | None, float | None]] = [
        ("provisioning", provisioning_low, provisioning_high),
        ("readiness", readiness_low, readiness_high),
        ("image_pull", image_pull_low, image_pull_high),
        ("model_download", download_low, download_high),
        ("training", training_low, training_high),
        ("teardown", teardown_low, teardown_high),
    ]

    phases: list[PhaseEstimate] = []
    total_low = 0.0
    total_high = 0.0
    cost_low = 0
    cost_high = 0
    for name, low, high in phase_durations:
        if low is None or high is None:
            phases.append(PhaseEstimate(name, None, None, None, None))
            continue
        total_low += low
        total_high += high
        cl, ch = _cost_minor(low, high, price_per_hour, minor)
        cost_low += cl
        cost_high += ch
        phases.append(PhaseEstimate(name, low, high, cl, ch))

    storage_low = round(
        storage_cost_usd_per_hour
        * (total_low / SECONDS_PER_HOUR)
        * MINOR_UNITS["USD"]
    )
    storage_high = round(
        storage_cost_usd_per_hour
        * (total_high / SECONDS_PER_HOUR)
        * MINOR_UNITS["USD"]
    )

    return Quote(
        currency=currency,
        minor_unit=minor,
        dataset_id=dataset_id,
        dataset_created_at=dataset_created_at,
        base_revision=base_revision,
        token_count=token_count,
        expires_at=expires_at,
        phases=tuple(phases),
        duration_low_s=total_low,
        duration_high_s=total_high,
        cost_low_minor=cost_low,
        cost_high_minor=cost_high,
        storage_cost_usd_per_hour=storage_cost_usd_per_hour,
        storage_cost_usd_total_low_minor=storage_low,
        storage_cost_usd_total_high_minor=storage_high,
    )
