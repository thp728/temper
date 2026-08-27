"""Measured ("actual") figures for one attempt, recorded against the prediction.

Issue #77's recording half. The duration and cost model rests on a handful of
measurements, and the mitigation is not a better constant, it is recording --
"calibrated against N real runs" is worth more than a better guess, but only if
recording starts with the first run rather than the last. This module turns a
job's own record into the measured half of that comparison, and the orchestrator
freezes the result onto the job row the moment the run ends (the mirror of the
quote frozen at launch), so no run is wasted even before anything consumes it.

Pure, no I/O, no clock: `measure` takes the job row, its state events and the
trainer's result document as arguments, which is what lets a test feed canned
events and a real run feed the database's -- and what keeps the recording
defensible under questioning, because the same function records both.

Three figures are measured and one is derived, and the difference is recorded,
not hidden (spec 005's "which numbers are measured and which are predicted"):

* **Duration** is measured twice: the wall total (`finished_at - created_at`,
  the job row's own timestamps) and a per-stage breakdown from the `state`
  events. The stages are the job's own -- provisioning, preparing, training,
  packaging -- and each stage's duration is the gap from its state event to
  the next one. These are deliberately *not* the quote's phases (issue #72):
  the quote prices provisioning, readiness, image pull, model download,
  training and teardown, while the machine can only be observed through the
  states it passes through (`preparing` bundles readiness, image pull and
  model download). Recording the observable stages and letting
  `temper_core.calibration` reconcile the two vocabularies in one place keeps
  this module honest about what it measured.
* **Peak memory** is measured on the machine by the trainer (`nvidia-smi`
  sampling during training, `apps/trainer/entrypoint.py`) and carried back in
  the run's result document. Absent -- honestly None, never a guess -- on a
  run that produced no result or a machine without `nvidia-smi`.
* **Cost** is derived, never measured: no code in this product reads a bill
  (spec 005's out-of-scope note). The derived figure is measured duration
  times the rate the job froze at launch (`price_per_hour`), converted into
  the currency's smallest unit exactly the way `temper_core.quote` does. The
  storage line is the same derivation against the separate USD rate
  (ADR-0030). Both are marked derived wherever they appear.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from temper_core import quote

# The states a job passes through, in order, whose duration the state events
# make measurable. `queued` and the terminal states frame them; teardown is
# not isolable from the terminal transition (it runs in the orchestrator's
# `finally`, immediately before the terminal state event) and is left out.
MEASURED_STAGES: tuple[str, ...] = (
    "provisioning",
    "preparing",
    "training",
    "packaging",
)

# Which currency is billed in the job's own rate, and its minor unit.
# Reused from the quote so the derivation cannot drift from the prediction it
# is compared against -- one definition of "paisa is a hundredth".
SECONDS_PER_HOUR = quote.SECONDS_PER_HOUR
USD_MINOR_UNIT = quote.MINOR_UNITS["USD"]


@dataclass(frozen=True)
class StageDuration:
    """One measured stage of a run: its name and how long the machine spent
    in it, from the state events. None when the stage was never reached (a
    job cancelled in provisioning has no `preparing` to measure)."""

    name: str
    duration_s: float | None


@dataclass(frozen=True)
class Actuals:
    """The measured half of issue #77's comparison, frozen at terminal.

    Every figure is marked by construction rather than by convention: the
    duration and peak fields are measured, the cost fields are derived from
    measured duration and the frozen rate. `currency` is the rate's currency,
    carried so a cost is never shown without the unit that gives it meaning.
    """

    duration_s: float | None
    peak_memory_gb: float | None
    cost_minor: int | None
    storage_cost_usd_minor: int | None
    currency: str | None
    # The measured stages are called stages, not phases, on purpose: the
    # quote prices *phases* (issue #72) and the machine passes through
    # *states* -- reconciling the two vocabularies is calibration's job, in
    # the one place they meet. Calling them the same word would hide that.
    stages: tuple[StageDuration, ...]


def _state_name(event: dict[str, Any]) -> str | None:
    """The state an event entered: its `data.state` where the control plane
    records it, else its message (the fallback a hand-built event or an older
    row provides)."""
    data = event.get("data")
    if isinstance(data, dict):
        value = data.get("state")
        if isinstance(value, str):
            return value
    message = event.get("message")
    return message if isinstance(message, str) else None


def _stage_durations(
    events: Sequence[dict[str, Any]],
) -> tuple[StageDuration, ...]:
    """Each measured stage's duration, from the job's state events.

    A stage's duration is the gap between its own state event and the next
    one: provisioning ends where preparing begins, and so on. Consecutive
    pairs are read in event order, so the timestamps are the state machine's
    own and a stage that was never entered (its event absent) or never left
    (no next event) records None rather than a number invented for it.
    """
    state_ts: list[tuple[str, float]] = []
    for e in events:
        if e.get("kind") != "state":
            continue
        name = _state_name(e)
        ts = e.get("ts")
        if name is None or not isinstance(ts, (int, float)):
            continue
        state_ts.append((name, float(ts)))

    durations: dict[str, float | None] = {
        name: None for name in MEASURED_STAGES
    }
    for (name, ts), (_, next_ts) in zip(state_ts, state_ts[1:], strict=False):
        if name in durations:
            durations[name] = next_ts - ts
    return tuple(
        StageDuration(name, durations[name]) for name in MEASURED_STAGES
    )


def _derived_cost_minor(
    duration_s: float | None,
    price_per_hour: float | None,
    currency: str | None,
) -> int | None:
    """The derived GPU cost: measured duration times the frozen hourly rate.

    Never a measured number -- nothing in the product reads a bill -- so it
    is always presented as derived from the measured duration. An unknown
    currency is refused rather than assumed, matching the quote's rule; a run
    with no duration or no rate derives nothing.
    """
    if duration_s is None or price_per_hour is None or currency is None:
        return None
    try:
        minor_unit = quote.minor_unit_for(currency)
    except ValueError:
        return None
    return round(duration_s / SECONDS_PER_HOUR * price_per_hour * minor_unit)


def measure(
    job: dict[str, Any],
    events: Sequence[dict[str, Any]],
    result: dict[str, Any] | None,
) -> Actuals:
    """The measured figures for one attempt, from the job's own record.

    `job` is the job row with at least `created_at`, `finished_at`,
    `price_per_hour`, `currency` and `storage_cost_usd_per_hour`; `events`
    are the state events the job's transitions appended; `result` is the
    trainer's result document (carrying the measured `peak_memory_gb`) or
    None when the run produced none.

    A job that never finished (no `finished_at`) measures no duration: the
    orchestrator only records actuals at a terminal state, so this is a guard
    rather than a path, but a function that would record a duration for a run
    that is still going is a function that lies.
    """
    finished = job.get("finished_at")
    created = job.get("created_at")
    duration_s: float | None = None
    if isinstance(finished, (int, float)) and isinstance(
        created, (int, float)
    ):
        duration_s = max(float(finished) - float(created), 0.0)

    peak_memory_gb: float | None = None
    if result:
        value = result.get("peak_memory_gb")
        if isinstance(value, (int, float)) and value > 0:
            peak_memory_gb = float(value)

    price = job.get("price_per_hour")
    currency = job.get("currency")
    storage_per_hour = job.get("storage_cost_usd_per_hour")
    cost_minor = _derived_cost_minor(
        duration_s,
        price if isinstance(price, (int, float)) else None,
        currency if isinstance(currency, str) else None,
    )
    storage_minor: int | None = None
    if (
        duration_s is not None
        and isinstance(storage_per_hour, (int, float))
        and storage_per_hour > 0
    ):
        storage_minor = round(
            duration_s
            / SECONDS_PER_HOUR
            * float(storage_per_hour)
            * USD_MINOR_UNIT
        )

    return Actuals(
        duration_s=duration_s,
        peak_memory_gb=peak_memory_gb,
        cost_minor=cost_minor,
        storage_cost_usd_minor=storage_minor,
        currency=currency if isinstance(currency, str) else None,
        stages=_stage_durations(events),
    )


def to_dict(actuals: Actuals) -> dict[str, Any]:
    """`actuals` as the dict the job row freezes and the API publishes."""
    return {
        "duration_s": actuals.duration_s,
        "peak_memory_gb": actuals.peak_memory_gb,
        "cost_minor": actuals.cost_minor,
        "storage_cost_usd_minor": actuals.storage_cost_usd_minor,
        "currency": actuals.currency,
        "stages": [
            {"name": p.name, "duration_s": p.duration_s}
            for p in actuals.stages
        ],
    }
