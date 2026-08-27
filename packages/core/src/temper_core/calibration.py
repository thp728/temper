"""Prediction against measurement, so systematic error is visible.

Issue #77's comparison half. `temper_core.actuals.measure` records what
happened; this module puts it beside what was predicted and aggregates across
runs, so a systematically wrong estimate is visible rather than absorbed into
a better-looking average -- the acceptance criterion the aggregate exists for.
The sentence "calibrated against N real runs" only holds if the runs it is
true of are actually compared, which is what this module does.

The comparison is per metric -- duration, peak memory, cost -- and sets the
measured figure beside the predicted one:

* **ratio** is actual / predicted, where predicted is the midpoint of the
  quote's range for the ranged metrics (duration, cost) and the point for
  peak memory. A ratio of 2.0 means the estimate under-predicted by 2x; 0.5
  means it over-predicted by 2x. The ratio, not the raw difference, is what
  aggregates across runs that are not directly comparable (a 4B run and a
  70B run differ in absolute seconds; their *error ratios* are comparable).
* **direction** says where the actual landed relative to the prediction:
  `under`, `inside` or `over` the predicted range. Ranged metrics get a
  direction; peak memory is a point (the memory half of the predictor blocks
  rather than warns, so it was never quoted as a range) and gets a ratio
  without one.

The phase buckets reconcile the quote's phases (issue #72) with the job's
measured stages (`temper_core.actuals`). The mapping is the orchestrator's
own: `preparing` bundles SSH readiness, image pull and model download, so
those three quote phases together predict the `preparing` stage. Teardown is
not isolable from the terminal transition (it runs in the `finally`
immediately before the terminal state event) and is left out. This is the one
place the two vocabularies meet, so it is the one place they are mapped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

# Which of the quote's phases (issue #72) predict each measured stage
# (temper_core.actuals). The mapping follows where the machine actually spends
# the time: the orchestrator's `preparing` state spans the SSH wait and the
# archive pushes, and its `training` state -- entered as "Building image and
# training" -- spans the on-machine image build (the quote's `image_pull`),
# the model weights download that happens as the container loads, and the
# training itself. So the quote's `readiness` predicts `preparing`, and
# `image_pull` + `model_download` + `training` together predict the measured
# `training` stage. Teardown is not isolable from the terminal transition and
# is left out. This is the one place the two vocabularies meet, so it is the
# one place they are mapped.
PHASE_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("provisioning", ("provisioning",)),
    ("preparing", ("readiness",)),
    ("training", ("image_pull", "model_download", "training")),
)

Direction = Literal["under", "inside", "over"]


@dataclass(frozen=True)
class Run:
    """One terminal job as the aggregate compares it: identity for the row,
    the frozen quote for the prediction, and the measured actuals."""

    job_id: str
    base_model: str
    status: str
    created_at: float
    quote: dict[str, Any]
    actuals: dict[str, Any]


def midpoint(low: float | None, high: float | None) -> float | None:
    """The middle of a predicted range, or None when either end is missing.

    A range, never a point, is what the quote predicts (spec 005: the
    throughput figures are the softest numbers in the model); the midpoint is
    the single number a ratio against the actual needs.
    """
    if low is None or high is None:
        return None
    return (float(low) + float(high)) / 2.0


def ratio(actual: float | None, predicted: float | None) -> float | None:
    """Actual over predicted: >1 means under-predicted, <1 over-predicted.

    None when either side is missing -- a run that measured no duration or
    was quoted none contributes no ratio, so the aggregate never averages a
    guess in with a measurement.
    """
    if actual is None or predicted is None or predicted == 0:
        return None
    return float(actual) / float(predicted)


def direction(
    actual: float | None, low: float | None, high: float | None
) -> Direction | None:
    """Where the actual landed relative to the predicted range.

    `under` and `over` are the systematic-error signals: an estimate that
    always lands `over` (the actual always exceeds the predicted high end) is
    wrong in a particular, fixable way, and that is exactly what the
    aggregate must not absorb. None when the range or the actual is missing.
    """
    if actual is None or low is None or high is None:
        return None
    if actual < float(low):
        return "under"
    if actual > float(high):
        return "over"
    return "inside"


def _metric_compare(
    predicted_low: Any,
    predicted_high: Any,
    actual: Any,
) -> dict[str, Any]:
    """One ranged metric's comparison record (duration, cost).

    The direction is the three-way landing of the actual on the predicted
    range; the ratio is against the midpoint. The midpoint appears alongside
    so a consumer can show its working rather than trust the ratio.
    """
    low = (
        float(predicted_low)
        if isinstance(predicted_low, (int, float))
        else None
    )
    high = (
        float(predicted_high)
        if isinstance(predicted_high, (int, float))
        else None
    )
    act = float(actual) if isinstance(actual, (int, float)) else None
    mid = midpoint(low, high)
    return {
        "predicted_low": low,
        "predicted_high": high,
        "predicted_midpoint": mid,
        "actual": act,
        "ratio": ratio(act, mid),
        "direction": direction(act, low, high),
    }


def _point_compare(predicted: Any, actual: Any) -> dict[str, Any]:
    """One point metric's comparison record (peak memory).

    Peak memory blocks rather than warns, so it is predicted as a point, not
    a range (spec 005's "memory blocks, time and cost warn"); without a range
    there is no `inside`, and the ratio is the honest comparison.
    """
    pred = float(predicted) if isinstance(predicted, (int, float)) else None
    act = float(actual) if isinstance(actual, (int, float)) else None
    return {
        "predicted": pred,
        "actual": act,
        "ratio": ratio(act, pred),
    }


def compare(quote: dict[str, Any], actuals: dict[str, Any]) -> dict[str, Any]:
    """The prediction-vs-measurement record for one job.

    Three metrics, each comparing the frozen quote's figure against the
    measured actuals: duration and cost against their ranges, peak memory
    against its point. A figure missing on either side stays missing -- a
    failed job that measured no peak memory still compares its duration.
    """
    return {
        "duration": _metric_compare(
            quote.get("duration_low_s"),
            quote.get("duration_high_s"),
            actuals.get("duration_s"),
        ),
        "peak_memory": _point_compare(
            quote.get("peak_memory_gb"), actuals.get("peak_memory_gb")
        ),
        "cost": _metric_compare(
            quote.get("cost_low_minor"),
            quote.get("cost_high_minor"),
            actuals.get("cost_minor"),
        ),
    }


def _phase_predicted_s(
    quote: dict[str, Any], bucket: tuple[str, ...]
) -> float | None:
    """The summed predicted duration of `bucket`'s quote phases.

    Sums the phase ranges' midpoints: the quote prices each phase as a range,
    and a bucket's prediction is the sum of its phases' midpoints, matched
    against the measured stage's single duration.
    """
    total = 0.0
    any_value = False
    phases = {
        p.get("name"): p
        for p in quote.get("phases") or []
        if isinstance(p, dict)
    }
    for name in bucket:
        phase = phases.get(name) or {}
        low = phase.get("duration_low_s")
        high = phase.get("duration_high_s")
        mid = midpoint(
            float(low) if isinstance(low, (int, float)) else None,
            float(high) if isinstance(high, (int, float)) else None,
        )
        if mid is not None:
            total += mid
            any_value = True
    return total if any_value else None


def _phase_actual_s(actuals: dict[str, Any], name: str) -> float | None:
    """The measured duration of stage `name`, or None when it was never reached."""
    for p in actuals.get("stages") or []:
        if isinstance(p, dict) and p.get("name") == name:
            value = p.get("duration_s")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _stat(
    predicted_values: Sequence[float],
    actual_values: Sequence[float],
    ratios: Sequence[float],
) -> dict[str, Any]:
    """One metric's aggregate row: how many runs, what was predicted on
    average, what actually happened, and the spread of the error ratio.

    `min_ratio`/`max_ratio` are the bounds a systematically wrong estimate
    shows up in -- a run that landed at 4x the prediction is a visible point
    in the range, not a number absorbed into the mean.
    """
    return {
        "count": len(ratios),
        "mean_predicted": _mean(predicted_values),
        "mean_actual": _mean(actual_values),
        "mean_ratio": _mean(ratios),
        "min_ratio": min(ratios) if ratios else None,
        "max_ratio": max(ratios) if ratios else None,
    }


def aggregate(runs: Sequence[Run]) -> dict[str, Any]:
    """Predictions against measurements across `runs`.

    Each metric aggregates only the runs that have both a prediction and a
    measurement for it, and reports the count so a reader knows how many runs
    a figure rests on -- "calibrated against N real runs" is only as honest
    as N is visible. The phases aggregate the same way per bucket. The runs
    themselves ride along so an outlier can be named rather than pointed at.
    """
    metrics: dict[str, dict[str, Any]] = {
        "duration": {"predicted": [], "actual": [], "ratios": []},
        "peak_memory": {"predicted": [], "actual": [], "ratios": []},
        "cost": {"predicted": [], "actual": [], "ratios": []},
    }
    run_rows: list[dict[str, Any]] = []
    for run in runs:
        comparison = compare(run.quote, run.actuals)
        for name, bucket in (
            ("duration", "predicted_midpoint"),
            ("peak_memory", "predicted"),
            ("cost", "predicted_midpoint"),
        ):
            record = comparison[name]
            pred = record.get(bucket)
            act = record.get("actual")
            r = record.get("ratio")
            if (
                isinstance(pred, (int, float))
                and isinstance(act, (int, float))
                and isinstance(r, (int, float))
            ):
                metrics[name]["predicted"].append(float(pred))
                metrics[name]["actual"].append(float(act))
                metrics[name]["ratios"].append(float(r))
        run_rows.append(
            {
                "job_id": run.job_id,
                "base_model": run.base_model,
                "status": run.status,
                "created_at": run.created_at,
                "comparison": comparison,
            }
        )

    phase_rows: list[dict[str, Any]] = []
    for stage, quote_names in PHASE_BUCKETS:
        predicted_values: list[float] = []
        actual_values: list[float] = []
        ratios: list[float] = []
        for run in runs:
            pred = _phase_predicted_s(run.quote, quote_names)
            act = _phase_actual_s(run.actuals, stage)
            r = ratio(act, pred)
            if pred is not None and act is not None and r is not None:
                predicted_values.append(pred)
                actual_values.append(act)
                ratios.append(r)
        phase_rows.append(
            {
                "name": stage,
                "quotes_phases": list(quote_names),
                **_stat(predicted_values, actual_values, ratios),
            }
        )

    return {
        "count": len(runs),
        "metrics": {
            name: _stat(
                bucket["predicted"], bucket["actual"], bucket["ratios"]
            )
            for name, bucket in metrics.items()
        },
        "phases": phase_rows,
        "runs": run_rows,
    }
