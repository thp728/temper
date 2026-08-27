"""Prediction against measurement -- `temper_core.calibration`.

The aggregate view's domain logic: put each run's frozen quote beside its
measured actuals (duration, peak memory, cost), and roll them up across runs so
a systematically wrong estimate is visible rather than absorbed into a
better-looking average. A good test asserts on the returned ratios and
directions, never on how a quote was produced -- the quote and the actuals are
inputs, and the seams that produce them are tested where they live.

Four properties are load-bearing and each is asserted directly:

* **The ratio is actual over predicted midpoint.** 2.0 means under-predicted
  by 2x; 0.5 over-predicted by 2x. The ratio, not the raw difference, is what
  aggregates across runs of different sizes.
* **Direction is the three-way landing of the actual on the predicted
  range** -- under, inside, over -- and `over` is the systematic-error signal
  the aggregate exists to surface.
* **A run with a missing figure contributes no ratio**, so the average never
  absorbs a guess in with a measurement.
* **The phase buckets reconcile the quote's phases with the job's measured
  stages** in exactly one place, mapping readiness + image_pull +
  model_download to the `preparing` stage (the orchestrator's own bundling).
"""

from __future__ import annotations

from temper_core import calibration
from temper_core.calibration import Run


def quote(**overrides) -> dict:
    """A frozen quote, shaped as the job row carries it."""
    base = {
        "currency": "INR",
        "minor_unit": 100,
        "dataset_id": "ds_abc",
        "dataset_created_at": 1000.0,
        "base_revision": "a" * 40,
        "token_count": None,
        "expires_at": 1000.0 + 86400,
        "phases": [
            {
                "name": "provisioning",
                "duration_low_s": 13,
                "duration_high_s": 17,
                "cost_low_minor": 15,
                "cost_high_minor": 20,
            },
            {
                "name": "readiness",
                "duration_low_s": 40,
                "duration_high_s": 68,
                "cost_low_minor": 46,
                "cost_high_minor": 78,
            },
            {
                "name": "image_pull",
                "duration_low_s": 87,
                "duration_high_s": 183,
                "cost_low_minor": 100,
                "cost_high_minor": 210,
            },
            {
                "name": "model_download",
                "duration_low_s": 9,
                "duration_high_s": 35,
                "cost_low_minor": 10,
                "cost_high_minor": 40,
            },
            {
                "name": "training",
                "duration_low_s": 80,
                "duration_high_s": 800,
                "cost_low_minor": 92,
                "cost_high_minor": 918,
            },
            {
                "name": "teardown",
                "duration_low_s": 5,
                "duration_high_s": 20,
                "cost_low_minor": 6,
                "cost_high_minor": 23,
            },
        ],
        "duration_low_s": 234,
        "duration_high_s": 1123,
        "cost_low_minor": 269,
        "cost_high_minor": 1289,
        "storage_cost_usd_per_hour": 0.0137,
        "storage_cost_usd_total_low_minor": 1,
        "storage_cost_usd_total_high_minor": 4,
        "is_estimate": True,
        "decisions": [],
        "peak_memory_gb": 5.4,
    }
    base.update(overrides)
    return base


def actuals(**overrides) -> dict:
    """Measured actuals, shaped as `temper_core.actuals.to_dict` produces."""
    base = {
        "duration_s": 300.0,
        "peak_memory_gb": 5.31,
        "cost_minor": 344,
        "storage_cost_usd_minor": 1,
        "currency": "INR",
        "phases": [
            {"name": "provisioning", "duration_s": 5.0},
            {"name": "preparing", "duration_s": 55.0},
            {"name": "training", "duration_s": 210.0},
            {"name": "packaging", "duration_s": 30.0},
        ],
    }
    base.update(overrides)
    return base


def run(job_id="job_a", q=None, a=None) -> Run:
    return Run(
        job_id=job_id,
        base_model="qwen3-4b",
        status="complete",
        created_at=1000.0,
        quote=q or quote(),
        actuals=a or actuals(),
    )


# --- the ratio and the direction ----------------------------------------------


def test_ratio_is_actual_over_predicted_midpoint():
    assert calibration.ratio(10.0, 5.0) == 2.0  # under-predicted by 2x
    assert calibration.ratio(2.5, 5.0) == 0.5  # over-predicted by 2x
    assert calibration.ratio(5.0, 5.0) == 1.0


def test_ratio_is_none_when_either_side_is_missing():
    assert calibration.ratio(None, 5.0) is None
    assert calibration.ratio(5.0, None) is None


def test_direction_is_the_landing_of_the_actual_on_the_range():
    assert calibration.direction(3.0, 10.0, 20.0) == "under"
    assert calibration.direction(15.0, 10.0, 20.0) == "inside"
    assert calibration.direction(30.0, 10.0, 20.0) == "over"


# --- compare ------------------------------------------------------------------


def test_compare_duration_uses_the_ranged_prediction():
    c = calibration.compare(quote(), actuals())["duration"]
    assert c["predicted_low"] == 234
    assert c["predicted_high"] == 1123
    # Midpoint of [234, 1123] = 678.5; actual 300 -> inside, ratio ~0.44.
    assert c["actual"] == 300.0
    assert c["direction"] == "inside"
    assert c["ratio"] == 300.0 / 678.5


def test_compare_peak_memory_uses_the_point_prediction():
    c = calibration.compare(quote(), actuals())["peak_memory"]
    assert c["predicted"] == 5.4
    assert c["actual"] == 5.31
    assert "direction" not in c  # a point has no range to land inside
    assert c["ratio"] == 5.31 / 5.4


def test_compare_cost_uses_the_ranged_prediction():
    c = calibration.compare(quote(), actuals())["cost"]
    assert c["predicted_low"] == 269
    assert c["predicted_high"] == 1289
    assert c["actual"] == 344
    assert c["direction"] == "inside"


def test_a_missing_actual_keeps_the_other_metrics_comparable():
    c = calibration.compare(quote(), actuals(peak_memory_gb=None))
    assert c["peak_memory"]["ratio"] is None
    assert c["duration"]["ratio"] is not None


# --- the phase buckets --------------------------------------------------------


def test_the_preparing_bucket_sums_readiness_image_pull_and_download():
    # Midpoints: (40+68)/2=54, (87+183)/2=135, (9+35)/2=22 -> 211 total.
    pred = calibration._phase_predicted_s(
        quote(), ("readiness", "image_pull", "model_download")
    )
    assert pred == 54 + 135 + 22


def test_phase_actual_reads_the_measured_stage():
    assert calibration._phase_actual_s(actuals(), "training") == 210.0
    assert calibration._phase_actual_s(actuals(), "teardown") is None


# --- aggregate ----------------------------------------------------------------


def test_aggregate_reports_counts_means_and_ratio_bounds():
    a1 = actuals(duration_s=678.5)  # exactly the midpoint: ratio 1.0
    a2 = actuals(duration_s=1357.0)  # twice the midpoint: ratio 2.0
    agg = calibration.aggregate([run("a", a=a1), run("b", a=a2)])
    duration = agg["metrics"]["duration"]
    assert duration["count"] == 2
    assert duration["mean_ratio"] == 1.5
    assert duration["min_ratio"] == 1.0
    assert duration["max_ratio"] == 2.0
    assert duration["mean_actual"] == (678.5 + 1357.0) / 2
    assert agg["count"] == 2


def test_a_run_without_a_measurement_contributes_no_ratio():
    agg = calibration.aggregate(
        [run("a"), run("b", a=actuals(peak_memory_gb=None))]
    )
    peak = agg["metrics"]["peak_memory"]
    assert peak["count"] == 1  # only the run that measured a peak counts
    duration = agg["metrics"]["duration"]
    assert duration["count"] == 2


def test_a_systematically_over_estimate_is_visible_in_the_ratio_bounds():
    # Every run lands far above the predicted high end: mean_ratio >> 1 and
    # min_ratio already above 1 -- the shape of a systematic error, not noise.
    high = actuals(duration_s=2000.0)
    agg = calibration.aggregate([run("a", a=high), run("b", a=high)])
    duration = agg["metrics"]["duration"]
    assert duration["min_ratio"] > 1.0
    assert duration["mean_ratio"] > 1.0


def test_aggregate_phase_rows_bucket_the_two_vocabularies():
    agg = calibration.aggregate([run()])
    names = [p["name"] for p in agg["phases"]]
    assert names == ["provisioning", "preparing", "training"]
    by_name = {p["name"]: p for p in agg["phases"]}
    # The preparing bucket's predicted is the summed midpoint (211), against
    # the measured preparing stage (55): over-predicted -> ratio < 1.
    assert by_name["preparing"]["mean_predicted"] == 211
    assert by_name["preparing"]["mean_actual"] == 55
    assert by_name["preparing"]["mean_ratio"] == 55 / 211


def test_aggregate_carries_the_runs_so_an_outlier_can_be_named():
    agg = calibration.aggregate([run("job_a"), run("job_b")])
    assert [r["job_id"] for r in agg["runs"]] == ["job_a", "job_b"]
    assert agg["runs"][0]["comparison"]["duration"]["actual"] == 300.0
