"""The feasibility warning. Spec 002, issue #10.

The size limit and the duration ceiling answer different questions and do not
reconcile: 1 GB is within memory and far beyond what finishes in 24 hours.
Job creation warns while the user can still act on it. A warning, not a
refusal: the estimate is crude, and a wrong block is worse than a wrong
warning.

Two seams, per the spec's testing decisions:

* the estimate itself, as pure functions over numbers, no I/O, no framework,
  so it survives the Phase B migration unchanged;
* job creation, through the HTTP seam; a dataset that plainly cannot finish
  yields a warning **and a launched job**, never a refusal.
"""

import pytest

from temper_core import feasibility, hyperparams

# The control plane's ceiling, as a number rather than as an import. The
# domain takes the ceiling as an argument precisely so it does not have to
# know where the value comes from, and a pure package's tests reaching into
# an application to fetch one would undo that (ADR-0010).
CEILING_S = 24 * 60 * 60

# --- the estimate ------------------------------------------------------------


def test_throughput_is_the_measured_one_not_a_round_number():
    """Derived from the one real run of 2026-08-19: 64 rows x 3 epochs =
    192 row-passes in the measured 161.4s training phase (trainer/README).
    If this ever becomes a chosen number, the derivation comment in
    feasibility.py has rotted."""
    assert feasibility.ROWS_PER_SECOND == pytest.approx(192 / 161.4)


def test_estimate_is_rows_times_epochs_at_measured_throughput():
    est = feasibility.estimated_duration_s(usable_rows=100, hyperparams={})
    assert est == pytest.approx(300 / feasibility.ROWS_PER_SECOND)


def test_estimate_honours_a_num_epochs_override():
    est = feasibility.estimated_duration_s(
        usable_rows=100, hyperparams={"num_epochs": 1}
    )
    assert est == pytest.approx(100 / feasibility.ROWS_PER_SECOND)


def test_estimate_ignores_a_junk_num_epochs():
    """The trainer refuses bad overrides later; the estimate must not crash
    on one before that."""
    est = feasibility.estimated_duration_s(
        usable_rows=100, hyperparams={"num_epochs": "lots"}
    )
    assert est == pytest.approx(300 / feasibility.ROWS_PER_SECOND)


def test_no_estimate_when_max_steps_caps_the_run():
    """With max_steps set, run length is bounded by steps, not data volume --
    a data-volume estimate would be wrong by orders of magnitude, and no
    per-step throughput was ever measured. No estimate, so no warning."""
    assert (
        feasibility.estimated_duration_s(
            usable_rows=10**9, hyperparams={"max_steps": 10}
        )
        is None
    )


# --- the warning -------------------------------------------------------------


def test_usable_rows_prefers_the_validation_report():
    ds = {"report": {"usable_rows": 90}, "row_count": 100}
    assert feasibility.usable_rows(ds) == 90


def test_usable_rows_falls_back_and_never_guesses_upward():
    assert feasibility.usable_rows({"row_count": 100}) == 100
    assert feasibility.usable_rows({}) == 0


def test_warning_fires_when_estimate_exceeds_the_ceiling():
    w = feasibility.warning(
        usable_rows=10**6,
        hyperparams={},
        max_duration_s=CEILING_S,
    )
    assert w is not None
    assert w["code"] == "duration_feasibility"
    assert w["estimated_duration_s"] > CEILING_S


def test_warning_names_itself_an_estimate_wherever_it_would_be_shown():
    w = feasibility.warning(
        usable_rows=10**6,
        hyperparams={},
        max_duration_s=CEILING_S,
    )
    assert "estimate" in w["message"].lower()
    assert "measured" in w["message"].lower()


def test_no_warning_when_the_run_plausibly_fits():
    assert (
        feasibility.warning(
            usable_rows=500,
            hyperparams={},
            max_duration_s=CEILING_S,
        )
        is None
    )


def test_warning_at_exact_equality_does_not_fire():
    """The estimate is crude; a boundary value is inside its own error bar.
    Only plainly-over fires."""
    rows = int(
        CEILING_S
        * feasibility.ROWS_PER_SECOND
        / hyperparams.DEFAULTS["num_epochs"]
    )
    assert (
        feasibility.warning(
            usable_rows=rows,
            hyperparams={},
            max_duration_s=CEILING_S,
        )
        is None
    )
