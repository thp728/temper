"""The feasibility estimate: will this dataset plainly not finish in time?

Spec 002's gap, stated: **the size limit and the duration limit answer
different questions and do not reconcile.** One gigabyte is within memory and
far beyond what finishes in 24 hours. Without a warning, a user discovers the
gap hours later as a `gpu_max_duration_exceeded` kill on a machine they paid
for. This module is the interim acknowledgement of that gap; closing it with a
real predictor is Phase B's quoting work, which needs a memory and throughput
model anyway.

Deliberate properties:

* **Pure functions over numbers.** No I/O, no framework imports, no clock --
  the caller supplies the ceiling. Domain logic that survives the Phase B
  migration unchanged, per the repo rule.
* **A warning, never a refusal.** The estimate is crude and derived from one
  measured run; a wrong block is worse than a wrong warning. The job launches
  regardless of what this module says.
* **Labelled an estimate everywhere it appears.** The message says "estimate"
  and names its measured basis, so no client can mistake it for a quote.

Where the throughput number comes from -- the only real run through the
product, 2026-08-19 (trainer/README.md): 64 rows x 3 epochs = 192 row-passes
in a measured **161.4s** training phase, hence ~1.19 row-passes/s. That figure
includes model load and other fixed overhead, so it *understates* true
steady-state throughput and therefore *overstates* duration for large
datasets -- the warning errs toward firing, which is the right direction for
a mechanism whose failure modes are "spurious warning" (mild) and "paid-for
kill" (the thing this exists to prevent).
"""

from __future__ import annotations

# Measured, not chosen: 192 row-passes / 161.4s. See the module docstring
# before changing this number or its derivation.
ROWS_PER_SECOND = 192 / 161.4

# Mirrors trainer/entrypoint.py DEFAULTS["num_epochs"]. Duplicated rather than
# imported because the control plane does not import trainer code -- but the
# two must agree, and the trainer's default is the documented one.
DEFAULT_EPOCHS = 3

WARNING_CODE = "duration_feasibility"


def usable_rows(dataset: dict) -> int:
    """The row count that actually trains, from a dataset record.

    Validation reports usable rows; a record from before that field existed
    falls back to raw row count, and 0 if neither is present -- which yields
    no warning rather than a wrong one.
    """
    report = dataset.get("report") or {}
    n = report.get("usable_rows")
    if n is None:
        n = dataset.get("row_count") or 0
    return n


def estimated_duration_s(usable_rows: int, hyperparams: dict) -> float | None:
    """Estimated wall-clock seconds for training this many rows.

    Returns None when no data-volume estimate can be made: with `max_steps`
    set, run length is bounded by steps rather than rows, and no per-step
    throughput was ever measured -- estimating from volume there would be
    wrong by orders of magnitude.
    """
    if hyperparams.get("max_steps") is not None:
        return None
    try:
        epochs = float(hyperparams.get("num_epochs", DEFAULT_EPOCHS))
    except (TypeError, ValueError):
        epochs = DEFAULT_EPOCHS
    # The trainer refuses bad overrides later; until then the estimate simply
    # falls back to the default rather than crashing job creation.
    if epochs <= 0:
        epochs = DEFAULT_EPOCHS
    return usable_rows * epochs / ROWS_PER_SECOND


def _fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} hours"
    if seconds >= 60:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds:.0f} seconds"


def warning(usable_rows: int, hyperparams: dict,
            max_duration_s: float) -> dict | None:
    """A warning dict when the dataset plainly cannot finish inside the
    ceiling; None otherwise.

    Fires only when the estimate strictly exceeds the ceiling -- at equality
    the value is inside the estimate's own error bar, and "plainly cannot"
    is the standard, not "might not".
    """
    estimate = estimated_duration_s(usable_rows, hyperparams)
    if estimate is None or estimate <= max_duration_s:
        return None
    return {
        "code": WARNING_CODE,
        "message": (
            f"Estimated training time is {_fmt_duration(estimate)} -- an "
            f"estimate from measured throughput "
            f"({ROWS_PER_SECOND:.2f} row-passes/s, one real run), not a "
            f"quote -- which exceeds the maximum job duration of "
            f"{_fmt_duration(max_duration_s)}. The job will be stopped if it "
            f"reaches that ceiling. It is launching anyway; cancel it if you "
            f"would rather not pay for a run that may be killed."),
        "estimated_duration_s": round(estimate, 1),
        "max_duration_s": max_duration_s,
        "rows_per_second": round(ROWS_PER_SECOND, 3),
    }
