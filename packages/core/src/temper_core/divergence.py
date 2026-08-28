"""Divergence detection against the measurements the platform already streams.

The brief says a run whose loss becomes meaningless burns its full duration and
hands back a worthless result. Detection uses the measurements the platform
already streams -- training loss as `metric` events -- against thresholds
published in the research rather than invented here.

Thresholds and their derivations live beside them (the way ADR-0036's dataset
ceiling records 21.6 MB/s times 60 seconds): a threshold with no derivation is
a magic number with a comment, and issue #36 exists to remove one.

Pure, no I/O, no framework imports -- domain logic that survives Phase B
unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Thresholds -- published, not invented.
# ---------------------------------------------------------------------------
#
# Derived from `docs/research-reports/report-b.md` Section 5.7
# "Divergence and NaN detection -- automatic abort":
#
# > Abort automatically if any of: **loss becomes NaN/Inf** (immediate abort --
# > unrecoverable without rollback); **loss increases >2x its trailing 50-step
# > average for >20 consecutive steps** (divergence); **grad_norm exceeds ~100
# > repeatedly** despite clipping (instability).
#
# And Section 6 "Reading a training run":
#
# > **Diverging:** loss rising, spiking, or NaN; grad_norm spiking to 10s-100s.
# > **Warn threshold:** loss > 2x trailing-average for 20 steps, or any NaN.
#
# The numbers below are those published figures, not choices made here:
#
# * **DIVERGENCE_MULTIPLIER = 2.0** -- the factor a loss must exceed its
#   trailing average by to count as diverging. Report-b: ">2x".
# * **DIVERGENCE_WINDOW = 50** -- the trailing steps the average is taken
#   over. Report-b: "trailing 50-step average".
# * **DIVERGENCE_CONSECUTIVE = 20** -- how many consecutive exceedances make
#   the run diverged. Report-b: "for >20 consecutive steps".
# * **WARNING_CONSECUTIVE = 5** -- instability short of divergence is the
#   *same* exceedance seen for fewer steps than a divergence. A run that has
#   spiked for 5 steps but not yet 20 is unstable, not yet diverged; it is
#   surfaced as a warning rather than an abort, exactly as the acceptance
#   criterion requires. 5 is chosen because a single spike is noise and 20 is
#   the proven divergence; 5 is one quarter of the divergence window, early
#   enough to warn while the user can still act.
#
# * **NaN / Inf is immediate divergence** -- Report-b's immediate abort, the
#   sophisticated rollback-to-100-steps is a v2 feature; for v1 we abort and
#   surface "training diverged -- try a lower learning rate."
#
# grad_norm is not streamed today (events.py deliberately does not promote it --
# it is diagnostic rather than progress), so the grad_norm >100 instability
# signal from the same section cannot be read without a second measurement path.
# The criterion says detection "uses the measurements the platform already
# streams", so this module reads only loss -- the second series (held_out_loss)
# is a separate signal owned by the overfitting detector (#53), not this one.
#
# All four are **configuration**: callers may override them, and the defaults
# are the published figures above. Changing a number changes the derivation
# comment here, not a silent literal elsewhere.
#
# ---------------------------------------------------------------------------

# The factor a loss must exceed its trailing average by.
DIVERGENCE_MULTIPLIER: float = 2.0
# How many trailing steps form the average.
DIVERGENCE_WINDOW: int = 50
# Consecutive exceedances that make the run diverged.
DIVERGENCE_CONSECUTIVE: int = 20
# Consecutive exceedances that make the run unstable (warning, not abort).
# Derived as one quarter of DIVERGENCE_CONSECUTIVE: early enough to warn while
# the user can still act, late enough that a single spike is not a warning.
WARNING_CONSECUTIVE: int = 5

# The stable code and the plain-language cause the abort carries.
# Every API error carries a code the caller can branch on; a human message
# alone forces callers to match on prose that will change. Both travel in the
# OrchestratorError that stops the job.
DIVERGED_CODE: str = "training_diverged"
DIVERGED_MESSAGE: str = (
    "Training diverged \u2014 the loss became meaningless (NaN or exploding). "
    "The run was stopped early so you are not billed for hours that cannot "
    "produce anything. Try a lower learning rate or check your data for "
    "issues. A single retry at half the learning rate is available as a "
    "choice \u2014 repeating the same rate is rarely the answer, because a "
    "diverging run usually means the data or the rate is wrong."
)

# Instability short of divergence is surfaced as a warning rather than an
# abort. It is the same exceedance seen for fewer steps than a divergence.
INSTABILITY_CODE: str = "training_instability"
INSTABILITY_MESSAGE: str = (
    "Training instability detected \u2014 loss is spiking well above its "
    "recent average. This may be early divergence; consider lowering the "
    "learning rate if it continues. The run is continuing, but the signal is "
    "worth watching."
)

# The reduced learning rate offered as a single retry choice. The research
# (report-b section 5.7) says "optionally auto-retrying once at half LR";
# the spec refines that to "offered as a choice rather than performed
# automatically" because a diverging run usually means the data or the rate is
# wrong. Halving is the published figure, not a choice made here.
RETRY_LR_FACTOR: float = 0.5


@dataclass(frozen=True)
class DivergenceResult:
    """What the detector thinks about the latest loss."""

    # "ok" | "warning" | "diverged"
    status: str
    # The stable code when status is not ok, else None.
    code: str | None = None
    # The plain-language sentence when status is not ok, else None.
    message: str | None = None
    # How many consecutive exceedances have been seen, for the caller to
    # decide whether to warn again or to record once.
    consecutive: int = 0


class DivergenceDetector:
    """Stateful detector over a single run's training-loss stream.

    Holds the loss history and the consecutive-exceedance counter. One
    detector per job, fed each training loss as it is streamed; it never
    looks at held-out loss (that series belongs to the overfitting detector).

    The detector reads the measurements the platform already streams (issue
    #53's `metric` events and the trainer's own NaN line), rather than adding
    a second measurement path.

    Non-finite loss (NaN / Inf) is immediate divergence -- unrecoverable
    without rollback, so v1 aborts rather than rolling back 100 steps and
    skipping the batch (the sophisticated recovery the report names as v2).

    A finite loss that exceeds ``multiplier * trailing_average(window)`` for
    ``consecutive`` steps in a row is divergence; the same exceedance for
    ``warning_consecutive`` steps is instability (warning, not abort). The
    trailing average excludes the current loss, because the current loss is
    what is being judged against where the run has been.
    """

    def __init__(
        self,
        multiplier: float = DIVERGENCE_MULTIPLIER,
        window: int = DIVERGENCE_WINDOW,
        consecutive: int = DIVERGENCE_CONSECUTIVE,
        warning_consecutive: int = WARNING_CONSECUTIVE,
    ) -> None:
        if multiplier <= 1:
            raise ValueError(
                "multiplier must be > 1; a loss must exceed its average to diverge"
            )
        if window < 1:
            raise ValueError("window must be >= 1")
        if consecutive < 1:
            raise ValueError("consecutive must be >= 1")
        if warning_consecutive < 1 or warning_consecutive >= consecutive:
            raise ValueError(
                "warning_consecutive must be >=1 and < consecutive"
            )
        self.multiplier = multiplier
        self.window = window
        self.consecutive = consecutive
        self.warning_consecutive = warning_consecutive
        self._losses: list[float] = []
        self._consecutive_exceed: int = 0
        self._warned: bool = False

    def observe(self, loss: float) -> DivergenceResult:
        """Feed one training loss and return the detector's verdict.

        Returns ``diverged`` immediately on a non-finite loss; otherwise
        returns ``diverged`` when the trailing average exists and the loss
        exceeds ``multiplier * average`` for ``consecutive`` steps in a row,
        ``warning`` when it has exceeded for ``warning_consecutive`` steps but
        not yet ``consecutive``, and ``ok`` otherwise.
        """
        # Immediate abort on a meaningless value -- the fault surface drives
        # loss to NaN, and the trainer's own check would see the same.
        if not math.isfinite(loss):
            return DivergenceResult(
                status="diverged",
                code=DIVERGED_CODE,
                message=DIVERGED_MESSAGE,
                consecutive=self.consecutive,
            )

        # Need enough history to form a trailing average that excludes this loss.
        if len(self._losses) < self.window:
            self._losses.append(loss)
            # Even before the window is full, a wildly exploding loss is worth
            # counting once the average exists over whatever we have: but the
            # spec's trailing-50 definition is honoured strictly so the detector
            # never fires on the warmup noise of the first few steps.
            return DivergenceResult(status="ok", consecutive=0)

        # Trailing average over the last `window` losses, excluding this one.
        trailing = sum(self._losses[-self.window :]) / self.window
        # A non-positive trailing average (loss should be positive) cannot be a
        # sensible denominator; treat it as not diverged rather than dividing by
        # zero or flipping the comparison.
        if trailing <= 0 or not math.isfinite(trailing):
            self._losses.append(loss)
            self._consecutive_exceed = 0
            self._warned = False
            return DivergenceResult(status="ok", consecutive=0)

        if loss > self.multiplier * trailing:
            self._consecutive_exceed += 1
        else:
            self._consecutive_exceed = 0
            self._warned = False

        self._losses.append(loss)

        if self._consecutive_exceed >= self.consecutive:
            return DivergenceResult(
                status="diverged",
                code=DIVERGED_CODE,
                message=DIVERGED_MESSAGE,
                consecutive=self._consecutive_exceed,
            )
        if (
            self._consecutive_exceed >= self.warning_consecutive
            and not self._warned
        ):
            self._warned = True
            return DivergenceResult(
                status="warning",
                code=INSTABILITY_CODE,
                message=INSTABILITY_MESSAGE,
                consecutive=self._consecutive_exceed,
            )
        return DivergenceResult(
            status="ok", consecutive=self._consecutive_exceed
        )

    def observe_many(self, losses: list[float]) -> DivergenceResult:
        """Feed many losses in order, returning the last verdict."""
        result = DivergenceResult(status="ok")
        for loss in losses:
            result = self.observe(loss)
            if result.status == "diverged":
                return result
        return result


def is_non_finite_loss_line(text: str) -> bool:
    """Whether ``text`` is a training loss line whose loss is non-finite.

    The trainer prints ``{'loss': nan, ...}`` when it diverges; the classifier
    deliberately does not promote non-finite numbers to metric events (a NaN is
    real information but not a point on a chart, and it survives in the log
    line), so the orchestrator sees it as a log line. This helper lets the
    detector read that line without a second measurement path -- the same loss
    the metric detector would have seen if it had been finite.

    The check is textual because the line has already been classified as log
    and its structured data is absent. It looks for a ``'loss'`` key and a
    non-finite token; ``inf`` and ``-inf`` are included because the trainer
    can produce either.
    """
    low = text.lower()
    if "'loss'" not in low and '"loss"' not in low:
        return False
    # A bare substring is enough: the line is ``{'loss': nan, ...}`` and the
    # classifier has already decided it is a loss line by the same key.
    return "nan" in low or "inf" in low


def retry_learning_rate(
    learning_rate: float, factor: float = RETRY_LR_FACTOR
) -> float:
    """The reduced learning rate offered as a single retry choice.

    Halving, as the research prescribes ("optionally auto-retrying once at
    half LR"), offered as a choice rather than performed automatically because
    a diverging run usually means the data or the rate is wrong and repeating
    it is rarely the answer.
    """
    return float(learning_rate) * factor
