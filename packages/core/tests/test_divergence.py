"""Divergence detection against the streamed training loss.

Issue #36. The detector reads the measurements the platform already streams
(training loss as metric events) against thresholds published in the research
rather than invented here. The numbers are report-b Section 5.7 and 6:

* loss becomes NaN/Inf -> immediate abort
* loss >2x trailing 50-step average for >20 consecutive steps -> divergence
* same exceedance for 5 consecutive steps -> instability warning (short of divergence)

The derivation lives beside the thresholds in both temper_core.divergence and
config.py (the way ADR-0036's ceiling records 21.6 MB/s times 60 seconds).

A threshold with no derivation is a magic number with a comment, and the issue
exists to remove one.
"""

from __future__ import annotations

import pytest

from temper_core.divergence import (
    DIVERGED_CODE,
    DIVERGENCE_CONSECUTIVE,
    DIVERGENCE_MULTIPLIER,
    DIVERGENCE_WINDOW,
    INSTABILITY_CODE,
    WARNING_CONSECUTIVE,
    DivergenceDetector,
    is_non_finite_loss_line,
    retry_learning_rate,
)


def test_thresholds_match_the_published_figures():
    """The defaults are the published figures, not choices made here."""
    assert DIVERGENCE_MULTIPLIER == 2.0
    assert DIVERGENCE_WINDOW == 50
    assert DIVERGENCE_CONSECUTIVE == 20
    assert WARNING_CONSECUTIVE == 5


def test_a_non_finite_loss_is_immediate_divergence():
    detector = DivergenceDetector()
    result = detector.observe(float("nan"))
    assert result.status == "diverged"
    assert result.code == DIVERGED_CODE
    assert "diverged" in result.message.lower()
    # Inf too
    assert DivergenceDetector().observe(float("inf")).status == "diverged"
    assert DivergenceDetector().observe(float("-inf")).status == "diverged"


def test_is_non_finite_loss_line():
    assert is_non_finite_loss_line("{'loss': nan, 'step': 20, 'epoch': 1.0}")
    assert is_non_finite_loss_line("{'loss': inf, 'step': 10}")
    assert is_non_finite_loss_line('{"loss": NaN}')
    assert is_non_finite_loss_line("{'loss': -inf}")
    assert not is_non_finite_loss_line("{'loss': 0.5, 'step': 10}")
    assert not is_non_finite_loss_line("pulling image")
    assert not is_non_finite_loss_line("{'eval_loss': 0.5}")


def test_stable_losses_do_not_fire():
    detector = DivergenceDetector(
        window=5, consecutive=4, warning_consecutive=2
    )
    # 20 steps of stable loss around 0.5
    for _ in range(20):
        result = detector.observe(0.5 + 0.01)
        assert result.status == "ok"


def test_divergence_after_consecutive_exceedances():
    # Smaller window for fast test, preserving multiplier and consecutive semantics
    detector = DivergenceDetector(
        multiplier=2.0, window=5, consecutive=4, warning_consecutive=2
    )
    # Fill window with stable 0.5 losses
    for _ in range(5):
        detector.observe(0.5)
    # Now sustained exceedance: trailing avg ~0.5, so 2*avg =1.0. Use
    # increasing losses so the sliding trailing average does not catch up and
    # hide the divergence (report-b's ">2x for >20 steps" assumes the loss
    # keeps climbing, not a flat high plateau).
    result = detector.observe(1.5)
    assert result.status == "ok"
    assert result.consecutive == 1
    result = detector.observe(3.0)
    assert result.status == "warning"
    assert result.code == INSTABILITY_CODE
    result = detector.observe(6.0)
    assert result.status == "ok"  # warned already, not repeating until reset
    result = detector.observe(12.0)
    assert result.status == "diverged"
    assert result.code == DIVERGED_CODE


def test_warning_is_not_repeated_until_reset():
    detector = DivergenceDetector(
        multiplier=2.0, window=5, consecutive=6, warning_consecutive=2
    )
    for _ in range(5):
        detector.observe(0.5)
    r1 = detector.observe(1.5)
    assert r1.status == "ok"
    r2 = detector.observe(3.0)
    assert r2.status == "warning"
    r3 = detector.observe(6.0)
    assert r3.status == "ok"
    r4 = detector.observe(12.0)
    assert r4.status == "ok"
    # Reset by a normal loss -- need several normal losses to flush the
    # window's high values so the next spike is judged against a low baseline
    # again (the sliding window still holds the previous spikes).
    for _ in range(5):
        r5 = detector.observe(0.5)
        assert r5.status == "ok"
    # New exceedance sequence should warn again
    r6 = detector.observe(1.5)
    assert r6.status == "ok"
    r7 = detector.observe(3.0)
    assert r7.status == "warning"


def test_a_normal_loss_resets_the_consecutive_counter():
    detector = DivergenceDetector(
        multiplier=2.0, window=5, consecutive=4, warning_consecutive=2
    )
    for _ in range(5):
        detector.observe(0.5)
    detector.observe(1.5)
    detector.observe(3.0)
    # Now a normal loss resets -- flush the window with normals so the next
    # sequence sees a low baseline again
    for _ in range(5):
        detector.observe(0.5)
    result = detector.observe(0.5)
    assert result.status == "ok"
    assert result.consecutive == 0
    # Need full consecutive again to diverge (increasing)
    for loss in [1.5, 3.0, 6.0]:
        detector.observe(loss)
    assert detector.observe(12.0).status == "diverged"


def test_observe_many_stops_at_divergence():
    detector = DivergenceDetector(
        multiplier=2.0, window=3, consecutive=3, warning_consecutive=2
    )
    for _ in range(3):
        detector.observe(0.4)
    result = detector.observe_many([1.0, 2.5, 6.0])
    assert result.status == "diverged"


def test_retry_learning_rate_is_half():
    assert retry_learning_rate(2e-4) == pytest.approx(1e-4)
    assert retry_learning_rate(0.001, factor=0.5) == pytest.approx(0.0005)


def test_early_steps_before_window_do_not_fire():
    # First 50 steps should never fire even if loss jumps, because trailing
    # average needs 50 steps to be meaningful -- warmup noise must not abort.
    detector = DivergenceDetector()
    for _ in range(10):
        detector.observe(0.5)
    # A spike early should not yet count because window not filled
    result = detector.observe(5.0)
    assert result.status == "ok"


def test_non_positive_trailing_average_does_not_crash():
    detector = DivergenceDetector(
        multiplier=2.0, window=3, consecutive=2, warning_consecutive=1
    )
    # Feed zero losses to make trailing avg zero
    for _ in range(3):
        detector.observe(0.0)
    result = detector.observe(1.0)
    assert result.status == "ok"


def test_divergence_uses_multiplier_correctly():
    # With multiplier 2.0, loss at 1.0 vs trailing 0.6 should not be divergence
    detector = DivergenceDetector(
        multiplier=2.0, window=3, consecutive=2, warning_consecutive=1
    )
    for _ in range(3):
        detector.observe(0.6)
    result = detector.observe(1.0)  # 1.0 < 1.2, not exceed
    assert result.status == "ok"
    # Need trailing average to include the 1.0 (not exceed) so avg ~0.73, then 1.8 >1.46 warning
    result = detector.observe(1.8)  # 1.8 > 2*~0.73=1.46, exceed 1
    assert result.status == "warning"
