"""Best-checkpoint selection: a pure function over checkpoints and losses.

Issue #62. The acceptance criterion is that selection is a pure function over
checkpoints and losses, tested without hardware -- so the tests below are a
table of losses and an expected winner. The rule, pinned here exactly as
ADR-0049 records it: the lowest held-out loss wins; a tie goes to the later
checkpoint; a checkpoint with no held-out loss is never a candidate; when no
checkpoint has a held-out loss the most-trained one is named as a fallback;
and the choice is a complete record (step, basis, reason), because the point
is that it is stored, not re-derived.
"""

from __future__ import annotations

import pytest

from temper_core.checkpoint import (
    BASIS_BEST,
    BASIS_FALLBACK,
    BASIS_NONE,
    BASIS_TIE,
    select_best_checkpoint,
)


def ckpt(step, held_out_loss=None):
    """One verified checkpoint record, shaped as the control plane stores it."""
    record = {"step": step}
    if held_out_loss is not None:
        record["held_out_loss"] = held_out_loss
    return record


def test_the_lowest_held_out_loss_wins():
    # The last checkpoint (step 30) is NOT the best: held-out loss rose after
    # step 20, which is exactly the overfitting case the feature exists for.
    selection = select_best_checkpoint(
        [ckpt(10, 0.44), ckpt(20, 0.39), ckpt(30, 0.52)]
    )
    assert selection.step == 20
    assert selection.held_out_loss == 0.39
    assert selection.basis == BASIS_BEST
    assert "Step 20" in selection.reason
    assert "0.39" in selection.reason


def test_a_tie_goes_to_the_later_checkpoint():
    """Two checkpoints with the same lowest loss: the one that trained further
    wins, because it matched the best loss while having seen more data."""
    selection = select_best_checkpoint(
        [ckpt(10, 0.5), ckpt(20, 0.3), ckpt(30, 0.3)]
    )
    assert selection.step == 30
    assert selection.held_out_loss == 0.3
    assert selection.basis == BASIS_TIE
    assert "Step 20" in selection.reason
    assert "tied" in selection.reason


def test_a_three_way_tie_names_all_of_them_and_takes_the_latest():
    selection = select_best_checkpoint(
        [ckpt(5, 0.4), ckpt(10, 0.4), ckpt(15, 0.4)]
    )
    assert selection.step == 15
    assert selection.basis == BASIS_TIE
    assert "Steps 10, 5 tied" in selection.reason


def test_a_checkpoint_without_a_held_out_loss_is_not_a_candidate():
    """A checkpoint that recorded no held-out loss offers no signal to be best
    on, so it can never win -- but it does not drag the others down either."""
    selection = select_best_checkpoint(
        [ckpt(10), ckpt(20, 0.31), ckpt(30, 0.28)]
    )
    assert selection.step == 30
    assert selection.basis == BASIS_BEST


def test_the_reason_names_the_pool_precisely_when_a_retained_checkpoint_lacks_a_loss():
    """The recorded reason must not understate either count: when one retained
    checkpoint carried no held-out loss, the reason says how many had a loss
    to choose on among how many were retained -- a claim that says 'of 3
    retained checkpoints' when only two had losses would be the recorded
    reason lying."""
    selection = select_best_checkpoint(
        [ckpt(10, 0.3), ckpt(20, 0.4), ckpt(30)]
    )
    assert selection.step == 10
    assert (
        "of 2 checkpoint(s) with a held-out loss among 3 retained"
        in selection.reason
    )


def test_a_non_finite_loss_is_treated_as_absent():
    """A NaN or infinite loss cannot be compared, so it is a gap, not a
    candidate -- a diverged run must not be named the best."""
    selection = select_best_checkpoint(
        [ckpt(10, float("nan")), ckpt(20, 0.3), ckpt(30, 0.5)]
    )
    assert selection.step == 20
    selection = select_best_checkpoint([ckpt(10, float("inf")), ckpt(20, 0.3)])
    assert selection.step == 20


def test_when_no_checkpoint_has_a_held_out_loss_the_last_is_named_as_a_fallback():
    """No signal at all: the most-trained checkpoint is the honest fallback,
    and the reason says it is a fallback rather than claiming a best."""
    selection = select_best_checkpoint([ckpt(10), ckpt(20), ckpt(30)])
    assert selection.step == 30
    assert selection.basis == BASIS_FALLBACK
    assert "No checkpoint recorded a held-out loss" in selection.reason
    assert "fallback" in selection.reason


def test_an_empty_set_chooses_nothing_and_says_so():
    selection = select_best_checkpoint([])
    assert selection.step is None
    assert selection.basis == BASIS_NONE
    assert selection.held_out_loss is None
    assert "No checkpoints were recorded" in selection.reason


def test_a_record_without_a_step_is_refused_not_skipped():
    """A checkpoint that cannot be named cannot be chosen, and a record
    silently skipped would let the selection pretend it did not exist."""
    with pytest.raises(ValueError):
        select_best_checkpoint([{"held_out_loss": 0.3}])


def test_the_selection_round_trips_through_its_stored_shape():
    """The recorded choice is what a run persists: step, basis, reason and the
    winner's loss -- the whole answer, never a step alone."""
    selection = select_best_checkpoint([ckpt(10, 0.44), ckpt(20, 0.39)])
    stored = selection.to_dict()
    assert stored == {
        "step": 20,
        "basis": BASIS_BEST,
        "reason": selection.reason,
        "held_out_loss": 0.39,
    }
    assert stored["step"] == 20
