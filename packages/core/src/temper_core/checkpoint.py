"""Choosing the run's result checkpoint by held-out loss (issue #62).

The last checkpoint is the result by default rather than by choice, and it is
frequently not the best one. This module holds the selection rule that fixes
that: given the run's retained, verified checkpoints -- each carrying a step
and, where one was recorded, a held-out loss -- choose the one with the best
held-out loss, and say why. It is a pure function over checkpoints and losses,
with no I/O, so its tests are a table of losses and an expected winner
(ADR-0049).

Two decisions this module makes, and why:

* **The choice is stored on the run, not re-derived at download time.** A
  user who returns a week later must see the same answer. A choice recomputed
  from whatever checkpoints still happen to be retained could change when
  retention evicts one (issue #37 keeps a bounded ring) or when the rule is
  edited, and a run's result must not move under it. `select_best_checkpoint`
  therefore produces the whole answer -- which step, and the reason -- once,
  at the moment the run's checkpoints are recorded, and the caller persists
  that record. This module never reads anything back and re-decides.
* **The rule distinguishes the three things that can be true.** The best
  checkpoint by held-out loss (`best_held_out_loss`, with `tie_latest` naming
  the tie-break); no checkpoint carrying a held-out loss at all, in which case
  the most-trained checkpoint is named as an honest fallback (`fallback_last`)
  rather than a best silently claimed; and nothing to choose from (`none`).

Ties and gaps are decisions, not oversights (ADR-0049): a tie on the lowest
held-out loss goes to the **later** checkpoint -- it has trained more while
achieving the same held-out loss, so it is at least as good and more fully
trained -- and a checkpoint with no held-out loss recorded is never a
candidate for "best by held-out loss", because it offers no signal to be best
on. A loss that is not a real, finite number (a NaN from a diverging run, say)
is treated the same as an absent one, for the same reason: a number that
cannot be compared is no basis for a choice.

The module is deliberately self-contained -- no imports from `temper_core`
siblings -- because it is one of the files shipped flat into the trainer
image (ADR-0010's rule that a value two components must agree on is defined
once), where the control plane's terminal selection runs in the same image
shape the rest of the domain uses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard

# The machine-readable bases a selection can rest on. One value per decision
# kind, defined once here and read wherever a selection is rendered or
# recorded, so the vocabulary cannot drift between the store and the surface.
BASIS_BEST = "best_held_out_loss"
BASIS_TIE = "tie_latest"
BASIS_FALLBACK = "fallback_last"
BASIS_NONE = "none"


def _usable_loss(value: Any) -> TypeGuard[float]:
    """Whether a reported held-out loss is a number a comparison can trust.

    `bool` is excluded because it is an int subclass and `True` would compare
    as 1.0; a non-finite value (NaN from a diverging run, infinity from a
    stalled one) is excluded because a number that cannot be compared to
    another is no basis for a choice -- the "no held-out loss recorded" gap,
    stated as data rather than guessed.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class CheckpointSelection:
    """The run's recorded answer to "which checkpoint is the result".

    Frozen at the moment the checkpoints are recorded and persisted beside
    them, so a run's choice cannot change later -- see the module docstring
    and ADR-0049. `step` is the chosen checkpoint's step (None only when there
    was nothing to choose from); `basis` names the rule that decided;
    `reason` says it in words a non-specialist can read; `held_out_loss` is
    the winner's held-out loss where one was the basis.
    """

    step: int | None
    basis: str
    reason: str
    held_out_loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"basis": self.basis, "reason": self.reason}
        if self.step is not None:
            out["step"] = self.step
        if self.held_out_loss is not None:
            out["held_out_loss"] = self.held_out_loss
        return out


def _reason(
    step: int,
    loss: float,
    candidates: int,
    retained: int,
    tied_with: Sequence[int],
) -> str:
    """The legible why for a held-out-loss choice, tie or not.

    The reason is part of the recorded choice, not a presentation layer's
    paraphrase: it is stored on the run beside the step, so the user's answer
    to "why this one" cannot drift from the rule that chose it. The candidate
    pool is stated precisely: when every retained checkpoint carried a usable
    held-out loss the pool *is* the retained set and is named so; when some
    did not, the reason says how many had a loss to choose on among how many
    were retained, rather than claiming a count that understates either.
    """
    tail = ""
    if tied_with:
        if len(tied_with) == 1:
            tail = (
                f" Step {tied_with[0]} tied for the lowest held-out loss; "
                "the later checkpoint won because it trained further while "
                "matching it."
            )
        else:
            names = ", ".join(str(s) for s in tied_with)
            tail = (
                f" Steps {names} tied for the lowest held-out loss; the "
                "later checkpoint won because it trained further while "
                "matching it."
            )
    if candidates == retained:
        pool = f"{candidates} retained checkpoint(s)"
    else:
        pool = (
            f"{candidates} checkpoint(s) with a held-out loss "
            f"among {retained} retained"
        )
    return (
        f"Step {step} has the lowest held-out loss ({loss}) of {pool}.{tail}"
    )


def select_best_checkpoint(
    checkpoints: Sequence[Mapping[str, Any]],
) -> CheckpointSelection:
    """The checkpoint with the best held-out loss, and why, as a pure function.

    `checkpoints` is the run's retained, verified checkpoint records -- each
    a mapping carrying an integer `step` and, where one was recorded, a
    numeric `held_out_loss`. Whether a checkpoint is retained and verified is
    the caller's filtering concern (issue #37 decides that); this function
    decides among what it is given, and never touches storage or anything a
    test would have to fake.

    Raises `ValueError` when a record carries no usable `step`, because a
    checkpoint that cannot be named cannot be chosen -- and a silently skipped
    record would let the selection pretend it did not exist.
    """
    candidates: list[tuple[int, float]] = []
    for record in checkpoints:
        step = record.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            raise ValueError(
                f"a checkpoint record carried no integer step: {record!r}"
            )
        loss = record.get("held_out_loss")
        if _usable_loss(loss):
            candidates.append((step, loss))

    if candidates:
        best_loss = min(loss for _, loss in candidates)
        tied = sorted(
            (step for step, loss in candidates if loss == best_loss),
            reverse=True,
        )
        chosen = tied[0]
        return CheckpointSelection(
            step=chosen,
            basis=BASIS_TIE if len(tied) > 1 else BASIS_BEST,
            reason=_reason(
                chosen,
                best_loss,
                len(candidates),
                len(checkpoints),
                tied[1:] if len(tied) > 1 else (),
            ),
            held_out_loss=best_loss,
        )

    steps = [record["step"] for record in checkpoints]
    if steps:
        last = max(steps)
        return CheckpointSelection(
            step=last,
            basis=BASIS_FALLBACK,
            reason=(
                f"No checkpoint recorded a held-out loss, so the most-trained "
                f"checkpoint (step {last}) is named as the fallback result "
                "rather than a best claimed."
            ),
        )
    return CheckpointSelection(
        step=None,
        basis=BASIS_NONE,
        reason="No checkpoints were recorded, so none could be chosen.",
    )
