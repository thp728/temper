# ADR-0049 — The best checkpoint is chosen by held-out loss, and the choice is recorded on the run

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/011-evaluation-and-delivery.md`
- **Issue:** [#62](https://github.com/thp728/temper/issues/62)

## Context

The last checkpoint is the result by default rather than by choice, and it is
frequently not the best one. Selecting it silently is a quality decision made
by omission: the product hands back a file without ever answering "did it
work, and is it better than what I started with", and the file it hands back
may be worse than one the same run produced earlier.

Two prior issues shape the room to work in, and neither is ours to change:

- **#53** produced the held-out split and streams held-out loss on the
  existing event stream. The loss we select on comes from there.
- **#37** writes checkpoints off the machine as they are produced, each
  recording its step and its held-out loss where one exists, with bounded ring
  retention. That record is what we select over; how checkpoints are written
  or retained is not changed by this decision.

The acceptance criteria carry two commitments that shape the whole design. The
first: **"the choice is stored on the run rather than inferred at download
time, so it cannot change later."** A choice re-derived at download time
silently changes when retention evicts a checkpoint or when the selection rule
is edited — a run's result must not move under it. The second: **"selection is
a pure function over checkpoints and losses, tested without hardware."** The
rule must live in `packages/core` with no I/O, so its tests are a table of
losses and an expected winner.

## Decision

**The checkpoint with the best held-out loss is chosen as the result, the
choice is made once at the moment the run's checkpoints are recorded, and the
whole choice — which step, and why — is stored on the run.**

- **The selection rule is a pure function in `temper_core.checkpoint`** —
  `select_best_checkpoint`, with no I/O and no framework imports, tested
  without hardware as a table of losses and an expected winner. The lowest
  held-out loss wins.
- **A tie is a decision, not an oversight.** Two checkpoints with the same
  lowest held-out loss: the **later** checkpoint wins, because it trained
  further while matching the best held-out loss — it is at least as good and
  more fully trained. The record's basis names `tie_latest` and its reason
  names the tied steps.
- **A checkpoint with no held-out loss recorded is never a candidate.** It
  offers no signal to be best on, so it cannot win; it remains downloadable,
  and its absence from the choice is stated as data rather than guessed. A
  non-finite loss (a NaN from a diverging run) is treated the same as an
  absent one, for the same reason: a number that cannot be compared is no
  basis for a choice.
- **When no checkpoint has a held-out loss at all, the most-trained
  checkpoint is named as a fallback** (`fallback_last`), and the reason says
  it is a fallback rather than a best silently claimed. There is no signal to
  choose on, so the honest answer is the last checkpoint, stated as such.
- **The choice is recorded on the run, not re-derived.** The orchestrator
  computes the selection once, at the same moment the verified checkpoints are
  recorded (on both terminal outcomes — a failed run's checkpoints are still
  its recovery material, and naming which one is best is as useful to a
  resumption as to a delivered result), and persists the step, the basis and
  the reason in the job's `best_checkpoint` record. No read path ever
  recomputes it. Retention evicting a checkpoint later, or the rule being
  edited, cannot move a choice that was already made.
- **Selection runs over the retained, verified checkpoints only.** A
  checkpoint whose bytes are not in storage — failed upload, or superseded by
  retention — is not something a run can stand behind, so it is not a
  candidate. The reason records how many retained checkpoints were considered.
- **Every retained checkpoint stays downloadable.** The download route serves
  any retained checkpoint by its step (the identifier the user sees), chosen
  or not; the storage key is read from the job row's own record and never
  published to the browser. A step that is not retained refuses with a stable
  code rather than a broken download.
- **The choice is published, and the chosen checkpoint flagged, without
  re-selecting.** The API publishes `checkpoints` (step, slot, losses,
  verified, and `selected` for the chosen one) and `best_checkpoint` (the
  stored step, basis and reason). The `selected` flag is set from the stored
  choice at presentation — presentation of a recorded fact, never a
  re-derivation of it — so the interface highlights the run's answer without
  holding the rule.
- **The choice is independent of the artifact's kind.** What the delivered
  artifact is (adapter versus fully trained model, and how it is packaged) is
  issue #32's and #66's concern; this decision records which checkpoint the
  run stands by, and leaves the artifact's kind alone.

## Alternatives considered

**Derive the choice at download time.**
Rejected: this is the whole point of the criterion. A choice recomputed from
whatever checkpoints still happen to be retained changes when retention
evicts one or when the selection rule is edited, and a user who returns a week
later must see the same answer. Deriving is cheaper to build and wrong in the
one way the criterion names.

**Select on the machine and ship the chosen checkpoint's weights as the
artifact.**
Rejected: the artifact's kind and packaging is #32's and #66's merged work,
and this issue's boundary is the run's record, not the artifact pipeline. The
choice is recorded against the run so that when the artifact flow reads it, it
reads a stored fact rather than recomputing one.

**Let the frontend pick the lowest held-out loss from the published list.**
Rejected: the rule would live in a component, be untested by the domain's
table-of-losses tests, and — the decisive objection — the choice would not be
stored anywhere, so it would be re-derived on every render by a different
piece of code than the one that "decided" it.

**Prefer the earliest checkpoint on a tie (less training, so closer to the
pre-overfit optimum).**
Rejected: a later checkpoint that matches the best held-out loss has also seen
more data; for any use of the result — serving, resumption, further training —
more training at an equal held-out loss is not a worse answer. The later
checkpoint wins, and the reason says so.

**Refuse to name any checkpoint when none has a held-out loss.**
Rejected: the product's answer would become "no result" for a run that
produced usable checkpoints, which is worse than the honest fallback. Naming
the most-trained checkpoint as a fallback keeps a deliverable while saying
plainly that no loss-based choice was possible — an omission stated is not the
silent default selection this issue exists to remove.

## Consequences

- `temper_core.checkpoint` ships a pure `select_best_checkpoint` and its
  tests are a table of losses and an expected winner — selection is tested
  without hardware.
- The control plane records `best_checkpoint` on the job row beside the
  verified checkpoints, on complete and failed runs, with a log event stating
  the choice in the run's own history.
- The API contract publishes `checkpoints` and `best_checkpoint`; the
  finished-record view shows which checkpoint was chosen and why, flags it,
  and offers every retained checkpoint for download by step.
- A new download route serves any retained checkpoint by step; a non-retained
  step is refused with `checkpoint_unavailable` or `checkpoint_missing` and
  never offered.
- Pre-existing rows read correctly: `best_checkpoint` is a new nullable
  column, and a row written before this record existed simply has no recorded
  choice and an empty checkpoint list — no migration re-derives anything.

## Rollback

Revert the selection recording (the `best_checkpoint` column and the
orchestrator's recording call), drop the published checkpoint fields and the
checkpoint download route, and the finished record returns to not naming a
result checkpoint. The pure selection function can stay or go; nothing else
depends on it.
