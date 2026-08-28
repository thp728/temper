# ADR-0048 — The held-out split is the platform's, and held-out loss rides the existing event stream

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/011-evaluation-and-delivery.md`
- **Issue:** [#53](https://github.com/thp728/temper/issues/53)

## Context

Issue #53 asks for a portion of the dataset held out automatically, an eval
loss charted against training loss, and a plain-language signal when held-out
loss stops improving. Two prior issues shape the room to work in.

The split's ordering is the substance of the issue, not a detail: the split
happens **after deduplication**, because splitting before it lets a duplicated
row land on both sides, which makes held-out loss optimistic in a way nothing
downstream can detect. The ordering has to be proven by a test, not asserted
in a comment.

And the flat-memory guarantee (#31, extended by #42) is a constraint on where
the split may run. Validation and token counting stream over the dataset and
never hold it; a split that materialises the dataset in the control plane
would undo both at once. The guarantee is guarded by `Report.retained_objects()`
and the streaming validation tests, and those must keep passing.

A third fact sets the surface: the live running-job view (#39) already
consumes one server-pushed event stream over the durable event log. Held-out
loss must ride that existing stream and be charted in that existing view —
not a second stream and not a second view.

## Decision

**The platform owns the split, it runs on the machine, and the held-out
signal travels through the same durable event log as training loss.**

- **The split lives in the trainer, not the control plane.** The control
  plane validates and stores the dataset streaming and never holds the rows;
  the trainer already materialises them (it reads the whole file to detect
  thinking mode), so dedup-and-split there adds no materialisation to the
  path #31 and #42 protect. The split logic is a pure module in
  `temper_core` and ships flat into the trainer image exactly like
  `thinking.py` — one file, two consumers (the domain's validation and the
  machine's trainer), so the two cannot drift.

- **Deduplication precedes the split, as one pipeline.** `held_out_split`
  dedups first and then splits; the raw splitter exists only so the ordering
  can be proven by a test that finds a seed under which the raw splitter puts
  a duplicate on both sides and asserts the pipeline never does. The dedup
  key is the normalised conversation, so two encodings of one visual string
  are one row.

- **The split is deterministic under a seed and its size is recorded.** The
  seed is the run's own (the training seed), so the same job spec reproduces
  the same split; the fraction is the effective `val_set_size` (the product's
  chosen eval proportion, exposed on the advanced surface); and the sizes —
  rows in, duplicates removed, trained, held out, fraction, seed — are
  recorded in the run's result document.

- **The held-out rows are excluded from training by construction.** The two
  sides are written to separate files. The training file is given to Axolotl
  with `val_set_size` zeroed (Axolotl accepts either `test_datasets` or a
  `val_set_size` split, not both) and the held-out file is given as
  `test_datasets` — the eval rows never enter the training file, so they
  cannot be trained on and cannot overlap the training side.

- **Held-out loss measured during the run rides the existing event stream.**
  The line classifier now promotes the evaluation row the trainer prints
  (`eval_loss`) into a metric event carrying `held_out_loss` — the domain's
  name, the one the checkpoint record already uses. The running view charts
  both series from the same event list it already appends to, so the chart
  and the plateau re-render as each measurement is pushed. No second stream,
  no second view.

- **The chart is on the finished record too, not only the live view.** The
  overfitting signal is read after the run — a user who returns tomorrow
  wants to see the divergence as much as someone watching it happen — so the
  finished record renders the same two series from the same durable history.
  That also makes a browser journey able to assert the chart race-free: it
  waits for the job's own terminal status (polling `GET /v1/jobs/:id`) and
  then reads the record, rather than racing the running view's hand-back.

- **A plateau is surfaced in plain language.** A patience rule — two
  consecutive evaluations without improvement — computed as a pure function
  over the held-out series and rendered as a note that says "overfitting" in
  words a non-specialist can act on. It is a live-region note on the running
  view and a plain paragraph on the finished record.

## Alternatives considered

**Split in the control plane during validation.**
Rejected: it would hold the dataset to split it, reintroducing exactly the
materialisation #31 and #42 exist to forbid. The split therefore runs where
the rows are already in memory — the trainer.

**Let Axolotl carve the split with `val_set_size`.**
Rejected: that split is not after our deduplication, is not deterministic
under a seed we control and record, and its size is not recorded. The issue
is explicit that the platform owns the split, its determinism and its
recorded size; letting the framework carve it silently would give away all
three.

**Open a second stream for eval loss.**
Rejected: #39's stream over the durable log is the channel, and the eval row
is the same kind of measurement as the training row. Extending the metric
event to carry `held_out_loss` keeps one stream, one replay story, and one
view.

**Detect the plateau on the backend and stream it as an event.**
Rejected: the frontend already holds the held-out series as events arrive,
the wording is a presentation concern that belongs with the surface that
renders it, and a stateful detector inside the streaming consume loop would
couple the domain to that loop. The tradeoff is recorded: the note is a live
view, not part of the durable run record — best-checkpoint selection (#62)
reads the recorded checkpoints, not this note.

**Add a chart library.**
Rejected: two polylines over a shared domain is the whole need, and a
dependency bought for one screen is a dependency maintained forever.

## Consequences

- The trainer dedups and splits at the start of every job, writes
  `train.jsonl`/`eval.jsonl`, narrates the split, and records it in
  `result.json`; the config hands the split to Axolotl as `test_datasets`.
- Eval rows become metric events carrying `held_out_loss`; the running view
  and the finished record both gain a chart and a plateau note, and the
  running view adds a held-out stat beside the training one.
- The journey fake emits an eval line and the split narration, so the
  journeys exercise held-out loss end to end, and the loss chart is asserted
  in a real browser against a job that really ran: the journey waits for the
  job's own terminal status and reads the finished record's chart, which is
  how a browser test asserts a two-series chart without racing the run.
- The one acceptance criterion that cannot be satisfied without hardware —
  *the chart is watched rendering live during a real run* — is stated as
  outstanding in the PR body rather than quietly reinterpreted as met by the
  fake run.
- The journeys' e2e ports move to this issue's 53xx range so parallel
  worktrees cannot collide; server reuse stays disabled.

## Rollback

Revert the trainer's split and the classifier's eval promotion, drop the
chart and plateau from the running view, and the previous single-series
behaviour returns; nothing else depends on the new event field.
