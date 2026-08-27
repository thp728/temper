# ADR-0037 — Every run records what was predicted against what happened

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/005-predictor-and-quote.md`
- **Issue:** [#77](https://github.com/thp728/temper/issues/77)

## Context

The duration and cost model rests on one measurement, from one model on one
device with one method, and extrapolating it crosses model size, hardware
generation, method and sharding at once. The spec names the mitigation
explicitly: it is not a better constant, it is recording — *"calibrated
against N real runs" is worth more than a better guess, but only if recording
starts with the first run rather than the last.* The risk is that the estimate
stays as accurate as it was on day one, with no way to see a systematically
wrong figure except as a string of quietly disappointing jobs.

The predictor's seams already exist on main: predicted peak VRAM from a
model-facts seam (ADR-0028), hardware from live availability and price
(ADR-0029), disk from model facts (ADR-0030), duration and cost as a range
broken down by phase (ADR-0031), and every decision carrying its reason
(ADR-0032). What none of them does is record what the prediction was *against*.
This record is the recording half of spec 005 — the half that has to ship
before anything consumes it, so no run is wasted.

## Decision

**Every attempt records predicted against actual — duration, peak memory and
cost — and the comparison is visible on a finished job and in an aggregate
view where a systematically wrong estimate shows up rather than being
absorbed.**

**The predicted half is the frozen quote.** It already carried the duration
and cost ranges per phase (ADR-0031); it now also carries the predicted peak
VRAM as a point, computed by the same selection that chose the card. Peak is a
point rather than a range because memory is the half of the predictor that
*blocks* (spec 005): it is arithmetic, not an estimate, so recording a range
for it would misstate how much evidence backs it.

**The measured half is frozen at terminal, the mirror of the quote frozen at
launch.** A new `actuals` block is written onto the job row the moment a run
reaches a terminal state, and never updated. It holds:

- **Duration, measured twice**: the wall total from the job's own timestamps,
  and a per-stage breakdown (provisioning, preparing, training, packaging)
  measured from the state events. The stages are the job's own, *not* the
  quote's phases — `preparing` bundles readiness, image pull and model
  download — and reconciling the two vocabularies happens in exactly one
  place, `temper_core.calibration`.
- **Peak memory, measured on the machine**: the trainer samples the
  machine's per-GPU used VRAM with `nvidia-smi` during training and records
  the peak in its result document. This is the same method the 5.31 GB anchor
  was measured with (spike 6), not a new technique, and a machine without
  `nvidia-smi` records no peak — honestly absent, never guessed.
- **Cost, derived, never measured**: measured duration times the rate the
  job froze at launch, converted into the currency's smallest unit exactly as
  the quote does. Nothing in the product reads a bill (spec 005's out-of-scope
  note), so the cost line is marked derived wherever it appears.

**Every figure is marked measured or derived in the shape itself**, not in a
comment: the published `actuals` distinguishes the measured duration and peak
from the derived cost, and the surfaces say so. A number without its basis is
a number a reader cannot judge.

**The state events now say which state they entered.** A state event's message
is often the human narration ("Selecting hardware", not "provisioning"), which
made it impossible to measure a stage from the events. The event now carries
the state as data (`{"state": "provisioning"}`), so the durations the
comparison rests on are the state machine's own.

**Actuals are frozen before the terminal status is written.** A reader can
never observe a terminal job without its actuals beside it, and the
orchestrator thread's database work ends the moment the status does. This is
also what keeps a background job thread from trailing writes into the next
test's database.

**The aggregate view makes systematic error visible.** `GET /v1/calibration`
rolls predictions against measurements across every terminal job that has
both, per metric (duration, peak memory, cost) and per phase bucket, with the
runs themselves riding along so an outlier can be named rather than pointed
at. The ratio is actual over predicted *midpoint*, so a systematically wrong
estimate appears as a mean ratio away from 1 — 0.44 across two runs is a
visible statement that the estimate over-predicted by more than 2x, not a
number folded into a better-looking average. Each roll reports its own count:
"calibrated against N real runs" is only as honest as N is visible.

The bucket mapping follows where the machine actually spends the time, not
where the quote's names would like it to be: the orchestrator's `preparing`
state spans the SSH wait and the archive pushes, so the quote's `readiness`
predicts it; its `training` state — entered as "Building image and training" —
spans the on-machine image build (the quote's `image_pull`), the weights
download as the container loads (`model_download`), and the training itself,
so those three quote phases together predict the measured `training` stage.
This is the one place the two vocabularies meet, so it is the one place they
are mapped.

**The comparison is visible on a finished job.** The record shows each metric
side by side — predicted (estimate) against actual (measured/derived) — with
the ratio and a sentence naming where the actual landed relative to the
predicted range, plus the measured stage-by-stage breakdown and a link to the
aggregate.

## Alternatives considered

**Record only the totals, not per-stage durations.**
Rejected: the quote's most useful claim is the cold-start breakdown, and a
comparison that cannot show a long `preparing` stage against its prediction
cannot show where a systematic error lives. The stages are the machine's own
observable states, which is what makes the measurement honest rather than a
re-mapping of the prediction onto itself.

**Measure actual peak memory in the control plane by polling `nvidia-smi`
during the run.**
Rejected: that is runtime monitoring, which the running-job route (issue #39)
owns, and it puts the measurement on the wrong side of the wire. The trainer
is co-located with the GPU and already writes the result document the
orchestrator reads, so the measurement travels with the only path that
guarantees it arrives.

**Derive actuals on read from the events and result instead of persisting
them.**
Rejected: the issue is explicit that recording ships first, and a frozen
record is the mirror of the frozen quote — an immutable "what happened" beside
the immutable "what was predicted". Recomputing on read would also make the
aggregate recompute differently from whatever a reader was shown last.

**Compute actuals after the terminal status is written.**
Rejected on a measured flake: a reader could see a terminal job without its
actuals, and a job thread could keep writing after a test (or a restart)
tore down its database. Freezing actuals first closes both windows by
construction.

## Consequences

- `temper_core.actuals` is new: `measure` turns a job row, its state events
  and the trainer's result into the frozen actuals. `temper_core.calibration`
  is new: the ratio/direction comparison and the aggregate. Both are pure and
  tested without I/O.
- The trainer samples `nvidia-smi` during training and records
  `peak_memory_gb` in its result document; a machine without `nvidia-smi`
  records none.
- `db.set_state` writes the state as event data; the job row gains an
  `actuals_json` column, frozen by the orchestrator at terminal.
- The quote publishes `peak_memory_gb`; `JobRecord` publishes `actuals`; the
  API gains `GET /v1/calibration`.
- The finished-job page gains a "Prediction vs what happened" section; a new
  `/calibration` view shows the aggregate.
- Calibration starts with the first real run: every subsequent run this week
  feeds it, and the risk note in spec 005 ("the riskiest number is
  throughput") becomes a measured claim instead of a caveat.

## Rollback

Drop the `actuals_json` column and `_record_actuals`, remove the quote's
`peak_memory_gb` and the trainer's sampler, and the surfaces return to
showing the quote alone. The state-event data addition is a superset that can
stay harmlessly.
