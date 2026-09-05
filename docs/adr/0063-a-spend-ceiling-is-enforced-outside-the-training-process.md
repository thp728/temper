# ADR-0063 — A spend ceiling is enforced outside the training process

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/010-recovery-and-fault-injection.md`
- **Issue:** [#46](https://github.com/thp728/temper/issues/46)

## Context

A job that has gone wrong keeps billing until somebody stops it. The stall
detector and the duration ceiling stop a job that is no longer making progress
or that runs too long; neither is a cap on *money*, and the scope boundary
names the missing control directly: a **hard spend cap enforced
outside the training process**, kept because it is a safety control, not
billing.

The reason "outside the training process" is the requirement, not a detail, is
in the spec's own one-liner: **a process that has stopped responding cannot
enforce its own limit.** A ceiling checked inside the trainer, or one that
trusts the trainer to report its own spend, fails exactly when it is needed —
the trainer is the thing that has gone wrong. The spec's implementation
decision names the order on reaching the ceiling: **checkpoint, terminate,
destroy**, and mark the job failed with the specific reason.

Two already-merged guarantees shape the room:

- **#34 (ADR-0057)** confirms teardown across consecutive absent observations,
  with `normalize_status()` defined once in `provider.py`. The ceiling's
  destruction step must go through that confirmation, not around it.
- **#62 (ADR-0049)** records the run's result checkpoint by held-out loss. A
  checkpoint written at the ceiling must be visible to that selection, not
  stored somewhere it cannot see.
- **#24 (ADR-0051)** built the fault surface. The path must be exercised
  through that surface — the fault that reaches the ceiling quickly is
  `machine_silent`, a provider-side fault that makes the machine go silent
  without ending, which is precisely the unresponsive case the criterion names.

## Decision

**The spend ceiling is a cost, configured as a named module-level constant,
enforced by the control plane from elapsed time and the job's frozen price,
and on reaching it the job runs checkpoint, terminate, destroy and fails with
`budget_exhausted`.**

- **The ceiling is a cost, not a number of minutes.** `SPEND_CEILING_MINOR`
  is the most a single job may cost, in the account currency's minor unit
  (paise for INR), and it is converted into a wall-clock deadline against the
  rate the job froze at provisioning (`price_per_hour`), the same derivation
  `temper_core.actuals` uses for the measured cost. An expensive machine
  exhausts the same ceiling sooner than a cheap one, which is the point of a
  money ceiling rather than a minute ceiling. It is a named module-level
  constant beside its derivation in `orchestrator.py`, the way
  `TEARDOWN_CONFIRM_SAMPLES` is — not added to `config.py`, which #29 owns.

- **The number is derived from two inherited facts and one judgment, and each
  half is labelled.** The most a *legitimate* job can cost is bounded by the
  duration ceiling (ADR-0002's 24h) and the most expensive card the platform
  has measured — H200 at ₹378.27/hr (measured 2026-08-17) — about ₹9,078.
  The ceiling is ₹10,000: roughly 10% above that worst legitimate cost, so it
  cannot fire on a legitimate run (a ceiling that fires on a legitimate run
  is a bug, not a safety net), and a runaway is stopped at roughly one job's
  worst-case cost rather than an open-ended bill. The two inherited facts are
  the adopted 24h ceiling and the measured H200 rate; the 10%-above-worst-
  legitimate bound is the judgment. If the catalog or the rates move, this is
  the number to revisit.

- **Enforcement is outside the training process, and the measurement never
  reads the trainer.** The spend check lives in the control plane's `guard`
  and at the stage boundaries, exactly where the duration ceiling lives, and
  it measures `elapsed x price_per_hour` on the control plane's own clock —
  never a figure the trainer reports. A machine that goes silent (the
  `machine_silent` fault) keeps billing, the guard keeps turning, and the
  ceiling fires before the stall detector has any reason to. The fault surface
  is exercised for this, not extended: no second way to cause a failure is
  added.

- **On reaching the ceiling the order is checkpoint, terminate, destroy.** The
  checkpoint half is issue #37's point: the machine writes checkpoints off
  itself as it trains, so the control plane asks it to report them
  (`request_checkpoint`, a new method on the provider seam), records what it
  can verify, and lets the held-out-loss selection (ADR-0049) see the result.
  The request is best-effort and bounded — a machine that does not answer
  records nothing and the shutdown proceeds, because an unresponsive trainer
  cannot be asked to save and the record says so. Terminate is the trainer
  finalizing gracefully (a SIGTERM handler converts the signal into the
  existing graceful-exit path that ships the final checkpoint and writes
  result.json); destroy is the existing confirmed teardown (ADR-0057), run in
  the `finally` before the terminal transition as every other outcome does.

- **The failure carries a specific reason.** `budget_exhausted`, the terminal
  code the reference architecture names, distinguishes a safety-limit stop
  from a stall (`gpu_stalled`), an over-long run
  (`gpu_max_duration_exceeded`), and a user's cancellation. The emergency
  checkpoint and the "no checkpoints saved" outcome are recorded in the job's
  history before the terminal state, so a job stopped by money reads
  differently from one that broke.

## Alternatives considered

**Enforce the ceiling inside the trainer, or from the trainer's own reports
of spend.** Rejected: it is the exact anti-pattern the criterion names. A
process that has stopped responding cannot enforce its own limit, and a ceiling
that trusts the trainer to report its own spend fails when the trainer is the
thing that has gone wrong. The control plane measures spend from its own clock
and the frozen price, so it never depends on the trainer answering.

**A check inside the streaming guard only, with no boundary checks.**
Rejected as insufficient: the guard only runs while the stream is open, and a
job spends time before that — provisioning, waiting for SSH, pushing sources —
during which a runaway can already accrue cost. The same boundaries that
check the duration ceiling check the spend ceiling too.

**A fixed GPU-minute cap.** Rejected: minutes are not spend. An H200 bills
roughly nine times an L4 per hour, and a ceiling that treats them alike is a
ceiling that lets the expensive runaway through. Converting the cost ceiling
into a deadline against the job's own frozen rate is the honest derivation and
is the same one the actuals use.

**Require a final checkpoint from the machine, blocking the shutdown until it
answers.** Rejected: it makes the shutdown wait on the very machine it exists
to stop. The emergency checkpoint is best-effort and bounded; a machine that
does not answer records "no checkpoints were saved" and the shutdown proceeds.
The checkpoints that count are the ones issue #37 already wrote off the
machine, which is why the ceiling stop does not also destroy the work.

**Let the duration ceiling subsume the spend ceiling (a 24h run on any card is
bounded).** Rejected: the duration ceiling is a time bound and is not
money-shaped. If the provider's prices rise, or a configuration error lets a
job provision something pricier than the plan, the duration ceiling does not
cap the bill. The spend ceiling is the money-shaped backstop and the only one
of the three that names the currency.

**Add a new fault to the surface for "spend fast".** Rejected: the fault
surface already has the fault that reaches the ceiling quickly —
`machine_silent` is the unresponsive machine that keeps billing, which is
exactly the criterion's scenario. A second way to cause a failure would be a
new mechanism parallel to the surface, which ADR-0051 says not to do.

**Destroy without recording the checkpoints.** Rejected: it fails both merged
guarantees — the checkpoint written at the ceiling would not be visible to
ADR-0049's selection, and reaching a limit would destroy the work. The
emergency checkpoint records what the machine already wrote off itself before
the destroy, and the recorded set feeds the held-out-loss choice.

## Consequences

- `RunLimits` gains an optional spend ceiling (off by default: no existing
  caller changes) and the `guard` checks it beside the duration ceiling; the
  orchestrator freezes the price onto the limits the moment provisioning
  chooses the machine, and refuses an unenforceable combination
  (`budget_unconfigurable`) before anything is provisioned.
- `orchestrator.SPEND_CEILING_MINOR` is the named configuration with its
  derivation beside it; `config.py` is untouched.
- `Provider` gains `request_checkpoint(machine, job_id)`; the real provider
  signals the job's container by name (the remote script now names it) and
  polls result.json for the manifest, bounded by
  `EMERGENCY_CHECKPOINT_TIMEOUT_S`; the fake provider writes the checkpoints
  and reports the manifest, or returns None when silent or explicitly
  unresponsive.
- The trainer installs a SIGTERM handler that converts the signal into the
  graceful-exit path, so the real machine can be asked to save a final
  checkpoint. This half is exercised against the fake only; the real-hardware
  proof remains outstanding by design and is stated as such.
- A job stopped by the ceiling fails with `budget_exhausted`, records the
  emergency checkpoint outcome in its history, and destroys its machine
  through the confirmed teardown (ADR-0057). A run that produced no
  checkpoint records that honestly.

## Rollback

Revert the spend fields and guard check in `limits.py`, drop
`SPEND_CEILING_MINOR` and the emergency-checkpoint handling in
`orchestrator.py`, remove `request_checkpoint` from the provider seam, the
container naming from the remote script, and the trainer's SIGTERM handler.
The stall and duration controls remain exactly as they were; jobs no longer
have a money-shaped cap, which is the gap this record closes.
