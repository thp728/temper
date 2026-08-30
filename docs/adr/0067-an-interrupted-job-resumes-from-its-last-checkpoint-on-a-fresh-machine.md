# ADR-0067 — An interrupted job resumes from its last checkpoint on a fresh machine

- **Status:** accepted
- **Date:** 2026-08-30
- **Spec:** `docs/specs/010-recovery-and-fault-injection.md`
- **Issue:** [#60](https://github.com/thp728/temper/issues/60)

## Context

A job that goes silent stops itself, a job that runs too long stops itself, a
job the user cancels destroys its machine — but a job that is *interrupted*
(its worker process killed, its machine dies) produced a result document
saying `training_failed` and stopped. The spec names the reason: resuming
from a checkpoint was proven on a probe (spike 4: a Qwen3-4B run resumed
from `checkpoint-2` to step 6) and had never happened through the product.

Resumption exists for a specific reason, and the reason dictates the shape.
**A resume that restores only the weights produces a run that silently
differs from an uninterrupted one** — the optimiser's momentum, the
scheduler's position and the step counter all live outside the weights. So
the recovery must bring back the whole checkpoint directory, and the product
must be able to prove it did.

Three prior decisions bound the design before this ticket started:

- **Checkpoints leave the machine as they are produced (ADR-0041/#37).** The
  machine writes each checkpoint to a scoped one-key grant, ring-buffered
  into a bounded number of slots, *during* training — so a checkpoint that
  only exists on a machine about to be destroyed is not a recovery
  mechanism. What survived an interruption is already in object storage,
  findable by slot.
- **A failure path is not done until it has been caused deliberately
  (ADR-0051/#24).** The fault surface's `worker_kill` fault kills the
  training process mid-run: no result document, only the checkpoints that
  had already left the machine. That is exactly the interruption a
  resumption exists to recover from, and it is how this recovery gets
  exercised without hardware.
- **Attempts are plural (ADR-0060/#35).** A memory failure's retry is a new
  attempt against the same job, each with its own machine, spec and outcome.
  The resumption ticket was recorded as "will record resumed runs the same
  way" — a resumed run is a second attempt, not a first attempt that got
  longer.

One honest gap inherited from ADR-0066, stated here so it is not silently
assumed solved: a *worker process* that crashes mid-job (as opposed to a
run within one job being interrupted) leaves the job claimed and non-terminal,
and reclaiming it is Spec 010's reconciler — sibling work, not this ticket's.
This ticket resumes a run that the orchestrator is already driving when its
attempt is interrupted.

## Decision

**A run that ends without a result document is an interruption, not a
training failure; the job discovers what survived in the checkpoint slots,
verifies it, and resumes from the latest checkpoint on a fresh machine —
restoring the whole checkpoint directory (optimiser, scheduler, step
position) rather than only the weights — as a new attempt with its own
machine, rate and outcome, bounded by a resume cap.**

- **The interruption is named, with its own stable code.** The trainer
  produces a result document on every path *including* failure; a stream
  that ends without one (the worker or machine died) is a different thing,
  and calling it `training_failed` hid the difference from clients that
  branch on codes. It now carries `interrupted`
  (`temper_core.resume.INTERRUPTED_CODE`), and the machine's own
  "no result.json" report (`trainer_no_result`, the shape a real killed
  trainer leaves) is mapped to the same code. A resumption can therefore
  key off exactly one stable signal: no result document arrived.

- **What survived is discovered, not assumed.** An interrupted run produced
  no manifest, so the control plane reads the checkpoint slots themselves
  (ADR-0041's bounded ring): each retained slot is streamed, hashed, and
  parsed for its `trainer_state.json` to learn the step and losses — one
  pass, streaming, never held whole (ADR-0010) — and anything that verifies
  becomes a checkpoint record exactly like a machine-reported one. The
  machine reported nothing, so the *slots* are the record; a slot that does
  not parse is skipped, because bytes that are not a checkpoint are nothing
  to resume from.

- **The resumed attempt is the frozen spec plus `resume_from_checkpoint`.**
  The optimisation must not change because the machine did: the resumed job
  spec is the frozen spec with the resume directive added (the trainer
  already passes it through to axolotl, which is what restores the optimiser,
  scheduler and step). The archive is streamed to the new machine and the
  remote script extracts it under the trainer's own output directory before
  the container starts — the same push channel the dataset uses, so the
  flat-memory rule holds for a large checkpoint as it does for a large
  dataset.

- **Resumption is a new attempt with its own machine, rate and outcome.**
  The existing attempts record (ADR-0060) gains the two fields a resumed run
  must carry: `rate`, the billing rate the attempt's own machine was
  provisioned at (the history of what ran is the history of what billed),
  and `resumed_from`, the step the attempt came back to. The interrupted
  attempt records `outcome: interrupted`, never `failed` — a run cut off is
  not a run that failed on its own terms.

- **Resumption is bounded, like memory retry.** Each resumption provisions a
  fresh machine, so an unbounded loop bills forever on a configuration the
  infrastructure keeps killing. `RESUME_RETRY_CAP = 2` bounds it (a
  judgment, not a measured figure, recorded with the constant): enough to
  recover from a single interruption — the adversarial tier's scenario — and
  small enough that a configuration that keeps getting interrupted surfaces
  with the attempts recorded rather than running up a bill. The job then
  fails with `resume_retries_exhausted`, distinct from a bare `interrupted`
  (nothing survived to resume from).

- **Teardown never stacks machines.** The interrupted attempt's machine is
  destroyed through the confirmed teardown path (ADR-0057) before the
  resumed attempt provisions, so the retry never runs two machines at once —
  the same guarantee the memory recovery already holds.

## The two acceptance criteria that could not be proven here

Two of issue #60's six criteria are trainer/hardware properties and are
documented here as **designed and unexercised, in those words** -- the
verification clause of spec 010 requires exactly that, not a claim:

- **Optimiser, scheduler and step are restored** is axolotl's behaviour when
  `resume_from_checkpoint` names a full checkpoint directory (proven once in
  spike 4). This ticket proves the *product* half -- the directive reaches
  the trainer, the archive it points at is the checkpoint that was actually
  written -- and the numerical half is unexercised: it needs the hardware
  tier to interrupt a real job and confirm the resumed run's loss curve is
  continuous.
- **A resumed run matches an uninterrupted one within a stated tolerance**
  is the same hardware property, and it too is unexercised. On the zero-cost
  tier nothing trains, so nothing can be compared numerically; the honest
  statement is that a resumption restores the whole checkpoint and therefore
  does not change the optimisation, and that this must be confirmed by
  interrupting a real job (a hardware run named in the PR body, owned by the
  submission's hardware-verification pass).

## Alternatives considered

**Resume only from machine-reported checkpoints (skip discovery).** A run
that *failed* with a result document already records its checkpoints, and
resuming from those is a smaller change. Rejected: the spec's headline
scenario is precisely the interruption — no result document — and its
checkpoints are in the slots but unrecorded. Without discovery the recovery
only works for runs that were not actually interrupted, which is the same
class of artifact as a large green suite over a product that could not train.

**Let the trainer download its own checkpoint from storage.** Rejected for
the reason ADR-0004 records: the machine is a pure compute node that holds no
read credential. The control plane already pushes the dataset; pushing the
checkpoint the same way (streamed, through the provider seam) adds no new
boundary and keeps the machine credential-free.

**Reuse the interrupted machine rather than provisioning a fresh one.** The
machine that died cannot be trusted to come back, and suspension (spike 8)
is not cheaper on this provider; resumption on a fresh machine is what the
spec mandates and what the money-safety guarantee (one machine at a time)
allows.

**Resume without a cap.** Rejected: each resume is a billed machine, and a
recovery that cannot stop billing on a pathological repeat is a recovery the
operator cannot leave unattended. The cap is small and recorded as a
judgment, the same way `MEMORY_RETRY_CAP` is.

## Consequences

- `temper_core.resume` holds the pure decisions (interruption code, latest
  checkpoint, resumed spec, cap) with no I/O, tested as a table.
- The orchestrator's `run_job` gains an interruption branch beside the memory
  and divergence branches; the attempts record carries `rate` and
  `resumed_from`; the remote script can extract a checkpoint and name
  `resume_from_checkpoint`.
- The `worker_kill` fault now *demonstrates* the recovery: the first machine
  is killed, the run resumes from the checkpoint that had left it, and the
  job completes. `test_fault_surface`'s expectation for `worker_kill` moves
  from `("failed", "training_failed")` to `("complete", None)` for exactly
  this reason — the fault is still named in the history, so a deliberately
  broken run is never mistaken for a real one.
- A job interrupted with nothing to resume from fails with `interrupted`
  (its checkpoints, if any, still recorded), and a job interrupted past the
  cap fails with `resume_retries_exhausted`; both surface the attempts.
- The fake provider writes checkpoints as the real trainer does — a tar of
  the checkpoint directory — so discovery is exercised against the object
  shape it will parse on hardware, and each `create` returns a distinct
  machine id so the attempts record names which machine ran which attempt.

## Rollback

Revert this issue's commits. `_consume`'s no-result path returns to
`training_failed`, the interruption branch and discovery helpers are removed
from the orchestrator, `resume_from_checkpoint`/resume extraction leave the
remote script, `rate` and `resumed_from` leave the attempts record (and
`AttemptRecord`), and the fake provider's checkpoints return to opaque bytes
with a single `MACHINE_ID`. The pure `temper_core.resume` module is removed
with them. The `worker_kill` fault surface expectation reverts to
`("failed", "training_failed")`.
