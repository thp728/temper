# ADR-0060 — An out-of-memory failure retries automatically with the effective batch preserved

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/010-recovery-and-fault-injection.md`
- **Issue:** [#35](https://github.com/thp728/temper/issues/35)

## Context

The single most valuable automatic recovery for a user with no background in
this is the one that recovers from running out of device memory. The
mechanism was understood and unwritten: halve the per-step batch, double the
accumulation, and the optimisation is unchanged because the effective batch
-- `micro_batch_size × gradient_accumulation_steps`, the number of rows one
optimiser step is taken over -- is fixed at launch and preserved across the
retry. What makes it a recovery rather than a restart is exactly that: a user
who did not choose the batch size should not see their training change
because of it, and the loss curve should be continuous across the retry.

The fault surface (#24, ADR-0051) already gives us a way to make the failure
happen on demand: the trainer-side `oom` fault exhausts device memory so the
training step genuinely fails. The recovery's job is to react; the fault's
job is only to be the exhaustion, on both tiers. This work therefore
exercises the existing surface and adds no second way to cause a failure and
weakens no guard.

Three facts shape the room to work in:

- **An OOM retry is a new attempt against the same job.** The domain glossary
  says attempts become plural when a memory failure retries, and the
  resumption ticket (#60) will record resumed runs the same way. The retry
  therefore lives in the control plane's orchestrator, which re-provisions a
  fresh machine per attempt, tears it down before the next one provisions,
  and records each attempt's own machine, spec and outcome.
- **The retry is automatic; a divergence retry is a choice.** ADR-0055
  shipped divergence detection with a single retry offered as a *choice*,
  deliberately, because a diverging run usually means the data or the rate is
  wrong. A memory retry is automatic because preserving the effective batch
  cannot change the optimisation. The two decisions must never look alike in
  the record or the database, so the codes are asserted disjoint and the
  automatic-vs-choice distinction is explicit in both code and copy.
- **The trainer names the failure.** "result.json is always written,
  including on failure, so the orchestrator never has to parse logs to find
  out what happened" is a trainer rule. The trainer detects a device-memory
  exhaustion in the failed run's tail and names it with a stable code -- the
  platform's `training_oom` for a genuine exhaustion, the fault surface's own
  `simulated_oom` for a deliberately caused one -- and the orchestrator reads
  the code rather than re-deriving the cause from prose.

## Decision

**An out-of-memory failure retries automatically with the effective batch
preserved: the per-step batch halves and the accumulation doubles, asserted
as an invariant over every reachable state rather than as a particular
sequence; the escalation ladder is ordered and bounded; retries are capped;
and the user is told a recovery happened and what changed.**

- **Detection is the trainer's, and it distinguishes the deliberate from the
  real.** On a failed training run the trainer scans the tail for a CUDA
  out-of-memory line (`is_oom_tail`): a line naming "CUDA" and "out of
  memory", so a host that ran out of RAM is never named as a device
  exhaustion. A genuine exhaustion carries `training_oom`
  (`temper_core.memory_retry.REAL_OOM_CODE`, pinned equal to the trainer's
  own constant by a test -- the FAULT_ENV pattern, since the trainer image
  cannot import `temper_core`, ADR-0010); a run broken by the `oom` fault
  keeps the contract's `simulated_oom`. The orchestrator retries on
  `MEMORY_FAILURE_CODES = {simulated_oom, training_oom}` and nothing else.

- **The transformation is a property, not a sequence.** The invariant that
  matters is `per_step × accumulation` unchanged before and after every
  reduction, including at the floors. `temper_core.memory_retry` is pure --
  no I/O, no framework imports -- and its tests walk the whole reachable
  state space (a grid of batches and accumulations, plus ladder walks to the
  floor) asserting the product never changes, rather than pinning the
  particular walk 8/1 → 4/2 → 2/4. Halving is exact only when the effective
  batch divides by the halved batch; an odd batch whose accumulation cannot
  compensate exactly is not quietly altered -- the rung escalates instead,
  because changing the effective batch is the very thing the recovery exists
  to prevent.

- **The escalation ladder is ordered and bounded, with the floors recorded.**
  Halve the per-step batch to its floor (1 -- the resolved default already
  sits there), then gradient checkpointing (already enabled by the platform's
  calculated tier, so it is recorded as such and passed over), then the
  sequence length to its floor (512 -- a judgment, labelled), then more
  capable hardware (a card with strictly more memory than the one that
  OOMed), then surface. Each retry climbs one rung; batch and sequence length
  repeat within their rung; gradient checkpointing and hardware fire once.
  The retry cap (`MEMORY_RETRY_CAP = 4`) bounds the whole thing so a
  pathological same-configuration repeat cannot provision machines forever.

- **The retry is a new attempt with its own machine, recorded.** The
  orchestrator re-provisions per attempt and tears each machine down before
  the next provisions. Every attempt is recorded (`attempts` on the job):
  its number, outcome, error code, machine, the memory-relevant spec it ran,
  and -- for the attempt that OOMed -- the recovery's step (which rung, what
  changed, the preserved effective batch). Exhausting the cap or the ladder
  fails the job with the stable code `memory_retries_exhausted`, the plain
  reason, and the attempts recorded.

- **The user is told, in the history and on the record.** Each recovery emits
  a `log` event carrying the code `memory_recovery` and a plain sentence:
  what was halved, what was already in force, and that this is a memory
  recovery, not a divergence retry. The finished record renders a "Memory
  recovery" banner from the attempts list, and the failure codes
  `training_oom` / `memory_retries_exhausted` carry plain-language
  explanations.

- **The loss curve is continuous across the retry, tested on the emitted
  series.** Continuity is a claim about data a user sees, so it is tested on
  the emitted series: the retried attempt (effective batch preserved)
  continues the pre-OOM loss series with no jump, and the test's double emits
  a *jumped* series whenever the effective batch is not preserved, so the
  test fails exactly when the recovery would change the optimisation. The
  retried attempt runs a fresh machine; the literal step-for-step continuation
  of the curve on a fresh machine is resumption from the last checkpoint,
  which is issue #60's mechanism and is stated as outstanding.

## Alternatives considered

**Retry inside the trainer on the same machine.** Rejected: the glossary says
a memory retry makes attempts plural, which is a job-level concept; the
trainer "resolves nothing" and holds no defaults (ADR-0025, #83), so a
batch-halving decision would be a second resolver inside the image; and the
escalation ladder's "more capable hardware" rung is provisioning, which only
the control plane can do. The retry is a control-plane re-attempt.

**Resume from the last checkpoint as part of this ticket.** Rejected: #60
owns resumption (interrupted job resumes from its last checkpoint on a fresh
machine, restoring optimiser, scheduler and step). The byte-level restore of
a recorded checkpoint to a new machine is that ticket's mechanism; this one
records the attempts and the continuity contract, and the real-hardware
continuity observation depends on #60. Stated as outstanding rather than
quietly reinterpreted.

**Make the retry a choice, like divergence.** Rejected: the whole point of
the memory recovery is that it is automatic and cannot change the
optimisation. Making it a choice would turn a recovery into a restart the
user has to notice and accept. The two stay distinguishable: automatic here,
a choice there (ADR-0055), with disjoint codes.

**Model the `oom` fault as failing every attempt.** Rejected: on the fake
tier the fault is the demonstration of the recovery, so the first machine
exhausts memory and the retried machine -- a smaller batch -- fits. Firing on
every retried machine would turn the demonstration into a retry-until-surface
loop and prove nothing about the recovery itself. (The cap test uses a
dedicated machine that OOMs every attempt, so the surface path is proven
too.)

**Add the recovery's tunables to `config.py`.** Rejected: #29 owns the typed
settings layer this wave. The floors and the cap are named module-level
constants in `temper_core.memory_retry` with their derivations recorded
beside them, the way `orchestrator.py` records `TEARDOWN_CONFIRM_SAMPLES`.

## Consequences

- `temper_core.memory_retry` ships the pure transformation, ladder, floors,
  cap and codes, tested without hardware: the effective-batch invariant as a
  property over the reachable state space, the ladder's order and floors, and
  the codes asserted disjoint from the divergence codes.
- The orchestrator's `run_job` drives the retry loop: one `_attempt` per
  machine, teardown between attempts, `attempts` recorded, and
  `memory_retries_exhausted` when the cap or the ladder binds. The trainer
  names the failure (`training_oom` / `simulated_oom`) from the run's own
  tail.
- The `oom` fault now demonstrates the recovery on the fake tier: the first
  attempt fails with the fault's own code and the job completes after an
  automatic retry, which changes the fault's previously-terminal observable
  outcome -- the expected shape once a recovery exists (ADR-0051 anticipated
  exactly this).
- The job record gains `attempts` (a new column, published on the API), and
  the finished-job view renders the "Memory recovery" banner plus the
  plain-language explanations for the two memory failure codes.
- The hardware-only acceptance criterion -- a real run made to exhaust device
  memory and recover and finish -- is stated as outstanding in the PR body,
  not claimed via the fake; so is the byte-level checkpoint resumption (#60)
  that a fresh machine's literal curve continuation depends on.

## Rollback

Revert `temper_core.memory_retry`, the retry loop in `orchestrator.run_job`,
the trainer's OOM naming, the `attempts` column and its published model, and
the web banner plus explanations; the `oom` fault reverts to failing its job
with `simulated_oom` and jobs return to failing on any memory exhaustion,
exactly the shape the fault surface was left in by #24 so that this recovery
could be demonstrated rather than reasoned about.
