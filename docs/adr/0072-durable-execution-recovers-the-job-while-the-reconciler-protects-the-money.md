# ADR-0072 — Durable execution recovers the job while the reconciler protects the money

- **Status:** accepted
- **Date:** 2026-08-31
- **Spec:** `docs/specs/010-recovery-and-fault-injection.md`
- **Issue:** [#68](https://github.com/thp728/temper/issues/68)

## Context

ADR-0066 moved orchestration into a worker process that claims jobs, and
recorded the gap this issue exists to close: **a worker crash mid-job leaves
the job's row non-terminal and unclaimable** -- `claim_next_job` only ever
looked at `queued` rows, by design, since a job actively being driven must
never be claimed twice. ADR-0068 then shipped the reconciler, which protects
*the money* for machines *no job owns*; it deliberately leaves a machine owned
by a non-terminal job alone, even a job whose worker crashed, because
recovering that job is durable execution's job, not the reconciler's. What did
not exist was that recovery.

Spec 010 names the distinction that shapes this record, and the distinction is
the point: **durable execution recovers the job -- the sequence resumes from
where it stopped rather than re-running side effects; the reconciler protects
the money -- it assumes nothing about whether orchestration is healthy and asks
only whether a billing machine has an owner.** They defend different failures,
and the reconciler would stay even with durable execution working perfectly.
An earlier argument that the reconciler made durable execution unnecessary was
retired during the build, and the retraction is kept visible because it was
reasoning working backwards from a price.

Three merged guarantees bound the room before this ticket started:

- **ADR-0057/0063:** a machine's identity is recorded before anything else can
  fail. Provisioning's retry safety is "check the row first": a retried step
  that finds `machine_id` already on the row knows a machine already exists and
  must not create a second one. The sub-second window between `create` and the
  record is acknowledged in ADR-0068 as a bounded, accepted gap.
- **ADR-0069/#60:** a run that ends without a result document is an
  interruption, and resumes from the latest checkpoint on a fresh machine,
  restoring the whole checkpoint directory. What survived an interruption is in
  the checkpoint slots because ADR-0041/#37 writes them off as training
  produces them.
- **ADR-0066/#51:** the claim is `SELECT ... FOR UPDATE SKIP LOCKED`, which
  makes two workers claim the same job impossible. Durable execution must not
  weaken either half of that.

## Decision

**The job row's `status` is the step cursor; the worker holds a claim lease it
refreshes while it drives; a job whose lease goes stale is reclaimed and
resumed from its recorded step -- with the machine torn down first so recovery
never stacks machines, the run handed to #60's checkpoint resumption rather
than duplicated, and the finished-but-uncollected run re-collected from
storage.**

- **A claim lease, refreshed by a heartbeat.** `jobs` gains `claimed_at`
  (migration 0006). The worker stamps it when it claims and a daemon heartbeat
  thread refreshes it every `HEARTBEAT_INTERVAL_S` (5s) while it drives; the
  orchestrator stamps it at entry so a direct driver owns its job too. A job
  whose lease has gone stale past `CLAIM_STALE_AFTER_S` (30s, six missed beats)
  is abandoned. The heartbeat is a separate thread precisely so a live driver
  never looks abandoned through the minutes-long stretches (`await_ready`, a
  silent stream) during which the job writes nothing to the database. The
  invariant `HEARTBEAT_INTERVAL_S < CLAIM_STALE_AFTER_S` is pinned by a test.

- **`claim_next_job` reclaims the oldest abandoned job after finding nothing
  queued.** Same `FOR UPDATE SKIP LOCKED`, so a stale job is never handed to
  two drivers (proved by the same concurrency tests as the fresh claim). The
  reclaim keeps the job's status -- that is the step cursor the new driver
  resumes from -- stamps a fresh lease, and records the reclaim in the job's
  own history. A *fresh* lease is never reclaimed (that is how a live worker is
  never double-driven) and a *NULL* lease (nobody ever claimed the row) is
  never treated as abandoned.

- **`run_job` resumes a reclaimed job from its recorded step.**
  - `provisioning`: nothing is recorded yet, so the attempt re-runs. The
    bounded create-to-record window ADR-0068 records is inherited unchanged:
    if the previous worker's machine was created but not recorded, the
    reconciler destroys it as unowned -- the money is protected, and the job
    proceeds.
  - `preparing`/`training`: the recorded machine is torn down through the
    confirmed teardown (ADR-0057) *before* anything new provisions, so
    recovery never stacks machines -- "no second machine appears as a result of
    recovery" is kept true by teardown-before-provision when the machine was
    already created. The run is then handed to #60's interruption handling --
    discover the surviving checkpoints, resume from the latest on a fresh
    machine -- which composes rather than duplicating it.
  - `packaging`: the run already finished training; only collection remains.
    The result manifest is now persisted the moment packaging begins (it is the
    record of what the run produced), so a packaging recovery re-verifies the
    stored artifact/checkpoints/delivery and finishes -- re-run, only
    re-collected. Nothing is re-trained.

- **The recovered drive continues the job's own history and bounds.** The
  attempts record, the resume count and the memory-retry count are seeded from
  the row, so a reclaimed job does not reset the caps (which bound the job's
  total, not one driver's work). The hyperparameter spec the last attempt ran
  is reconstructed from the attempts record, so a resumed run keeps a memory
  recovery's halved batch rather than quietly reverting to the frozen request
  -- the exact "recovery must not change the optimisation" ADR-0069 exists to
  prevent.

- **The worker is already a separate process, and stays one.** It polls the
  database and calls `orchestrator.run_job`; a control-plane restart does not
  stop it. This is proved by a test that creates a job, takes the control plane
  down, lets the worker drive it to completion, and reads the finished record
  after a restart.

## The two-recovery split, kept visible

Durable execution (this record) and the reconciler (ADR-0068) both act on a
job whose worker died, and the reason both stay is the failure each is for.
The reconciler answers "is a machine billing with no owner?" -- it is a
financial control, running on its own schedule, assuming nothing about whether
orchestration is healthy. Durable execution answers "can the job continue?" --
it is the workflow, and it is the only thing that can finish the job. The
reconciler must not destroy the machine of a job being recovered, and it does
not: its ownership query reads every non-terminal state, and a job this path is
actively recovering is non-terminal. This path must not rely on the reconciler
to clean up after a recovery, and it does not: the recorded machine is torn
down by the recovery itself, before anything new provisions. The two are
complementary, not redundant, and the reconciler would stay even with durable
execution working perfectly.

## What the double proves, and what still needs hardware

The zero-cost tier proves the orchestration logic (Spec 010's testing
decision): the reclaim finds an abandoned job and only an abandoned one; a live
job is never stolen; a reclaimed `training` job's machine is destroyed before
the resumption provisions (no second machine), the run resumes from its last
checkpoint and completes with its record and artifact intact; a reclaimed
`packaging` job is re-collected without provisioning a single machine; the
heartbeat keeps a driving worker from looking abandoned; a worker survives a
control-plane restart.

What the double cannot prove is the criterion the issue states as the finish
line -- **a worker killed during a live job on real hardware, and the job
completing anyway, observed rather than asserted.** A simulated interruption is
not an interruption, and saying so unprompted is more useful than the tests
are. That run is the adversarial tier's fixture, stated here as designed and
unexercised, in those words, until it has been run. The same is true of the
#60 half this recovery composes with: restoring the optimiser, scheduler and
step position is the trainer's behaviour with `resume_from_checkpoint`, proven
once on a probe and needing the hardware tier to confirm on a live job.

## Alternatives considered

**A task queue or workflow engine (Temporal was in ADR-0020's plan) instead of
a lease and a step cursor.** Rejected for the reason ADR-0066 rejected a queue:
the job row already *is* the workflow state, `SELECT FOR UPDATE SKIP LOCKED`
already *is* the claim, and a lease is the one missing primitive. Introducing a
workflow engine would duplicate a state machine that already exists and is
already tested, for machinery this issue does not need.

**Reclaim every non-terminal job unconditionally (no lease).** Rejected: it
would let a second worker steal a job a live worker is driving. Two workers
driving one job is the double-machine failure the reconciler and this path both
exist to prevent; the lease is what tells abandoned from merely quiet.

**Re-attach to the machine a killed worker left behind and continue the stream
in place.** Rejected: the stream is a one-shot SSH session; there is no way to
resume a dead session, and re-running the script against a machine whose
container is still training is not idempotent (`docker run --name` collides).
Teardown-before-provision plus #60's checkpoint resumption is the honest
recovery: what survived is in the slots, and the run continues from there.

**Persist nothing at packaging and treat a packaging recovery as a resumption.**
Rejected: that would re-train a run whose training already completed, wasting
the hours spent and handing back a *different* run. Persisting the result
manifest the moment packaging begins is the small change that makes collection
exactly resumable.

**Let a reclaimed job restart from the beginning.** Rejected outright: it
contradicts the criterion ("resumes from its exact step rather than from the
beginning") and would re-provision a machine when one is already recorded --
the double-machine failure the whole design exists to avoid.

## Consequences

- `jobs` gains `claimed_at`; migration 0006 lays it down and 0001's baseline
  already anticipates nothing -- the column is nullable and existing rows read
  NULL, which for them is the honest "no lease was ever taken".
- `claim_next_job` now reclaims abandoned non-terminal jobs; the reclaim is
  visible in the job's history (a `job_reclaimed` event).
- `run_job` resumes a reclaimed job from its recorded step; `_attempt` is
  otherwise unchanged (the seam ADR-0066 proved holds). The result manifest is
  persisted when packaging begins and cleared on a cancelled terminal state.
- The worker runs a heartbeat thread per claimed job and a reconciler thread;
  both stop with the process.
- A job whose worker dies is reclaimed within a staleness window and resumes
  rather than leaking; a job the infrastructure keeps killing still surfaces
  with the attempts recorded (the #60 resume cap is not reset by a reclaim).

## Rollback

Revert this issue's commits. `jobs` loses `claimed_at` (migration 0006 rolls
back), `claim_next_job` returns to queued-only, the worker's heartbeat thread
and the orchestrator's reclaim dispatch are removed, the result manifest is
persisted only at the terminal transition again, and the durable-execution
tests leave the suite. The named gap ADR-0066 recorded returns: a worker that
crashes mid-job leaves that job stuck and its machine unwatched until the
reconciler's owner query sees the job reach terminal -- which it would not.
