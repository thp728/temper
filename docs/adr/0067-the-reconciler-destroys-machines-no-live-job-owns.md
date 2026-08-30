# ADR-0067 — The reconciler destroys machines no live job owns

- **Status:** accepted
- **Date:** 2026-08-30
- **Spec:** `docs/specs/010-recovery-and-fault-injection.md`
- **Issue:** [#61](https://github.com/thp728/temper/issues/61)

## Context

Nothing looked for a machine that is billing with no job that owns it. The
orchestrator's final-block teardown covers the failures it can see; it cannot
cover the process disappearing. A worker that crashes mid-job leaves that
job's row non-terminal and its machine, if any, unwatched — the named,
accepted gap ADR-0066 recorded: `claim_next_job` only ever looks at `queued`
rows, so a machine owned by a `provisioning`/`preparing`/`training`/
`packaging` job is never reclaimed by anything that pass touches. An orphaned
GPU bills until someone notices, and billing is what this product is graded
on.

Spec 010 names the reconciler as a financial control, deliberately distinct
from durable execution: **durable execution recovers *the job*; the
reconciler protects *the money*.** An earlier argument that the reconciler
made durable execution unnecessary was retired during the build, and Spec 010
keeps that retraction visible because it was reasoning working backwards from
a price.

Two merged guarantees shape the room:

- **ADR-0057:** teardown is confirmed by absence across consecutive
  observations, `destroying` is not yet confirmed, and `normalize_status` is
  defined once in `provider.py` so the teardown and the reconciler cannot
  drift about whether a `Destroying` status is a stray.
- **ADR-0065:** a served endpoint is a billed, warm machine the control
  plane provisions and arms with idle/max timers. It is not a job, so a
  reconciler that matched machines only against jobs would destroy a machine
  a user is actively serving.

The criterion that shapes the matching is the one the issue states as the
reason the reconciler exists: a machine owned by a live job — even a job
whose worker crashed and is waiting for resumption (#60) — is matched and
left alone, because recovering that job is the resumption path's job, not
this pass's. The reconciler asks only whether a billing machine has an owner.

## Decision

**A scheduled reconciler lists machines, matches them against jobs that are
not in a terminal state and against served endpoints, destroys anything it
cannot account for, records what it did, and marks a job whose machine was
destroyed as unowned failed with a reason.**

- **It runs on a schedule, independent of any workflow.** It is a daemon
  thread in the worker process (ADR-0010 hosts the reconciler beside the
  claim loop), started at worker startup and stopping with the process. It
  assumes nothing about whether orchestration is healthy, because the case
  it exists for is that it is not. A long-running job claim cannot stall it,
  because it is not in the claim loop.

- **A machine is accounted for if it is owned by a job that is not in a
  terminal state, or by a running endpoint.** The ownership query reads every
  non-terminal state — never a `queued`-only subset — because
  `claim_next_job` only looks at `queued` rows and a machine owned by a
  `provisioning`/`preparing` job must therefore be matched here or this pass
  would destroy a machine a worker is actively driving. A served endpoint's
  machine is matched by the endpoint row (ADR-0065), never destroyed.

- **Anything it cannot account for is destroyed through the confirmed
  path.** The retry/escalation and consecutive-absence rules are extracted
  from `orchestrator._teardown` into `_confirmed_destroy`, so the two cannot
  drift about what a destroy looks like or when it is confirmed. A machine
  reported as `destroying` is not issued a second destroy — that is the
  double-destroy ADR-0057 exists to stop.

- **What it did is recorded, so an orphan is visible rather than silently
  cleaned.** Each decision lands one row in a new `machine_reconciliation`
  log (machine id, provider status, action, reason, owning job or endpoint,
  timestamp), and a destroyed machine with an owner also gets an event on the
  owner's own history.

- **A job that still claims a destroyed machine is marked failed with a
  reason.** When the reconciler destroys an orphan whose `machine_id` is
  recorded on a job row that is not terminal, it marks that job `failed`
  with the stable code `orphaned_machine` and a plain sentence — the "rather
  than left running forever" clause. Under the matching above this branch is
  reached by the record-first race, where a worker writes its `machine_id`
  onto a live row in the same instant the pass is acting on the unowned
  machine; the branch is exercised deterministically in the reconciler tests
  by simulating the stale ownership snapshot.

## Alternatives considered

**Reclaim machines owned by non-terminal jobs after a worker crash.** That is
the gap ADR-0066 names, but recovering a stuck job is durable execution's job
(the resumption path, #60), not this pass's. Destroying a machine whose job
row is non-terminal would wreck a job the resumption path is about to resume,
and marking every such job failed would fight the resumption path. The
reconciler protects the money for machines *no job owns*; a machine a job
owns, even a crashed one, has an owner who may yet resume it.

**Match machines only against `queued` jobs (what `claim_next_job` sees).**
Rejected: it would treat every machine owned by an already-claimed job as an
orphan and destroy machines a worker is actively driving. The ownership
query reads all non-terminal states for exactly this reason.

**Destroy machines owned by served endpoints.** Rejected: an endpoint is the
billed, warm machine ADR-0065 exists to protect. Matching the endpoint row is
the whole point of distinguishing endpoint machines from job machines.

**A silent clean-up (destroy without recording).** Rejected: the criterion is
explicitly that an orphan is *visible* rather than silently cleaned, and user
story 19 requires that every recovery that has happened can be seen. The
`machine_reconciliation` log is that record, and an owner job's history names
the destruction of its own machine.

**A fresh teardown loop in the reconciler.** Rejected: it would let the two
paths drift about what a destroy looks like and when it is confirmed
(ADR-0057's own warning). `_confirmed_destroy` is the single definition, and
`_teardown` and the reconciler both call it.

**A grace period keyed to provider-side creation time to close the
create-to-record window.** Considered and deferred: the window between
`provider.create` and the `machine_id` write is sub-second by the record-first
ordering the platform already commits to, and the provider's listing does not
expose creation timestamps through the seam. It is recorded here as a known,
bounded gap rather than silently assumed closed.

## Consequences

- The worker process now runs a reconciler thread; `reconcile_once()` is a
  testable pure pass, and the schedule is a domain constant
  (`RECONCILE_INTERVAL_S`, 30s) with its derivation recorded, not a
  deployment setting (ADR-0062).
- `db.py` gains the two ownership queries (`list_non_terminal_machine_ids`,
  `list_active_endpoint_machine_ids`), the owner lookup
  (`job_rows_for_machine`), and the reconciliation log
  (`record_reconciliation`/`list_reconciliations`); migration 0005 lays down
  the table.
- `orchestrator._confirmed_destroy` is the extracted retry/confirm loop;
  `_teardown` keeps its exact external behaviour and records on the job's
  history. The fake provider now removes a destroyed orphan from its listing,
  so the reconciler's fixture can be confirmed gone.
- An orphan is found within a pass interval and destroyed through the same
  confirmed path a job's own teardown uses; a job whose machine is destroyed
  as unowned is marked `failed` with `orphaned_machine` rather than left
  non-terminal forever.
- What the double proves is the orchestration logic — matching, confirmed
  destroy, recording, the endpoint protection, the mark-failed branch. What
  it cannot prove is the recovery on real hardware: a machine deliberately
  orphaned on a live provider and destroyed there. That is the adversarial
  tier's fixture, and the verification clause of Spec 010 keeps it stated as
  designed and unexercised until it has been run.

## Rollback

Revert this issue's commits. The reconciler thread stops starting at worker
startup, `_teardown` returns to its inline retry/confirm loop, the
`machine_reconciliation` table and its `db.py` functions are removed
(migration 0005 rolls back), and the fake provider stops removing destroyed
orphans. Machines owned by non-terminal jobs and served endpoints become
unwatched again — the gap this record exists to close.
