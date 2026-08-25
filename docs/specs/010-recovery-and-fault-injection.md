# Spec 010 — Recovery, and the faults that prove it

**Status:** ready for tickets
**Phase:** B, band 2 (the defensibility path)
**Depends on:** Spec 008 (a worker process separate from the request path), Spec 006 (object storage for checkpoints), Spec 004 spike 8 (whether the provider can suspend a machine more cheaply than destroying it)
**Produces:** ADRs that a failure path is not done until it has been caused deliberately, that durable execution recovers the job while the reconciler protects the money, and that teardown is confirmed across consecutive observations
**Assumes:** ADR-0002 (a stalled job and an over-long job are stopped separately, and named separately), ADR-0003 (cancellation is destructive — reopened only if spike 8 finds suspension is cheaper)

## Problem Statement

The brief says every flow except authentication and billing must be complete, and
the failure paths are flows. They are also the flows this product currently
handles worst, and the reason is structural: **there is no way to make a failure
happen on purpose.**

What exists today: a job that goes silent stops itself; a job that runs too long
stops itself; a job the user cancels destroys its machine; the teardown path runs
in a final block and has fired for real, on a real failure, with money on the
line. That is a genuine foundation and more than most demonstrations have.

What does not exist:

- **Out-of-memory recovery.** The single most valuable automatic recovery for a
  user with no background in this. The mechanism is understood and unwritten.
- **Divergence handling.** A run whose loss becomes meaningless burns its full
  duration and hands back a worthless result.
- **Resumption after an interruption.** Resuming from a checkpoint is proven on
  a probe and has never happened through the product. The control plane cannot
  currently survive its own restart, let alone drive a resumption.
- **A reconciler.** Nothing looks for a machine that is billing with no job that
  owns it. The final-block teardown covers the failures it can see; it cannot
  cover the process disappearing.
- **A cap on spend enforced outside the training process.**

And the teardown confirmation itself has a known weakness recorded during the
build: after a destroy, the provider's listing is eventually consistent — the
machine reads absent, then reappears as destroying, then goes absent for good.
The check reads at the moment it returns absent, and **has never been tested
against a destroy that actually failed.**

Underneath all of it is the reason none of these are finished: **every one is
verified by something going wrong, and nothing in this system can make something
go wrong on demand.** A recovery path that has only ever been reasoned about is
the same class of artifact as a large green test suite over a product that could
not train.

## Solution

**Build the ability to cause failures first, then build the recoveries, then
cause them.**

A fault-injection surface, driven by configuration and documented, that makes
each failure happen at a chosen point: exhaust memory during training, drive the
loss to a meaningless value, kill the worker mid-run, stop the machine
responding, leave a machine running with no job that owns it, make the provider
refuse a destroy. It is not test scaffolding hidden in a test directory — it is
part of the product, off by default, and its presence in the repository is the
answer to *"how do you know your recovery works?"*

On top of it, the recoveries:

- **Out of memory retries with the effective batch preserved.** Halve the
  per-step batch, double the accumulation, so the optimisation is unchanged and
  the loss curve is continuous across the retry. If the per-step batch is already
  at its floor, escalate through gradient checkpointing, then sequence length,
  then more capable hardware, then surface it. This is what makes it a recovery
  rather than a restart: a user who did not choose the batch size should not see
  their training change because of it.
- **Divergence aborts with a plain cause**, rather than burning the remaining
  hours, with a single optional retry at a reduced learning rate.
- **An interrupted job resumes from its last checkpoint on a fresh machine**,
  restoring not only weights but the optimiser, scheduler and step position — a
  resume that restores only weights produces a run that silently differs from an
  uninterrupted one.
- **A durable workflow drives the job**, so the sequence survives the process
  that started it. This is the answer to the question a reader whose business is
  GPU hours asks first: *what happens when your control plane dies while my
  machine is billing?*
- **A reconciler independently destroys any machine with no live job that owns
  it**, and marks that job failed with a reason.

**The last two are not redundant and the distinction is the point.** Durable
execution recovers *the job*: the workflow resumes from where it stopped. The
reconciler protects *the money*: it assumes nothing about whether a workflow
exists and asks only whether a billing machine has an owner. They defend
different failures, and the reconciler would stay even with durable execution
working perfectly. An earlier argument that the reconciler made durable
execution unnecessary was retired during the build, and it is worth keeping that
retraction visible because it was reasoning working backwards from a price.

**Teardown confirmation requires absence across consecutive observations**, and
treats a machine reported as destroying as not yet confirmed.

## User Stories

1. As a user, I want a job that runs out of memory to recover automatically, so that a configuration I did not choose does not end my run.
2. As a user, I want a memory recovery to leave my training mathematically unchanged, so that the result is what I asked for and not a quietly different one.
3. As a user, I want to be told that a recovery happened and what changed, so that the run's history is honest.
4. As a user, I want repeated memory failures to stop rather than retry forever, so that I am not billed for a loop.
5. As a user, I want a job whose training becomes meaningless to stop early, so that I do not pay for hours that cannot produce anything.
6. As a user, I want a plain-language explanation of why it stopped, so that I know whether to change my data or my settings.
7. As a user, I want an interrupted job to continue rather than start over, so that an infrastructure problem does not cost me the hours already spent.
8. As a user, I want a resumed job to produce the same result as an uninterrupted one, so that recovery is not a quiet change in what I trained.
9. As a user, I want my job to survive the platform restarting, so that an operational event on their side is not my problem.
10. As a user, I want a machine that stops responding to be detected and my job ended with a reason, so that I am not billed for silence.
11. As a user, I want a hard ceiling on what a single job can spend, so that a runaway cannot consume everything.
12. As a user, I want a job stopped by that ceiling to save its progress first, so that reaching a limit does not also destroy the work.
13. As a user, I want a job stopped by a safety limit reported as a failure with a specific reason, so that it is distinguishable from something I chose.
14. As a user, I want cancelling a job to stop the spending immediately, so that the control means what it says.
15. As a user, I want every failure to carry a stable code and a human sentence, so that I can act on it and so that it can be looked up.
16. As an operator, I want any machine without a live job that owns it destroyed automatically, so that an orphan cannot bill indefinitely.
17. As an operator, I want teardown confirmed by more than one observation, so that eventual consistency cannot make a failed destroy look like a success.
18. As an operator, I want a failed destroy retried and escalated, so that a machine the provider refused to remove is not silently forgotten.
19. As an operator, I want to see every recovery that has happened, so that a systematically failing configuration is visible rather than absorbed.
20. As an operator, I want to cause each failure deliberately, so that a recovery path can be proven rather than argued.
21. As an operator, I want the fault surface off by default and obvious when on, so that it cannot be mistaken for real behaviour.
22. As a developer of this product, I want an interrupted workflow to resume from its exact step, so that recovery is not re-running side effects.
23. As a developer of this product, I want provisioning a machine to happen once even if its step is retried, so that a retry cannot double the bill.
24. As a reviewer, I want to see how each failure was caused, so that I can tell a proven recovery from a described one.

## Implementation Decisions

**The fault surface is configuration-driven and lives inside the existing seams.**
Provider-side faults are behaviour of the provider seam; trainer-side faults are
an environment switch the trainer reads. No new module and no new boundary — a
fault-injection framework would be a seam introduced to test seams.

**Every fault is named and its name appears in the run's history**, so a run that
was deliberately broken can never be mistaken for one that broke.

**Retry preserves the effective batch as an invariant, not as a behaviour.** The
product of per-step batch and accumulation is fixed at launch and every escalation
step maintains it. Asserting the invariant is more useful than asserting a
particular sequence of reductions, because the reason the recovery exists is that
the optimisation must not change.

**Escalation is ordered and bounded**, and when the order is exhausted the job
fails with the reason and the attempts recorded.

**Divergence is detected on the measurements the platform already streams**,
using thresholds published in the research rather than invented here, and it
aborts rather than retrying by default. A single reduced-rate retry is offered as
a choice rather than performed automatically, because a diverging run usually
means the data or the rate is wrong and repeating it is rarely the answer.

**Checkpoints go to object storage as they are written**, not at the end. A
checkpoint that only exists on a machine that is about to be destroyed is not a
recovery mechanism.

**Resumption is a new attempt against the same job.** The job record already
distinguishes the job from its executions; a resumed run is a second attempt
carrying its own machine, rate and outcome, so the history says what actually
happened rather than presenting one continuous run that was not.

**The workflow's steps are individually idempotent, and provisioning is the one
that matters.** A retried step that creates a second machine is the failure that
costs money, so the machine's identity is recorded before anything else can fail
— which is why readiness was separated from creation in the first place.

**The reconciler runs on a schedule, independent of any workflow.** It lists
machines, matches them against jobs that are not in a terminal state, destroys
what it cannot account for, and records what it did. It assumes nothing about
whether the orchestration is healthy, because the case it exists for is that the
orchestration is not.

**The spend ceiling is enforced outside the training process** — a process that
has stopped responding cannot enforce its own limit. On reaching it: checkpoint,
terminate, destroy, and mark the job failed with the specific reason.

**Teardown confirmation requires consecutive absent observations and treats a
destroying state as not yet confirmed.** This closes a known weakness recorded
during the build, and the confirmation path is itself exercised by a fault that
makes the provider refuse.

**If the provider can suspend a machine more cheaply than destroying and
recreating one, the cancellation decision is reopened**, and a retry stops paying
a cold start per attempt. That finding comes from spike 8; if suspension does not
exist or is not cheaper, the current decision stands unchanged and the spike is
the evidence for it.

## Testing Decisions

**A good test here asserts on the outcome and the invariant, never on the
sequence of internal steps.** For memory recovery the assertion is that the
effective batch is unchanged and the job completes; for teardown it is that no
machine remains; for resumption it is that the result matches an uninterrupted
run. Tests that pin the order of escalation will fail the first time the order is
improved, which is exactly when they should not.

**Two tiers, and the split is deliberate.** The zero-cost provider proves the
orchestration logic — that a memory failure triggers a retry with the invariant
held, that a refused destroy is retried and escalated, that the reconciler
destroys an unowned machine, that a workflow resumes from its step. These run on
every push and cost nothing. **They also cannot prove the recovery works**,
because a simulated memory failure is not a memory failure, and saying so
unprompted is more useful than the tests are.

**The second tier is real hardware, deliberately broken.** This is the
adversarial run cluster, and it is the whole point of the spec:

- A configuration that genuinely exhausts device memory, recovering and finishing.
- A learning rate that genuinely diverges, aborting with the right reason.
- A worker killed mid-training on a live job, resuming and completing without a
  second machine appearing.
- A machine deliberately orphaned, found and destroyed by the reconciler.
- A destroy the provider refuses, retried and escalated.

**These share machines.** One provisioned job can absorb several of these faults
in sequence, and batching them is the difference between an afternoon and a week.

**Prior art:** the existing runaway-limit tests are the closest model — a
simulated clock and a controlled stream, asserting the outcome rather than the
mechanism. The zero-cost provider already supports failing at a chosen stage,
going silent without ending, and refusing a destroy a chosen number of times;
this spec extends that vocabulary rather than replacing it.

**The verification clause.** Not done until every failure in the second tier has
been caused on real hardware and the recovery observed through the interface. A
recovery path proven only against the double is explicitly not done, and any
that cannot be exercised before submission is documented as designed and
unexercised, in those words.

## Out of Scope

- **Automatic hyperparameter tuning.** Recovering from a failure is not the same
  as searching for a better configuration.
- **Preemption recovery.** Interruptible machines are not available for this
  workload on this provider — not deferred, unavailable.
- **Multi-machine training.** A job spans one machine; more devices, not more
  hosts.
- **Alerting and on-call.** There is no operator to page.
- **Reconciling spend against an invoice.** Costs stay derived from the event
  record against a stored rate.

## Further Notes

The fault surface is worth defending on its own terms, because it will look like
test code that leaked into the product. It is not: it is the mechanism by which a
claim about reliability becomes checkable by someone who did not write it. A
reviewer can turn on a memory fault and watch the recovery happen. That is a
categorically different kind of evidence from a passing test, and it directly
answers the strongest question this project has to answer about itself — the one
raised by a large green suite over a product that could not train.

The scheduling note that matters: **the fault surface ships before the
recoveries, not alongside them.** Building the recovery first means verifying it
by reasoning, and the entire argument of this spec is that reasoning about
recovery paths is what got this project into trouble once already.
