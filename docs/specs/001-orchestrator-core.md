# Spec 001 — Orchestrator core: streaming, cancellation, and runtime limits

**Status:** ready for tickets
**Phase:** A (target: Thu 2026-08-20 evening)
**Depends on:** nothing
**Produces:** ADR-0001 (event channel over SSH stdout), ADR-0002 (stall and duration limits), and the decision-record index
**Assumes:** ADR-0004 (the VM is a pure compute node), produced by Spec 002

## Problem Statement

A user who launches a job watches nothing happen for roughly four minutes, cannot stop it, and is not protected from a job that hangs and bills until somebody notices.

Concretely, from the user's side:

- **The run is silent.** The whole build-and-train phase — measured at 251 seconds on 2026-08-19 — produces no output at all. Events do appear afterwards, but all of them carry the same timestamp, so even the record of what happened cannot say how long anything took.
- **The loss is nowhere.** The training framework's own output never leaves the machine; it is written to a file on a VM that is then destroyed. There is no loss number available during a run or after one. A fine-tuning product that cannot show a loss curve is a hello-world version, and this project rules that out.
- **There is no way to stop a job.** A `cancelled` state exists in the lifecycle and no path reaches it. A user who realises they picked the wrong dataset can only wait and pay.
- **There is no protection against a hung job.** A safety constant exists and is never read. A job that wedges bills for as long as nobody is watching.
- **Teardown confirmation arrives after the job says it finished.** The "VM destroyed" event is appended after the terminal transition, so any client that stops polling on a terminal status — which is what a reasonable UI does — never sees the one confirmation it most wants.

Underneath all five: the money-spending path has **no test coverage**. The orchestration function builds its provider client internally and shells out directly, so it cannot be exercised without provisioning a real GPU. Every behaviour above would be added to code that nothing tests.

## Solution

Open the output channel, and use it as the place where the other four problems are solved.

The remote command stops being one blocking call that returns everything at the end, and becomes a stream of lines that arrive as they are produced. That stream gives the orchestrator a loop — and a loop is somewhere to check whether the user has asked to cancel, whether output has stopped arriving, and whether the job has been running too long. Three problems that look separate in the backlog are consequences of the same missing structure.

To make any of it testable, the provider becomes an injected dependency rather than something the orchestrator constructs. One seam, one fake, and the entire money-spending path becomes exercisable with no GPU and no spend.

From the user's perspective:

- Log lines appear as the job produces them, each stamped when it actually happened.
- Training step, loss and epoch are surfaced as structured metrics, not buried in prose.
- A cancel action exists, stops the job promptly, and says plainly that it produces no adapter.
- A job that stops producing output is killed automatically and reports that it stalled — which is a different outcome from a user cancelling and from a job failing.
- A job that runs longer than the maximum permitted duration fails with a named reason.
- The confirmation that the machine was destroyed arrives **before** the job is reported finished.

## User Stories

1. As a user watching a job, I want log lines to appear as they are produced, so that I can tell the job is progressing rather than hung.
2. As a user watching a job, I want each event to carry the time it actually occurred, so that I can see how long each phase took.
3. As a user watching a job, I want to see the current training step, so that I can estimate how far through the run I am.
4. As a user watching a job, I want to see the current loss value, so that I can judge whether the model is learning.
5. As a user watching a job, I want to see the current epoch, so that I can relate progress to the number of epochs I asked for.
6. As a user watching a job, I want structured metrics kept distinct from ordinary log output, so that a chart can be added later without re-parsing prose.
7. As a user watching a job, I want to know whether I am in image build, model download, or training, so that I know which phase is slow.
8. As a user watching a job, I want quiet periods to be explicable, so that I do not mistake a legitimate image pull for a hang.
9. As a user who picked the wrong dataset, I want to cancel a running job, so that I stop paying for a result I do not want.
10. As a user who cancels, I want the machine destroyed promptly, so that billing stops rather than continuing to the end of training.
11. As a user who cancels, I want to be told plainly that no adapter will be produced, so that I do not go looking for one.
12. As a user who cancels, I want the job to end in a state that is clearly distinct from failure, so that my own action is not recorded as a defect.
13. As a user who cancels twice by double-clicking, I want the second request to be accepted quietly, so that I do not see a spurious error.
14. As a user who cancels a job that has already finished, I want a clear refusal with a stable reason, so that I understand nothing was undone.
15. As a user who cancels during provisioning, before any machine exists, I want the job to stop cleanly, so that cancellation is not only available once training starts.
16. As a user whose job hangs, I want it killed automatically after a defined period of silence, so that a wedged job does not bill overnight.
17. As a user whose job is killed for stalling, I want that reported as stalling specifically, so that I can distinguish it from a crash or my own cancellation.
18. As a user whose job runs longer than the permitted maximum, I want it stopped with a named reason, so that I understand it was a policy limit and not a bug.
19. As a user, I want the duration ceiling to be generous enough that no legitimate run on the supported models reaches it, so that the limit never surprises me.
20. As a user whose job ends for any reason, I want confirmation the machine was destroyed **before** the job is reported finished, so that a client which stops polling on a terminal state still sees it.
21. As a user reading a failed job, I want a stable machine-readable code alongside the human message, so that I can handle categories of failure programmatically.
22. As a user, I want every terminal outcome to record why it ended, so that no job reaches a terminal state without an explanation.
23. As an operator, I want teardown to run even when training raises an unexpected exception, so that no machine is orphaned by a code path nobody anticipated.
24. As an operator, I want teardown attempted repeatedly before giving up, so that a single transient provider error does not leave a machine running.
25. As an operator, I want a machine that survives teardown reported loudly and unmistakably, so that I destroy it manually before it bills for hours.
26. As an operator, I want the stall detector's resets visible in the event stream, so that I can see the mechanism working rather than trusting it silently.
27. As an operator, I want the runtime limits configurable without a code change, so that they can be tuned per deployment.
28. As a developer, I want the orchestration path exercisable without provisioning a GPU, so that the suite that guards it actually gets run.
29. As a developer, I want to simulate a stalled job in a test without waiting the real timeout, so that the suite stays fast.
30. As a developer, I want to simulate a provider that fails to destroy a machine, so that the stray-instance path is tested rather than assumed.
31. As a developer, I want the streaming interface to be identical whether output arrives over SSH or an HTTPS push, so that deploying the control plane later does not require rearchitecting.
32. As a developer, I want a single injection point for all provider interaction, so that there is one thing to fake and one thing to reimplement.
33. As a reviewer, I want the ordering of teardown relative to the terminal transition asserted by a test, so that a future refactor cannot silently reintroduce the original bug.

## Implementation Decisions

### The Provider seam

All interaction with the compute provider moves behind one protocol, injected into the orchestration function with the real implementation as the default. This is the only new seam in the change.

```
select_gpu(preference)        -> GpuChoice      # type, price, currency
create(gpu, storage, name)    -> Machine
destroy(machine_id)           -> None
list_machine_ids()            -> list[int]
push(machine, payload, dest)  -> None           # bytes to a path on the machine
stream(machine, script)       -> Iterator[str]  # lines, yielded as they arrive
```

`stream` is deliberately shaped so that its contract — an iterator of lines — holds whether the underlying transport is SSH stdout today or an outbound HTTPS push from a deployed control plane later. Swapping transports becomes a new implementation of this protocol rather than a change to the orchestration logic.

The default implementation wraps the JarvisLabs SDK and SSH. It keeps the two properties that were learned expensively and must not regress: readiness distinguishes *unreachable* from *authentication failed*, and teardown is independently confirmed by listing instances rather than trusting the destroy call's return value.

### Opening the channel

Output is currently redirected three times between the training framework and the orchestrator, and all three must be opened for any of them to matter:

1. The trainer writes the framework's output to a file handle rather than to the container's standard output.
2. The remote script redirects the container's output to a file on the machine.
3. The orchestrator captures the remote command's output and only reads it after the command exits.

All three change. The container's output is unbuffered explicitly, because reintroducing buffering at the innermost layer would silently undo the other two.

### Line classification

A pure function maps one output line to a classified event. Lines carrying training step, loss, or epoch become `metric` events with those values extracted into structured fields; everything else becomes a `log` event. Grad norm, learning rate and throughput are recognised but not promoted — they stay in `log` output for now, and promoting them later is a change to this one function.

Every event is stamped at the moment its line is read, not when the command completes.

### Cancellation

Cancellation is **destructive**: the machine is destroyed and no adapter is produced. A partially trained adapter handed to a user who asked to stop invites them to mistake it for a finished model, which is the same class of confusion that the adapter-selection and adapter-config work already exists to prevent.

- A cancellation request sets a flag on the job record; the streaming loop observes it on each iteration and abandons the run.
- The request is accepted for any non-terminal job, in any state including before a machine exists.
- Repeated requests on an already-cancelling job succeed quietly — cancellation is idempotent.
- A request against a job in a terminal state is refused with a stable code.
- The resulting terminal state is `cancelled`, distinct from `failed`, because a user's own decision is not a defect.

### Runtime limits

The unused single safety constant is deleted and replaced by two live ones with distinct meanings and distinct outcomes:

- **Stall timeout** — if no output line arrives for the configured period, the job is killed and ends `failed` with code `gpu_stalled`. Default 15 minutes: an order of magnitude above the longest legitimately quiet stretch measured (a 183-second image build), and far below an unattended overnight.
- **Maximum duration** — if total elapsed time exceeds the ceiling, the job is killed and ends `failed` with code `gpu_max_duration_exceeded`. Default 24 hours, matching the industry default for managed training jobs, so the figure is not invented.

Both are configuration, not constants. Each time the stall detector resets, it emits a log event, so the mechanism is observable rather than a silent watchdog.

These are **circuit breakers, not spend policy**. They exist to catch a wedged job, not to limit what a user may legitimately train.

### Teardown ordering

Teardown and its independent confirmation run **before** the terminal state transition, so that the destroy confirmation is visible to a client that stops polling once the job reports a terminal status. Teardown continues to run in a path that executes regardless of how the run ended, including on unexpected exceptions.

### Event emission

Events continue to be emitted through the existing event-append function rather than a new protocol. That function is already a single point of interception — tests already replace things at that level, and a push-based implementation would replace it in the same way. Adding a formal protocol here would create a second seam for no testing benefit.

### Error codes

New stable codes introduced: `gpu_stalled`, `gpu_max_duration_exceeded`, `job_already_terminal`. Existing codes are unchanged.

### API surface

One new endpoint: a cancel action on a job. It returns the job's resulting state, and refuses terminal jobs with `job_already_terminal`.

### Decision records

This spec is not complete until its decision records exist. They are written **as each change lands, not afterwards** — entries written during the build record reasoning, entries written after it reconstruct reasoning, and reconstruction is what fails under questioning.

- **ADR-0001 — Event channel over SSH stdout.** Why the stream is pulled over the connection that already exists rather than pushed to the control plane; that a push requires a publicly addressable control plane, which is out of scope in both phases; that the `stream` contract is shaped so a push implementation is a substitution rather than a rewrite; and the alternatives rejected — polling a file on the machine, and a reverse tunnel.
- **ADR-0002 — Stall detection and a duration ceiling.** Why one unread wall-clock constant is replaced by two live controls with different meanings; why a stalled job and an over-long job are different outcomes with different codes; how each default was derived; and why these are circuit breakers rather than spend policy.
- **The decision-record index.** This spec creates the decision-record directory, so it also creates the index that explains the numbering. See `docs/adr/README.md` for why the numbers are not in date order.

## Testing Decisions

**What makes a good test here.** Tests assert what a user or operator can observe — the HTTP responses, the job record's state and error code, and the ordered event log. They do not assert how many times an internal helper was called, which subprocess flags were used, or the shape of intermediate strings. The one exception is the stray-instance path, where the observable outcome *is* an event, and the fact that destroy was attempted is the behaviour under test.

**Prior art.** The control-plane suite already tests at the HTTP seam using a test client with the database, upload directory and artifact directory redirected to temporary paths. That fixture is the model; these tests extend it rather than introducing a new style. The thinking-mode detection tests are the model for the pure-function tests.

**Modules under test.**

- *The orchestration function*, through the HTTP seam, with a fake provider supplied at the new seam. The fake yields a scripted sequence of lines with controllable delays, and can be told to fail at any stage, to stop producing output, or to fail the destroy call.
- *The line classifier*, directly, as a pure function over representative training output — including malformed lines, partial lines, and lines that resemble metrics but are not.
- *The cancel endpoint*, through the HTTP seam.

**Cases that must exist.**

- Log lines appear as events with distinct, increasing timestamps.
- Step, loss and epoch are extracted into metric events; unrecognised lines remain log events.
- Cancellation during provisioning, during image build, and during training each reach `cancelled` with the machine destroyed and no artifact recorded.
- Cancelling an already-cancelling job succeeds; cancelling a terminal job is refused with `job_already_terminal`.
- A provider that stops yielding lines produces `gpu_stalled` within the configured timeout.
- A job exceeding the duration ceiling produces `gpu_max_duration_exceeded`.
- The destroy-confirmation event precedes the terminal state event in the ordered event log. This is asserted explicitly, because it is the bug being fixed.
- Teardown occurs when the run raises an unexpected exception.
- A provider whose destroy fails and whose instance remains listed produces the stray-instance error event.

**Time is injected**, so stall and duration tests run instantly rather than sleeping. No test may construct a real provider client; the suite should fail loudly if one is attempted.

## Out of Scope

- **HTTPS push transport and any tunnel.** The `stream` contract is shaped to accept one later; building one now adds a second failure domain for a capability SSH already provides.
- **Packaging a checkpoint on cancellation.** Deliberately excluded — see the cancellation decision above. Revisit in Phase B when a UI can label a partial artifact unambiguously.
- **The reconciler.** Destroying machines that no job owns is a separate financial control with a separate failure mode, and belongs with Phase B's durable execution work.
- **Eventually-consistent teardown confirmation.** The existing confirmation may sample during a gap in which a destroyed machine is briefly absent before reappearing as destroying. Requiring absence across consecutive samples is the fix, and it belongs with the reconciler rather than here — noted so it is not mistaken for solved.
- **Charting metrics.** Metric events are structured from the start so a chart is a read-only addition; drawing one is Phase B.
- **Durable execution.** Recovering a job across a process restart is Phase B's Temporal work. Today a restart still orphans in-flight jobs, which the startup path surfaces rather than hides.
- **A feasibility estimate at job creation.** Predicting whether a job can finish inside the duration ceiling requires a throughput model that does not exist yet. Spec 002 adds a warning; a real estimate is Phase B.

## Further Notes

The 251 seconds of silence, the 183-second image build, the 161-second training phase and the identical timestamps on all 13 events are measured figures from the run of 2026-08-19, not estimates.

The three-layer redirection was found by reading the code during design, not by observing a failure. It is the reason this work is larger than "parse the stream" implies, and it is worth keeping in the record: the backlog entry that described this as one task was describing the last of three.

Deleting the unused safety constant is not merely cleanup. Its presence made the codebase read as though a spend limit existed, which is a worse state than having none — a control that looks implemented and is not is indistinguishable from one that works until the day it matters.
