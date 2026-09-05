# ADR-0002 — Stall detection and a duration ceiling

- **Status:** accepted
- **Date:** 2026-08-21
- **Spec:** `docs/specs/001-orchestrator-core.md`
- **Issue:** [#7](https://github.com/thp728/temper/issues/7)

## Context

The orchestrator carried this line:

```python
MAX_GPU_MINUTES = 90  # safety control, not billing
```

Nothing read it. No code path compared anything to it, and no test would have
noticed if the number had been 9 or 9000. It had been there since the module
was first written, and it read exactly like a spend control: a plausible name, a
plausible value, a comment explaining its intent.

**That is worse than having no limit at all.** A missing control is visible to
anyone reading the module; a control that looks implemented and is not is
indistinguishable from a working one until the day it matters, and the day it
matters is the day a job wedges overnight on a machine billing per minute.
The person reading the code that evening would have
concluded the job was already capped at ninety minutes.

Two facts made this reachable rather than theoretical. First, the job's work
happens inside a single loop over lines arriving from the machine (ADR-0001), so
a machine that goes quiet blocks that loop with no upper bound. Second, the only
thing that had ever bounded a run was the SSH transport's own 90-minute
subprocess watchdog — an implementation detail of one provider, which a push
transport would not inherit, and which reports as a `TimeoutExpired` rather than
as anything a user could act on.

## Decision

The unread constant is deleted and replaced by **two live controls with
different meanings, different codes, and different remedies.**

- **Stall timeout** — no output line for the configured period. The job is
  killed, the machine destroyed, and the job ends `failed` with
  `gpu_stalled`. Default **15 minutes**.
- **Duration ceiling** — total elapsed time past the configured maximum,
  whether or not output is still arriving. The job is killed, the machine
  destroyed, and the job ends `failed` with `gpu_max_duration_exceeded`.
  Default **24 hours**.

Both are **configuration** (`TEMPER_STALL_TIMEOUT_S`,
`TEMPER_MAX_JOB_DURATION_S`), not constants, because the right number depends on
the catalog and the catalog will grow.

### Why two codes and not one

The remedies are unrelated. A stall points at the machine or the training
process — something is wedged, and the run is unlikely to be reproducible by
repeating it unchanged. Hitting the ceiling points at the job being larger than
the platform's policy allows, which is a conversation about the limit, not about
a defect. Collapsing them into one "took too long" hands the user the wrong
question. This is the same mistake as collapsing *unreachable* and
*authentication failed* into "no answer", which cost an evening and produced a
wrongly-filed platform bug.

### These are circuit breakers, not spend policy

They exist to stop a job that is no longer making progress, not to cap what a
user may legitimately train. That distinction is what sets the defaults: a spend
cap would be derived from a budget, and would need to be visible to the user
before they launched. These are derived from what a healthy run looks like.

### How each default was derived

- **15 minutes.** The longest legitimately quiet stretch on the measured run of
  2026-08-19 was a **183-second image build**. Fifteen minutes is roughly an
  order of magnitude above the worst observed silence, and far below an
  unattended overnight, which is the loss this exists to prevent. It is a
  measured floor with a wide margin, not a guess.
- **24 hours.** An **adopted convention, not a measured or cited figure** — it
  is the ceiling commonly used for managed fine-tuning jobs, and it is recorded
  here as borrowed rather than derived so that nobody later mistakes it for
  evidence. What supports it is weaker and sufficient: nothing on the current
  4B/8B catalog comes close, the measured training phase being **161 seconds**,
  and a backstop that never fires on a legitimate run does its job either way.
  This is the number to revisit when the catalog grows, which is why it is
  configuration.

### Time is injected

`RunLimits` carries its own clock. A limit whose test has to wait fifteen
minutes is a limit whose test gets skipped, and a skipped test is how the
constant above stayed unread for a week. Both limits are exercised in
milliseconds against a clock that advances a fixed step per reading, and the
guard reads the clock **exactly once per iteration** so that simulated time per
loop is deterministic rather than dependent on how many times the code happens
to look.

### Where the enforcement lives

In `api/limits.py`, between the provider's line iterator and everything that
reads it — not in the SSH implementation. A push transport inherits both limits
without knowing they exist, which is the same reason `stream` is an iterator in
the first place.

The source is drained on its own thread, because there is no way to ask a plain
iterator whether the next item is late. When a limit trips, that thread is left
blocked rather than reached into: closing a generator another thread is
executing is not something Python permits, and it is unnecessary here — the
caller's next act is to destroy the machine, which drops the connection the
thread is waiting on. The thread is a daemon either way.

### The two limits measure from different origins

**The ceiling counts from the start of the job.** Provisioning and waiting for
SSH are part of a job's elapsed time, so they are part of its budget.

**The stall budget counts from the start of the stream.** This is the opposite
choice and it is deliberate: the stall timeout is a statement about the stream,
and charging four minutes of provisioning against the first line's grace period
would silently shorten the timeout by however long the machine took to arrive —
so a slow provision would present as a stall. The first version of this change
had both counting from job start, which is the bug described; review caught it
before it shipped, and both origins now have a test naming which one they are.

**What the ceiling does *not* cover, stated rather than implied.** The guard can
only observe the ceiling while it is reading lines, and the between-stage checks
(`RunLimits.check_duration`) only fire between provider calls. A job blocked
*inside* a single provider call is bounded by that call's own timeout — 300s for
SSH readiness, 180s for the source push, 600s for a fetch — not by the ceiling.
Those bounds are short and the ceiling is long, so in practice the gap does not
matter; it is written down because the alternative is a record claiming a
coverage the code does not have, which is the failure mode this whole change
exists to correct.

## Deviation from the acceptance criteria, stated plainly

The ticket asks that the stall detector **"emit a log event each time it
resets"**. Taken literally that is one bookkeeping event per output line, which
would more than double the event log and bury the training output it exists to
make legible — defeating the stated purpose of the criterion, which is that the
mechanism be *observable*.

What is implemented instead: a reset emits an event when the silence that
preceded it was **at least a quarter of the stall budget** — far longer than any
gap a healthy run produces, and the first warning that the next gap might not
end. Short gaps pass silently. The threshold is a constant in `limits.py`
(`REPORT_FRACTION`) and is pinned by its own test.

## Alternatives considered

**Enforce it in the SSH provider's existing subprocess watchdog.** It already
killed the process at 90 minutes, so this looked close to free. Rejected: it is
one provider's implementation detail, a push transport would not inherit it, it
cannot distinguish silence from length, and it surfaces as `TimeoutExpired`
rather than as a coded, user-facing outcome. The seam exists precisely so that
orchestration policy does not live inside one transport.

**That watchdog could not simply be left alone, and the first version of this
change left it alone.** A 90-minute transport timeout sitting *below* a 24-hour
ceiling means the ceiling can never fire against a real machine: the transport
kills first, and the job fails with an uncoded `TimeoutExpired` mapped to
`internal_error` — precisely the unnamed outcome this record objects to. No test
could catch it, because the fake provider has no watchdog. It is now derived
from the ceiling and sits ten minutes above it, so it is a genuine backstop for
a stream that outlives even the guard, and the coded outcome always wins the
race. Recorded because it is the kind of error the rest of this document is
about: a control that was correct until a second control appeared beside it.

**A single limit — total duration only.** Simpler, and it would have caught the
overnight case. Rejected: it is the slowest possible detector of the most common
failure. A wedged trainer would burn the full ceiling before anyone found out,
which on a 24-hour ceiling is the entire loss the control exists to prevent.
Silence is the early signal; duration is the backstop.

**A single limit — stall only.** Rejected in the other direction: a job that
emits a progress bar every few seconds while making no real progress is silent
to a stall detector and would run forever.

**A spend cap in currency rather than time.** Genuinely the control a user
wants, and it is what a commercial platform exposes. Rejected for now because it
needs a price-per-second model, a running-cost meter, and a policy about what
happens to a partially trained job when the money runs out — that is the billing
work that is out of scope. Recorded here so the absence is a decision
rather than an oversight.

**A watchdog thread that kills the run from outside.** Rejected: it needs a way
to interrupt a blocked reader, which in Python means either killing a
provider-specific subprocess (see above) or asynchronous exceptions. Draining
the iterator into a queue makes lateness observable with no interruption
primitive at all.

**Make the limits per-job parameters at creation.** Rejected for v1: a
user-supplied ceiling is a spend conversation, and until there is a feasibility
estimate to check it against, a per-job number would be a knob with nothing
behind it. Spec 002 adds a warning at job creation; a real estimate is Phase B.

## Consequences

- Neither failure message claims the machine **has been** destroyed. Both say it
  is *being* destroyed and point at the teardown confirmation in the job's
  events, because the message is constructed before teardown runs and the
  stray-machine path — a destroy call that fails and a machine that stays
  listed — is real and tested. A message asserting teardown as a completed fact
  would be the one thing an operator trusted and the one thing that was false.
- Two new stable error codes: `gpu_stalled`, `gpu_max_duration_exceeded`. Both
  end the job `failed`, not `cancelled` — the platform stopped it, the user did
  not, and a job stopped by a safety limit is a failure by this project's own
  glossary.
- The machine is destroyed on both paths, through the same `finally` that
  already covers every other failure, and the destroy confirmation still
  precedes the terminal state event.
- The guard adds one thread per streaming job. With one process and one job at a
  time this is free; it is bounded by the same thing that bounds jobs.
- The stall timeout is now the effective cap on how long a hung job can bill.
  Fifteen minutes at ₹41.31/hr is about ₹10 of exposure per wedged job, down
  from unbounded.
- A legitimate run that is quiet for more than fifteen minutes — a very large
  model download on a slow link, say — would now be killed. The current catalog
  cannot produce that; a larger catalog might, and the fix is configuration
  rather than code. This is the tradeoff accepted, and the reason the value is
  not a constant.

### A misconfigured limit refuses to start

`TEMPER_STALL_TIMEOUT_S=15m` is the realistic typo. Falling back to the default
would leave an operator believing a limit was in force that was not — the same
shape as the unread constant above, and *less* visible, because there would be
nothing wrong in the source to find. A value that cannot be honoured raises at
import, which is one legible line at boot. There is deliberately no value
meaning "no limit": a limit that a typo can switch off is not a limit.

## Rollback

Set `TEMPER_STALL_TIMEOUT_S` and `TEMPER_MAX_JOB_DURATION_S` to values large
enough to be unreachable. That disables both controls without a code change,
which is the point of them being configuration. Removing the mechanism means
deleting `api/limits.py` and the `guard` call in `_attempt` — and would return
the system to the state this record exists to describe, so it supersedes rather
than reverts.
