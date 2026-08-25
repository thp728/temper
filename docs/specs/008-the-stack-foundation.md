# Spec 008 — The stack foundation

**Status:** ready for tickets
**Phase:** B, band 2 (the defensibility path)
**Depends on:** Spec 006 (the storage seam this implements behind), Spec 007 (the stream this fans out to)
**Produces:** [ADR-0010](../adr/0010-the-repository-is-laid-out-as-apps-and-packages.md) and [ADR-0011](../adr/0011-one-command-runs-every-task-and-one-defines-green.md) (landed), plus an ADR that an event is persisted before it is published with replay by last-seen identifier, and one that the migration is bounded by seams that already exist
**Assumes:** the Phase A rule that no persistence code lives in request handlers — that rule is the reason this spec is a migration rather than a rewrite

## Problem Statement

The control plane is one process holding a local file. Every job runs on a
thread inside the same process that serves requests. Every stored object is a
path under a working directory. Every log line is text on standard output.

That was the correct shape for proving a loop, and it was chosen deliberately
with a written argument for not migrating until the loop ran. The loop ran. What
the shape now costs:

- **A restart loses running jobs.** Job state lives in a file, but the thread
  driving the job does not survive the process. A control plane that stops
  mid-run leaves a machine provisioned, billing, and unowned. For a product
  whose reader sells GPU hours, that is the first question asked and the worst
  available answer.
- **One writer, and no way to hand work between processes.** The local database
  cannot express a claim on a row that survives contention, so there is no path
  to running the orchestration anywhere but inside the request-serving process.
- **Watching is polling.** The interface asks for events after an identifier on a
  timer. It works for one watcher on one machine and does not survive a second
  process holding the events.
- **Objects are paths.** Datasets and artifacts live on whichever filesystem the
  process happens to have, which is not a place a second process or a scoped
  write URL can reach.
- **Logs are unstructured and uncorrelated.** A failure spanning a request, a
  job and a machine cannot be reassembled from them.

None of these is urgent for a demonstration and all of them are the difference
between a working pipeline and a platform. The repository is the work sample,
read by a company whose product is the infrastructure underneath this one.

## Solution

Replace what the domain logic stands on, without changing the domain logic.

This is bounded because the seams already exist and were built with this in
mind: persistence is behind functions rather than scattered through handlers,
validation is pure, and orchestration is a function over a job record that does
not know what called it or what it persists to. **What changes is what those
functions write to and what runs them. The validation rules, the state machine,
the orchestration sequence and thinking-mode detection carry over unchanged** —
and if they do not, the migration has been done wrong.

The stack, every piece conventional and each with a reason:

- **A relational database with real migrations**, replacing the local file.
  Real constraints, real concurrency, and the ability for one process to claim
  work another can see. Migrations exist because a schema that changes by
  editing a create statement cannot be rolled forward or back.
- **An object store speaking the standard protocol**, behind Spec 006's storage
  seam, so that datasets, checkpoints and artifacts are objects with keys rather
  than paths on a particular disk — and so that the scoped write URL has
  something to be scoped against.
- **A publish-subscribe channel** fanning persisted events out to every watcher,
  so a second process holding the events is not a problem and a second watcher
  is not either.
- **Structured logs with a correlation identifier** threaded from request
  through job through machine, so one failure reads as one story.
- **One command starts everything**, which is the deployment target and also the
  thing that makes a cold clone possible for someone who is not the author.

Two properties are worth stating as rules rather than as configuration:

**An event is persisted before it is published.** The store is the truth and the
channel is a notification. A watcher that misses a message, reconnects, or
arrives late replays from the store by the last identifier it saw, and loses
nothing. Publishing first and persisting after would make a dropped connection
lossy in a way no watcher could detect.

**Every state change and its event are written together or not at all.** A job
whose state moved with no event recorded is a job that cannot be explained, and
explaining runs is the product.

## User Stories

1. As a user, I want a running job to survive the control plane restarting, so that an operational event does not destroy my run.
2. As a user, I want my job's machine to be destroyed even if the control plane stops, so that I am not billed for something nobody is watching.
3. As a user, I want to watch a job from two places at once, so that I can check on it from another device without breaking the first.
4. As a user, I want a dropped connection to resume where it left off, so that a brief network problem does not leave a hole in the history.
5. As a user, I want to close my laptop and come back to a complete record, so that watching is optional rather than required.
6. As a user, I want my dataset and artifact to remain available regardless of which process handled them, so that storage is a property of the product rather than of a machine.
7. As a user, I want the product to behave identically after an upgrade, so that a schema change does not lose my history.
8. As an operator, I want database changes applied as versioned migrations, so that an upgrade is repeatable and reversible.
9. As an operator, I want to roll a schema change back, so that a bad migration is recoverable.
10. As an operator, I want one command to start every part of the system, so that setting it up is not a sequence of manual steps.
11. As an operator, I want every configuration value read from the environment, so that nothing needs a code change to point at a different deployment.
12. As an operator, I want no secret to appear in a log, a page, or a response, so that a public repository and a running system are both safe.
13. As an operator, I want each log line structured and machine-readable, so that failures can be searched rather than skimmed.
14. As an operator, I want a correlation identifier connecting a request, its job and its machine, so that one failure reads as one story.
15. As an operator, I want work claimed exactly once when more than one process is running, so that two workers cannot provision two machines for one job.
16. As an operator, I want a health check reporting on each dependency, so that a broken database is distinguishable from a broken application.
17. As a developer of this product, I want lint, type checking and tests to run on every push, so that a regression is caught before it is reviewed.
18. As a developer of this product, I want the trainer image built and published by the pipeline and referenced by digest, so that what runs is exactly what was built.
19. As a developer of this product, I want the domain logic unchanged by this migration, so that the behaviour proven on hardware is the behaviour that survives.
20. As a reviewer, I want to start the whole system from a clone without configuring anything by hand, so that evaluating it does not begin with a debugging session.
21. As a reviewer, I want to read how the migration was bounded, so that I can tell whether the seams were designed for it or claimed after the fact.

## Implementation Decisions

**Persistence moves behind the existing functions, not around them.** The
function names and their return shapes stay; their bodies change. The measure of
success is that the orchestration, validation and state-machine code is
untouched by this spec. Anywhere it is not, the seam was leakier than claimed and
that is worth recording rather than patching over.

**Migrations start from a baseline that matches the current schema**, then add
what Phase B needs — the quote, the attempt, the artifact and its manifest, the
metric series, the checkpoint. Every migration is reversible; a migration that
cannot be rolled back is an outage waiting for a bad afternoon.

**Money is stored as an integer in the smallest unit with its currency
alongside.** This account bills in rupees, and a floating-point amount with an
assumed currency is a defect that only shows up in front of someone who cares.

**The job spec is written once at launch and never updated.** A completed job
must describe what it actually did, so a later change to a default cannot
retroactively rewrite history.

**Work is claimed with a row-level lock that skips what is already claimed.**
This is the specific capability the local file could not provide, and it is why
the database change is a prerequisite for running orchestration outside the
request-serving process rather than a tidiness exercise.

**Orchestration moves out of the request-serving process into a worker**, with
the same entry point it has now: a function over a job record. Durable
execution and recovery are Spec 010; this spec establishes that the worker is a
separate process reading claimed work, which is what makes that spec possible
without further surgery.

**Events are persisted and then published; watchers replay by last-seen
identifier.** The stream endpoint reads history from the store first, then
follows the channel. A watcher that reconnects sends what it last saw and
receives the gap.

**Configuration is typed and read from the environment, with no literals in
code.** Local defaults are set so the system starts with nothing configured,
because a reviewer's first run should not require a decision.

**Logs are structured, and telemetry carries numbers only.** No prompt, no
completion, no dataset row ever appears in a log line — a training platform that
logs user data has a problem no amount of access control fixes.

**Error reporting is wired as an integration point with no credential set
locally.** Present so that the shape is visible, inert so that nothing is
required to run the system.

## Testing Decisions

**A good test here uses the real dependency.** Faking a database or an object
store in tests for a spec whose entire content is *which* database and object
store defeats the purpose. These tests run against real services started as
throwaway containers by the suite itself, so every run gets clean state and
cleanup survives a crash.

That covers the code but not the composition. A broken environment variable or
healthcheck in the compose file would leave every test green and still fail for
the reviewer who types the documented start command. So the pipeline also
starts the whole stack from that command and polls the health endpoint until
every dependency reports ready. Two mechanisms, two different failures. See
ADR-0011.

**What gets tested, and what the assertion is:**

- **Migrations** — a fresh database migrates to head, and every migration rolls
  back. The assertion is that the schema round-trips, not that a particular
  statement was issued.
- **Claiming work** — concurrent workers claiming from the same set never claim
  the same job twice, and none blocks on another. This is the property the
  database was changed for; asserting it directly is asserting the reason.
- **Transactions** — a state change and its event are both present or both
  absent, verified by forcing a failure between them.
- **Object round-trip** — an object written through the seam is readable
  unchanged, and a scoped write URL cannot read, list, or write another key.
- **Stream catch-up** — a watcher that disconnects and reconnects with its last
  seen identifier receives exactly the events it missed, in order, once. Tested
  by dropping the connection mid-stream rather than by simulating it.
- **Correlation** — one request produces one identifier that appears on every
  log line the resulting job emits.

**Domain behaviour is not re-tested here.** If the existing suite still passes
unchanged against the new persistence, the migration is bounded as claimed. That
suite passing is itself this spec's most meaningful assertion, and any test it
requires changing should be examined rather than edited.

**Prior art:** the existing persistence tests define the contract the new
implementation must satisfy. They should need no changes; where they do, the
change is evidence about the seam.

**The verification clause.** Done means a job launched through the interface,
running on real hardware, survives the control plane being restarted mid-run —
observed, not asserted — and its record and artifact are intact afterwards.
That is not a test-double claim and cannot be made by one. It shares a machine
with Spec 010's adversarial cluster.

## Out of Scope

- **Durable orchestration and recovery from an interrupted run.** This spec
  makes a worker process possible; Spec 010 makes an interrupted job resumable.
- **The reconciler.** Spec 010, where it belongs beside the failure it defends
  against.
- **Deployment anywhere.** The stack starts locally and is written to be
  deployable; deploying it is deliberately out of scope for this build.
- **Multi-tenancy, authentication and everything downstream of them.** Cutting
  authentication properly means there are no tenants, not that tenancy is
  half-built.
- **Backups, retention and disaster recovery.** Retention is touched in Spec 012.
- **Horizontal scaling.** The claim mechanism makes more than one worker
  possible; running more than one is not a goal of this build.

## Further Notes

The honest risk in this spec is that it is the largest change with the least
visible result. Nothing a user can see improves, and the days it consumes come
out of flows that would raise the parity number. It is in band 2 for exactly
that reason, and the ordering rule is worth restating: **nothing here starts
while a band 1 spec is unproven on hardware.**

The counter-argument, which is why it is in scope at all: the orchestration layer
is the work sample, the reader's own product is the infrastructure underneath
this one, and *"a thread in the web process, a local file, and a directory"* is
not a production answer to any question that will be asked about it. The
mitigation for its cost is that the migration is bounded by seams built for it —
and if that turns out to be untrue in practice, that finding is worth recording
prominently, because a claim about a design that did not survive contact is
exactly the kind of correction this project treats as an asset.
