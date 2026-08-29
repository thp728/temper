# ADR-0066 — Orchestration moves into a worker process that claims work

- **Status:** accepted
- **Date:** 2026-08-29
- **Spec:** `docs/specs/008-the-stack-foundation.md`
- **Issue:** [#51](https://github.com/thp728/temper/issues/51)

## Context

Every job ran on a thread inside the process that serves requests. A restart
of that process loses the thread driving the job and leaves any machine it
provisioned billing, unowned, with nobody watching. For a product whose
reader sells GPU hours, that is the first question asked and the worst
available answer.

ADR-0064 moved persistence to PostgreSQL for exactly this reason, stated in
its own context: a local file "cannot express a claim on a row surviving
contention," so there was no path to running orchestration anywhere but
inside the request-serving process. This record is the claim ADR-0064 made
possible: a separate process claims queued rows with a lock that survives
contention, and the request path stops starting threads at all.

Spec 008 names the seam this issue tests: **the orchestration entry point is
already a function over a job record that does not know what called it.**
`orchestrator.run_job(job_id, provider=None, limits=None, models=None)` reads
the row, drives the state machine, and writes the result back through
`db.py`. Nothing about its signature or body names a thread, a process, or a
caller. If making this work required editing `run_job`, the seam would have
leaked — it did not.

## Decision

**A worker process polls for queued jobs, claims one with a row-level lock
that skips what is already claimed, and calls the unchanged `run_job`.**

- **`db.claim_next_job()`** is `SELECT ... FOR UPDATE SKIP LOCKED` on the
  oldest `queued` row, followed by the `provisioning` transition and its
  event in the same transaction (a status that moved with no event recorded
  is a run that cannot be explained, ADR-0064's own rule, unchanged here).
  `FOR UPDATE` is what makes two claims of the same row impossible; `SKIP
  LOCKED` is the specific half that makes the second half of the acceptance
  criterion true — without it, a second worker racing for a locked row would
  *block* until the first transaction finished, not skip past it and look
  for other work. Both halves are proved directly, under real concurrency
  (a `threading.Barrier` forces the race rather than hoping for one), in
  `test_worker_claim.py`: two workers claiming one job, twenty workers
  claiming one job with the whole race finishing in under a second (a
  blocking `FOR UPDATE` alone would have serialized behind the first
  transaction's lifetime), and the ordering claimed oldest-first.

- **`apps/worker/src/temper_worker/worker.py`** is the process:
  `run_once()` claims and drives one job, `run_forever(stop)` polls every
  `WORKER_POLL_INTERVAL_S` (0.5s, tuning not contract) until told to stop,
  and `main()` is the container's entry point. A job that raises past
  `run_job`'s own error handling — a defect in the worker or the
  orchestrator itself, not a job's own failure — is caught and marked
  `failed` with `internal_error` rather than left claimed and non-terminal
  forever with no worker ever coming back to it; `run_job`'s ordinary
  failures already record themselves on the row before raising, so this is
  the belt under a belt, exercised by its own test rather than left to
  chance.

- **`jobs.create` and `jobs.retry_diverged_job` never call into the
  orchestrator.** They insert the row and return. No conditional, no
  fallback path, no "unless a test replaced this" branch — the request
  path does not start a thread, full stop, and `test_worker_claim.py`
  proves the negative directly: a job created through the API stays
  `queued`, with no machine and no event beyond the one creation itself
  wrote, until something claims it.

- **`orchestrator.launch` — the thread-starter — is deleted, not left
  unused.** Nothing in production calls it once `jobs.create` no longer
  does, and a function whose docstring claims "one process is the whole
  deployment" would be actively misleading once that stops being true. Kept
  code nobody calls is a claim about the architecture that the architecture
  no longer makes.

- **Provisioning's retry safety is unchanged, because it was already
  correct.** `orchestrator._attempt` provisions the machine and records
  `machine_id` on the row *before* `await_ready` can fail — this ordering
  predates this issue (ADR-0057/#34, ADR-0063/#46 both depend on it) and
  this issue does not touch it. What's new is the test that states the
  property directly rather than trusting it: `_attempt` is driven against a
  provider that fails at `await_ready`, and the row already carries
  `machine_id` before the exception propagates, so a retry that checked the
  row first would see a machine already exists rather than provisioning a
  second one.

## Whether the seam held

**Completely, for the orchestrator itself.** `run_job`'s signature and body
are untouched by this issue. Every existing orchestrator test — the fault
surface, memory recovery, divergence retry, teardown confirmation, the spend
ceiling — passes unchanged against a job driven by `run_job` called directly,
which is the same call the worker makes.

**Not for the test suite's driving mechanism, and the WIP commit this issue
inherited only partly caught it.** Roughly a dozen test files reached past
`orchestrator.launch` as an informal seam: some monkeypatched it to a no-op
so `POST /v1/jobs` couldn't provision a real machine, others replaced it with
a closure that called `run_job` synchronously (or on a thread) so the test
could observe a finished job. Both patterns depended on the request path
calling `launch` — once it does not, a monkeypatch that used to run
automatically on every `POST /v1/jobs` runs never. The commit this issue
picked up fixed the seven files it touched with a conditional shim in
`jobs.py` (call the monkeypatched `launch` only if a test had replaced it):
that shim kept those seven passing but left the other five broken outright
(nothing had exercised them, because the shim only ever ran when a test
_did_ monkeypatch `launch`, which happened to cover the touched files by
coincidence of what that worker had gotten to). It was also, on its own
terms, wrong: a request path that starts a thread conditionally on test
state is still a request path that starts threads. This record's fix deletes
the shim and updates every affected file (twelve, not seven) to drive
`run_job` explicitly after posting, the way the worker does — the same
technique `test_worker_claim.py`'s own tests use.

**One more thing the seam did not survive on its own: `compose.yaml`'s
worker healthcheck ran `ps aux | grep temper_worker`.** `python:3.12-slim`
carries no `procps`, so `ps` does not exist in the image and the healthcheck
failed unconditionally — verified by running the base image directly and
finding no `ps` binary. Not something a unit test would catch (the
`one-command stack` CI job that runs `docker compose up` for real is the one
that would have), so this is recorded rather than left to be found on the
next full-stack run. Replaced with a check against the dependency that
actually determines whether the worker can do anything: `db.ping()`, the
same probe `/health` already uses for the control plane, run through
`python -c` for consistency with that healthcheck's own style rather than
installing `procps` to keep a process-table check.

## The three questions this change forces

**Who owns the served-endpoint timers and sweep thread after orchestration
leaves this process?** They stay in the control plane, unmoved. A served
endpoint (ADR-0065/#78) is not a job: it is a warm machine the control plane
itself provisioned and is billing for, armed with `threading.Timer`s in
*that* process because that is the process a browser's request to start or
stop it reaches. Nothing about this issue moves endpoint serving anywhere —
`create_endpoint`, `stop_endpoint` and the idle/max timers untouched in
`serving.py`, still called from `main.py`'s handlers. `lifespan` still
re-arms timers for any endpoint left running across a control-plane restart
and starts the sweep thread that catches ones already past their deadline
while the process was down. The worker never touches an endpoint; a served
model and a training job are different machines with different owners by
construction, and this issue does not change who owns which.

**Who watches spend and teardown after this change, and does a restart
reconcile?** For a *job in flight*, the worker does — `_attempt`'s guard
(duration ceiling, spend ceiling, stall detection) and the confirmed
teardown in `run_job`'s `finally` are the same code, called from the same
`run_job`, now running in the worker's process instead of the control
plane's. Nothing about moving the caller changes what watches while a job is
running. **What changed, honestly:** the control plane's restart used to
mark every non-terminal job `orphaned_by_restart` at startup, because a
restart of that process used to mean the one thread that could finish the
job had died with it. That marking is gone (`main.py`'s `lifespan`, and
`db.active_jobs()` — the query that only existed to back it — deleted
outright, nothing else called it). It is gone correctly for a **control-plane**
restart: the job's thread was never in that process to begin with, so a
control-plane restart no longer orphans anything, which is the entire point
of this issue. It is a real, named gap for a **worker** crash mid-job: a job
claimed and left in a non-terminal, non-`queued` state (`provisioning`,
`preparing`, `training`, `packaging`) when the worker driving it dies is not
reclaimed by anything — `claim_next_job` only ever looks at `queued` rows,
by design, since a job actively being driven must never be claimed twice.
Recovering that job, and the machine it may have provisioned, is Spec 010's
reconciler, named explicitly as out of scope by Spec 008 itself ("Durable
orchestration and recovery from an interrupted run… makes a worker process
possible; Spec 010 makes an interrupted job resumable"). Recorded here rather
than silently assumed solved: **a worker that crashes mid-job currently
leaves that job stuck and its machine, if any, unwatched, until Spec 010
ships the reconciler that watches for exactly this.**

**Does `test_orchestrator.py`'s `Harness.run_on_a_thread` race — the thread
outliving the test, writing to a torn-down database — become obsolete or
worse?** Neither cleanly: the harness is still needed (mid-flight output is
only observable while a job is genuinely running on another thread, and
cancellation can only be exercised against a job that is), but the mechanism
it used to lean on (the request path starting a thread via `launch`) is
gone, so it now starts and tracks its own thread explicitly via
`orchestrator.run_job`. That is the fix, not a new instance of the bug: the
harness now keeps a list of the threads it starts and joins every one of
them, with a timeout, in the fixture's teardown, before the test's database
is dropped. The race was never about the harness needing a thread; it was
about nothing owning that thread's lifetime. Something does now.

## Alternatives considered

**A task queue (Celery, RQ, arq) instead of a polling worker.** Rejected for
the same reason `orchestrator.launch`'s own retired docstring gave for
threads over Celery: one worker, no retry semantics beyond what `run_job`
already owns internally, and a queue framework buys infrastructure this
issue does not need. `SELECT ... FOR UPDATE SKIP LOCKED` against a table
that already exists is the whole mechanism a queue framework would also
reduce to underneath, without a second dependency or a second failure mode
to reason about.

**Keep `orchestrator.launch` for tests, rather than deleting it and updating
every call site.** Considered, and it is what the WIP commit half-did. Rejected
on reflection: a thread-starter kept alive only because tests still called it
is production code shaped by its own test suite, and its docstring ("one
process is the whole deployment") would be actively wrong the moment it
shipped. Updating twelve test files once is a bounded, one-time cost; leaving
a misleading function around is a standing one.

**Give the worker its own reconciliation pass for jobs stuck by a crash,
inside this issue.** Rejected: Spec 008 states explicitly that this is Spec
010's scope, and building a partial reconciler here would be doing that
spec's work without its adversarial cluster to test against. The honest
choice is to build the claim mechanism Spec 010 needs and name the gap it
leaves, not to half-solve the next spec's problem under this one's ticket.

**A worker healthcheck via `procps` in the shared Dockerfile instead of
`db.ping()`.** Considered once the `ps`-not-found failure was found.
Rejected: installing a package to ask "does a process with this name exist"
is a weaker signal than asking the worker's actual dependency whether it can
be reached, and it would touch the Dockerfile both images share for a check
this container has a better answer to already available.

## Consequences

- `apps/worker` ships a running process for the first time; `compose.yaml`
  gains a fourth service, `playwright.config.ts` a third `webServer` (the
  worker has no URL to wait on health at, so it starts once the backend
  reports healthy and stays up for the suite's duration), and the control
  plane's own Dockerfile now installs both packages so one image serves
  both entry points.
- `db.claim_next_job` is new; `db.active_jobs()` is deleted (nothing calls
  it once the control plane no longer marks restarts as orphaning).
- `orchestrator.launch` is deleted. Any external code depending on it (none,
  inside this repository) would need to call `run_job` directly, on a thread
  it owns, exactly as the worker and the test suite now do.
- A worker crash mid-job leaves that job's row non-terminal and unclaimable
  until Spec 010 ships the reconciler that watches for it — a named,
  accepted gap, not a silent one.
- Running more than one worker is now possible (the acceptance criterion);
  running more than one in this build's deployment is not exercised beyond
  the concurrency tests, which is consistent with Spec 008 naming horizontal
  scaling itself out of scope.

## Rollback

Revert this issue's commits. `jobs.create` and `retry_diverged_job` resume
calling `orchestrator.launch` (restored from history), `db.claim_next_job`
and the worker process are removed, `compose.yaml` drops the `worker`
service and `playwright.config.ts` its third `webServer`. `main.py`'s
restart-time `orphaned_by_restart` marking and `db.active_jobs()` are
restored alongside it, since that marking's premise (the driving thread
lives in the control plane) becomes true again. Nothing in
`packages/core` or `orchestrator.run_job` needs to change either direction,
which is the seam holding exactly where Spec 008 drew it.
