# ADR-0003 — Cancellation is destructive, and is not a failure

- **Status:** accepted
- **Date:** 2026-08-21
- **Spec:** `docs/specs/001-orchestrator-core.md`
- **Issue:** [#6](https://github.com/thp728/temper/issues/6)

## Context

`cancelled` has been in the job lifecycle since the state machine was written,
and until now no code path reached it. A user who launched a job against the
wrong dataset — or with a rank they immediately regretted — could do nothing but
watch it run and pay for it, on a machine billing ₹41.31 an hour against a fixed
grant.

Two things made this the right moment to fix it rather than a fourth thing to
add later. The streaming loop (ADR-0001) gave the run its first place to look at
anything mid-flight, and the runtime limits (ADR-0002) had already established
what stopping a job in progress has to do: kill it, destroy the machine, confirm
the teardown, and report the outcome with a name.

What was left undecided was the part that is not mechanical — what a user gets
back when they stop a run that had already started training, and what the job's
history should say happened.

## Decision

**Cancellation destroys the machine and produces no adapter.** No checkpoint is
packaged, nothing is fetched from `/tmp/out`, and the job's `adapter_path` stays
null.

**The terminal state is `cancelled`, and it carries no error code.** Not
`failed`, and not `failed` with a friendly code — the row's `error_code` and
`error_message` are left null, because every value that could go there names a
defect, and this is not one.

**The consequence is stated at the moment it is chosen.** The cancel request
answers with the same sentence that is written into the job's own event log:
*"Cancellation requested. The machine is being destroyed and no adapter will be
produced."* One string in both places, so what the button said and what the run's
history says cannot drift.

**The request is a flag on the job row; the run reads it.** The user's request
sets `jobs.cancel_requested` and returns immediately. The thread running the job
reads it at every boundary between stages and once per trip round the streaming
loop, and raises a dedicated `Cancelled` exception — which is deliberately *not*
a subclass of `OrchestratorError`, so nothing that catches failures generically
can catch a user's decision by inheritance and relabel it.

**Honoured until the last moment an adapter could appear.** Packaging is not
instantaneous — it is a download from a machine that is still billing — so a
cancellation can genuinely arrive during it. If one does, the fetched adapter is
deleted and the job still ends `cancelled` with a null `adapter_path`. Keeping
it, on the grounds that it is trained and paid for, would make the answer a user
gets depend on how many seconds their download took, which is the one thing
about their own decision they cannot see. A promise that holds only outside a
race is not a promise.

**Accepted in any non-terminal state, refused in a terminal one.** Including
before a machine exists: the flag is honoured before the provider client is even
built, so a job cancelled while it sat in `queued` costs nothing at all.
Repeating the request succeeds quietly and appends no second event — a
double-clicked button is one decision. A request against a job that has already
ended is refused `409` with `job_already_terminal`, because nothing was undone
and the user should not be left thinking otherwise.

## Alternatives considered

**Package the checkpoint and hand it over.** Rejected. A partially trained
adapter looks exactly like a finished one — same file, same config, same
extension — and there is nothing in the artifact to say it stopped early. Handing
it to the user who just asked to stop invites precisely the mistake that the
adapter-config work already exists to prevent: taking something unfinished for
the deliverable. It is also not free — fetching it keeps the machine alive
past the moment the user asked for the billing to end, which is what they were
actually buying. Revisit in Phase B, when a UI can label a partial artifact
unambiguously and the user can choose.

**Record cancellation as `failed` with a code like `cancelled_by_user`.**
Rejected. It collapses two things a user needs to keep apart: a run that broke
and a run they stopped. A job list where their own decisions are red is a job
list they stop reading. It also makes every client that branches on `failed`
wrong about the one case it did not need to handle.

**Interrupt the thread, or kill it.** Rejected. Python offers no safe way to
interrupt a thread blocked in a socket read, and the approximations
(`PyThreadState_SetAsyncExc` via `ctypes`, killing the process group) leave the
teardown path — the only part that stops the billing — unrun. Cooperative
checking means cancellation always exits through the same `finally` that
destroys the machine, on the same code path as every other outcome, which is
the path that is already tested.

**Signal it in memory — an `Event` per running job.** Rejected. It works only
while the request and the run are in one process, which is true today and is
explicitly not the target: Phase B moves the run into a Temporal activity. The
row is where the two already meet, the read is one indexed lookup by primary key
against a local file, and a flag that survives a restart is strictly more
correct than one that does not.

**Have the cancel request itself destroy the machine.** Rejected. Two threads
racing to tear down one machine is how a machine gets destroyed twice, or
neither time. Teardown stays in one place — the run's own `finally` — and the
request's only job is to say so.

**Return the job as `cancelled` from the cancel endpoint.** Rejected. At the
moment the request returns, the machine is still running. Reporting the terminal
state there would claim a teardown that has not happened, and teardown is the
entire point of cancelling. The endpoint returns the state the job is actually
in, plus `cancel_requested: true`.

## Consequences

- Cancellation is prompt but not instantaneous. It is observed within one poll
  interval of the streaming loop (one second by default) at any point during
  training, and at the next stage boundary otherwise. What is not interruptible
  is the inside of a single provider call — a `create` that is waiting on the
  platform finishes first, and the machine it returns is then destroyed. That
  costs a machine's minimum billing increment in the worst case, and the
  alternative is a machine created by a call nobody was waiting on any more,
  which nobody destroys.
- A cancelled job's event log ends the way every other outcome does: the
  destroy confirmation, then the terminal state. A user who cancels is the most
  likely to stop reading the moment the job says it is over, and the most likely
  to want proof the billing stopped.
- The database gained a column on an existing table, which `CREATE TABLE IF NOT
  EXISTS` will not add to a database that already exists. `db.init()` now checks
  for it and issues the `ALTER TABLE`. This is a migration mechanism in
  everything but name, and it is four lines rather than Alembic because Alembic
  arrives in Phase B with Postgres; naming it `ADDED_COLUMNS` rather than
  pretending the schema is declarative is the honest version.
- Nothing exists yet that cancels a job automatically. The two runtime limits
  stop a job that has stopped making progress, and they end it `failed`,
  correctly — the platform stopped it, the user did not.

## Rollback

Remove the endpoint and the `check=` argument the streaming guard takes; the
column becomes inert and the `Cancelled` path unreachable. The guard's hook is
the only change to code that is not about cancellation, and it is one call to an
optional callable.
