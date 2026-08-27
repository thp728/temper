# ADR-0034 — The running-job surface consumes a server-pushed event stream

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/007-the-application-shell.md`
- **Issue:** [#39](https://github.com/thp728/temper/issues/39)

## Context

Issue #39 ports the last unported screen: watching a running job. The old
watch page polled two endpoints every two seconds -- the event log and the job
record -- and re-rendered with JavaScript. Spec 007 is explicit that the new
surface replaces that poll with a **server-pushed stream**: output arrives
without the page being refreshed, the latest loss appears as it is measured,
leaving and returning shows continuous history rather than a view that starts
where the visitor rejoined, and a dropped connection reconnects on its own.

Three existing facts shape the port. The event log is durable and monotonic
(`GET /v1/jobs/:id/events?after=N`), so a reconnecting client can be caught up
from the database rather than from whatever one connection happened to see.
The job record, not any event's prose, is authoritative for status -- a state
event says *what is happening*, the record says *what the job is*. And the
finished-job record is owned elsewhere (#40, extended by #77), so the running
view must hand back to it rather than re-build it, or two views of one record
drift.

A testing constraint completes the picture: the journey fake completes a job
in milliseconds, so a browser journey cannot watch a run or cancel one mid-run
unless the simulated machine is slowed enough to act on.

## Decision

**The running-job surface consumes a server-pushed event stream over the
durable log, and hands back to the server-rendered finished record at a
terminal state.**

- **One endpoint, `GET /v1/jobs/:id/stream`, a server-sent event stream.**
  Each event carries the same `JobEvent` shape the polling endpoint publishes,
  plus an SSE `id`. The server polls the durable log every quarter second and
  pushes what is new; this process has no broker to subscribe to (that is Spec
  008's), and the poll is one indexed query. The stream ends only when the job
  reaches a terminal state, having delivered the terminal transition itself.

- **Replay is the durable log, in both directions.** On first connection the
  page has already rendered everything recorded so far, so it passes its last
  event id as `after`. On reconnect, EventSource replays `Last-Event-ID`, and
  the endpoint honours that header over the query parameter. A visitor
  returning to a running job finds continuous history from the first paint,
  and a dropped connection resumes where it stopped without re-delivering.

- **The status word is refetched on state transitions, never read from
  prose.** A state event triggers one `GET /v1/jobs/:id`, so the State stat
  shows the record's status and the machine line appears as provisioning
  reports it. The refetch is driven by the stream, not a timer, so the two
  cannot drift about when to look.

- **The stream's close is the hand-back.** When the job reaches a terminal
  state, the server closes the stream and the page reloads into the
  server-rendered finished record. The running view never re-implements the
  finished view; it hands off to it.

- **The journey fake is slowed by configuration, not by a new reserved
  hyperparameter.** `TEMPER_FAKE_LINE_DELAY_S` (default 0) spaces the simulated
  machine's output lines, so a launched job lives long enough to watch and to
  cancel. The e2e control plane is booted with a positive value; every journey
  tolerates the extra seconds.

## Alternatives considered

**Keep the two-second poll the old page used.**
Rejected: Spec 007 names the server-pushed stream as the replacement for the
poll, and the acceptance criterion is that output arrives without the page
being refreshed. A poll is also a timer that has to know when to look, whereas
a state-event-driven record refetch looks exactly when the stream says
something changed.

**Push the full job record on the stream alongside each state event.**
Rejected: it couples the stream's wire format to the record's shape, and the
record is already fetched on every page load and on every state transition.
The stream should carry events; the record is a fetch.

**Re-implement the finished view inside the running view once the job ends.**
Rejected: that duplicates the finished-job page, which a sibling owns (#77),
and two renderings of one record are the drift this repo exists to prevent.
The reload hand-back keeps the finished record the single finished view.

**Pause the fake through a new reserved hyperparameter, like
`simulated_failure_code`.**
Rejected: that key lives in `trainer-defaults.json`'s `allowed_overrides`,
which is the trainer configuration schema -- #33's surface this wave. Adding
another fake-only key to it would leak test plumbing into the schema the
advanced surface is generated from. Slowing the shared fake via environment
achieves the same watchable run without touching #33's boundary.

**Run a second, paused control plane for the cancel journey.**
Rejected: two control-plane processes in one worktree share one SQLite file
(`DB_PATH` is not configurable), and concurrent writers invite exactly the
database-is-locked flake the journeys exist to catch. One slowed fake is
simpler and no journey measures the difference.

## Consequences

- The API gains the stream endpoint; the contract documents it even though the
  generated fetch client cannot consume it (an EventSource is not a fetch, the
  same reason the adapter download is an anchor rather than a client call).
  The stream URL and parser live in `src/lib/jobs/stream.ts`, transport only.
- The running half of `/jobs/:id` is `RunningJobView`; the finished half
  remains `JobRecordView`. The old watch page's cancel-form proxy and its
  stylesheet proxy are deleted, leaving `/v1/:path*` the only rewrite.
- The fake provider is slowed by configuration for the journeys; unit tests
  inject their own fakes and are unaffected.
- A latent `active_jobs` startup path can raise when a stale non-terminal job
  exists in a development database (the row lacks a mapped `warnings` key).
  Pre-existing, unrelated to this issue; encountered while the journey's own
  database still held a job from a killed process, and resolved by clearing
  the disposable dev database rather than by changing the record mapper here.

## Rollback

Remove the stream endpoint and the client's `EventSource`, restore the
server-rendered meta-refresh or poll for non-terminal jobs, and the running
view is back to #40's interim behaviour with no other code to unwind.
