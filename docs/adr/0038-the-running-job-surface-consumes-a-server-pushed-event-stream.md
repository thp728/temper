# ADR-0038 — The running-job surface consumes a server-pushed event stream

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

- **The terminal hand-back is a record refetch, not the connection closing.**
  When the stream reports a terminal state transition, the view refetches the
  record and reloads on a terminal status into the server-rendered finished
  record. The refetch retries on every reconnect -- a dropped connection
  re-delivers the terminal state event, so a transient refetch failure
  self-heals. The server also emits an explicit `event: end` marker after the
  terminal transition, and the page reloads on it where the transport
  delivers it, but nothing relies on the connection closing as a signal: a
  browser's EventSource reconnects on a server-initiated close instead of
  reporting `CLOSED`, and a proxy in the path can fail to propagate the final
  chunk or the close. The running view never re-implements the finished view;
  it hands off to it.

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

**React to the connection closing when the job reaches a terminal state.**
Rejected once observed: a browser's EventSource reconnects on a
server-initiated close instead of reporting `CLOSED`, so close alone could
never hand the page back -- it would reconnect forever against a terminal
job. The hand-back therefore rides on the terminal state event itself (a
record refetch that reloads on a terminal status), with an explicit
`event: end` marker as a supplementary signal rather than the sole one.

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
Rejected: it would have doubled the processes the journeys boot for one test,
and two writers over one SQLite file invite exactly the database-isolated
flake the journeys exist to catch. The single slowed fake needs no second
server.

**Delete the journey database from the Playwright config before each run.**
Rejected once observed: a Playwright config is loaded more than once per run,
and a wipe at load time deleted the backend's freshly-created database out
from under it (surfaced as `no such table: datasets` on every request). The
reset therefore lives in the backend's own `init()`, once per process start.

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
- The journeys' control plane now owns its runtime state the way it owns its
  ports: a database of its own (`TEMPER_DB_PATH`) that it resets at startup
  (`TEMPER_DB_RESET`), and object storage of its own
  (`TEMPER_STORAGE_ROOT`). A journey never writes to the developer's
  `data/`, and a run killed mid-job cannot leave an orphaned non-terminal job
  that trips the orphaned-job scan on the next run's startup. The journeys'
  e2e ports were already derived from the issue number (3930/3931) with
  `reuseExistingServer: false`, and the web server is pointed at the backend
  it booted (`TEMPER_BACKEND_URL`) -- without that, the app proxied to the
  developer-facing 8000 while the journey backend ran elsewhere.
- A latent `active_jobs` startup path can raise when a stale non-terminal job
  exists in a development database (the row lacks a mapped `warnings` key).
  Pre-existing, unrelated to this issue; encountered while the journey's own
  database still held a job from a killed process. This issue does not fix it;
  it makes the journeys immune to it, and a developer who hits it clears the
  disposable dev database.

## Rollback

Remove the stream endpoint and the client's `EventSource`, restore the
server-rendered meta-refresh or poll for non-terminal jobs, and the running
view is back to #40's interim behaviour with no other code to unwind.
