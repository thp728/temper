# ADR-0070 - Events are persisted before they are published, and watchers replay by last-seen identifier

- **Status:** accepted
- **Date:** 2026-08-30
- **Spec:** `docs/specs/008-the-stack-foundation.md`
- **Issue:** [#57](https://github.com/thp728/temper/issues/57)

## Context

The watch interface (ADR-0039/#39) reads a running job's events from
`GET /v1/jobs/{id}/stream` instead of polling. But the stream behind that
endpoint *polled the durable log*: it re-read `db.get_events` every quarter
second, because "a worker thread writes events and this process has no broker
to subscribe to" (the stream's own pre-change docstring). That was the honest
description of the gap this record closes.

Two properties of the running system made the poll not just inelegant but
wrong to keep:

- **The worker that records events is a separate process** (ADR-0066/#51). A
  channel living inside the control-plane process could never carry the
  worker's events to the streams the control plane serves. The channel had
  to live in the one component both processes already share.
- **A poll has latency equal to its interval and scales with its readers.**
  Spec 008 names the target: a publish-subscribe channel "fanning persisted
  events out to every watcher, so a second process holding the events is not
  a problem and a second watcher is not either." Two watchers on one job each
  polling the log is two polls, and neither learns anything until its next
  tick.

And the rule the spec states twice, as the load-bearing contract: **an event
is persisted before it is published.** The store is the truth and the channel
is a notification, so a watcher that misses a message, reconnects, or arrives
late replays from the store by the last identifier it saw and loses nothing.
Publishing first and persisting after would make a dropped connection lossy
in a way no watcher could detect.

## Decision

**The channel is PostgreSQL's LISTEN/NOTIFY, and persist-then-publish is a
property of the database, not a discipline.**

- **`db._append_event` — the single choke point through which every event is
  inserted — now issues the NOTIFY in the same transaction as the INSERT**,
  after it, with the event's id as the payload. PostgreSQL delivers a NOTIFY
  only when its transaction commits: an event whose transaction rolls back
  publishes nothing, and a subscriber can never receive an event that is not
  already durable. Publishing is therefore ordered after persisting by the
  database itself, not by a call ordering a future editor could invert.
  `db.py` has exactly one place that appends an event (`_append_event`, used
  by `create_job`, `set_state`, `request_cancel`, `claim_next_job` and
  `add_event`), so one statement publishes every event ever recorded.

- **The channel is named per job** (`job_events_<job_id>`, built once in
  `channel.channel_for` and read by both the publisher and the subscriber, so
  the two cannot drift apart in spelling — ADR-0010's one-definition rule). A
  NOTIFY on a job's channel reaches every connection LISTENing on it and only
  those, which is what makes two watchers on one job both receive everything
  with neither affecting the other: each holds its own connection and its own
  cursor.

- **The stream endpoint reads history from the store first, then follows the
  channel** (`main._job_event_stream`). Each stream opens a dedicated
  subscription, waits for it to confirm its LISTEN, then does its catch-up
  read from the store — the ordering that closes the subscribe/catch-up race
  (an event persisted between the read and the LISTEN would otherwise be
  missed by both). From then on the stream waits on the channel: every
  notification wakes it to re-read the store from its cursor and yield
  whatever is new, so a live event is delivered within a re-read of the
  moment its transaction commits. The store is the source of the data; the
  channel is the signal that data may exist. The stream's idle wake (its
  heartbeat timeout) also re-reads the store — that is the self-healing net,
  not the delivery mechanism: a channel that dropped while no connection was
  listening loses nothing, because re-reading an idempotent cursor is the
  same replay a watcher's own reconnection performs. This is what makes the
  dropped-connection criterion testable for real: the client's replay is the
  product, and the channel is only ever an acceleration on top of it.

- **A dropped channel is not a dead stream.** The subscription runs a
  dedicated connection on a daemon thread and reconnects (re-LISTEN) on its
  own if the connection dies; the stream's next store re-read delivers
  anything published in the gap. The same property, on the client side, is
  the acceptance criterion: a watcher whose connection drops replays by its
  `Last-Event-ID` (or `after`) and receives exactly what it missed, in order,
  once.

- **The subscription is thread-bridged, not an async psycopg connection.**
  psycopg's `AsyncConnection` cannot run on Windows' `ProactorEventLoop`
  (the default uvicorn loop on this development machine), so the
  subscription runs a *synchronous* connection on a daemon thread and wakes
  the asyncio loop with `call_soon_threadsafe` — the same thread-offload the
  control plane already uses for all blocking database work (ADR-0006,
  ADR-0064). The thread and its connection are owned by the stream: created
  on enter, stopped and closed on exit, so a dropped connection cleans up
  after itself.

## Why the seam held

The seam was the stream endpoint's own docstring, which said the poll existed
because "this process has no broker to subscribe to." The stream's contract —
events pushed as recorded, `after`/`Last-Event-ID` resume without
re-delivery, ending only at a terminal state with the terminal transition
delivered first — is untouched by this change. The five pre-existing
`test_stream_endpoint.py` tests pass unchanged against the channel-driven
implementation, which is the same assertion ADR-0066 makes about `run_job`:
the contract the prior issue pinned is the contract this issue's mechanism
satisfies. The interface side (ADR-0039's EventSource client) needed no
change at all: from the browser's seat, nothing about the stream changed.

One thing the seam did not carry: the notification had to be bound into the
INSERT path itself, which meant `_append_event`'s shape changed from "one
INSERT" to "INSERT ... RETURNING id, then NOTIFY". That is a change to
persistence internals, not to any function's public contract — every caller
of `db.add_event`, `db.set_state` and the rest is unchanged.

## Alternatives considered

**An in-process pub-sub (a `threading.Condition` or an `asyncio` fan-out hub)
so the stream could be notified without a broker.** Rejected immediately: the
worker is a separate process (ADR-0066), and an in-process hub would notify
nobody when events were written by the worker. This was the *first* rejected
option precisely because the polling stream's own comment ("a worker thread
writes events and this process has no broker") stopped being true the moment
the worker shipped. A channel had to be cross-process.

**Redis pub/sub.** A real broker, and the one the broader Phase B stack
contemplates. Rejected for this issue because PostgreSQL LISTEN/NOTIFY does
the job with a dependency the system already runs and already healthchecks:
Redis would be a second stateful service added for one feature, with its own
connection handling and its own failure mode to reason about, and it would
not make the core guarantee (persist-then-publish) any stronger — that comes
from the store, which is Postgres either way. The channel being the same
database that holds the truth means the notification can never be delivered
ahead of the persist that is its precondition.

**A dedicated table polled by the stream but written by the worker (the
status quo).** The previous implementation. Rejected because it is the thing
being replaced: bounded latency, no way for a second process's writes to
wake a watcher, and one query per stream per interval that a channel makes
unnecessary when idle.

**Server-side change-stream protocols (Postgres logical decoding / `wal2json`
/ triggers writing an outbox table).** The outbox table (a triggers-or-application
second write that a poller reads) is a strictly weaker form of what NOTIFY
already gives: NOTIFY is the outbox where the "notification" delivery is
handled by the database, transactional with the event, with no second table
to lag behind. Logical decoding exists for replaying a *whole* database to
another system, which this issue does not need; it would add a replication
slot to manage for the privilege of re-implementing what NOTIFY does natively.

**Keep the poll but shorten it.** Considered only to be dismissed: it does
not scale to readers (spec 008's "a second watcher is not either"), and it
does not change the delivery mechanism's character. The acceptance criteria
for this issue are about the channel fanning out and replay working — neither
improves by polling faster.

## Consequences

- Every event recorded anywhere (`db.add_event`, `db.set_state`,
  `request_cancel`, `claim_next_job`, `create_job`) now publishes to its
  job's channel. The cost is one more statement in the same transaction as
  the INSERT; the guarantee is that no watcher sees an event before it is
  durable.
- The stream endpoint now holds one dedicated PostgreSQL connection per open
  stream (plus one daemon thread). For a live-watch surface this is
  proportionate — streams are long-lived and few — and the connection is
  closed on stream exit. The pool is untouched; a subscription never borrows
  a pooled connection, because LISTEN is a long-lived session state that must
  not be returned to a shared pool.
- A worker crash mid-job is unaffected: the worker's events stop being
  written (and published) when it dies, exactly as before. The reconciler
  that recovers the stuck row is Spec 010's, as ADR-0066 already records.
- A notification can be missed while a connection is down (a restart of the
  database, a killed connection) — that is not data loss, because the store
  is the truth and the stream re-reads it. This is recorded so the two are
  not confused: the channel optimizes latency; the store guarantees
  correctness.
- The web interface needed no change: it already consumed the stream through
  EventSource (ADR-0039), and the stream's wire shape is unchanged.

## Rollback

Revert this issue's commits. `db._append_event` returns to a plain INSERT
(the NOTIFY disappears), the stream endpoint returns to its store-polling
loop, and `channel.py` is deleted. No migration, no schema change, and the
contract the interface consumes (`after`/`Last-Event-ID`, terminal end
marker) is byte-identical either direction — the seam held on both sides.
