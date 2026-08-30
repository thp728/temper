"""The publish-subscribe channel a job's live data fans out over.

Spec 008 names the rule this module exists for: **an event is persisted
before it is published.** The store is the truth and the channel is a
notification, so a watcher that misses a message, reconnects, or arrives late
replays from the store by the last identifier it saw and loses nothing
(ADR-0067).

The channel is PostgreSQL's LISTEN/NOTIFY, and the persist-then-publish rule
falls out of it rather than being a discipline: a NOTIFY issued in the same
transaction as the write it announces is delivered to listeners only when
that transaction commits, so a subscriber can never receive a notification
for a change that is not already durable. `db.py` issues the NOTIFY for every
change the stream renders -- events, and the progress/output snapshots that
ride the same connection (issue #49) -- after the write, in the same
transaction; this module only listens.

Why the database rather than an in-process broker: the worker that records a
job's events (issue #51) is a *separate process* from the control plane that
serves streams, so an in-process channel would carry the worker's events
nowhere. PostgreSQL is the one component both processes already share, and a
NOTIFY on a shared server reaches every connection that is LISTENing on the
channel -- two watchers on one job both receive everything, and neither
affects the other.

The stream is driven by this channel, not by a timer: when nothing is written
the stream waits and makes no store reads, and each publish wakes it to
re-read the store (the truth) from its cursor. The channel is therefore the
delivery mechanism, not an ornament on a poll (ADR-0067).
"""

from __future__ import annotations

import asyncio
import threading
import time

import psycopg

# How long the listener thread blocks waiting for a notification before it
# re-checks for a stop request. Tuning, not contract: the stream's wake
# re-reads the store (the truth) on a notification or a reconnect, and this
# only bounds how quickly an idle subscription notices a shutdown.
LISTEN_WAIT_S = 0.25
# The same ceiling, applied to the subscription's own startup and shutdown.
# A LISTEN registration is a local round-trip measured in milliseconds; these
# are safety ceilings so that a stuck database fails the caller rather than
# hanging it, not timings the correct path ever reaches.
READY_TIMEOUT_S = 5.0
STOP_JOIN_S = 1.0
READY_POLL_S = 0.01


def channel_for(job_id: str) -> str:
    """The PostgreSQL channel one job's live data publishes to.

    Built from the job id, which `db.new_id` always shapes as
    ``<prefix>_<hex>`` -- characters PostgreSQL accepts in a quoted
    identifier -- and quoted at every use site regardless, so nothing
    user-shaped ever reaches a statement unquoted. The events table's own
    foreign key to `jobs(id)` already bounds the id to a real job. Defined
    once and read by both the publisher (`db.py`) and the subscriber, so the
    two cannot drift apart in spelling (ADR-0010).
    """
    return f"job_events_{job_id}"


class EventSubscription:
    """A live subscription to one job's channel.

    The stream endpoint catches up from the store first (the truth), then
    follows this subscription for what happens next. `wait()` blocks until a
    notification wakes it or a timeout passes, and reports which; the stream
    re-reads the store on a wake and stays silent on a timeout, which is what
    makes the channel the delivery mechanism rather than a poll. `take_
    reconnected()` reports a re-established LISTEN, so the stream re-reads
    once to close the gap a dropped channel opens (ADR-0067).

    psycopg's async connection cannot run on Windows' ProactorEventLoop (the
    default uvicorn loop), so the subscription runs a *synchronous* connection
    on a daemon thread and wakes the asyncio loop with
    ``call_soon_threadsafe`` -- the same thread-offload the rest of the
    control plane uses for blocking database work (ADR-0006, ADR-0064).

    A dropped channel (a restarted database, a killed connection) is not a
    dead stream: the listener reconnects and re-LISTENs on its own, and the
    stream's next wake re-reads the store -- the same replay that serves a
    watcher whose own connection dropped, because both are the store being
    the truth.
    """

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id
        self._queue: asyncio.Queue[None] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._reconnected = threading.Event()
        self._thread = threading.Thread(
            target=self._listen, name=f"stream-{job_id[:8]}", daemon=True
        )

    async def __aenter__(self) -> EventSubscription:
        self._thread.start()
        # The catch-up read must not race the LISTEN registration: an event
        # persisted before the read but after the LISTEN is both in the store
        # (caught by the read) and notified (a no-op re-read later); an event
        # persisted after the read is notified. Waiting for the listener to
        # confirm its LISTEN makes that ordering certain.
        deadline = time.monotonic() + READY_TIMEOUT_S
        while not self._ready.is_set():
            if time.monotonic() > deadline:
                self._stop.set()
                self._thread.join(timeout=STOP_JOIN_S)
                raise RuntimeError(
                    f"the event channel for {self._job_id} never became ready"
                )
            await asyncio.sleep(READY_POLL_S)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=STOP_JOIN_S)

    def _listen(self) -> None:
        # The first successful LISTEN is the subscription itself, accounted
        # for by the stream's catch-up read; only a *re*connect raises the
        # reconnect flag the stream re-reads on, so the two are told apart.
        first_connect = True
        while not self._stop.is_set():
            conn: psycopg.Connection | None = None
            try:
                # A dedicated connection, not the pool: LISTEN is a
                # long-lived session state that must not be returned to a
                # shared pool, and this connection is held for the stream's
                # lifetime. `db.DATABASE_URL` is read at call time so the
                # per-test database redirect applies (ADR-0064).
                from . import db

                conn = psycopg.connect(db.DATABASE_URL, autocommit=True)
                conn.execute(f'LISTEN "{channel_for(self._job_id)}"')
                self._ready.set()
                if not first_connect:
                    self._reconnected.set()
                first_connect = False
                # Keep listening on the *same* connection: `notifies(timeout)`
                # ends after the wait interval, and LISTEN is sticky, so the
                # interval just means "wait again" -- reconnecting on it
                # would needlessly re-raise the reconnect flag the stream
                # re-reads on. Only an error drops the connection.
                while not self._stop.is_set():
                    for _notify in conn.notifies(timeout=LISTEN_WAIT_S):
                        if self._stop.is_set():
                            break
                        # The notification's payload (an event id, or a
                        # progress/output marker) is informational: the
                        # stream re-reads the store either way, because the
                        # store is the truth and the channel only says that
                        # new data exists.
                        self._loop.call_soon_threadsafe(
                            self._queue.put_nowait, None
                        )
            except psycopg.Error:
                # A dropped channel is not a dead stream: the store is the
                # truth, and the stream re-reads it on the wake that follows
                # a reconnect. Back off briefly and re-LISTEN.
                if not self._stop.is_set():
                    self._ready.clear()
                    self._stop.wait(LISTEN_WAIT_S)
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except psycopg.Error:
                        pass

    async def wait(self, timeout: float) -> bool:
        """Block until a notification wakes the subscription or `timeout`
        passes. True when a notification arrived; the caller re-reads the
        store then. A timeout means nothing was published, so the store has
        not changed and the caller reads nothing."""
        try:
            await asyncio.wait_for(self._queue.get(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def take_reconnected(self) -> bool:
        """Whether the listener re-established its LISTEN since the last
        check. True after a drop-and-reconnect: the stream re-reads the store
        once to deliver anything published while no connection was listening
        (lost from the channel, never from the store -- ADR-0067)."""
        if self._reconnected.is_set():
            self._reconnected.clear()
            return True
        return False
