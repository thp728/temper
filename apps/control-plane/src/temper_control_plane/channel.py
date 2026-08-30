"""The publish-subscribe channel a job's events fan out over.

Spec 008 names the rule this module exists for: **an event is persisted
before it is published.** The store is the truth and the channel is a
notification, so a watcher that misses a message, reconnects, or arrives late
replays from the store by the last identifier it saw and loses nothing
(ADR-0067).

The channel is PostgreSQL's LISTEN/NOTIFY, and the persist-then-publish rule
falls out of it rather than being a discipline: a NOTIFY issued in the same
transaction as the event INSERT is delivered to listeners only when that
transaction commits, so a subscriber can never receive an event that is not
already durable. `db._append_event` issues both statements in that order; this
module only listens.

Why the database rather than an in-process broker: the worker that records a
job's events (issue #51) is a *separate process* from the control plane that
serves streams, so an in-process channel would carry the worker's events
nowhere. PostgreSQL is the one component both processes already share, and a
NOTIFY on a shared server reaches every connection that is LISTENing on the
channel -- two watchers on one job both receive everything, and neither
affects the other.
"""

from __future__ import annotations

import asyncio
import threading
import time

import psycopg

from . import db

# How long the listener thread blocks waiting for a notification before it
# re-checks for a stop request. Tuning, not contract: the stream's own wake
# re-reads the store (the truth) regardless, so this only bounds how quickly
# an idle subscription notices a shutdown.
LISTEN_WAIT_S = 0.25


def channel_for(job_id: str) -> str:
    """The PostgreSQL channel one job's events publish to.

    Built from the job id, which `db.new_id` always shapes as
    ``<prefix>_<hex>`` -- characters PostgreSQL accepts in a quoted
    identifier -- and quoted at every use site regardless, so nothing
    user-shaped ever reaches a statement unquoted. The events table's own
    foreign key to `jobs(id)` already bounds the id to a real job.
    """
    return f"job_events_{job_id}"


class EventSubscription:
    """A live subscription to one job's event channel.

    The stream endpoint catches up from the store first (the truth), then
    follows this subscription for what happens next. `wait()` blocks until a
    notification wakes it or a timeout passes; the stream re-reads the store
    on either outcome, because the store is the source of the data and the
    channel is only the signal that new data may exist.

    psycopg's async connection cannot run on Windows' ProactorEventLoop (the
    default uvicorn loop), so the subscription runs a *synchronous* connection
    on a daemon thread and wakes the asyncio loop with
    ``call_soon_threadsafe`` -- the same thread-offload the rest of the
    control plane uses for blocking database work (ADR-0006, ADR-0064).

    A dropped channel (a restarted database, a killed connection) is not a
    dead stream: the listener reconnects and re-LISTENs on its own, and the
    stream's next store re-read delivers anything published in the gap -- the
    same replay that serves a watcher whose own connection dropped, because
    both are the store being the truth.
    """

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id
        self._queue: asyncio.Queue[None] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._stop = threading.Event()
        self._ready = threading.Event()
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
        deadline = time.monotonic() + 5.0
        while not self._ready.is_set():
            if time.monotonic() > deadline:
                self._stop.set()
                self._thread.join(timeout=1.0)
                raise RuntimeError(
                    f"the event channel for {self._job_id} never became ready"
                )
            await asyncio.sleep(0.01)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _listen(self) -> None:
        while not self._stop.is_set():
            conn: psycopg.Connection | None = None
            try:
                # A dedicated connection, not the pool: LISTEN is a
                # long-lived session state that must not be returned to a
                # shared pool, and this connection is held for the stream's
                # lifetime. `db.DATABASE_URL` is read at call time so the
                # per-test database redirect applies (ADR-0064).
                conn = psycopg.connect(db.DATABASE_URL, autocommit=True)
                conn.execute(f'LISTEN "{channel_for(self._job_id)}"')
                self._ready.set()
                for _notify in conn.notifies(timeout=LISTEN_WAIT_S):
                    if self._stop.is_set():
                        break
                    self._loop.call_soon_threadsafe(
                        self._queue.put_nowait, None
                    )
            except psycopg.Error:
                # A dropped channel is not a dead stream: the store is the
                # truth, and the stream re-reads it on its next wake. Back
                # off briefly and re-LISTEN.
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
        store either way."""
        try:
            await asyncio.wait_for(self._queue.get(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
