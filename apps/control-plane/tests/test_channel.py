"""The event channel: persisted first, published second (ADR-0070).

Spec 008's load-bearing rule -- an event is persisted before it is published;
the store is the truth and the channel is a notification -- is a property of
PostgreSQL's LISTEN/NOTIFY here, not a discipline: `db._append_event` issues
the NOTIFY in the same transaction as the INSERT, and PostgreSQL delivers a
NOTIFY only when its transaction commits. These tests prove that against the
real store rather than by reading source: a rolled-back event publishes
nothing, and an event is already durable in the store the moment its
notification arrives.

The channel itself is exercised end to end by `test_stream_endpoint.py`
(two watchers, replay after a dropped connection); this file pins the one
property the stream tests can only assume.
"""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from temper_control_plane import channel, db


def _writer() -> psycopg.Connection:
    """A write connection shaped like the pool's (dict rows, one transaction),
    for the call into `db._append_event` -- which reads its RETURNING row by
    column name, the shape the pool always gives it."""
    return psycopg.connect(db.DATABASE_URL, row_factory=dict_row)


def _listening(job_id: str):
    """A real listener on the job's channel, the same connection shape the
    stream's subscription uses (autocommit, LISTEN, dedicated)."""
    conn = psycopg.connect(db.DATABASE_URL, autocommit=True)
    conn.execute(f'LISTEN "{channel.channel_for(job_id)}"')
    return conn


def test_a_rolled_back_event_publishes_nothing(isolated):
    ds_id = db.create_dataset("d.jsonl", "datasets/ds_chan.jsonl", "ds_chan")
    job_id = db.create_job(ds_id, "qwen3-4b", {})
    conn = _listening(job_id)
    try:
        # The subscription starts empty: the `queued` event was published by
        # create_job *before* this connection LISTENed, so it is not queued
        # on the channel -- history lives in the store and is replayed, not
        # re-notified (ADR-0070).
        assert list(conn.notifies(timeout=0.2)) == []

        # An event whose transaction rolls back must publish nothing: the
        # NOTIFY was issued, but PostgreSQL discards it with the rollback.
        with _writer() as writer:
            db._append_event(writer, job_id, "log", "will roll back")
            writer.rollback()

        # Nothing arrived. A watcher that had relied on the channel alone for
        # this event would never know it happened -- which is exactly why the
        # store, not the channel, is the truth, and why replay is by the
        # store: a dropped connection loses nothing because the store never
        # loses anything (ADR-0070).
        assert list(conn.notifies(timeout=0.4)) == []
    finally:
        conn.close()

    # The store agrees: no such event was ever durable.
    assert [e["message"] for e in db.get_events(job_id)] == ["queued"]


def test_a_committed_event_is_durable_when_its_notification_arrives(
    isolated,
):
    ds_id = db.create_dataset("d.jsonl", "datasets/ds_chan2.jsonl", "ds_chan2")
    job_id = db.create_job(ds_id, "qwen3-4b", {})
    conn = _listening(job_id)
    try:
        list(conn.notifies(timeout=0.2))  # drain the (empty) backlog

        with _writer() as writer:
            db._append_event(writer, job_id, "log", "committed")

        notifies = list(conn.notifies(timeout=2.0))
        assert len(notifies) == 1
        # The notification names the job's channel and carries the event's
        # id as its payload.
        assert notifies[0].channel == channel.channel_for(job_id)
        payload = int(notifies[0].payload)

        # The event is already durable the moment the notification arrives:
        # the notification cannot outrun its own transaction's commit, which
        # is the persist-then-publish ordering made a database property.
        events = db.get_events(job_id)
        assert events[-1]["id"] == payload
        assert events[-1]["message"] == "committed"
    finally:
        conn.close()


def test_the_channel_name_is_derived_from_the_job_id(isolated):
    # A value two components must agree on -- the store publishing and the
    # stream subscribing -- is defined once (ADR-0010): the publisher builds
    # the name with `channel_for`, the subscriber does too, and a fixed
    # spelling would let the two drift apart.
    assert channel.channel_for("job_abc123") == "job_events_job_abc123"
    assert channel.channel_for("job_abc123") != channel.channel_for(
        "job_abc124"
    )
