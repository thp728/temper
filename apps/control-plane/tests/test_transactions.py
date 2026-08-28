"""A state change and its event are written in one transaction, both present
or both absent.

`db.set_state` documents the intent: "a status that moved with no event
recorded is a run you cannot account for." This proves it by forcing the
event write to fail after the status update has already been sent to the
server, and asserting the status update did not survive either -- not by
reading `set_state`'s source and trusting the `with connect()` block.
"""

from __future__ import annotations

import pytest

from temper_control_plane import db


def test_a_failed_event_write_rolls_back_the_state_change_too(isolated):
    ds_id = db.create_dataset("d.jsonl", "datasets/ds_txn.jsonl", "ds_txn")
    job_id = db.create_job(ds_id, "qwen3-4b", {})
    assert db.get_job(job_id)["status"] == "queued"

    # `events.job_id` carries a foreign key to `jobs(id)`; a job id the
    # table rejects makes the INSERT inside `_append_event` fail exactly
    # where the real failure this criterion is about would happen -- after
    # the status UPDATE has been sent, before the transaction commits.
    with pytest.raises(Exception):
        db.set_state("job_does_not_exist", "training")

    # The row for `job_does_not_exist` was never going to exist, so this
    # alone would not prove atomicity. The real test is the job that *does*
    # exist: its own `set_state` call, forced to fail the same way.
    import temper_control_plane.db as db_module

    original_append = db_module._append_event

    def failing_append(conn, job_id_, kind, message, data=None):
        if job_id_ == job_id and kind == "state":
            raise RuntimeError("simulated event-write failure")
        return original_append(conn, job_id_, kind, message, data)

    db_module._append_event = failing_append
    try:
        with pytest.raises(RuntimeError, match="simulated event-write failure"):
            db.set_state(job_id, "training")
    finally:
        db_module._append_event = original_append

    # Both absent: the status update inside the same `with connect()` block
    # rolled back with the event write that failed after it.
    job = db.get_job(job_id)
    assert job["status"] == "queued", (
        "the state change survived a failed event write -- the transaction "
        "did not roll back both together"
    )
    events = db.get_events(job_id)
    assert [e["kind"] for e in events] == ["state"], (
        "an extra event landed despite the failure"
    )
    assert events[0]["message"] == "queued"
