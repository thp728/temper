"""Helpers shared across the control-plane test files.

Validation is asynchronous (issue #31): an upload returns while validation
runs in the background, and the report lands on `GET /v1/datasets/{id}`.
Tests that need a validated dataset therefore poll the record instead of
reading the upload response -- the one thing every file with a `valid_dataset`
helper needs, defined once.

Token counting is asynchronous too (issue #42): it runs as its own phase after
validation, so tests that need a counted dataset poll for the phase's `done`
state the same way.
"""

from __future__ import annotations

import time

_IN_PROGRESS_STATUSES = ("importing", "validating")


def wait_validated(client, ds_id: str, timeout: float = 10.0) -> dict:
    """Poll the dataset record until it reaches a terminal status. Returns
    the record.

    An import (issue #45) passes through "importing" (its bytes are still
    being fetched) before "validating", so both count as still in progress;
    an upload skips straight to "validating". Raises instead of returning an
    unfinished record, because every caller wants a verdict and a test that
    silently used an unfinished one would be testing nothing.
    """
    deadline = time.time() + timeout
    last_status = "unknown"
    while time.time() < deadline:
        record = client.get(f"/v1/datasets/{ds_id}").json()
        last_status = record["status"]
        if last_status not in _IN_PROGRESS_STATUSES:
            return record
        time.sleep(0.02)
    raise AssertionError(
        f"dataset {ds_id} was still {last_status!r} after {timeout:.0f}s"
    )


def wait_counted(client, ds_id: str, timeout: float = 10.0) -> dict:
    """Poll until the counting phase reaches a terminal state. Returns the
    record.

    Raises instead of returning a still-counting record: a caller that wanted
    the count but got the phase in flight would be testing nothing."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = client.get(f"/v1/datasets/{ds_id}").json()
        status = record.get("token_count_status")
        if status in ("done", "failed"):
            return record
        time.sleep(0.02)
    raise AssertionError(
        f"dataset {ds_id} was still counting after {timeout:.0f}s"
    )
