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


def wait_validated(client, ds_id: str, timeout: float = 10.0) -> dict:
    """Poll the dataset record until validation finishes. Returns the record.

    Raises instead of returning a still-validating record, because every
    caller wants a verdict and a test that silently used an unfinished one
    would be testing nothing.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = client.get(f"/v1/datasets/{ds_id}").json()
        if record["status"] != "validating":
            return record
        time.sleep(0.02)
    raise AssertionError(
        f"dataset {ds_id} was still validating after {timeout:.0f}s"
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
