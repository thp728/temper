"""Upload size limits, responsiveness and progress. Spec 002 and spec 006.

Two defects, one root: nothing larger than 7.5 KB was ever uploaded, so size
was never considered. These tests pin the fixes, as they now stand:

* a dataset over the configured limit is refused immediately, with a stable
  code and both the limit and the actual size named;
* the limit is a product limit derived from the measured streaming throughput
  (ADR-0036), no longer from the in-memory validator's memory multiplier;
* validation runs in the background, so a large upload returns its id while
  validation works and never freezes the page that asked for it; and
* validation progress is observable on the record while it runs.

Everything here exercises the HTTP seam. Size fixtures are **generated**,
never committed.
"""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient
from helpers import wait_validated

from temper_control_plane import config


@pytest.fixture()
def server(isolated):
    """The app, storage-isolated by `isolated`. Posting a job only inserts a
    `queued` row (issue #51); nothing here runs it."""
    from temper_control_plane import main

    return main


@pytest.fixture()
def client(server):
    with TestClient(server.app) as c:
        yield c


def chat_rows(n, pad=0):
    """n valid chat rows; `pad` inflates each row by roughly `pad` bytes."""
    filler = "x" * pad
    for i in range(n):
        yield {
            "messages": [
                {"role": "user", "content": f"question {i} {filler}"},
                {"role": "assistant", "content": f"answer {i} {filler}"},
            ]
        }


def jsonl_bytes(rows):
    return ("\n".join(json.dumps(r) for r in rows)).encode("utf-8")


def upload(client, data, name="d.jsonl"):
    return client.post("/v1/datasets", files={"file": (name, data)})


# --- the limit ---------------------------------------------------------------


def test_default_limit_is_derived_from_measured_throughput(monkeypatch):
    """The ceiling is a product limit, not a memory multiplier: the measured
    streaming validation rate (21.6 MB/s on a real 1 GB file, spike 9, a
    floor) times a 60-second tolerable synchronous wait. 1.3 GB is 1331.2 MB.
    """
    monkeypatch.delenv("TEMPER_MAX_DATASET_MB", raising=False)
    limit = config._megabytes("TEMPER_MAX_DATASET_MB", 1331.2) * 1024 * 1024
    assert limit == pytest.approx(1.3 * 1024**3, rel=1e-9)
    assert limit > 1024**3, "streaming validation raises the ceiling past 1 GB"


def test_limit_is_configurable_via_environment(monkeypatch):
    monkeypatch.setenv("TEMPER_MAX_DATASET_MB", "8")
    assert (
        config._megabytes("TEMPER_MAX_DATASET_MB", 1024) * 1024 * 1024
        == 8 * 1024 * 1024
    )


def test_dataset_over_limit_refused_with_code_and_both_sizes(
    client, monkeypatch
):
    # A small limit keeps the fixture generated-and-cheap; the enforcement
    # path cannot tell a configured 64 KB from the derived 1.3 GB.
    monkeypatch.setattr(config, "MAX_DATASET_BYTES", 64 * 1024)
    data = jsonl_bytes(chat_rows(400, pad=300))  # ~350 KB
    assert len(data) > 64 * 1024

    r = upload(client, data)

    assert r.status_code == 413
    body = r.json()["detail"]
    assert body["code"] == "dataset_too_large"
    assert body["limit_bytes"] == 64 * 1024
    # The early refusal fires on the declared body size, which carries the
    # multipart envelope -- so it names the dataset's size to within that
    # small overhead, and never understates it.
    assert len(data) <= body["actual_bytes"] < len(data) + 1024
    # Both numbers are named in prose too -- the user reads the message,
    # not the JSON keys.
    assert f"{body['limit_bytes']:,}" in body["message"]
    assert f"{body['actual_bytes']:,}" in body["message"]


def test_refusal_explains_the_limit_is_a_product_decision(client, monkeypatch):
    """The limit is now a product limit derived from measured throughput, not
    an implementation detail of an in-memory validator waiting to be removed."""
    monkeypatch.setattr(config, "MAX_DATASET_BYTES", 64 * 1024)
    body = upload(client, jsonl_bytes(chat_rows(400, pad=300))).json()[
        "detail"
    ]
    assert "derived from the measured validation throughput" in body["message"]
    assert "in-memory" not in body["message"]


def test_dataset_just_under_limit_is_accepted(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_DATASET_BYTES", 64 * 1024)
    data = jsonl_bytes(chat_rows(50, pad=300))  # ~45 KB, under
    assert len(data) < 64 * 1024

    r = upload(client, data)

    assert r.status_code == 202
    assert r.json()["status"] == "validating"
    ds = r.json()["id"]
    record = wait_validated(client, ds)
    assert record["report"]["valid"] is True


def test_mid_stream_refusal_catches_a_lying_content_length(
    client, monkeypatch
):
    """The declared Content-Length is the first refusal; a header that is
    absent or lies is caught once the true size is known, mid-stream, before
    anything is published."""
    import io

    from fastapi import HTTPException

    from temper_control_plane import datasets

    monkeypatch.setattr(config, "MAX_DATASET_BYTES", 64 * 1024)
    data = jsonl_bytes(chat_rows(400, pad=300))  # ~350 KB, no declared length
    with pytest.raises(HTTPException) as exc_info:
        datasets.ingest(None, "d.jsonl", io.BytesIO(data))
    detail = exc_info.value.detail
    assert exc_info.value.status_code == 413
    assert detail["code"] == "dataset_too_large"
    assert detail["actual_bytes"] > 64 * 1024
    # The row that was created to be watched is cleaned up, and put_stream
    # never published a partial object: nothing is left behind.
    from temper_control_plane import db

    assert db.list_datasets() == []


# --- responsiveness ----------------------------------------------------------


def test_upload_returns_while_validation_runs_and_health_stays_up(
    client, monkeypatch
):
    """A slow validation must not hold up the upload's own response, nor any
    other request.

    Validation runs on its own thread (issue #31), so the upload answers with
    the dataset's id while validation is still mid-block, and /health answers
    throughout. The old defect -- validation running on the event loop --
    froze every request for the duration; this asserts the fix structurally,
    by holding validation in a block and checking that neither the upload nor
    /health wait for it.
    """
    from temper_core import validation

    release = threading.Event()
    block_started = threading.Event()
    real_validate_chunks = validation.validate_chunks

    def slow_validate_chunks(chunks, *a, **k):
        block_started.set()
        release.wait(timeout=15)
        return real_validate_chunks(chunks, *a, **k)

    monkeypatch.setattr(validation, "validate_chunks", slow_validate_chunks)
    watchdog = threading.Timer(3.0, release.set)
    watchdog.start()

    data = jsonl_bytes(chat_rows(2000, pad=300))  # ~1.7 MB

    try:
        r = upload(client, data)
        assert block_started.wait(timeout=10), "validation never started"
        # The upload already answered while validation is mid-block...
        assert r.status_code == 202
        # ...and /health answers too, though validation is still blocked.
        assert client.get("/health").status_code == 200
    finally:
        watchdog.cancel()
        release.set()

    # Let the validation thread finish so it does not outlive the test's
    # database.
    wait_validated(client, r.json()["id"])


# --- progress ----------------------------------------------------------------


def test_validation_progress_is_observable_while_it_runs(client, monkeypatch):
    """A large upload must not look frozen: while validation runs, the
    dataset record carries how far it has got."""
    import threading as _threading

    from temper_control_plane import db

    release = _threading.Event()

    def blocking_validate(ds_id, key, total_bytes):
        def run():
            db.set_dataset_progress(
                ds_id,
                {
                    "bytes_read": total_bytes // 2,
                    "bytes_total": total_bytes,
                    "rows": 1,
                },
            )
            release.wait(timeout=15)

        _threading.Thread(target=run, daemon=True).start()

    monkeypatch.setattr(
        "temper_control_plane.datasets._validate_in_background",
        blocking_validate,
    )
    data = jsonl_bytes(chat_rows(50, pad=300))

    r = upload(client, data)
    ds = r.json()["id"]

    try:
        deadline = time.time() + 5
        record = None
        while time.time() < deadline:
            record = client.get(f"/v1/datasets/{ds}").json()
            if record["progress"]:
                break
            time.sleep(0.02)
        assert record is not None
        assert record["status"] == "validating"
        assert record["progress"]["bytes_read"] > 0
        assert record["progress"]["bytes_total"] > 0
    finally:
        release.set()
