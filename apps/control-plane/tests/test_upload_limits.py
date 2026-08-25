"""Upload size limits and responsiveness — spec 002, issue #9.

Two defects, one root: nothing larger than 7.5 KB was ever uploaded, so size
was never considered. These tests pin the fixes:

* a dataset over the configured limit is refused immediately, with a stable
  code and both the limit and the actual size named;
* validation no longer runs on the event loop, so a large upload blocks only
  its own request.

Everything here exercises the HTTP seam, per the spec's testing decisions.
Size fixtures are **generated**, never committed.
"""

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from temper_control_plane import config  # noqa: E402


@pytest.fixture()
def server(tmp_path, monkeypatch):
    """The app with storage redirected to tmp_path and the GPU stubbed out."""
    from temper_control_plane import datasets, db, main, orchestrator

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(datasets, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(orchestrator, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(orchestrator, "launch", lambda job_id: None)
    db.init()
    datasets.UPLOADS.mkdir(parents=True, exist_ok=True)
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


def test_default_limit_is_one_gigabyte(monkeypatch):
    monkeypatch.delenv("TEMPER_MAX_DATASET_MB", raising=False)
    assert (
        config._megabytes("TEMPER_MAX_DATASET_MB", 1024) * 1024 * 1024
        == 1024**3
    )


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
    # path cannot tell a configured 64 KB from the default 1 GB.
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


def test_refusal_names_the_removal_path(client, monkeypatch):
    """The limit is an implementation limit, and says what removes it."""
    monkeypatch.setattr(config, "MAX_DATASET_BYTES", 64 * 1024)
    body = upload(client, jsonl_bytes(chat_rows(400, pad=300))).json()[
        "detail"
    ]
    assert "streaming" in body["message"].lower()


def test_dataset_just_under_limit_is_accepted(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_DATASET_BYTES", 64 * 1024)
    data = jsonl_bytes(chat_rows(50, pad=300))  # ~45 KB, under
    assert len(data) < 64 * 1024

    r = upload(client, data)

    assert r.status_code == 201
    assert r.json()["valid"] is True


# --- responsiveness ----------------------------------------------------------


def test_large_upload_does_not_block_concurrent_requests(server, monkeypatch):
    """A slow validation must hold up only its own request.

    Validation is stubbed to block until released -- standing in for the
    measured ~80 ms/MB of real CPU work. A watchdog releases it after 3s no
    matter what, so both worlds terminate deterministically. The measure is
    **wall-clock from the moment validation begins blocking to the moment
    /health answers** -- not a latency measured from the test coroutine,
    because when the loop is frozen the test coroutine is frozen with it and
    a naive clock starts late enough to hide the defect entirely:

    * validation off the loop (correct): /health answers within milliseconds
      of the block starting;
    * validation on the loop (the defect): the loop itself is frozen inside
      the stub, so /health cannot answer until the watchdog fires -- ~3s.
    """
    import threading

    from temper_core import validation

    real_validate = validation.validate
    block_started = threading.Event()
    release = threading.Event()
    block_t0 = 0.0

    def slow_validate(path, *a, **k):
        nonlocal block_t0
        block_t0 = time.monotonic()
        block_started.set()
        release.wait(timeout=15)
        return real_validate(path, *a, **k)

    monkeypatch.setattr(validation, "validate", slow_validate)
    watchdog = threading.Timer(3.0, release.set)
    watchdog.start()

    data = jsonl_bytes(chat_rows(2000, pad=300))  # ~1.7 MB

    async def scenario():
        transport = ASGITransport(app=server.app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as c:
            upload_task = asyncio.create_task(
                c.post("/v1/datasets", files={"file": ("big.jsonl", data)})
            )
            # Returns only once validation is verifiably mid-block.
            await asyncio.to_thread(block_started.wait, 10)
            r = await c.get("/health")
            # Measured here, inside the scenario: after asyncio.run returns
            # the clock includes teardown, which would mask a healthy result.
            health_done_t = time.monotonic()
            await upload_task
            return r.status_code, health_done_t

    try:
        health_status, health_done_t = asyncio.run(scenario())
    finally:
        watchdog.cancel()

    delay_to_health = health_done_t - block_t0
    assert health_status == 200
    assert delay_to_health < 1.0, (
        f"/health could not answer for {delay_to_health:.2f}s while an "
        f"upload was validating; validation is running on the event loop."
    )
