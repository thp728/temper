"""The live event stream: a server-pushed channel over the durable log.

The running-job surface (issue #39) reads events from `GET /v1/jobs/{id}/stream`
instead of polling. The contract is: events are pushed oldest-first as they are
recorded; each event carries an `id` a reconnecting EventSource replays; the
`after` query and the `Last-Event-ID` header resume without re-delivery; and
the stream ends exactly when the job reaches a terminal state, having
delivered the terminal transition itself.
"""

import json
import threading
import time

import pytest
from helpers import wait_validated

from temper_control_plane.fake_provider import (
    DEMO_ADAPTER_BYTES,
    DEMO_LINES,
    DEMO_RESULT,
    FakeProvider,
    completed_run,
)


def _jsonl(rows):
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def _chat_rows(n):
    return [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(n)
    ]


def _upload_and_create(client, driver=None):
    """A job created through the API. The request path only inserts a
    `queued` row (issue #51); if `driver` is given, it is called with the
    job id right after creation, the way the worker would drive it."""
    r = client.post(
        "/v1/datasets",
        files={
            "file": (
                "d.jsonl",
                _jsonl(_chat_rows(12)).encode("utf-8"),
                "application/octet-stream",
            )
        },
    )
    # 202, not 201: since #31 the upload returns as soon as the bytes are
    # stored and validation runs in the background, so a job cannot be created
    # from the response alone -- wait for the verdict first.
    assert r.status_code == 202, r.text
    ds_id = r.json()["id"]
    wait_validated(client, ds_id)
    r = client.post("/v1/jobs", json={"dataset_id": ds_id})
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    if driver is not None:
        driver(job_id)
    return job_id


def _run_to_completion(job_id):
    """Drive a job synchronously to completion on the canned fake."""
    from temper_control_plane import fake_models, orchestrator

    orchestrator.run_job(
        job_id,
        provider=completed_run(),
        models=fake_models.catalog_models(),
    )


def _wait_terminal(client, job_id):
    deadline = time.time() + 20
    while time.time() < deadline:
        if client.get(f"/v1/jobs/{job_id}").json()["status"] in (
            "complete",
            "failed",
            "cancelled",
        ):
            return
        time.sleep(0.01)
    raise AssertionError("job never reached a terminal state")


def _stream_body(client, url, **kwargs):
    """The whole stream body for a stream that ends (a terminal job's)."""
    with client.stream("GET", url, **kwargs) as r:
        assert r.status_code == 200, r.text
        return r.read().decode("utf-8")


def _payloads(text):
    """The job-event payloads in an SSE body.

    The stream also carries progress snapshots (issue #49) as their own
    `event: progress` blocks; only `event: job` payloads are events, so the
    two are told apart by the block's event type rather than by the `data:`
    prefix alone.
    """
    out = []
    current = None
    for line in text.splitlines():
        if line.startswith("event: "):
            current = line[7:].strip()
        elif line.startswith("data: ") and current == "job":
            out.append(json.loads(line[6:]))
    return out


def _ids(text):
    return [
        int(line[4:]) for line in text.splitlines() if line.startswith("id: ")
    ]


@pytest.fixture()
def finished_job_id(client):
    job_id = _upload_and_create(client, driver=_run_to_completion)
    _wait_terminal(client, job_id)
    return job_id


def test_stream_404s_for_a_missing_job(client):
    assert client.get("/v1/jobs/job_nope/stream").status_code == 404


def test_a_completed_job_streams_its_whole_history_then_ends(
    client, finished_job_id
):
    body = _stream_body(client, f"/v1/jobs/{finished_job_id}/stream")
    # The events endpoint is the durable truth; the stream must deliver
    # exactly the same history, in the same order, each with an `id` the
    # browser would replay on reconnect.
    page = client.get(f"/v1/jobs/{finished_job_id}/events").json()
    expected = page["events"]
    assert _ids(body) == [e["id"] for e in expected]
    assert _payloads(body) == expected
    # The terminal transition itself is delivered before the stream ends, and
    # an explicit end marker follows it -- the interface's hand-back signal,
    # which cannot rely on the connection closing (a browser's EventSource
    # reconnects on a server-initiated close rather than reporting it).
    assert _payloads(body)[-1]["kind"] == "state"
    assert "event: end" in body


def test_after_resumes_without_redelivery(client, finished_job_id):
    page = client.get(f"/v1/jobs/{finished_job_id}/events").json()
    mid = page["events"][1]["id"]
    body = _stream_body(
        client, f"/v1/jobs/{finished_job_id}/stream", params={"after": mid}
    )
    delivered = _payloads(body)
    assert delivered, (
        "a catch-up client should receive the rest of the history"
    )
    assert all(e["id"] > mid for e in delivered)


def test_last_event_id_header_wins_over_after(client, finished_job_id):
    page = client.get(f"/v1/jobs/{finished_job_id}/events").json()
    mid = page["events"][1]["id"]
    # `after` points back at the start (everything would be re-sent); the
    # reconnected client's Last-Event-ID is the truth about what it holds.
    body = _stream_body(
        client,
        f"/v1/jobs/{finished_job_id}/stream",
        params={"after": 0},
        headers={"Last-Event-ID": str(mid)},
    )
    delivered = _payloads(body)
    assert all(e["id"] > mid for e in delivered)


def _paused_live_provider():
    """A live job's fake that holds still after the image build, so a test
    can open watchers and act on the run at a known point (issue #57: the
    two-watcher and dropped-connection tests both need a job they can hold
    still while connections come and go)."""
    return FakeProvider(
        lines=DEMO_LINES,
        result=DEMO_RESULT,
        adapter_bytes=DEMO_ADAPTER_BYTES,
        pause_at_line=2,
    )


def _drive_live_job(client, provider):
    """Start a job on a daemon thread the way the worker would (issue #51),
    and return the job id once the provider reports it is paused."""
    from temper_control_plane import fake_models, orchestrator

    threads: list[threading.Thread] = []

    def start(job_id):
        t = threading.Thread(
            target=orchestrator.run_job,
            args=(job_id, provider, None, fake_models.catalog_models()),
            daemon=True,
            name=f"job-{job_id[:8]}",
        )
        t.start()
        threads.append(t)

    job_id = _upload_and_create(client, driver=start)
    assert provider.wait_until_paused(), "the job never reached its pause"
    return job_id, threads


def test_two_watchers_on_one_job_both_receive_everything(client, tmp_path):
    """Two streams on one job both deliver the whole history, and neither
    affects the other (spec 008's story of the second device).

    Both watchers are open while the job is still working -- the same live
    run, not one after the other -- and each ends on the terminal transition
    with the full record, identical to the other's and to the store's. A
    fan-out that shared a cursor or consumed a notification out from under
    the other watcher would fail this with a gap in one of the two.
    """
    provider = _paused_live_provider()
    job_id, threads = _drive_live_job(client, provider)

    with (
        client.stream("GET", f"/v1/jobs/{job_id}/stream") as first,
        client.stream("GET", f"/v1/jobs/{job_id}/stream") as second,
    ):
        assert first.status_code == 200
        assert second.status_code == 200
        # Both watchers are subscribed while the job is held still, so every
        # event from here on is fanned out to both -- and both replay the
        # history recorded before they connected from the store (the stream
        # reads history first, then follows the channel; ADR-0067).
        provider.resume.set()
        body_first = first.read().decode("utf-8")
        body_second = second.read().decode("utf-8")

    for t in threads:
        t.join(timeout=5.0)

    page = client.get(f"/v1/jobs/{job_id}/events").json()
    expected = page["events"]
    assert expected, "the run recorded no events"
    # Each watcher got everything, in order, once -- and they agree with each
    # other (neither affected the other) and with the store (the truth).
    assert _payloads(body_first) == expected
    assert _payloads(body_second) == expected
    assert _ids(body_first) == [e["id"] for e in expected]
    assert _ids(body_second) == [e["id"] for e in expected]
    # Both ended on their own: the terminal transition, then the explicit end
    # marker -- one watcher's end did not close the other.
    assert _payloads(body_first)[-1]["kind"] == "state"
    assert _payloads(body_second)[-1]["kind"] == "state"
    assert "event: end" in body_first
    assert "event: end" in body_second


def test_a_dropped_connection_replays_exactly_what_it_missed(client, tmp_path):
    """A connection that is actually dropped mid-stream -- closed, not
    simulated -- replays exactly what it missed, in order, once, on
    reconnecting with its last seen identifier (spec 008's replay-by-last-
    seen rule; ADR-0067).

    The drop is real: the stream is closed while the job is held still, the
    job then runs on to completion out of any connection's sight, and the
    reconnect asks for everything after the last id the dropped watcher had
    delivered. The store is the truth, so the gap is filled exactly --
    nothing re-sent, nothing skipped, nothing doubled.
    """
    provider = _paused_live_provider()
    job_id, threads = _drive_live_job(client, provider)

    # Watch until the image build is done, then drop the connection for real
    # by closing the stream while the job is still paused.
    with client.stream("GET", f"/v1/jobs/{job_id}/stream") as r:
        assert r.status_code == 200
        seen: list[str] = []
        for line in r.iter_lines():
            seen.append(line)
            if "image built" in "\n".join(seen):
                break
    # The connection is closed now. Everything recorded from here on is
    # genuinely missed by the dropped watcher.
    last_seen = _ids("\n".join(seen))[-1]

    provider.resume.set()
    _wait_terminal(client, job_id)
    for t in threads:
        t.join(timeout=5.0)

    page = client.get(f"/v1/jobs/{job_id}/events").json()
    expected = page["events"]
    missed = [e for e in expected if e["id"] > last_seen]
    assert missed, (
        "the job recorded nothing after the drop; the test needs a gap"
    )

    # Reconnect with the last id the dropped watcher had seen: exactly the
    # gap, in order, once.
    body = _stream_body(
        client,
        f"/v1/jobs/{job_id}/stream",
        headers={"Last-Event-ID": str(last_seen)},
    )
    delivered = _payloads(body)
    assert delivered == missed, (
        "the reconnecting watcher did not receive exactly the events it "
        "missed, in order, once"
    )
    assert _ids(body) == [e["id"] for e in missed]
    # And the whole history is still continuous from the drop point: the
    # first missed event is the one after the last seen, with nothing lost
    # in between.
    assert missed[0]["id"] == last_seen + 1


def test_the_stream_carries_live_events_and_ends_at_terminal(
    client, monkeypatch, tmp_path
):
    """The one test that watches a running job: events arrive while it is
    working, and the stream ends of its own accord at the terminal state."""
    from temper_control_plane import fake_models, orchestrator
    from temper_control_plane.fake_provider import DEMO_ADAPTER_BYTES

    provider = FakeProvider(
        lines=DEMO_LINES,
        result=DEMO_RESULT,
        adapter_bytes=DEMO_ADAPTER_BYTES,
        pause_at_line=2,
    )

    threads: list[threading.Thread] = []

    def start(job_id):
        t = threading.Thread(
            target=orchestrator.run_job,
            args=(job_id, provider, None, fake_models.catalog_models()),
            daemon=True,
            name=f"job-{job_id[:8]}",
        )
        t.start()
        threads.append(t)

    # The request path only inserts a `queued` row (issue #51); start the
    # job on a thread directly, the way the worker would.
    job_id = _upload_and_create(client, driver=start)
    assert provider.wait_until_paused(), "the job never reached its pause"

    # The stream is read on the main thread, line by line, and the pause is
    # released from *inside* the read loop the moment the last pre-pause event
    # has arrived: the events recorded while the job was still working reach
    # the open stream without the reader doing anything, and the stream then
    # runs on to the terminal transition and closes of its own accord.
    with client.stream("GET", f"/v1/jobs/{job_id}/stream") as r:
        assert r.status_code == 200
        seen: list[str] = []
        for line in r.iter_lines():
            seen.append(line)
            text = "\n".join(seen)
            if "image built" in text and not provider.resume.is_set():
                provider.resume.set()
            if "Training complete" in text:
                break

    payloads = _payloads("\n".join(seen))
    assert "image built" in "\n".join(seen), (
        "the running job's events did not arrive on the open stream"
    )
    assert payloads[-1]["kind"] == "state"
    assert payloads[-1]["message"] == "Training complete"
    # The stream delivered the whole history, ending at the terminal event.
    page = client.get(f"/v1/jobs/{job_id}/events").json()
    assert _ids("\n".join(seen)) == [e["id"] for e in page["events"]]
    for t in threads:
        t.join(timeout=5.0)
