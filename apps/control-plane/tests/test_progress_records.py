"""Progress is promoted, not filtered (issue #49).

The orchestrator classifies each line; a layer-pull or model-download line
updates the phase's progress record -- one superseding row per phase, with the
rate measured live -- and is retained as the job's collapsed detail, instead of
becoming a log event. These tests pin that wiring: nothing a pull produces ever
reaches the event log, the phase record replaces itself rather than
accumulating, the raw lines are all kept, and the live channel carries the
snapshot.
"""

import hashlib
import json
import threading

import pytest
from helpers import wait_validated

from temper_control_plane.fake_provider import FakeProvider
from temper_core.progress import PHASE_IMAGE_PULL, PHASE_MODEL_DOWNLOAD

ADAPTER_BYTES = b"weights"
RESULT = {
    "ok": True,
    "stage": "train",
    "adapter_path": "run/adapter_model.safetensors",
    "adapter_sha256": hashlib.sha256(ADAPTER_BYTES).hexdigest(),
    "adapter_config": {"r": 16, "lora_alpha": 32},
}

# Docker's classic pull output (documented format; the repo holds no captured
# pull transcript, see test_events.py): layers pull in parallel, so the lines
# interleave across two layers.
PULL_LINES = [
    "9b829b73a52f: Pulling fs layer",
    "1fe172e4850f: Pulling fs layer",
    "9b829b73a52f: Downloading [===============> ] 15.19MB/42.42MB",
    "1fe172e4850f: Downloading [======>            ]  8.5MB/25.54MB",
    "9b829b73a52f: Downloading [==================> ] 28.1MB/42.42MB",
    "1fe172e4850f: Downloading [================>   ] 17.2MB/25.54MB",
    "9b829b73a52f: Extracting [========================> ] 35.2MB/42.42MB",
    "9b829b73a52f: Pull complete",
    "1fe172e4850f: Pull complete",
]

# The captured huggingface_hub download bar (see test_events.py).
DOWNLOAD_LINES = [
    "model.safetensors:  10%|█         | 400M/4.00G [00:05<00:45]",
    "model.safetensors:  45%|████▌     | 1.8G/4.00G [00:22<00:27]",
]

# What the image-pull phase must aggregate to: layer 9b8 peaked at 35.2MB of
# 42.42MB, layer 1fe at 17.2MB of 25.54MB.
PULL_DONE = (35.2 + 17.2) * 1e6
PULL_TOTAL = (42.42 + 25.54) * 1e6


def _chat_rows(n=12):
    return [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(n)
    ]


def _upload_and_create(client):
    payload = "\n".join(json.dumps(r) for r in _chat_rows()) + "\n"
    r = client.post(
        "/v1/datasets",
        files={
            "file": (
                "d.jsonl",
                payload.encode("utf-8"),
                "application/octet-stream",
            )
        },
    )
    assert r.status_code == 202, r.text
    ds_id = r.json()["id"]
    wait_validated(client, ds_id)
    r = client.post("/v1/jobs", json={"dataset_id": ds_id})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _run_sync(client, monkeypatch, provider):
    """Launch every job synchronously on the given provider."""
    from temper_control_plane import fake_models, orchestrator

    monkeypatch.setattr(
        orchestrator,
        "launch",
        lambda job_id: orchestrator.run_job(
            job_id,
            provider=provider,
            models=fake_models.catalog_models(),
        ),
    )
    return _upload_and_create(client)


def _progress(client, job_id):
    from temper_control_plane import db

    return {p["phase"]: p for p in db.get_progress(job_id)}


def _output_lines(client, job_id):
    from temper_control_plane import db

    return [o["line"] for o in db.get_output(job_id)]


def _event_messages(client, job_id):
    return [
        e["message"]
        for e in client.get(f"/v1/jobs/{job_id}/events").json()["events"]
    ]


def test_pull_and_download_lines_become_progress_not_log_events(
    client, monkeypatch
):
    lines = (
        PULL_LINES
        + ["[00:00:02] running training"]
        + DOWNLOAD_LINES
        + ["{'loss': 0.6931, 'step': 10, 'epoch': 0.5}"]
    )
    job_id = _run_sync(
        client, monkeypatch, FakeProvider(lines=lines, result=RESULT)
    )

    # None of the promoted lines is an event: the flood that truncated the
    # finished page no longer exists in the event log.
    messages = _event_messages(client, job_id)
    assert not any(
        ": Downloading" in m or ": Pull complete" in m for m in messages
    )
    assert not any("model.safetensors" in m for m in messages)

    # Both phases have exactly one superseding record, aggregated correctly.
    progress = _progress(client, job_id)
    assert set(progress) == {PHASE_IMAGE_PULL, PHASE_MODEL_DOWNLOAD}
    pull = progress[PHASE_IMAGE_PULL]
    assert pull["done"] == pytest.approx(PULL_DONE)
    assert pull["total"] == pytest.approx(PULL_TOTAL)
    download = progress[PHASE_MODEL_DOWNLOAD]
    assert download["done"] == pytest.approx(1.8e9)
    assert download["total"] == pytest.approx(4e9)

    # Every promoted raw line is retained as the job's collapsed detail.
    kept = _output_lines(client, job_id)
    assert kept == PULL_LINES + DOWNLOAD_LINES

    # Ordinary classification still works beside it.
    metrics = [
        e["data"]
        for e in client.get(f"/v1/jobs/{job_id}/events").json()["events"]
        if e["kind"] == "metric"
    ]
    assert metrics == [{"loss": 0.6931, "step": 10, "epoch": 0.5}]


def test_progress_supersedes_rather_than_accumulates(client, monkeypatch):
    job_id = _run_sync(
        client,
        monkeypatch,
        FakeProvider(
            lines=PULL_LINES
            + DOWNLOAD_LINES
            + ["[00:00:02] running training"],
            result=RESULT,
        ),
    )
    progress = _progress(client, job_id)
    assert len(progress) == 2, (
        "one superseding row per phase, never one per line"
    )


def test_the_measured_rate_reaches_the_record(client, monkeypatch):
    """Spaced lines let the tracker measure a live rate; the estimate uses it."""
    job_id = _run_sync(
        client,
        monkeypatch,
        FakeProvider(
            lines=DOWNLOAD_LINES + ["[00:00:02] running training"],
            result=RESULT,
            line_delay=0.05,
        ),
    )
    download = _progress(client, job_id)[PHASE_MODEL_DOWNLOAD]
    assert download["rate"] is not None and download["rate"] > 0
    assert download["eta_s"] is not None and download["eta_s"] > 0


def test_the_events_page_publishes_progress_and_output(client, monkeypatch):
    job_id = _run_sync(
        client,
        monkeypatch,
        FakeProvider(
            lines=PULL_LINES
            + DOWNLOAD_LINES
            + ["[00:00:02] running training"],
            result=RESULT,
        ),
    )
    page = client.get(f"/v1/jobs/{job_id}/events").json()
    by_phase = {p["phase"]: p for p in page["progress"]}
    assert set(by_phase) == {PHASE_IMAGE_PULL, PHASE_MODEL_DOWNLOAD}
    pull = by_phase[PHASE_IMAGE_PULL]
    assert pull["done"] == pytest.approx(PULL_DONE)
    assert pull["total"] == pytest.approx(PULL_TOTAL)
    assert "phase" in pull and "ts" in pull and "message" in pull
    assert [o["line"] for o in page["output"]] == PULL_LINES + DOWNLOAD_LINES
    assert [o["phase"] for o in page["output"]] == [PHASE_IMAGE_PULL] * len(
        PULL_LINES
    ) + [PHASE_MODEL_DOWNLOAD] * len(DOWNLOAD_LINES)


def test_the_stream_pushes_the_progress_snapshot(client, monkeypatch):
    """The live view gets progress over the same connection as the events."""
    from temper_control_plane import fake_models, orchestrator

    provider = FakeProvider(
        lines=(
            PULL_LINES
            + ["[00:00:02] running training"]
            + DOWNLOAD_LINES
            + ["{'loss': 0.6931, 'epoch': 0.5}"]
        ),
        result=RESULT,
        pause_at_line=len(PULL_LINES) + 1,
    )

    def start(job_id):
        threading.Thread(
            target=orchestrator.run_job,
            args=(job_id, provider, None, fake_models.catalog_models()),
            daemon=True,
            name=f"job-{job_id[:8]}",
        ).start()

    monkeypatch.setattr(orchestrator, "launch", start)
    job_id = _upload_and_create(client)
    assert provider.wait_until_paused(), "the job never reached its pause"

    with client.stream("GET", f"/v1/jobs/{job_id}/stream") as r:
        assert r.status_code == 200
        seen: list[str] = []
        for line in r.iter_lines():
            seen.append(line)
            text = "\n".join(seen)
            if "event: progress" in text and not provider.resume.is_set():
                provider.resume.set()
            if "Training complete" in text:
                break

    text = "\n".join(seen)
    # The snapshot arrived on the open stream, naming the phase that was
    # working while the job was paused.
    assert "event: progress" in text
    assert PHASE_IMAGE_PULL in text
