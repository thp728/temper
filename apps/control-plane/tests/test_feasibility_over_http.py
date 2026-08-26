"""Feasibility warnings as a user meets them: on job creation, over HTTP.

These moved out of `packages/core/tests` with the reorg. They drive the API with
a TestClient and assert on what a job carries afterwards, which makes them
control-plane tests that happened to be filed under the rule they exercise.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from temper_control_plane import config

# --- job creation, through the HTTP seam --------------------------------------


@pytest.fixture()
def client(isolated, monkeypatch):

    from temper_control_plane import main, orchestrator

    monkeypatch.setattr(orchestrator, "launch", lambda job_id: None)
    with TestClient(main.app) as c:
        yield c


def chat_rows(n):
    for i in range(n):
        yield {
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ]
        }


def upload(client, rows):
    data = ("\n".join(json.dumps(r) for r in rows)).encode("utf-8")
    r = client.post("/v1/datasets", files={"file": ("d.jsonl", data)})
    assert r.status_code == 201
    return r.json()["id"]


def test_large_dataset_yields_a_warning_and_a_launched_job(
    client, monkeypatch
):
    """The acceptance criterion, verbatim: a large dataset yields a warning
    and a launched job, not a refusal. The ceiling is pulled down to seconds
    so a generated fixture -- never a committed one -- trips it."""
    monkeypatch.setattr(config, "MAX_JOB_DURATION_S", 1.0)
    ds_id = upload(client, chat_rows(50))

    r = client.post("/v1/jobs", json={"dataset_id": ds_id})

    assert r.status_code == 201, "a warning must not veto the user's judgement"
    body = r.json()
    assert body["status"] == "queued"
    warnings = body["warnings"]
    assert len(warnings) == 1
    assert warnings[0]["code"] == "duration_feasibility"
    assert "estimate" in warnings[0]["message"].lower()


def test_warning_is_on_the_job_afterwards_and_in_its_history(
    client, monkeypatch
):
    """Shown wherever the job is shown, and recorded as an event, so the
    warning the user accepted is part of the run's own account of itself."""
    monkeypatch.setattr(config, "MAX_JOB_DURATION_S", 1.0)
    ds_id = upload(client, chat_rows(50))
    job_id = client.post("/v1/jobs", json={"dataset_id": ds_id}).json()["id"]

    fetched = client.get(f"/v1/jobs/{job_id}").json()
    assert fetched["warnings"][0]["code"] == "duration_feasibility"

    events = client.get(f"/v1/jobs/{job_id}/events").json()["events"]
    assert any("estimate" in (e["message"] or "").lower() for e in events)


def test_normal_dataset_gets_no_warning(client):
    ds_id = upload(client, chat_rows(50))

    r = client.post("/v1/jobs", json={"dataset_id": ds_id})

    assert r.status_code == 201
    assert r.json()["warnings"] == []
