"""The event log publishes typed shapes.

The finished-job view renders a job's history, which it reads through the
generated client -- and ADR-0023's rule is that nothing is hand-typed. The
events endpoint was the last job endpoint returning raw rows: no schema in
the contract, `unknown` in the client. These tests pin what crosses the
boundary once it is published.
"""

import json

import pytest


@pytest.fixture()
def finished_job_id(client, monkeypatch):
    """One job driven to `complete` on the journey fake, created via API."""
    from temper_control_plane import orchestrator
    from temper_control_plane.fake_provider import completed_run

    payload = "\n".join(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            }
        )
        for i in range(12)
    )
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
    ds_id = r.json()["id"]

    def start(job_id):
        from temper_control_plane import fake_models

        orchestrator.run_job(
            job_id,
            provider=completed_run(),
            models=fake_models.catalog_models(),
        )

    monkeypatch.setattr(orchestrator, "launch", start)
    r = client.post("/v1/jobs", json={"dataset_id": ds_id})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_the_event_page_is_published_typed(client, finished_job_id):
    body = client.get(f"/v1/jobs/{finished_job_id}/events")
    assert body.status_code == 200
    page = body.json()
    assert set(page.keys()) == {"events", "last_id"}
    assert page["events"]
    for event in page["events"]:
        assert set(event.keys()) == {
            "id",
            "job_id",
            "ts",
            "kind",
            "message",
            "data",
        }
        assert event["job_id"] == finished_job_id
    kinds = {e["kind"] for e in page["events"]}
    # A completed job's history carries its transitions, its output and the
    # numbers promoted out of that output.
    assert {"state", "log", "metric"} <= kinds
    metric = next(e for e in page["events"] if e["kind"] == "metric")
    assert isinstance(metric["data"]["loss"], (int, float))
    assert page["last_id"] == page["events"][-1]["id"]


def test_after_resumes_from_where_the_client_stopped(client, finished_job_id):
    page = client.get(f"/v1/jobs/{finished_job_id}/events").json()
    mid = page["events"][1]["id"]
    tail = client.get(
        f"/v1/jobs/{finished_job_id}/events", params={"after": mid}
    ).json()
    assert tail["last_id"] == page["last_id"]
    assert all(e["id"] > mid for e in tail["events"])
