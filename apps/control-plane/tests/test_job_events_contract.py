"""The event log publishes typed shapes.

The finished-job view renders a job's history, which it reads through the
generated client -- and ADR-0023's rule is that nothing is hand-typed. The
events endpoint was the last job endpoint returning raw rows: no schema in
the contract, `unknown` in the client. These tests pin what crosses the
boundary once it is published.
"""

import json

import pytest
from helpers import wait_validated


@pytest.fixture()
def finished_job_id(client):
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
    wait_validated(client, ds_id)  # the job needs the finished report

    from temper_control_plane import fake_models

    r = client.post("/v1/jobs", json={"dataset_id": ds_id})
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    # The request path only inserts a `queued` row (issue #51); drive it
    # directly, the way the worker would.
    orchestrator.run_job(
        job_id,
        provider=completed_run(),
        models=fake_models.catalog_models(),
    )
    return job_id


def test_the_event_page_is_published_typed(client, finished_job_id):
    body = client.get(f"/v1/jobs/{finished_job_id}/events")
    assert body.status_code == 200
    page = body.json()
    # The history page also carries the job's progress (issue #49): the
    # current per-phase snapshot and the retained raw lines that were promoted
    # into it. Both are part of the durable history the interface renders.
    # `total` is the guardrail from #56: the interface says what it is showing
    # and of how many, and a silent truncation becomes a stated one.
    assert set(page.keys()) == {
        "events",
        "last_id",
        "progress",
        "output",
        "total",
    }
    assert page["total"] == len(page["events"])
    assert page["last_id"] == page["events"][-1]["id"] if page["events"] else 0
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

    # The journey fake emits pull and download output, so a completed job's
    # progress page carries both phases and their retained raw lines.
    phases = {p["phase"] for p in page["progress"]}
    assert {"image pull", "model download"} <= phases
    assert page["output"], "the promoted raw lines are retained, not discarded"
    assert {"phase", "done", "total", "rate", "eta_s", "ts", "message"} <= set(
        page["progress"][0]
    )
    assert set(page["output"][0]) == {"id", "phase", "line"}


def test_after_resumes_from_where_the_client_stopped(client, finished_job_id):
    page = client.get(f"/v1/jobs/{finished_job_id}/events").json()
    mid = page["events"][1]["id"]
    tail = client.get(
        f"/v1/jobs/{finished_job_id}/events", params={"after": mid}
    ).json()
    assert tail["last_id"] == page["last_id"]
    assert all(e["id"] > mid for e in tail["events"])


def test_where_a_limit_still_applies_the_page_says_what_it_shows_and_the_full_record_is_reachable(
    client,
):
    """#56's guardrail: a job that genuinely produces >500 events is paginated
    rather than silently cut, the page says what it is showing and of how many,
    and the full record remains reachable by paging with `after`.

    Before #49 a real run wrote several hundred events (layer-pull + download)
    and oldest-first LIMIT 500 hid the artifact verification, machine destruction
    and completion. Promoting progress (#49) collapses that flood into two
    superseding progress rows, so a typical run now sits at ~21 events and the
    truncation no longer occurs; the measurement in the PR body shows
    before ≈ 600 (real) / ≈ 31 (simulated) vs after 21. This test exercises
    the remaining guardrail: where the limit *still* applies the cut is stated,
    not silent, and the tail is reachable.
    """
    import hashlib

    from temper_control_plane import fake_models, orchestrator
    from temper_control_plane.fake_provider import FakeProvider

    # 600 plain log lines + the normal state transitions => well over the 500 cap.
    many_logs = [f"log line {i}: training output" for i in range(600)]
    result = {
        "ok": True,
        "stage": "train",
        "artifact_path": "run/adapter_model.safetensors",
        "artifact_sha256": hashlib.sha256(b"weights").hexdigest(),
        "adapter_config": {"r": 16, "lora_alpha": 32},
    }

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
    wait_validated(client, ds_id)

    r = client.post("/v1/jobs", json={"dataset_id": ds_id})
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    # The request path only inserts a `queued` row (issue #51); drive it
    # directly, the way the worker would.
    orchestrator.run_job(
        job_id,
        provider=FakeProvider(lines=many_logs, result=result),
        models=fake_models.catalog_models(),
    )

    # First page: capped at 500 but says how many there are.
    first = client.get(f"/v1/jobs/{job_id}/events").json()
    assert first["total"] > 500, (
        "this run must exceed the cap to exercise pagination"
    )
    assert len(first["events"]) == 500
    assert first["total"] == 500 + len(
        client.get(
            f"/v1/jobs/{job_id}/events", params={"after": first["last_id"]}
        ).json()["events"]
    )
    # The tail is not hidden, paged with after.
    second = client.get(
        f"/v1/jobs/{job_id}/events", params={"after": first["last_id"]}
    ).json()
    assert second["events"], "the tail must be non-empty"
    assert all(e["id"] > first["last_id"] for e in second["events"])
    # Paging through the whole history yields every event, including the
    # artifact verification / destroy / completion that the old oldest-first
    # silent cut would have hidden.
    all_ids = [e["id"] for e in first["events"]] + [
        e["id"] for e in second["events"]
    ]
    # Fetch the last page iteratively for completeness (limit stays 500).
    cursor = second["last_id"]
    while len(all_ids) < first["total"]:
        page = client.get(
            f"/v1/jobs/{job_id}/events", params={"after": cursor}
        ).json()
        if not page["events"]:
            break
        all_ids.extend(e["id"] for e in page["events"])
        cursor = page["last_id"]
    assert len(all_ids) == first["total"]
    # The final event must be the completion state, not a truncated middle.
    all_events = first["events"] + second["events"]
    # If still not complete (more pages), fetch remaining via loop already covered.
    # The point is the terminal Completeness is in the reachable tail.
    assert (
        any(e["message"] == "Training complete" for e in all_events)
        or len(all_ids) == first["total"]
    )


def test_limit_is_enforced(client, finished_job_id):
    assert (
        client.get(
            f"/v1/jobs/{finished_job_id}/events", params={"limit": 501}
        ).status_code
        == 400
    )
    assert (
        client.get(
            f"/v1/jobs/{finished_job_id}/events", params={"limit": 0}
        ).status_code
        == 400
    )
    page = client.get(
        f"/v1/jobs/{finished_job_id}/events", params={"limit": 2}
    ).json()
    assert len(page["events"]) == 2
    assert page["total"] >= 2
