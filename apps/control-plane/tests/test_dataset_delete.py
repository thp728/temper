"""Deleting a dataset (CRUD gap: create and read existed, delete did not).

A dataset can be removed only when nothing points at it: `jobs.dataset_id` is
`NOT NULL REFERENCES datasets(id)` with no cascade, so a completed job's
record must keep being able to say what it trained on. Delete is refused,
not archived -- the guard is checked before anything is removed rather than
letting the database's own constraint raise mid-delete.
"""

from __future__ import annotations

import pytest
from helpers import wait_validated

from temper_control_plane import db, storage
from temper_control_plane.storage import ObjectNotFound


@pytest.fixture()
def server(isolated):
    from temper_control_plane import main

    return main


@pytest.fixture()
def client(server):
    from fastapi.testclient import TestClient

    with TestClient(server.app) as c:
        yield c


def upload_dataset(client, content: bytes = b'{"messages": []}\n') -> str:
    """A stored, *settled* dataset -- waited past "validating" so a delete
    right after this helper exercises the guards this file is about, not the
    unrelated "still being ingested" refusal a delete immediately after
    upload would otherwise race against (a background thread still holds the
    object open at that instant)."""
    r = client.post("/v1/datasets", files={"file": ("d.jsonl", content)})
    assert r.status_code == 202
    ds_id = r.json()["id"]
    wait_validated(client, ds_id)
    return ds_id


def test_a_dataset_with_no_jobs_can_be_deleted(client):
    ds_id = upload_dataset(client)

    r = client.delete(f"/v1/datasets/{ds_id}")

    assert r.status_code == 204
    assert client.get(f"/v1/datasets/{ds_id}").status_code == 404
    assert db.get_dataset(ds_id) is None


def test_deleting_removes_the_stored_object_too(client):
    ds_id = upload_dataset(client)
    key = storage.dataset_key(ds_id)
    assert storage.STORE.get(key)  # present before delete

    client.delete(f"/v1/datasets/{ds_id}")

    with pytest.raises(ObjectNotFound):
        storage.STORE.get(key)


def test_a_dataset_used_by_a_job_cannot_be_deleted(client):
    ds_id = upload_dataset(client)
    db.create_job(ds_id, "qwen3-4b", {})

    r = client.delete(f"/v1/datasets/{ds_id}")

    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "dataset_in_use"
    assert detail["job_count"] == 1
    # Refused, not archived: the row and its object both survive.
    assert db.get_dataset(ds_id) is not None
    assert storage.STORE.get(storage.dataset_key(ds_id))


def test_deleting_a_dataset_used_by_several_jobs_names_the_count(client):
    ds_id = upload_dataset(client)
    db.create_job(ds_id, "qwen3-4b", {})
    db.create_job(ds_id, "qwen3-4b", {})

    r = client.delete(f"/v1/datasets/{ds_id}")

    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["job_count"] == 2
    assert "2 jobs" in detail["message"]


def test_deleting_a_dataset_that_does_not_exist_is_refused_with_404(client):
    r = client.delete("/v1/datasets/ds_nope")

    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "not_found"


def test_a_dataset_still_being_ingested_cannot_be_deleted(client, monkeypatch):
    """A background thread still holds the object open while a dataset is
    `importing` or `validating` (issue: "dataset import from HF" widened
    ingest to include a fetch phase). Deleting under it is a race with no
    defined outcome, so it is refused until the row reaches a terminal
    status. Validation is held open on its own thread so the window this
    test needs is not a race against how fast a tiny file validates."""
    import threading

    release = threading.Event()

    def blocking_validate(ds_id, key, total_bytes):
        threading.Thread(
            target=lambda: release.wait(timeout=10), daemon=True
        ).start()

    monkeypatch.setattr(
        "temper_control_plane.datasets._validate_in_background",
        blocking_validate,
    )

    r = client.post(
        "/v1/datasets", files={"file": ("d.jsonl", b'{"messages": []}\n')}
    )
    ds_id = r.json()["id"]

    try:
        r = client.delete(f"/v1/datasets/{ds_id}")
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "dataset_not_ready"
    finally:
        release.set()
