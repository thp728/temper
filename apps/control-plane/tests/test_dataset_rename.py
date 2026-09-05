"""Renaming a dataset -- the one update a dataset can have.

The stored object never moves; a rename only changes the `filename` column
and stamps `updated_at`, so it is allowed regardless of ingest status (no
race with a background thread the way delete has).
"""

from __future__ import annotations

import pytest

from temper_control_plane import db


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
    r = client.post("/v1/datasets", files={"file": ("d.jsonl", content)})
    assert r.status_code == 202
    return r.json()["id"]


def test_a_dataset_can_be_renamed(client):
    ds_id = upload_dataset(client)
    before = db.get_dataset(ds_id)

    r = client.patch(
        f"/v1/datasets/{ds_id}", json={"filename": "renamed.jsonl"}
    )

    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "renamed.jsonl"
    assert body["updated_at"] > before["updated_at"]
    assert body["created_at"] == before["created_at"]
    # The rename is durable, not just echoed back.
    assert db.get_dataset(ds_id)["filename"] == "renamed.jsonl"


def test_renaming_does_not_move_the_stored_object(client):
    from temper_control_plane import storage

    ds_id = upload_dataset(client)
    key = storage.dataset_key(ds_id)
    before = storage.STORE.get(key)

    client.patch(f"/v1/datasets/{ds_id}", json={"filename": "renamed.jsonl"})

    assert storage.STORE.get(key) == before


def test_renaming_is_allowed_while_still_ingesting(client, monkeypatch):
    """Contrast delete: a rename touches only the `filename` column, never
    the object a background thread might still hold open, so it does not
    need to wait for a terminal status."""
    import threading

    release = threading.Event()
    monkeypatch.setattr(
        "temper_control_plane.datasets._validate_in_background",
        lambda ds_id, key, total_bytes: threading.Thread(
            target=lambda: release.wait(timeout=10), daemon=True
        ).start(),
    )

    ds_id = upload_dataset(client)
    try:
        r = client.patch(
            f"/v1/datasets/{ds_id}", json={"filename": "renamed.jsonl"}
        )
        assert r.status_code == 200
        assert r.json()["status"] == "validating"
        assert r.json()["filename"] == "renamed.jsonl"
    finally:
        release.set()


def test_renaming_to_a_non_jsonl_name_is_refused(client):
    ds_id = upload_dataset(client)

    r = client.patch(f"/v1/datasets/{ds_id}", json={"filename": "renamed.csv"})

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "unsupported_extension"
    assert db.get_dataset(ds_id)["filename"] != "renamed.csv"


def test_renaming_a_dataset_that_does_not_exist_is_refused_with_404(client):
    r = client.patch("/v1/datasets/ds_nope", json={"filename": "x.jsonl"})

    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "not_found"
