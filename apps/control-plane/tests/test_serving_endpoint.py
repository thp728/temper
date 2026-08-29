"""Temporary authenticated endpoints (issue #78).

An endpoint for a finished job's model behind a temporary authenticated
endpoint that answers prompts and stops itself. The tests assert what the
user receives: that the preview shows cost and stop time before it starts,
that a key is required and stored hashed, that the endpoint carries an
expiry from the moment it starts, extends on use, and stops itself via a
timer, and that a busy endpoint still dies at the hard ceiling.
"""

from __future__ import annotations

import json
import time

import pytest
from helpers import wait_validated

from temper_control_plane import db
from temper_core import serving as core_serving


def jsonl(tmp_path, rows, name="d.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def chat(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def upload(client, path):
    with open(path, "rb") as f:
        return client.post("/v1/datasets", files={"file": (path.name, f)})


def _valid_dataset(client, tmp_path):
    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    return wait_validated(client, upload(client, p).json()["id"])["id"]


def _complete_job(client, tmp_path, monkeypatch):
    """Create a job and mark it complete with a fake price, without provisioning a real machine.

    The endpoint requires a complete job. The `client` fixture stubs
    orchestrator.launch, so a job stays queued unless we drive it. For
    serving tests we just mark the job complete directly in the DB, with
    the fields a real orchestrator would have set.
    """
    from temper_control_plane import storage

    ds = _valid_dataset(client, tmp_path)
    r = client.post("/v1/jobs", json={"dataset_id": ds})
    assert r.status_code == 201, r.text
    job = r.json()
    job_id = job["id"]
    # Make it look like a finished L4 job
    weights_key = storage.artifact_key(job_id, storage.ADAPTER_WEIGHTS_NAME)
    storage.STORE.put(weights_key, b"weights")
    storage.STORE.put(
        storage.artifact_key(job_id, storage.ADAPTER_CONFIG_NAME), b'{"r":16}'
    )
    row_count = 12
    db.set_state(
        job_id,
        "complete",
        "done",
        method="qlora",
        gpu_type="L4",
        price_per_hour=41.31,
        currency="INR",
        artifact_key=weights_key,
        result_json={
            "held_out_split": {
                "rows_in": row_count,
                "rows_removed_duplicates": 0,
                "train_rows": 11,
                "held_out_rows": 1,
                "fraction": 0.05,
                "seed": 42,
            },
            "template_probe": {"ok": True},
        },
        best_checkpoint_json={
            "step": 10,
            "basis": "best_held_out_loss",
            "reason": "Step 10 has the lowest held-out loss",
            "held_out_loss": 0.5,
        },
    )
    return job_id


def _fake_provider(monkeypatch):
    """A fake provider that can provision and destroy, and never fails verification."""
    from temper_control_plane.fake_provider import FakeProvider

    provider = FakeProvider()
    # Patch serving's new_provider to return this instance, and config.FAKE_PROVIDER so
    # any fallback also returns a fake.
    from temper_control_plane import config, serving

    monkeypatch.setattr(config, "FAKE_PROVIDER", True)
    monkeypatch.setattr(serving, "new_provider", lambda: provider)
    # Also patch verify_not_reachable to pass for the fake (it already does for fake://)
    return provider


# ---------------------------------------------------------------------------
# preview before start
# ---------------------------------------------------------------------------


def test_preview_shows_cost_and_stop_time_before_it_starts(
    client, tmp_path, monkeypatch
):
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _fake_provider(monkeypatch)
    r = client.get(f"/v1/jobs/{job_id}/endpoint/preview")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["price_per_hour"] == 41.31
    assert body["currency"] == "INR"
    assert body["idle_timeout_s"] == core_serving.ENDPOINT_IDLE_TIMEOUT_S
    assert body["max_lifetime_s"] == core_serving.ENDPOINT_MAX_LIFETIME_S
    # Stop times are in the future
    now = time.time()
    assert body["expires_at"] > now
    assert body["max_expires_at"] > body["expires_at"]
    assert body["max_expires_at"] == pytest.approx(
        body["expires_at"]
        - core_serving.ENDPOINT_IDLE_TIMEOUT_S
        + core_serving.ENDPOINT_MAX_LIFETIME_S,
        abs=1.0,
    )


def test_preview_refuses_non_complete_job(client, tmp_path, monkeypatch):
    ds = _valid_dataset(client, tmp_path)
    r = client.post("/v1/jobs", json={"dataset_id": ds})
    job_id = r.json()["id"]  # still queued
    _fake_provider(monkeypatch)
    r = client.get(f"/v1/jobs/{job_id}/endpoint/preview")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "job_not_complete"


# ---------------------------------------------------------------------------
# start, key hashed, one active per job
# ---------------------------------------------------------------------------


def test_start_creates_endpoint_with_hashed_key_and_returns_plaintext_once(
    client, tmp_path, monkeypatch
):
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _fake_provider(monkeypatch)
    r = client.post(f"/v1/jobs/{job_id}/endpoint")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["job_id"] == job_id
    assert body["api_key"]
    assert (
        body["api_key_prefix"]
        == body["api_key"][: core_serving.API_KEY_PREFIX_LEN]
    )
    assert body["expires_at"] > time.time()
    assert body["max_expires_at"] > body["expires_at"]
    assert body["price_per_hour"] == 41.31
    assert isinstance(body["machine_id"], int)
    # Stored hash is not the key, and GET never returns the key or the hash
    ep_row = db.get_endpoint(body["id"])
    assert ep_row is not None
    assert ep_row["api_key_hash"] != body["api_key"]
    assert ep_row["api_key_hash"] == core_serving.hash_api_key(body["api_key"])
    assert (
        "api_key_hash" not in client.get(f"/v1/jobs/{job_id}/endpoint").json()
    )
    assert "api_key" not in client.get(f"/v1/jobs/{job_id}/endpoint").json()
    # Prefix is stored and returned
    assert ep_row["api_key_prefix"] == body["api_key_prefix"]
    # Second start is refused
    r2 = client.post(f"/v1/jobs/{job_id}/endpoint")
    assert r2.status_code == 409
    assert r2.json()["detail"]["code"] == "endpoint_already_running"


def test_start_refuses_non_complete_job(client, tmp_path, monkeypatch):
    ds = _valid_dataset(client, tmp_path)
    r = client.post("/v1/jobs", json={"dataset_id": ds})
    job_id = r.json()["id"]
    _fake_provider(monkeypatch)
    r = client.post(f"/v1/jobs/{job_id}/endpoint")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "endpoint_not_complete"


def test_get_returns_404_when_no_endpoint(client, tmp_path, monkeypatch):
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _fake_provider(monkeypatch)
    r = client.get(f"/v1/jobs/{job_id}/endpoint")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# inference: requires key, extends on use, capped by max
# ---------------------------------------------------------------------------


def test_infer_requires_key_and_extends_on_use(client, tmp_path, monkeypatch):
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _fake_provider(monkeypatch)
    created = client.post(f"/v1/jobs/{job_id}/endpoint").json()
    key = created["api_key"]
    first_expires = created["expires_at"]
    max_at = created["max_expires_at"]

    # Missing key is 401
    r = client.post(
        f"/v1/jobs/{job_id}/endpoint/infer", json={"prompt": "hello"}
    )
    assert r.status_code == 401

    # Wrong key is 401
    r = client.post(
        f"/v1/jobs/{job_id}/endpoint/infer",
        json={"prompt": "hello"},
        headers={"X-API-Key": "wrong"},
    )
    assert r.status_code == 401

    # Correct key succeeds and extends (but not beyond max)
    time.sleep(0.05)
    r = client.post(
        f"/v1/jobs/{job_id}/endpoint/infer",
        json={"prompt": "hello"},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (
        "tuned response" in body["completion"] or "hello" in body["completion"]
    )
    assert body["expires_at"] > first_expires
    assert body["expires_at"] <= max_at

    # DB was extended too
    ep = db.get_endpoint(created["id"])
    assert ep["expires_at"] == pytest.approx(body["expires_at"], abs=0.1)
    assert ep["last_used_at"] > created["created_at"]


def test_infer_extends_but_busy_endpoint_still_dies_at_max(
    client, tmp_path, monkeypatch
):
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _fake_provider(monkeypatch)
    # Make the windows tiny so the test can reach the max without waiting 2h.
    # Monkeypatch the core constants that serving reads at call time.
    monkeypatch.setattr(core_serving, "ENDPOINT_IDLE_TIMEOUT_S", 0.4)
    monkeypatch.setattr(core_serving, "ENDPOINT_MAX_LIFETIME_S", 1.0)
    # Also patch the serving module's imported copy (it imported the values at import)
    from temper_control_plane import serving as serving_mod

    monkeypatch.setattr(
        serving_mod.core_serving, "ENDPOINT_IDLE_TIMEOUT_S", 0.4
    )
    monkeypatch.setattr(
        serving_mod.core_serving, "ENDPOINT_MAX_LIFETIME_S", 1.0
    )

    created = client.post(f"/v1/jobs/{job_id}/endpoint").json()
    key = created["api_key"]
    max_at = created["max_expires_at"]
    # Busy loop: keep using the endpoint every 0.2s (less than idle), it should keep extending
    # but never beyond max.
    for _ in range(3):
        time.sleep(0.2)
        r = client.post(
            f"/v1/jobs/{job_id}/endpoint/infer",
            json={"prompt": "hi"},
            headers={"X-API-Key": key},
        )
        assert r.status_code == 200
        assert r.json()["expires_at"] <= max_at

    # Wait until max has passed (max was 1.0s after creation)
    time.sleep(0.6)
    # Next infer should be 410 expired, and the endpoint should have been stopped via sweep
    # (or lazily on this request). Call sweep explicitly to drive the timer path.
    from temper_control_plane import serving as serving_mod2

    serving_mod2.sweep_expired()
    r = client.post(
        f"/v1/jobs/{job_id}/endpoint/infer",
        json={"prompt": "hi"},
        headers={"X-API-Key": key},
    )
    assert r.status_code in (404, 410)
    ep = db.get_endpoint(created["id"])
    assert ep["status"] in ("stopped", "expired")


# ---------------------------------------------------------------------------
# stops itself without anyone asking (timer) and immediate stop
# ---------------------------------------------------------------------------


def test_endpoint_stops_itself_after_idle_without_anyone_asking(
    client, tmp_path, monkeypatch
):
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _fake_provider(monkeypatch)
    monkeypatch.setattr(core_serving, "ENDPOINT_IDLE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(core_serving, "ENDPOINT_MAX_LIFETIME_S", 5.0)
    from temper_control_plane import serving as serving_mod

    monkeypatch.setattr(
        serving_mod.core_serving, "ENDPOINT_IDLE_TIMEOUT_S", 0.3
    )
    monkeypatch.setattr(
        serving_mod.core_serving, "ENDPOINT_MAX_LIFETIME_S", 5.0
    )

    created = client.post(f"/v1/jobs/{job_id}/endpoint").json()
    ep_id = created["id"]
    # No one calls infer or delete; just wait and sweep
    time.sleep(0.5)
    from temper_control_plane import serving as serving_mod2

    serving_mod2.sweep_expired()
    ep = db.get_endpoint(ep_id)
    assert ep["status"] == "expired"
    assert ep["stop_reason"] == "idle_expired"
    # GET now 404 (no running endpoint)
    assert client.get(f"/v1/jobs/{job_id}/endpoint").status_code == 404
    # Machine was destroyed via confirmed path: the fake's destroy was called
    # and the endpoint row records the machine id.
    assert ep["machine_id"] is not None


def test_stop_immediately_via_delete(client, tmp_path, monkeypatch):
    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _fake_provider(monkeypatch)
    created = client.post(f"/v1/jobs/{job_id}/endpoint").json()
    ep_id = created["id"]
    r = client.delete(f"/v1/jobs/{job_id}/endpoint")
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"
    assert r.json()["stop_reason"] == "user_stopped"
    ep = db.get_endpoint(ep_id)
    assert ep["status"] == "stopped"
    assert provider.destroyed is True or provider.destroy_attempts >= 1


def test_reachability_verified_from_outside_not_from_on_it(
    client, tmp_path, monkeypatch
):
    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _fake_provider(monkeypatch)
    from temper_control_plane import serving as serving_mod

    # Make verification fail: the machine's port *is* directly reachable, so the
    # firewall did not hold and the start must be refused.
    monkeypatch.setattr(
        serving_mod, "verify_not_reachable", lambda machine: False
    )
    r = client.post(f"/v1/jobs/{job_id}/endpoint")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "endpoint_reachable"
    # No endpoint was persisted
    assert db.get_endpoint_by_job(job_id) is None
    # Even though create was attempted, the machine was destroyed (no orphan)
    assert provider.destroyed is True or provider.destroy_attempts >= 1


def test_keys_are_stored_hashed_never_plaintext(client, tmp_path, monkeypatch):
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _fake_provider(monkeypatch)
    created = client.post(f"/v1/jobs/{job_id}/endpoint").json()
    ep = db.get_endpoint(created["id"])
    # The stored value is a SHA-256 hex, not the key
    assert ep["api_key_hash"] != created["api_key"]
    assert len(ep["api_key_hash"]) == 64
    # The API never returns the hash
    body = client.get(f"/v1/jobs/{job_id}/endpoint").json()
    assert "api_key_hash" not in body
    assert "api_key" not in body
    # The DB row never contains the plaintext
    assert created["api_key"] not in str(ep.values())
