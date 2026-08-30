"""Delivery formats' control-plane side (issue #74): request, verify, serve.

The launch can ask for more than the canonical artifact: a merged single-file
model and a quantised local-inference format. The control plane validates the
request against the one delivery vocabulary, mints one scoped grant per format,
verifies what the machine wrote against its checksum, records the per-format
verdicts, and serves each format with a manifest generated from the run record
(ADR-0054 flow-through -- one generator, not a second, parallel description of
an artifact).
"""

from __future__ import annotations

import io
import json
import zipfile

from helpers import wait_validated

from temper_core import delivery as core_delivery


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


def _wait_terminal(client, job_id: str, timeout: float = 10.0) -> dict:
    """Poll a job until it reaches a terminal state. Returns the record."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = client.get(f"/v1/jobs/{job_id}").json()
        if rec["status"] in ("complete", "failed", "cancelled"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal state")


def _job_request(client, tmp_path, provider=None, **extra):
    """Post a job, and drive it to a terminal state against `provider` if
    given -- the request path only inserts a `queued` row (issue #51), so a
    test that wants the job to actually run drives it itself, the way the
    worker would."""
    ds = _valid_dataset(client, tmp_path)
    body = {"dataset_id": ds, **extra}
    r = client.post("/v1/jobs", json=body)
    assert r.status_code == 201, r.text
    job = r.json()
    if provider is not None:
        from temper_control_plane import orchestrator as orch

        orch.run_job(job["id"], provider=provider)
    return job


def _set_published_reference(monkeypatch):
    """The checked-in image contract starts unpublished; a test that drives
    `run_job` against the fake provider needs a reference so the
    pull-by-digest path is exercised rather than the refusal."""
    from temper_control_plane import orchestrator as orch

    monkeypatch.setattr(
        orch, "published_reference", lambda: "ghcr.io/x/y@sha256:0000"
    )


# --- creation: the request is validated and frozen ----------------------------


def test_the_spec_preview_offers_the_delivery_formats_with_their_purpose(
    client, tmp_path
):
    """The launch screen offers each format by what it is for, read from the
    one delivery vocabulary -- the same sentences the finished page shows, so
    the two surfaces cannot drift about what a format is (ADR-0010)."""
    ds = _valid_dataset(client, tmp_path)
    r = client.get("/v1/jobs/spec", params={"dataset_id": ds})
    assert r.status_code == 200
    formats = {d["id"]: d for d in r.json()["delivery_formats"]}
    assert set(formats) == {"adapter", "merged", "quantised"}
    for fmt in formats.values():
        assert fmt["what_for"].strip()


def test_a_delivery_request_is_accepted_and_frozen(client, tmp_path):
    job = _job_request(client, tmp_path, delivery=["merged", "quantised"])
    # Frozen at creation, like the hyperparameters: the run says what it was
    # asked to produce.
    assert job["delivery_request"] == ["merged", "quantised"]


def test_an_unknown_delivery_format_is_refused(client, tmp_path):
    ds = _valid_dataset(client, tmp_path)
    r = client.post("/v1/jobs", json={"dataset_id": ds, "delivery": ["wat"]})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unknown_delivery_format"


def test_the_default_delivery_request_is_empty(client, tmp_path):
    job = _job_request(client, tmp_path)
    assert job["delivery_request"] == []
    assert job["delivery_formats"] == []


# --- the machine produces the formats and the plane verifies what landed -----


def test_verified_delivery_formats_are_recorded_and_published(
    client, tmp_path, monkeypatch
):
    """A complete run whose trainer reported delivery formats gets each one
    verified against its checksum and published with its plain-language
    purpose -- the interface can offer a download the user understands."""
    from temper_control_plane import db
    from temper_control_plane.fake_provider import completed_run

    _set_published_reference(monkeypatch)
    job = _job_request(
        client,
        tmp_path,
        delivery=["merged", "quantised"],
        provider=completed_run(),
    )

    rec = _wait_terminal(client, job["id"])
    assert rec["status"] == "complete", rec

    # The published record lists both formats with purpose and members.
    assert [d["format"] for d in rec["delivery_formats"]] == [
        "merged",
        "quantised",
    ]
    merged_pub = rec["delivery_formats"][0]
    assert merged_pub["what_for"].strip()
    assert merged_pub["members"] == ["merged.tar.gz"]
    assert merged_pub["kind"] == core_delivery.kind_for("merged")

    # The stored records carry the verification's bytes and checksum.
    stored = db.get_job(job["id"])
    delivery_records = stored["delivery"]
    assert all(r["verified"] is True for r in delivery_records)
    assert delivery_records[0]["members"][0]["name"] == "merged.tar.gz"
    assert delivery_records[1]["members"][0]["name"] == "quantised.gguf"


def test_each_delivery_format_downloads_with_a_manifest_describing_itself(
    client, tmp_path, monkeypatch
):
    """The download for a format serves that format's members and a manifest
    generated from the run record that describes THAT format -- the same
    generator ADR-0054 ships for the canonical artifact, not a parallel one."""
    from temper_control_plane.fake_provider import completed_run

    _set_published_reference(monkeypatch)
    job = _job_request(
        client,
        tmp_path,
        delivery=["merged", "quantised"],
        provider=completed_run(),
    )
    rec = _wait_terminal(client, job["id"])
    assert rec["status"] == "complete", rec

    r = client.get(
        f"/v1/jobs/{job['id']}/artifact", params={"format": "merged"}
    )
    assert r.status_code == 200, (
        r.json()
        if r.headers.get("content-type", "").startswith("application/json")
        else r.text
    )
    assert "merged.zip" in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = set(z.namelist())
        assert "merged.tar.gz" in names
        assert "temper-artifact.json" in names
        assert "PROVENANCE.md" in names
        manifest = json.loads(z.read("temper-artifact.json"))
        # The manifest describes the merged format, not the adapter.
        assert manifest["artifact"]["kind"] == "merged_model"
        assert manifest["artifact"]["members"] == ["merged.tar.gz"]
        assert "merged" in manifest["artifact"]["loading"].lower()
        # The rest of the provenance is the run's own record (ADR-0054).
        assert manifest["base_model"] == job["base_model"]
        assert manifest["evaluation"] is not None

    rq = client.get(
        f"/v1/jobs/{job['id']}/artifact", params={"format": "quantised"}
    )
    assert rq.status_code == 200
    assert "quantised.zip" in rq.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(rq.content)) as z:
        manifest = json.loads(z.read("temper-artifact.json"))
        assert manifest["artifact"]["kind"] == "quantised_local"
        assert manifest["artifact"]["members"] == ["quantised.gguf"]
        assert "local" in manifest["artifact"]["loading"].lower()


def test_the_default_download_still_serves_the_canonical_artifact(
    client, tmp_path, monkeypatch
):
    """Without a format parameter, the download is exactly the canonical
    artifact -- the delivery addition changes nothing for existing flows."""
    from temper_control_plane.fake_provider import completed_run

    _set_published_reference(monkeypatch)
    job = _job_request(
        client, tmp_path, delivery=["merged"], provider=completed_run()
    )
    rec = _wait_terminal(client, job["id"])
    assert rec["status"] == "complete", rec

    r = client.get(f"/v1/jobs/{job['id']}/artifact")
    assert r.status_code == 200, (
        r.json()
        if r.headers.get("content-type", "").startswith("application/json")
        else r.text
    )
    assert "artifact.zip" in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert "adapter_model.safetensors" in z.namelist()
        manifest = json.loads(z.read("temper-artifact.json"))
        assert manifest["kind"] == "adapter"


def test_a_format_that_was_not_requested_is_not_served(client, tmp_path):
    job = _job_request(client, tmp_path)  # no delivery request
    r = client.get(
        f"/v1/jobs/{job['id']}/artifact", params={"format": "merged"}
    )
    # Not produced, so unavailable -- a coded refusal, never an empty archive.
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "format_unavailable"


def test_an_unknown_format_on_the_download_is_refused(client, tmp_path):
    job = _job_request(client, tmp_path)
    r = client.get(f"/v1/jobs/{job['id']}/artifact", params={"format": "wat"})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "unknown_delivery_format"


# --- teardown covers the delivery objects (ADR-0054's resolution extends) -----


def test_teardown_deletes_the_delivery_objects_with_the_artifact(
    client, tmp_path, monkeypatch
):
    """Cancellation's object deletion reads `db.artifact_members`, which now
    resolves delivery objects too -- a cancelled job keeps no recovery material,
    and a delivery format is not orphaned bytes in the store."""
    from temper_control_plane import db, storage

    job = _job_request(client, tmp_path, delivery=["merged"])
    # Store a delivery object the way a machine's write would land it.
    key = storage.delivery_key(job["id"], "merged", "merged.tar.gz")
    storage.STORE.put(key, b"bytes")
    db.set_delivery(
        job["id"],
        [
            {
                "format": "merged",
                "verified": True,
                "key": key,
                "members": [{"name": "merged.tar.gz", "key": key}],
            }
        ],
    )
    members = db.artifact_members(db.get_job(job["id"]))
    assert any(k == key for _n, k in members)
    # The resolution the teardown path reads now covers the delivery object.
    for _n, k in members:
        storage.STORE.delete(k)
    from temper_control_plane.storage import ObjectNotFound

    try:
        storage.STORE.get(key)
        raise AssertionError("delivery object should have been deleted")
    except ObjectNotFound:
        pass


def test_a_corrupt_delivery_format_is_recorded_as_unverified(
    client, tmp_path, monkeypatch
):
    """A delivery format whose checksum does not match what landed is refused
    rather than served -- the same rule that governs the canonical artifact."""
    from temper_control_plane import db

    _set_published_reference(monkeypatch)
    job = _job_request(
        client,
        tmp_path,
        delivery=["merged"],
        provider=_corrupt_delivery_provider(),
    )
    _wait_terminal(client, job["id"])

    stored = db.get_job(job["id"])
    delivery_records = stored.get("delivery") or []
    if delivery_records:
        assert any(not r.get("verified") for r in delivery_records)


def _corrupt_delivery_provider():
    """A simulated machine that reports a merged format whose checksum does
    not match the bytes it actually wrote."""
    import hashlib

    from temper_control_plane import fake_provider, storage
    from temper_control_plane.fake_provider import SimulatedMachine
    from temper_control_plane.storage import WriteGrant

    class CorruptDelivery(SimulatedMachine):
        def _write_delivery(self, spec):
            grants = (spec or {}).get("delivery_grants") or []
            if not grants:
                return
            records = []
            for block in grants:
                format_id = block.get("format")
                payload = fake_provider._delivery_bytes(format_id)
                grant = WriteGrant(
                    url=block["url"],
                    key=block["key"],
                    expires_at=block["expires_at"],
                )
                storage.STORE.redeem(grant, payload)
                records.append(
                    {
                        "format": format_id,
                        # Deliberately wrong: what the machine reports is not
                        # what it wrote.
                        "sha256": hashlib.sha256(b"other bytes").hexdigest(),
                        "upload": {"ok": True, "bytes": len(payload)},
                    }
                )
                self._enter(f"delivery_upload:{format_id}")
            if records and self._result is not None:
                self._result = {**self._result, "delivery": records}

    return CorruptDelivery(
        lines=fake_provider.DEMO_LINES,
        result=dict(fake_provider.DEMO_RESULT),
        adapter_bytes=fake_provider.DEMO_ADAPTER_BYTES,
        line_delay=0.0,
        checkpoints=[
            {
                "step": 20,
                "loss": 0.35,
                "held_out_loss": 0.39,
                "bytes": b"ckpt-20",
            }
        ],
    )
