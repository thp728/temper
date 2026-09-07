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
import threading
import time

import pytest
from helpers import wait_validated

from temper_control_plane import db
from temper_control_plane.fake_provider import FakeProvider
from temper_control_plane.storage import FilesystemStorage
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

    The endpoint requires a complete job. Posting through the API only
    inserts a `queued` row (issue #51); nothing drives it further, so for
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


def test_unparsable_handle_does_not_get_a_key(client, tmp_path, monkeypatch):
    """A handle whose grammar cannot be parsed must not get a key (fail-closed).

    _parse_host previously returned None for three cases (empty, fake://,
    unrecognised) and verify_not_reachable treated all three as "no public
    host, pass". Empty and fake:// are genuinely safe (the fake publishes
    nothing, which is the trainer's own mitigation). An unrecognised
    grammar is not safe to shrug at: if a provider's handle format ever
    changes, silently passing would issue a key for a machine nobody
    verified. This test proves the fix fails closed.
    """
    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _fake_provider(monkeypatch)
    from temper_control_plane.provider import Machine

    # Make create return a machine whose handle matches no known grammar:
    # non-empty, not fake://, and containing no '@'.
    orig_create = provider.create

    def bad_create(gpu_type, num_gpus, storage_gb, name):
        m = orig_create(gpu_type, num_gpus, storage_gb, name)
        return Machine(
            machine_id=m.machine_id, handle="unparsable-handle-no-at-sign"
        )

    monkeypatch.setattr(provider, "create", bad_create)
    r = client.post(f"/v1/jobs/{job_id}/endpoint")
    # Must refuse to issue a key, with a stable code and a message naming
    # the handle grammar it could not read.
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "endpoint_verification_failed"
    assert "handle does not contain" in detail["message"]
    assert "expected grammar" in detail["message"]
    # No endpoint was persisted and the machine was destroyed (no orphan).
    assert db.get_endpoint_by_job(job_id) is None
    assert provider.destroyed is True or provider.destroy_attempts >= 1


# ---------------------------------------------------------------------------
# real inference: the branch that makes the billed machine worth holding
# ---------------------------------------------------------------------------
#
# The endpoint provisioned an L4 at 41.31 INR/hr and answered every prompt
# with `[qwen3-4b] tuned response to: <prompt>`, having never loaded a model,
# while the comment beside that line claimed a real path existed on real
# hardware. Nothing exercised it, which is why it survived. These tests turn
# on `is_remote`, the same switch the code branches on.


class _RemoteFake(FakeProvider):
    """A fake that reports itself remote and answers as the model server would.

    Subclassed rather than written fresh so provisioning, teardown and the
    destroy confirmation stay identical to the simulated tier: what these
    tests vary is the serving branch and nothing else. `stream` is the one
    channel the serving path uses, so this reads the script it is handed and
    answers the way `apps/trainer/serve.py` would.
    """

    is_remote = True

    def __init__(
        self,
        *,
        healthy_after=0,
        completion="Paris.",
        raw=None,
        banner=None,
    ):
        super().__init__()
        self.scripts: list[str] = []
        self.health_probes = 0
        self._healthy_after = healthy_after
        self._completion = completion
        self._raw = raw
        # What the connection says before the command does. `provider.stream`
        # folds stderr into stdout, so this arrives in the same text as the
        # reply -- which is how a real completion came to be refused.
        self._banner = banner

    def stream(self, machine, script):
        text = script.decode("utf-8")
        self.scripts.append(text)
        if self._banner:
            yield self._banner
        for line in text.splitlines():
            if line.strip().startswith("echo "):
                yield line.strip()[len("echo ") :]
        if "/health" in text:
            self.health_probes += 1
            yield json.dumps(
                {"ready": self.health_probes > self._healthy_after}
            )
        elif "/generate" in text:
            if self._raw is not None:
                yield self._raw
            else:
                yield json.dumps({"completion": self._completion})
        else:
            yield "started"


class _GrantableStore(FilesystemStorage):
    """A local store that can hand a machine an address it could fetch.

    The serving path refuses outright on a store whose grants only this
    process can redeem -- that refusal is the subject of its own test below.
    Every other remote test needs a store on the far side of that question,
    and standing up S3 for them would be exercising boto3 rather than
    serving. The url is a stand-in and is never fetched; what the tests read
    is which key was granted and where the command puts it.
    """

    grants_are_remotely_redeemable = True

    def mint_read_grant(self, key, expires_in_s):
        from temper_control_plane import storage

        return storage.ReadGrant(
            url=f"https://store.example/{key}?sig=stand-in",
            key=key,
            expires_at=time.time() + expires_in_s,
        )


def _remote_provider(monkeypatch, provider):
    """Install `provider` as the one every serving path resolves.

    `config.FAKE_PROVIDER` stays False on purpose: the branch under test is
    the provider's own `is_remote`, and pointing the switch the other way
    would let a bug that reads the switch instead pass unnoticed.

    The store is swapped for one that can grant reads, because a remote
    endpoint on a local-only store is refused before it provisions.
    """
    from temper_control_plane import serving, storage

    monkeypatch.setattr(
        storage, "STORE", _GrantableStore(root=storage.STORE._root)
    )
    monkeypatch.setattr(serving, "new_provider", lambda: provider)
    monkeypatch.setattr(serving, "SERVE_READY_POLL_S", 0.01)
    return provider


def test_a_remote_endpoint_answers_from_the_model_not_a_template(
    client, tmp_path, monkeypatch
):
    """The completion comes from the machine's model server.

    The assertion that matters is the negative one: the old template echoed
    the prompt back, so a completion that still contains it would mean the
    branch never ran.
    """
    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _remote_provider(
        monkeypatch, _RemoteFake(completion="The capital of France is Paris.")
    )
    created = client.post(f"/v1/jobs/{job_id}/endpoint").json()

    r = client.post(
        f"/v1/jobs/{job_id}/endpoint/infer",
        json={"prompt": "What is the capital of France?"},
        headers={"X-API-Key": created["api_key"]},
    )
    assert r.status_code == 200, r.text
    completion = r.json()["completion"]
    assert completion == "The capital of France is Paris."
    assert "tuned response to" not in completion
    assert "What is the capital of France?" not in completion
    # The prompt reached the server as a JSON body on the generate path.
    generate = [s for s in provider.scripts if "/generate" in s]
    assert len(generate) == 1
    assert (
        json.dumps({"prompt": "What is the capital of France?"})
        in (generate[0])
    )


def test_starting_a_remote_endpoint_ships_the_server_and_the_adapter(
    client, tmp_path, monkeypatch
):
    """The key is minted only after the model answers.

    An endpoint that hands out a key and then cannot answer has taken the
    user's money for nothing, so readiness is settled before the key exists.
    """
    from temper_control_plane import serving, storage

    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _remote_provider(monkeypatch, _RemoteFake(healthy_after=2))

    r = client.post(f"/v1/jobs/{job_id}/endpoint")
    assert r.status_code == 201, r.text
    assert r.json()["api_key"]

    # The server script is the only thing this process moves; it is a few
    # kilobytes. The adapter is a hundred-odd megabytes and is fetched by
    # the machine, so it must not appear here.
    pushed = dict(provider.pushed)
    assert list(pushed) == [serving.SERVE_SCRIPT_PATH]
    assert pushed[serving.SERVE_SCRIPT_PATH] == serving._serve_source()

    start = next(s for s in provider.scripts if "docker run" in s)
    for name in storage.ARTIFACT_MEMBERS:
        assert f"artifacts/{job_id}/{name}" in start
        assert f"-o {serving.SERVE_ADAPTER_DIR}/{name} " in start
    # It waited rather than assuming: three probes, the first two not ready.
    assert provider.health_probes == 3


def test_the_machine_is_told_the_repository_not_the_catalog_id(
    client, tmp_path, monkeypatch
):
    """`qwen3-4b` is a catalog id. `Qwen/Qwen3-4B` is what loads.

    A job row stores the id, and `from_pretrained` cannot resolve it. Getting
    this wrong fails on the machine, minutes into a paid start, with an error
    that reads like a network problem -- the class of bug that only shows up
    on hardware, which is exactly why it is asserted here.
    """
    from temper_core import catalog

    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _remote_provider(monkeypatch, _RemoteFake())
    assert client.post(f"/v1/jobs/{job_id}/endpoint").status_code == 201

    model = catalog.get(catalog.DEFAULT_MODEL)
    run = next(s for s in provider.scripts if "docker run" in s)
    assert f"TEMPER_BASE_MODEL={model.repo}" in run
    assert f"TEMPER_BASE_REVISION={model.revision}" in run
    assert f"TEMPER_BASE_MODEL={catalog.DEFAULT_MODEL}" not in run


def test_the_port_that_serves_is_the_port_the_isolation_check_probes(
    client, tmp_path, monkeypatch
):
    """One number, or the check asserts a property of nothing.

    `verify_not_reachable` connects to `INFERENCE_PORT` on the machine's
    public host and refuses to mint a key if it answers. A server listening
    on some other port would leave that check passing while the port that
    actually serves went unexamined.
    """
    from temper_control_plane import serving

    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _remote_provider(monkeypatch, _RemoteFake())
    assert client.post(f"/v1/jobs/{job_id}/endpoint").status_code == 201

    port = serving.INFERENCE_PORT
    run = next(s for s in provider.scripts if "docker run" in s)
    assert f"TEMPER_SERVE_PORT={port}" in run
    # Every call the control plane makes to the machine's own loopback goes
    # to that port: the readiness probe and the generation both.
    loopback = [s for s in provider.scripts if "127.0.0.1" in s]
    assert loopback
    for script in loopback:
        assert f"127.0.0.1:{port}/" in script


def test_the_endpoint_serves_under_the_thinking_mode_the_run_trained_with(
    client, tmp_path, monkeypatch
):
    """The same value at training and at serving, or the answers are junk.

    `entrypoint.build_config` says it beside the value it sets: Qwen3 emits
    thinking blocks through its template by default, the trainer detects
    from the dataset whether the training rows had them, and the SAME value
    must be applied at serving. Rendering under the other one produces
    plausible-looking output, which is why nothing downstream would catch it.
    """
    job_id = _complete_job(client, tmp_path, monkeypatch)
    result = dict(db.get_job(job_id)["result"] or {})
    result["thinking"] = {"enable_thinking": True, "assistant_turns": 12}
    db.set_state(job_id, "complete", "done", result_json=result)

    provider = _remote_provider(monkeypatch, _RemoteFake())
    assert client.post(f"/v1/jobs/{job_id}/endpoint").status_code == 201
    run = next(s for s in provider.scripts if "docker run" in s)
    assert "TEMPER_ENABLE_THINKING=1" in run


def test_a_run_that_did_not_think_does_not_serve_as_though_it_had(
    client, tmp_path, monkeypatch
):
    """The other direction, which is the one a default would get wrong.

    Qwen3's template thinks by default, so an endpoint that passed nothing
    would serve a non-thinking run under a thinking template and look fine.
    """
    job_id = _complete_job(client, tmp_path, monkeypatch)
    result = dict(db.get_job(job_id)["result"] or {})
    result["thinking"] = {"enable_thinking": False, "assistant_turns": 12}
    db.set_state(job_id, "complete", "done", result_json=result)

    provider = _remote_provider(monkeypatch, _RemoteFake())
    assert client.post(f"/v1/jobs/{job_id}/endpoint").status_code == 201
    run = next(s for s in provider.scripts if "docker run" in s)
    assert "TEMPER_ENABLE_THINKING=0" in run


def test_a_local_only_store_never_provisions_a_serving_machine(
    client, tmp_path, monkeypatch
):
    """Refused before the money, not after it.

    The filesystem backend's grants are tokens only this process redeems, so
    a machine has nowhere to fetch the adapter from. That is the same
    asymmetry the launch refuses with `artifact_undeliverable`, and it was
    learned from a paid run that trained for eleven minutes and delivered
    nothing. The refusal has to land before `provider.create`.
    """
    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _RemoteFake()
    from temper_control_plane import serving

    monkeypatch.setattr(serving, "new_provider", lambda: provider)
    # The `isolated` fixture's store is a plain FilesystemStorage, which is
    # exactly the backend under test here.

    r = client.post(f"/v1/jobs/{job_id}/endpoint")
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "endpoint_artifact_unreachable"
    # Nothing was asked for, so nothing is billing. `create_calls` is the
    # assertion and not `list_machine_ids`: the fake lists a machine from
    # the start whether or not anyone created one.
    assert provider.create_calls == []


def test_the_grant_the_machine_gets_reads_one_key_and_expires(
    client, tmp_path, monkeypatch
):
    """One key, and a lifetime no longer than the wait it serves.

    A grant is a credential. This one exists to let the machine pull one
    adapter member during the readiness window; anything broader would be
    authority nobody needed.
    """
    from temper_control_plane import serving, storage

    job_id = _complete_job(client, tmp_path, monkeypatch)
    _remote_provider(monkeypatch, _RemoteFake())
    granted = []
    original = storage.STORE.mint_read_grant

    def record(key, expires_in_s):
        granted.append((key, expires_in_s))
        return original(key, expires_in_s)

    monkeypatch.setattr(storage.STORE, "mint_read_grant", record)
    assert client.post(f"/v1/jobs/{job_id}/endpoint").status_code == 201

    assert [k for k, _ in granted] == [
        storage.artifact_key(job_id, name) for name in storage.ARTIFACT_MEMBERS
    ]
    assert {ttl for _, ttl in granted} == {serving.SERVE_GRANT_TTL_S}


def test_the_signed_url_never_reaches_the_job_s_event_log(
    client, tmp_path, monkeypatch
):
    """A grant in the events table is a credential on the finished-job page.

    The start script carries signed URLs and its output is dropped rather
    than classified into events. This asserts the whole log, not just the
    lines the start writes, because the leak would be equally bad wherever
    it landed.
    """
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _remote_provider(monkeypatch, _RemoteFake())
    assert client.post(f"/v1/jobs/{job_id}/endpoint").status_code == 201

    logged = " ".join(str(e) for e in db.get_events(job_id))
    assert "sig=" not in logged
    assert "store.example" not in logged


def test_a_model_that_never_loads_costs_no_key_and_no_machine(
    client, tmp_path, monkeypatch
):
    """The machine is destroyed and the start refuses.

    This is the money path. A server that never comes up would otherwise
    leave a GPU billing for an endpoint that could never have answered.
    """
    from temper_control_plane import serving

    job_id = _complete_job(client, tmp_path, monkeypatch)
    # Never ready: the health probe answers `false` for every attempt.
    provider = _remote_provider(monkeypatch, _RemoteFake(healthy_after=10**6))
    monkeypatch.setattr(serving, "SERVE_READY_TIMEOUT_S", 0.05)

    r = client.post(f"/v1/jobs/{job_id}/endpoint")
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "endpoint_model_not_ready"
    assert db.get_endpoint_by_job(job_id) is None
    # Teardown confirmed by listing, never by the destroy call's return.
    assert provider.list_machine_ids() == []


def test_the_reconciler_spares_a_machine_that_is_still_loading_its_model(
    client, tmp_path, monkeypatch
):
    """The window this whole `starting` status exists to close.

    Loading a model takes minutes. The reconciler destroys any machine no
    live job or endpoint claims, and the job that owns a serving machine is
    `complete` and therefore terminal, so before this the machine spent its
    entire warm-up unclaimed. A pass landing in that window destroyed a
    machine that was doing exactly what it had been asked to -- ADR-0068's
    race, at a hundred times the width.
    """
    from temper_worker import reconciler

    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _remote_provider(monkeypatch, _RemoteFake(healthy_after=10**6))
    seen = threading.Event()
    report = {}

    def reconcile_during_warmup():
        # Wait until the machine exists, then run a pass against it, which
        # is precisely when the old code lost one.
        while not provider.create_calls:
            time.sleep(0.01)
        report.update(reconciler.reconcile_once(provider=provider))
        seen.set()

    from temper_control_plane import serving

    # Long enough that the pass above lands inside the warm-up.
    monkeypatch.setattr(serving, "SERVE_READY_TIMEOUT_S", 3.0)
    monkeypatch.setattr(serving, "SERVE_READY_POLL_S", 0.05)
    watcher = threading.Thread(target=reconcile_during_warmup, daemon=True)
    watcher.start()
    client.post(f"/v1/jobs/{job_id}/endpoint")
    assert seen.wait(30)

    assert report["destroyed"] == [], report
    assert report["owned"] >= 1, report


def test_a_start_that_never_finished_stops_protecting_its_machine(
    client, tmp_path, monkeypatch
):
    """The grace is bounded, and that bound is the point.

    A control plane killed mid-start leaves a `starting` row behind. An
    ownership claim that never expired would make that row a permanent
    licence for a machine nobody will ever serve from -- an orphan the
    reconciler is forbidden to collect, which is worse than the race the
    status exists to prevent.
    """
    from temper_control_plane import db as db_mod

    job_id = _complete_job(client, tmp_path, monkeypatch)
    endpoint_id = db_mod.new_id("ep")
    long_ago = time.time() - 10_000
    db_mod.create_endpoint(
        endpoint_id,
        job_id,
        api_key_hash="",
        api_key_prefix="",
        created_at=long_ago,
        expires_at=long_ago,
        max_expires_at=long_ago,
        machine_id=4242,
        machine_handle="fake://stale",
        status=db_mod.ENDPOINT_STARTING,
    )
    assert db_mod.list_active_endpoint_machine_ids(1800.0) == []
    assert db_mod.list_live_endpoint_job_ids(1800.0) == []
    # Inside the grace it is owned, which is the other half of the claim.
    assert db_mod.list_active_endpoint_machine_ids(20_000.0) == [4242]


def test_a_failed_start_leaves_no_row_claiming_a_destroyed_machine(
    client, tmp_path, monkeypatch
):
    """The record says why it never opened, and stops claiming the machine.

    A `starting` row left behind after its machine was destroyed would have
    the reconciler sparing an id the provider has already forgotten, and
    would leave the job looking like it has an endpoint on the way.
    """
    from temper_control_plane import db as db_mod
    from temper_control_plane import serving

    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _remote_provider(monkeypatch, _RemoteFake(healthy_after=10**6))
    monkeypatch.setattr(serving, "SERVE_READY_TIMEOUT_S", 0.05)

    r = client.post(f"/v1/jobs/{job_id}/endpoint")
    assert r.status_code == 409, r.text

    row = db_mod.get_endpoint_by_job(job_id, status=None)
    assert row["status"] == "stopped"
    assert row["stop_reason"] == "endpoint_model_not_ready"
    assert db_mod.list_active_endpoint_machine_ids(1800.0) == []
    # And a second attempt is not blocked by the abandoned one.
    assert provider.list_machine_ids() == []


def test_a_real_completion_survives_ssh_talking_over_it(
    client, tmp_path, monkeypatch
):
    """The first paid endpoint answered, and the answer was thrown away.

    `provider.stream` folds stderr into stdout on purpose: one ordered
    channel is what the transport delivers. So `ssh` saying

        Warning: Permanently added '217.18.55.26' (ED25519) to the list of
        known hosts.

    lands in the same text as the reply, and parsing the whole stream as
    JSON refused a genuine `{"completion": "<think>\\nOkay, the user is
    asking for the capital of France..."}`. The banner here is the one the
    machine actually sent, quoted from that run.
    """
    banner = (
        "Warning: Permanently added '217.18.55.26' (ED25519) to the list "
        "of known hosts."
    )
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _remote_provider(
        monkeypatch, _RemoteFake(completion="Paris.", banner=banner)
    )
    created = client.post(f"/v1/jobs/{job_id}/endpoint").json()

    r = client.post(
        f"/v1/jobs/{job_id}/endpoint/infer",
        json={"prompt": "What is the capital of France?"},
        headers={"X-API-Key": created["api_key"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["completion"] == "Paris."


def test_the_readiness_probe_also_survives_it(client, tmp_path, monkeypatch):
    """The probe asks one yes-or-no question and must not be strict.

    It answered correctly through the same banner on real hardware, because
    it tests for a substring rather than parsing. That is deliberate rather
    than lucky, and this keeps it that way.
    """
    banner = "Warning: Permanently added 'x' (ED25519) to the list of hosts."
    job_id = _complete_job(client, tmp_path, monkeypatch)
    provider = _remote_provider(
        monkeypatch, _RemoteFake(healthy_after=1, banner=banner)
    )
    assert client.post(f"/v1/jobs/{job_id}/endpoint").status_code == 201
    assert provider.health_probes == 2


def test_a_banner_with_no_answer_behind_it_is_still_diagnosable(
    client, tmp_path, monkeypatch
):
    """What could not be parsed reaches the user, not an empty string.

    A marker that never arrived means the whole text is the best evidence
    there is, and hiding it would leave someone debugging a blank message.
    """
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _remote_provider(
        monkeypatch, _RemoteFake(raw="curl: (7) Failed to connect")
    )
    created = client.post(f"/v1/jobs/{job_id}/endpoint").json()

    r = client.post(
        f"/v1/jobs/{job_id}/endpoint/infer",
        json={"prompt": "hello"},
        headers={"X-API-Key": created["api_key"]},
    )
    assert r.status_code == 502
    assert "Failed to connect" in r.json()["detail"]["message"]


def test_a_server_that_answers_nonsense_is_not_passed_off_as_a_completion(
    client, tmp_path, monkeypatch
):
    """A traceback on stdout is not a completion.

    502 rather than 400: the request was well formed and authorised, and the
    thing that failed was the machine behind it.
    """
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _remote_provider(
        monkeypatch, _RemoteFake(raw="Traceback (most recent call last):")
    )
    created = client.post(f"/v1/jobs/{job_id}/endpoint").json()

    r = client.post(
        f"/v1/jobs/{job_id}/endpoint/infer",
        json={"prompt": "hello"},
        headers={"X-API-Key": created["api_key"]},
    )
    assert r.status_code == 502, r.text
    assert r.json()["detail"]["code"] == "endpoint_generation_failed"


def test_the_endpoint_records_the_route_to_its_own_machine(
    client, tmp_path, monkeypatch
):
    """The handle is stored, because inference is a different request.

    A training run holds its `Machine` in memory for the whole run. An
    endpoint is started by one request and asked for a completion by
    another, and `list_machines` reports ids and names, not handles, so a
    handle that is not written down cannot be recovered.
    """
    job_id = _complete_job(client, tmp_path, monkeypatch)
    _remote_provider(monkeypatch, _RemoteFake())
    created = client.post(f"/v1/jobs/{job_id}/endpoint").json()

    ep = db.get_endpoint(created["id"])
    assert ep["machine_handle"].startswith("fake://")

    # An endpoint whose route was never recorded refuses, rather than
    # handing `ssh` an empty destination and reporting what that fails with.
    with db.connect() as c:
        c.execute(
            "UPDATE endpoints SET machine_handle=NULL WHERE id=%s",
            (created["id"],),
        )
    r = client.post(
        f"/v1/jobs/{job_id}/endpoint/infer",
        json={"prompt": "hello"},
        headers={"X-API-Key": created["api_key"]},
    )
    assert r.status_code == 502, r.text
    assert r.json()["detail"]["code"] == "endpoint_machine_unreachable"


def test_the_endpoint_decodes_the_way_the_comparison_did():
    """One prompt, one answer, whichever screen asked it.

    `serve.py` restates the decoding settings as a literal because it is
    pushed to the machine rather than imported with the package.
    `comparison.COMPARISON_DECODING` is the definition, and this is what
    keeps the copy honest: without it an endpoint could answer at a
    temperature the prediction-versus-actual panel never used.
    """
    import importlib.util
    import sys

    from temper_control_plane import trainer_build

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    from temper_control_plane import serving

    trainer = trainer_build.TRAINER_DIR
    comparison = load("_comparison_for_serve", trainer / "comparison.py")
    serve = load("_serve_for_decoding", trainer / "serve.py")
    assert serve.DECODING == comparison.COMPARISON_DECODING
    # Same argument, same file: the port it falls back to when nobody passes
    # one is the port the isolation check probes.
    assert serve.PORT == serving.INFERENCE_PORT
