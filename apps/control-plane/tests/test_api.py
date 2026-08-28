"""End-to-end tests for the control plane, with the GPU stubbed out.

Everything except the provider call is exercised for real: upload, validation,
line-numbered errors, job creation, state transitions, the event log and
artifact download. Provisioning is the one thing mocked, because a test suite
that costs money per run does not get run.
"""

import json

import pytest
from helpers import wait_validated


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


def report_of(client, path):
    """Upload `path` and return the finished validation report.

    Validation is asynchronous: the upload returns the dataset's id while the
    report is being produced in the background, so the report is read back off
    the record once it lands."""
    r = upload(client, path)
    assert r.status_code == 202
    return wait_validated(client, r.json()["id"])["report"]


# --- catalog ---------------------------------------------------------------


def test_catalog_lists_pinned_models(client):
    from temper_core.catalog import is_pinned_revision

    body = client.get("/v1/models").json()
    ids = [m["id"] for m in body["models"]]
    assert body["default"] in ids
    for m in body["models"]:
        assert m["license"] == "Apache-2.0"  # licence flows to derivatives
        assert m["revision"]  # never unpinned
        assert is_pinned_revision(m["revision"]), (
            f"{m['id']} revision '{m['revision']}' is not a pinned commit SHA"
        )
        assert m["revision"] != "main", "branch name is not a pinned revision"


def test_models_response_is_exactly_the_published_model(client):
    from temper_control_plane.contracts_models import ModelCatalog

    body = client.get("/v1/models").json()
    parsed = ModelCatalog.model_validate(body)
    assert parsed.default in [m.id for m in parsed.models]
    assert set(body) == set(ModelCatalog.model_fields)


def test_catalog_entries_carry_a_computed_peak_memory_not_a_stored_one(
    client,
):
    """Spec 005, issue #48: the per-entry stored VRAM figure is gone, and
    every entry's peak is computed through the `models` seam instead --
    exercised here through the default fake, which the whole suite runs
    against (`conftest.no_real_models`)."""
    from temper_core import memory

    body = client.get("/v1/models").json()
    by_id = {m["id"]: m for m in body["models"]}
    assert "est_peak_vram_gb" not in by_id["qwen3-4b"]

    peak = by_id["qwen3-4b"]["peak_memory"]
    assert peak["trainable_params"] == 33_030_144
    assert peak["gpu_type"] == "L4"
    assert peak["gpu_capacity_gb"] == 24.0
    assert peak["headroom_gb"] == pytest.approx(
        24.0 - peak["total_gb"], abs=1e-6
    )
    error = abs(peak["total_gb"] - 5.31) / 5.31
    assert error <= memory.PEAK_TOLERANCE


# --- validation ------------------------------------------------------------


def test_valid_dataset_accepted(client, tmp_path):
    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    report = report_of(client, p)
    assert report["valid"] is True
    assert report["usable_rows"] == 12
    assert report["schema_type"] == "chat"
    assert report["enable_thinking"] is False


def test_too_few_rows_blocks(client, tmp_path):
    p = jsonl(tmp_path, [chat("q", "a") for _ in range(3)])
    report = report_of(client, p)
    assert report["valid"] is False
    assert any(e["code"] == "too_few_rows" for e in report["errors"])


def test_malformed_json_names_the_line(client, tmp_path):
    p = tmp_path / "bad.jsonl"
    good = "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(11))
    p.write_text(good + "\n{not json}\n", encoding="utf-8")
    report = report_of(client, p)
    err = [e for e in report["errors"] if e["code"] == "invalid_json"]
    assert err and err[0]["line"] == 12, "the offending line must be named"


def test_empty_assistant_turn_is_an_error(client, tmp_path):
    rows = [chat(f"q{i}", f"a{i}") for i in range(11)] + [chat("q", "   ")]
    report = report_of(client, jsonl(tmp_path, rows))
    assert any(
        e["code"] == "empty_target" and e["line"] == 12
        for e in report["errors"]
    )


def think_content(i: int) -> str:
    """An assistant turn carrying a reasoning trace, built from chr() so the
    literal angle brackets survive every edit to this file."""
    return f"{chr(60)}think{chr(62)}r{i}{chr(60)}/think{chr(62)}a{i}"


def test_mixed_thinking_dataset_blocks(client, tmp_path):
    rows = [chat(f"q{i}", think_content(i)) for i in range(6)]
    rows += [chat(f"q{i}", f"a{i}") for i in range(6)]
    report = report_of(client, jsonl(tmp_path, rows))
    assert report["valid"] is False
    assert any(e["code"] == "mixed_thinking" for e in report["errors"])


def test_all_thinking_dataset_enables_thinking(client, tmp_path):
    rows = [chat(f"q{i}", think_content(i)) for i in range(11)]
    report = report_of(client, jsonl(tmp_path, rows))
    assert report["valid"] is True and report["enable_thinking"] is True


def test_wrong_schema_explains_the_expected_shape(client, tmp_path):
    p = jsonl(
        tmp_path, [{"instruction": "x", "output": "y"} for _ in range(11)]
    )
    report = report_of(client, p)
    e = [x for x in report["errors"] if x["code"] == "unrecognised_schema"]
    assert e and "instruction" in e[0]["message"]  # tells them what it saw


# --- the published contract -------------------------------------------------
# The web client is generated from this API's schema (issue #28). A response
# whose shape is whatever a dict merge happened to produce generates to
# `unknown`, and the interface ends up hand-typing what the contract refused
# to -- so the dataset endpoints publish models, and these tests pin them.


def test_upload_response_is_exactly_the_published_model(client, tmp_path):
    """The upload answer is where validation is happening, not the report --
    the report lands on the record so a large upload can be watched. The
    response must parse against the published model and carry no field
    outside it, in particular not the stored file path, which is server
    state."""
    from temper_control_plane.contracts_models import DatasetAccepted

    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    r = upload(client, p)
    assert r.status_code == 202
    body = r.json()
    parsed = DatasetAccepted.model_validate(body)
    assert parsed.id == body["id"]
    assert parsed.filename == body["filename"]
    assert parsed.status == "validating"
    assert set(body) == set(DatasetAccepted.model_fields)
    assert "report" not in body
    # Let the background validation finish before this test's database is torn
    # down -- a straggler thread writing to the next test's tables is worse
    # than a test that waits.
    wait_validated(client, parsed.id)


def test_dataset_record_response_is_exactly_the_published_model(
    client, tmp_path
):
    from temper_control_plane.contracts_models import DatasetRecord

    ds = valid_dataset(client, tmp_path)
    body = client.get(f"/v1/datasets/{ds}").json()
    record = DatasetRecord.model_validate(body)
    assert record.report is not None and record.report.valid is True
    assert set(body) == set(DatasetRecord.model_fields)


def test_dataset_record_hides_the_server_file_path(client, tmp_path):
    """The stored path is where the server keeps bytes. It reaches no page and
    no client: an absolute filesystem path in a response is a leak of machine
    layout into something a browser renders."""
    ds = valid_dataset(client, tmp_path)
    assert "path" not in client.get(f"/v1/datasets/{ds}").json()


def test_dataset_list_response_is_exactly_the_published_model(
    client, tmp_path
):
    from temper_control_plane.contracts_models import DatasetList

    valid_dataset(client, tmp_path)
    body = client.get("/v1/datasets").json()
    parsed = DatasetList.model_validate(body)
    assert len(parsed.datasets) == 1
    assert set(body) == {"datasets"}
    # The list is a published shape like any other: no raw row leaks through.
    assert all("path" not in ds for ds in body["datasets"])


def test_missing_dataset_is_404(client):
    r = client.get("/v1/datasets/ds_nope")
    assert r.status_code == 404


def test_preview_turns_are_published_typed(client, tmp_path):
    """Preview rows are the weakest-typed corner of the report -- they hold
    rows validation has not judged -- so this pins what the contract claims:
    turns render as strings to the client, whatever the JSON line held."""
    from temper_control_plane.contracts_models import DatasetReport

    rows = [chat(f"q{i}", f"a{i}") for i in range(11)]
    # Inside MAX_PREVIEW's window of three, so it actually reaches the report.
    rows[2] = {"messages": ["a bare string turn", {"role": "user"}]}
    report = report_of(client, jsonl(tmp_path, rows))
    parsed = DatasetReport.model_validate(report)
    turn = parsed.preview[2].messages[0]
    assert turn.role is None
    assert turn.content.startswith('"')  # preserved as its JSON form


# --- the launch preview -----------------------------------------------------
# Issue #38. The shell's model-choice screen shows everything a job would
# train with before launching: the dataset it would train on, the effective
# specification, and any feasibility warning -- while there is still time to
# act on it. One endpoint answers that question, through the same
# usable_dataset refusal the launch itself applies, so the browser page and
# the API cannot disagree about whether a dataset may start a job.


def test_launch_preview_shows_dataset_spec_and_no_warning(client, tmp_path):
    from temper_control_plane.contracts_models import JobSpecPreview
    from temper_core import hyperparams

    ds = valid_dataset(client, tmp_path)
    body = client.get("/v1/jobs/spec", params={"dataset_id": ds}).json()
    parsed = JobSpecPreview.model_validate(body)
    assert parsed.dataset.id == ds
    # The specification shown is the one the launch would freeze: resolved
    # exactly as the trainer resolves overrides, which here are none.
    assert parsed.hyperparameters == hyperparams.effective({})
    assert parsed.warning is None


def test_launch_preview_hides_the_stored_file_path(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    body = client.get("/v1/jobs/spec", params={"dataset_id": ds}).json()
    assert "path" not in body["dataset"]


def test_launch_preview_warns_before_launch_not_after(
    client, tmp_path, monkeypatch
):
    from temper_control_plane import config

    monkeypatch.setattr(config, "MAX_JOB_DURATION_S", 10)
    ds = valid_dataset(client, tmp_path)
    body = client.get("/v1/jobs/spec", params={"dataset_id": ds}).json()
    assert body["warning"]["code"] == "duration_feasibility"


def test_launch_preview_refuses_an_invalid_dataset_with_its_code(
    client, tmp_path
):
    p = jsonl(tmp_path, [chat("q", "a")])  # too few rows
    ds = upload(client, p).json()["id"]
    r = client.get("/v1/jobs/spec", params={"dataset_id": ds})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "dataset_invalid"


def test_launch_preview_for_an_unknown_dataset_404s(client):
    r = client.get("/v1/jobs/spec", params={"dataset_id": "ds_nope"})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "not_found"


# --- jobs ------------------------------------------------------------------


def valid_dataset(client, tmp_path):
    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    return wait_validated(client, upload(client, p).json()["id"])["id"]


def test_job_creation_freezes_hyperparameters(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs", json={"dataset_id": ds, "hyperparameters": {"lora_r": 32}}
    )
    assert r.status_code == 201
    job = r.json()
    assert job["status"] == "queued"
    assert job["hyperparameters"] == {"lora_r": 32}


def test_unknown_hyperparameter_is_refused_at_creation(client, tmp_path):
    """Issue #83: resolution happens before launch, so an unknown key can no
    longer fall through to the trainer's guard to be echoed there -- it would
    be silently dropped by the resolver first. Refused here, named back, and
    nothing is launched."""
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "hyperparameters": {"lora_r": 32, "maxSteps": 5},
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "unknown_hyperparameter"
    assert detail["unknown"] == ["maxSteps"]
    # A misspelling must not cost a provisioned machine: nothing was launched.
    assert client.get("/v1/jobs").json()["jobs"] == []


def test_a_known_but_unsupported_key_is_refused_with_its_reason(
    client, tmp_path
):
    """Issue #33: a key the trainer knows but the platform does not expose is
    refused with the reason rather than 'unknown key' -- the refusal says why,
    and nothing is launched."""
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "hyperparameters": {"wandb_project": "my-project"},
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "unsupported_hyperparameter"
    assert detail["name"] == "wandb_project"
    assert "reason" in detail
    assert client.get("/v1/jobs").json()["jobs"] == []


def test_a_calculated_correctness_setting_is_refused_at_creation(
    client, tmp_path
):
    """Issue #33: a field the platform sets (a correctness setting) is refused
    as an override with its reason before launch."""
    from temper_core import surface

    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "hyperparameters": {"train_on_inputs": True},
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "unsupported_hyperparameter"
    assert detail["name"] == "train_on_inputs"
    assert detail["reason"] == surface.calculated_fields()["train_on_inputs"]
    assert client.get("/v1/jobs").json()["jobs"] == []


def test_job_records_exact_revision_it_trained_against(client, tmp_path):
    """A job records the exact revision it trained against, frozen at creation."""
    from temper_core import catalog

    ds = valid_dataset(client, tmp_path)
    model_id = catalog.DEFAULT_MODEL
    expected = catalog.get(model_id).revision
    job = client.post(
        "/v1/jobs", json={"dataset_id": ds, "base_model": model_id}
    ).json()
    assert job["base_revision"] == expected
    fetched = client.get(f"/v1/jobs/{job['id']}").json()
    assert fetched["base_revision"] == expected
    assert fetched["base_revision"] != "main"


def test_job_on_invalid_dataset_is_refused(client, tmp_path):
    p = jsonl(tmp_path, [chat("q", "a")])  # too few rows
    ds = upload(client, p).json()["id"]
    wait_validated(client, ds)  # the refusal needs the finished report
    r = client.post("/v1/jobs", json={"dataset_id": ds})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "dataset_invalid"


def test_unknown_model_is_refused_with_alternatives(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    r = client.post("/v1/jobs", json={"dataset_id": ds, "base_model": "gpt-9"})
    assert r.status_code == 400
    assert r.json()["detail"]["available"]


def test_job_response_is_exactly_the_published_model(client, tmp_path):
    from temper_control_plane.contracts_models import JobRecord

    ds = valid_dataset(client, tmp_path)
    body = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    JobRecord.model_validate(body)
    assert set(body) == set(JobRecord.model_fields)
    # The stored artifact path is server state; it reaches no client.
    assert "artifact_path" not in body


def test_created_and_fetched_jobs_publish_the_same_shape(client, tmp_path):
    """One concept, one published shape: creating a job and fetching it later
    answer with the same fields, so a client cannot be right about one and
    wrong about the other."""
    from temper_control_plane.contracts_models import JobRecord

    ds = valid_dataset(client, tmp_path)
    created = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    fetched = client.get(f"/v1/jobs/{created['id']}").json()
    assert set(fetched) == set(created) == set(JobRecord.model_fields)


def test_job_list_response_is_exactly_the_published_model(client, tmp_path):
    from temper_control_plane.contracts_models import JobList

    ds = valid_dataset(client, tmp_path)
    client.post("/v1/jobs", json={"dataset_id": ds})
    body = client.get("/v1/jobs").json()
    parsed = JobList.model_validate(body)
    assert len(parsed.jobs) == 1
    assert all("artifact_path" not in j for j in body["jobs"])


def test_queued_job_has_no_artifact_yet(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    r = client.get(f"/v1/jobs/{job['id']}/artifact")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "no_artifact"


def test_events_record_every_transition(client, tmp_path):
    from temper_control_plane import db

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    db.set_state(job["id"], "provisioning", "Selecting a GPU")
    db.set_state(job["id"], "failed", "boom", error_code="training_failed")

    events = client.get(f"/v1/jobs/{job['id']}/events").json()["events"]
    states = [e["message"] for e in events if e["kind"] == "state"]
    assert states == ["queued", "Selecting a GPU", "boom"]
    assert (
        client.get(f"/v1/jobs/{job['id']}").json()["error_code"]
        == "training_failed"
    )


def test_event_polling_is_incremental(client, tmp_path):
    from temper_control_plane import db

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    first = client.get(f"/v1/jobs/{job['id']}/events").json()
    db.add_event(job["id"], "log", "later")
    nxt = client.get(
        f"/v1/jobs/{job['id']}/events", params={"after": first["last_id"]}
    ).json()
    assert [e["message"] for e in nxt["events"]] == ["later"]


def test_missing_job_is_404(client):
    assert client.get("/v1/jobs/job_nope").status_code == 404


def test_an_upload_without_a_filename_is_refused_by_name(client):
    """A multipart part can carry no filename -- the type says `str | None`
    even though Starlette's current parser rejects such parts before the
    handler runs. The handler's own answer to None must still be a coded
    400, never a crash on None."""
    import io

    import pytest
    from fastapi import HTTPException
    from starlette.requests import Request

    from temper_control_plane import main

    class Unnamed:
        filename = None
        file = io.BytesIO(b"{}\n")

    with pytest.raises(HTTPException) as exc:
        main.upload_dataset(
            Request({"type": "http", "headers": []}), Unnamed()
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "unsupported_extension"


def test_cancelling_a_job_that_vanishes_mid_request_is_a_404(
    client, tmp_path, monkeypatch
):
    """The cancel handler re-reads the row after request_cancel answered;
    if it vanished in between, the answer is a named 404, not a TypeError."""
    from temper_control_plane import db

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()

    real_get_job = db.get_job

    def vanished(job_id):
        real_get_job(job_id)
        return None

    monkeypatch.setattr(db, "get_job", vanished)
    r = client.post(f"/v1/jobs/{job['id']}/cancel")
    assert r.status_code == 404
    assert r.json()["detail"] == "No such job."


def test_a_completed_adapter_job_downloads_a_loadable_artifact(
    client, tmp_path
):
    """The download is a zip carrying the adapter's config and a manifest.

    Regression test for shipping a bare .safetensors: PEFT cannot load weights
    without the config that records rank, alpha and target modules, so a
    download missing it looks like the deliverable and is not one. Both files
    are stored as objects behind the storage seam; the endpoint reads them by
    key and must not know where they live. The manifest declares the artifact's
    kind and its load path (issue #32) -- the artifact is the deliverable, and
    the adapter is one kind of it.
    """
    import io
    import zipfile

    from temper_control_plane import db, storage

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()

    weights_key = storage.artifact_key(job["id"], storage.ADAPTER_WEIGHTS_NAME)
    storage.STORE.put(weights_key, b"weights")
    storage.STORE.put(
        storage.artifact_key(job["id"], storage.ADAPTER_CONFIG_NAME),
        b'{"r": 16, "lora_alpha": 32}',
    )
    db.set_state(
        job["id"],
        "complete",
        "done",
        method="qlora",
        artifact_key=weights_key,
    )

    r = client.get(f"/v1/jobs/{job['id']}/artifact")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert job["id"] in r.headers["content-disposition"]
    assert "adapter.zip" not in r.headers["content-disposition"]
    assert "artifact.zip" in r.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert sorted(z.namelist()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
            "temper-artifact.json",
        ]
        assert z.read("adapter_model.safetensors") == b"weights"
        manifest = json.loads(z.read("temper-artifact.json"))
        assert manifest["kind"] == "adapter"
        assert manifest["base_model"] == job["base_model"]
        assert manifest["loading"]


def test_a_job_whose_stored_weights_are_gone_refuses_loudly(client, tmp_path):
    """A download that 'succeeds' with an empty archive would look like the
    deliverable and is not one -- the failure shape the zip exists to prevent.
    The refusal carries a stable code like every other one."""
    from temper_control_plane import db, storage

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    key = storage.artifact_key(job["id"], storage.ADAPTER_WEIGHTS_NAME)
    db.set_state(job["id"], "complete", "done", artifact_key=key)

    r = client.get(f"/v1/jobs/{job['id']}/artifact")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "artifact_missing"


def test_a_recorded_full_model_artifact_downloads_without_special_casing(
    client, tmp_path
):
    """The download path serves any kind by streaming whatever the artifact
    record names -- the endpoint carries no per-kind branch.

    A full fine-tune produces a fully trained model, not an adapter; its
    members are recorded at packaging time and served exactly like an
    adapter's. The manifest says which kind it is, so what a user downloads
    tells them how to load it. This is issue #32's "download path serves any
    kind without special-casing" made testable before a full-model run exists.
    """
    import io
    import zipfile

    from temper_control_plane import db, storage

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()

    keys = {
        name: storage.artifact_key(job["id"], name)
        for name in ("model.safetensors", "config.json")
    }
    storage.STORE.put(keys["model.safetensors"], b"full weights")
    storage.STORE.put(keys["config.json"], b'{"architectures": ["Qwen3"]}')
    db.set_state(
        job["id"],
        "complete",
        "done",
        method="full",
        artifact_key=keys["model.safetensors"],
        artifact_json={
            "members": [
                {
                    "name": "model.safetensors",
                    "key": keys["model.safetensors"],
                },
                {"name": "config.json", "key": keys["config.json"]},
            ],
            "bytes": len(b"full weights")
            + len(b'{"architectures": ["Qwen3"]}'),
            "sha256": "x",
        },
    )

    r = client.get(f"/v1/jobs/{job['id']}/artifact")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert sorted(z.namelist()) == [
            "config.json",
            "model.safetensors",
            "temper-artifact.json",
        ]
        assert z.read("model.safetensors") == b"full weights"
        manifest = json.loads(z.read("temper-artifact.json"))
        assert manifest["kind"] == "full_model"
        assert "PeftModel" not in manifest["loading"]


def test_the_published_job_record_carries_the_artifact_and_its_kind(
    client, tmp_path
):
    """`artifact` is part of the published record: the interface can name the
    deliverable -- its kind and its load path -- without touching where it
    lives (the storage keys stay server-side)."""
    from temper_control_plane import db, storage

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    assert job["artifact"] is None

    weights_key = storage.artifact_key(job["id"], storage.ADAPTER_WEIGHTS_NAME)
    storage.STORE.put(weights_key, b"weights")
    storage.STORE.put(
        storage.artifact_key(job["id"], storage.ADAPTER_CONFIG_NAME),
        b'{"r": 16}',
    )
    db.set_state(
        job["id"],
        "complete",
        "done",
        method="qlora",
        artifact_key=weights_key,
        artifact_json={
            "members": [
                {"name": "adapter_model.safetensors", "key": weights_key}
            ],
            "bytes": 7,
            "sha256": "x",
        },
    )

    body = client.get(f"/v1/jobs/{job['id']}").json()
    artifact = body["artifact"]
    assert artifact["kind"] == "adapter"
    assert artifact["members"] == ["adapter_model.safetensors"]
    assert artifact["bytes"] == 7
    assert artifact["loading"]
    # The storage address never reaches the client.
    assert "artifact_key" not in body and "artifact_record" not in body


def test_a_job_that_is_not_complete_publishes_no_artifact(client, tmp_path):
    """An artifact is available exactly when the job is complete.

    A cancelled or failed row can retain keys whose objects teardown already
    deleted; publishing an artifact there would offer a download that cannot
    succeed. The record is only published beside the `complete` state.
    """
    from temper_control_plane import db, storage

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    weights_key = storage.artifact_key(job["id"], storage.ADAPTER_WEIGHTS_NAME)
    storage.STORE.put(weights_key, b"weights")
    db.set_state(
        job["id"],
        "cancelled",
        "Cancelled at your request. No artifact was produced.",
        method="qlora",
        artifact_key=weights_key,
    )

    assert client.get(f"/v1/jobs/{job['id']}").json()["artifact"] is None


def test_the_download_zip_streams_without_holding_it_whole(
    client, tmp_path, peak_memory
):
    """The download path hands the browser chunks, not a buffered whole.

    This is the guard against the regression the streaming read exists to
    prevent: the artifact is the payload that grows without bound, and a
    `STORE.get` into an in-memory zip would hold every byte of a full
    fine-tune per request behind an API whose name claims otherwise. The
    real route handler runs against the real (filesystem) backend seeded by
    the app's own fixtures; what is drained is the response body it built.
    Bounds match test_orchestrator's -- independent measurements of the same
    clause at another leg.
    """
    from temper_control_plane import db, main, storage

    FLAT_SPREAD = 1 << 21
    ABS_CEIL = 8 << 20

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    weights_key = storage.artifact_key(job["id"], storage.ADAPTER_WEIGHTS_NAME)
    storage.STORE.put(
        storage.artifact_key(job["id"], storage.ADAPTER_CONFIG_NAME),
        b'{"r": 16}',
    )
    db.set_state(job["id"], "complete", "done", artifact_key=weights_key)

    block = bytes(range(256)) * 1024

    def drive():
        # The real handler, against the real backend: member peeking, key
        # derivation and the streamed zip all run; only the HTTP layer is
        # skipped, because a test client buffering the response would put
        # the payload inside the measurement. Starlette wraps a sync
        # iterator in an async generator eagerly, so the body is consumed
        # the way the server itself would.
        import asyncio

        response = main.download_artifact(job["id"])

        async def total() -> int:
            seen = 0
            async for chunk in response.body_iterator:
                seen += len(chunk)
            return seen

        return asyncio.run(total())

    peaks = []
    for size in (1 << 20, 32 << 20):
        storage.STORE.put(weights_key, block * (size >> 20))
        peaks.append(peak_memory(drive))
        assert peaks[-1] > 0

    # And the stream is a valid archive either way.
    storage.STORE.put(weights_key, block * 3)
    import asyncio
    import io
    import zipfile

    async def collect() -> bytes:
        parts = []
        async for chunk in main.download_artifact(job["id"]).body_iterator:
            parts.append(chunk)
        return b"".join(parts)

    joined = asyncio.run(collect())
    with zipfile.ZipFile(io.BytesIO(joined)) as z:
        assert sorted(z.namelist()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
            "temper-artifact.json",
        ]
        assert z.read("adapter_model.safetensors") == block * 3

    assert max(peaks) - min(peaks) < FLAT_SPREAD, f"peaks grew: {peaks}"
    assert peaks[-1] < ABS_CEIL, f"peak scaled with the payload: {peaks}"


# --- ops --------------------------------------------------------------------


def test_health_advertises_which_provider_would_run(client, monkeypatch):
    """A launch driven by the browser journeys must never be able to reach the
    billing account, so they boot the control plane with TEMPER_FAKE_PROVIDER
    and refuse to proceed unless /health says the switch took effect."""
    from temper_control_plane import config

    monkeypatch.setattr(config, "FAKE_PROVIDER", False)
    assert client.get("/health").json()["provider"] == "real"
    monkeypatch.setattr(config, "FAKE_PROVIDER", True)
    assert client.get("/health").json()["provider"] == "fake"
