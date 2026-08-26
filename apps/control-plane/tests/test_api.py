"""End-to-end tests for the control plane, with the GPU stubbed out.

Everything except the provider call is exercised for real: upload, validation,
line-numbered errors, job creation, state transitions, the event log and
artifact download. Provisioning is the one thing mocked, because a test suite
that costs money per run does not get run.
"""

import json


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


def test_catalog_revision_shown_alongside_licence_at_model_choice(
    client, tmp_path
):
    """The revision is shown to the user at model choice, alongside the licence.

    The create-job page renders the catalog entries with both fields, so a user
    choosing a model sees what revision they are about to train against.
    """
    ds = valid_dataset(client, tmp_path)
    html = client.get(f"/jobs/new?dataset_id={ds}").text
    for m in client.get("/v1/models").json()["models"]:
        assert m["revision"] in html
        assert m["license"] in html


# --- validation ------------------------------------------------------------


def test_valid_dataset_accepted(client, tmp_path):
    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    r = upload(client, p)
    assert r.status_code == 201
    body = r.json()
    assert body["valid"] is True
    assert body["usable_rows"] == 12
    assert body["schema_type"] == "chat"
    assert body["enable_thinking"] is False


def test_too_few_rows_blocks(client, tmp_path):
    p = jsonl(tmp_path, [chat("q", "a") for _ in range(3)])
    body = upload(client, p).json()
    assert body["valid"] is False
    assert any(e["code"] == "too_few_rows" for e in body["errors"])


def test_malformed_json_names_the_line(client, tmp_path):
    p = tmp_path / "bad.jsonl"
    good = "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(11))
    p.write_text(good + "\n{not json}\n", encoding="utf-8")
    body = upload(client, p).json()
    err = [e for e in body["errors"] if e["code"] == "invalid_json"]
    assert err and err[0]["line"] == 12, "the offending line must be named"


def test_empty_assistant_turn_is_an_error(client, tmp_path):
    rows = [chat(f"q{i}", f"a{i}") for i in range(11)] + [chat("q", "   ")]
    body = upload(client, jsonl(tmp_path, rows)).json()
    assert any(
        e["code"] == "empty_target" and e["line"] == 12 for e in body["errors"]
    )


def test_mixed_thinking_dataset_blocks(client, tmp_path):
    rows = [chat(f"q{i}", f"<think>r</think>a{i}") for i in range(6)]
    rows += [chat(f"q{i}", f"a{i}") for i in range(6)]
    body = upload(client, jsonl(tmp_path, rows)).json()
    assert body["valid"] is False
    assert any(e["code"] == "mixed_thinking" for e in body["errors"])


def test_all_thinking_dataset_enables_thinking(client, tmp_path):
    rows = [chat(f"q{i}", f"<think>r{i}</think>a{i}") for i in range(11)]
    body = upload(client, jsonl(tmp_path, rows)).json()
    assert body["valid"] is True and body["enable_thinking"] is True


def test_wrong_schema_explains_the_expected_shape(client, tmp_path):
    p = jsonl(
        tmp_path, [{"instruction": "x", "output": "y"} for _ in range(11)]
    )
    body = upload(client, p).json()
    e = [x for x in body["errors"] if x["code"] == "unrecognised_schema"]
    assert e and "instruction" in e[0]["message"]  # tells them what it saw


# --- the published contract -------------------------------------------------
# The web client is generated from this API's schema (issue #28). A response
# whose shape is whatever a dict merge happened to produce generates to
# `unknown`, and the interface ends up hand-typing what the contract refused
# to -- so the dataset endpoints publish models, and these tests pin them.


def test_upload_response_is_exactly_the_published_model(client, tmp_path):
    from temper_control_plane.contracts_models import (
        DatasetUploaded,
        ValidationIssue,
    )

    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    body = upload(client, p).json()
    # Parses against the published model and carries no field outside it --
    # in particular not the stored file path, which is server state.
    parsed = DatasetUploaded.model_validate(body)
    assert parsed.id == body["id"]
    assert parsed.filename == body["filename"]
    assert set(body) == set(DatasetUploaded.model_fields)
    # An issue parses as the published shape, line reference included.
    bad = upload(client, jsonl(tmp_path, [chat("q", "a")])).json()
    issues = DatasetUploaded.model_validate(bad).errors
    assert any(
        isinstance(i, ValidationIssue) and i.code == "too_few_rows"
        for i in issues
    )


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


def test_preview_turns_are_published_typed(client, tmp_path):
    """Preview rows are the weakest-typed corner of the report -- they hold
    rows validation has not judged -- so this pins what the contract claims:
    turns render as strings to the client, whatever the JSON line held."""
    from temper_control_plane.contracts_models import DatasetUploaded

    rows = [chat(f"q{i}", f"a{i}") for i in range(11)]
    # Inside MAX_PREVIEW's window of three, so it actually reaches the report.
    rows[2] = {"messages": ["a bare string turn", {"role": "user"}]}
    body = upload(client, jsonl(tmp_path, rows)).json()
    parsed = DatasetUploaded.model_validate(body)
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
    return upload(client, p).json()["id"]


def test_job_creation_freezes_hyperparameters(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs", json={"dataset_id": ds, "hyperparameters": {"lora_r": 32}}
    )
    assert r.status_code == 201
    job = r.json()
    assert job["status"] == "queued"
    assert job["hyperparameters"] == {"lora_r": 32}


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
    assert "adapter_path" not in body


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
    assert all("adapter_path" not in j for j in body["jobs"])


def test_queued_job_has_no_adapter_yet(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    r = client.get(f"/v1/jobs/{job['id']}/adapter")
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


def test_completed_job_downloads_a_loadable_adapter(client, tmp_path):
    """The download is a zip, and the zip carries adapter_config.json.

    Regression test for shipping a bare .safetensors: PEFT cannot load weights
    without the config that records rank, alpha and target modules, so a
    download missing it looks like the deliverable and is not one.
    """
    import io
    import zipfile

    from temper_control_plane import db, orchestrator

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()

    art = orchestrator.ARTIFACTS / job["id"]
    art.mkdir(parents=True)
    (art / "adapter_model.safetensors").write_bytes(b"weights")
    (art / "adapter_config.json").write_text('{"r": 16, "lora_alpha": 32}')
    db.set_state(
        job["id"],
        "complete",
        "done",
        adapter_path=str(art / "adapter_model.safetensors"),
    )

    r = client.get(f"/v1/jobs/{job['id']}/adapter")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert job["id"] in r.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert sorted(z.namelist()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]
        assert z.read("adapter_model.safetensors") == b"weights"


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
