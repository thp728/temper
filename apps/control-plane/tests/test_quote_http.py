"""The quote as a user meets it: on the plan screen, and frozen at launch.

Issue #72. The preview endpoint computes a quote per catalog model and the
plan shows it; launching freezes the quote into the job row, written once and
never updated. Three properties are asserted here, all through the HTTP seam:

* The plan screen carries a quote for each catalog model -- a duration range
  and a per-phase cost breakdown in the account's currency.
* Launching freezes that quote onto the job, and it is still there afterwards
  (a completed job says what it was predicted to cost).
* A configuration that cannot be priced never blocks: no quote is a valid
  outcome for both surfaces.
"""

from __future__ import annotations

import json

from temper_control_plane.fake_provider import FakeProvider


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


def valid_dataset(client, tmp_path):
    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    with open(p, "rb") as f:
        r = client.post("/v1/datasets", files={"file": (p.name, f)})
    assert r.status_code == 201
    return r.json()["id"]


def preview(client, ds):
    return client.get("/v1/jobs/spec", params={"dataset_id": ds}).json()


def job_quote(client, job_id):
    return client.get(f"/v1/jobs/{job_id}").json()["quote"]


# --- the plan carries a quote per model ---------------------------------------


def test_the_plan_shows_a_quote_for_every_catalog_model(client, tmp_path):
    from temper_core import catalog

    ds = valid_dataset(client, tmp_path)
    body = preview(client, ds)
    assert set(body["quotes"]) == set(catalog.CATALOG)
    for model_id, q in body["quotes"].items():
        assert q is not None
        assert q["is_estimate"] is True
        assert q["currency"] == "INR"
        # Duration is a range, never a point.
        assert q["duration_low_s"] < q["duration_high_s"]
        # Cost is composed per phase; the phases are spec 005's list.
        names = [p["name"] for p in q["phases"]]
        assert names == [
            "provisioning",
            "readiness",
            "image_pull",
            "model_download",
            "training",
            "teardown",
        ]
        # The quote pins what it was computed against.
        assert q["dataset_id"] == ds
        assert q["base_revision"] == catalog.get(model_id).revision


def test_the_quote_breaks_cost_down_by_phase_in_minor_units(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    q = preview(client, ds)["quotes"]["qwen3-4b"]
    for phase in q["phases"]:
        assert phase["duration_low_s"] < phase["duration_high_s"]
        assert phase["cost_low_minor"] < phase["cost_high_minor"]
    # Total is the sum of the phases.
    assert q["cost_low_minor"] == sum(p["cost_low_minor"] for p in q["phases"])
    assert q["cost_high_minor"] == sum(
        p["cost_high_minor"] for p in q["phases"]
    )


def test_the_quote_expires(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    q = preview(client, ds)["quotes"]["qwen3-4b"]
    assert q["expires_at"] > q["dataset_created_at"]


# --- the plan preview is exactly the published model ---------------------------


def test_preview_response_is_exactly_the_published_model(client, tmp_path):
    from temper_control_plane.contracts_models import JobSpecPreview

    ds = valid_dataset(client, tmp_path)
    body = preview(client, ds)
    parsed = JobSpecPreview.model_validate(body)
    assert set(body) == set(JobSpecPreview.model_fields)
    assert parsed.quotes is not None


# --- launching freezes the quote ----------------------------------------------


def test_launching_copies_the_quote_onto_the_job(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    q = job_quote(client, job["id"])
    assert q is not None
    # The frozen quote is the one the plan would have shown for the default
    # model: same inputs, same prediction.
    plan_quote = preview(client, ds)["quotes"][job["base_model"]]
    assert q["base_revision"] == plan_quote["base_revision"]
    assert q["duration_low_s"] == plan_quote["duration_low_s"]
    assert q["cost_low_minor"] == plan_quote["cost_low_minor"]


def test_the_frozen_quote_is_part_of_the_immutable_job_spec(client, tmp_path):
    """A quote is written once, at launch, and never updated -- the same rule
    as the hyperparameters and warnings it sits beside."""
    from temper_control_plane import db

    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    stored = db.get_job(job["id"])
    assert stored["quote"] == job_quote(client, job["id"])
    # A later state transition does not touch it.
    db.set_state(job["id"], "training", "running")
    fetched = db.get_job(job["id"])
    assert fetched["quote"] == stored["quote"]


def test_job_record_response_is_exactly_the_published_model(client, tmp_path):
    from temper_control_plane.contracts_models import JobRecord

    ds = valid_dataset(client, tmp_path)
    body = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    parsed = JobRecord.model_validate(body)
    assert set(body) == set(JobRecord.model_fields)
    assert parsed.quote is not None


# --- a configuration that cannot be priced never blocks ------------------------


def test_an_unquotable_configuration_still_launches_without_a_quote(
    client, tmp_path, monkeypatch
):
    """The estimate never blocks a launch (spec 005): if nothing can be
    priced -- here, no hardware fits because the fake has only a card too
    small -- the job still launches, and carries no quote rather than a
    refusal."""
    from temper_control_plane import quote as quote_mod

    monkeypatch.setattr(
        quote_mod,
        "QUOTE_PROVIDER",
        FakeProvider(
            availability=[],
            currency="INR",
        ),
    )
    ds = valid_dataset(client, tmp_path)
    # The plan still renders, with no quote for any model...
    body = preview(client, ds)
    assert all(v is None for v in body["quotes"].values())
    # ...and the launch still happens.
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    assert job["status"] == "queued"
    assert job_quote(client, job["id"]) is None


def test_an_unreachable_provider_means_no_quote_not_a_broken_page(
    client, tmp_path, monkeypatch
):
    """The provider being unreachable is a fact about the world, not about the
    user's job: the plan still renders and the launch still goes ahead."""
    from temper_control_plane import fake_provider
    from temper_control_plane import quote as quote_mod

    def boom():
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr(quote_mod, "QUOTE_PROVIDER", FakeProvider())
    monkeypatch.setattr(
        fake_provider.FakeProvider, "gpu_availability", boom, raising=False
    )
    ds = valid_dataset(client, tmp_path)
    body = preview(client, ds)
    assert all(v is None for v in body["quotes"].values())
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    assert job["status"] == "queued"
