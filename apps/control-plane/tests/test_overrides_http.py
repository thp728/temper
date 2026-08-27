"""Overrides over HTTP -- spec 005, issue #79.

The experienced-user half of the plan, through the seam a user meets: an
override is re-requested (`POST /v1/quotes`) rather than applied locally, so
the recomputation rules live in the server; a launch freezes the overrides and
the recomputed quote into the job spec; and an override that cannot be
honoured is refused with the same arithmetic the predictor used, never
silently fallen back to its pick.

Three things are asserted through the HTTP seam:

* `POST /v1/quotes` recomputes around an override, marks the overridden
  decision, and publishes the legal values each control can offer.
* A refusal -- unknown decision, inconsistent precision/method, a
  configuration that cannot fit on memory grounds, a disk below the need or
  the minimum -- is a 400 with a stable code and the arithmetic that refused
  it, on both the recompute and the launch.
* A launch freezes the overrides beside the hyperparameters and the quote,
  and the orchestrator provisions against them rather than re-picking.
"""

from __future__ import annotations

import json

from temper_core.selection import GpuAvailability


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


def quote_post(client, ds, overrides, model="qwen3-4b"):
    return client.post(
        "/v1/quotes",
        json={"dataset_id": ds, "base_model": model, "overrides": overrides},
    )


def decision_of(q, named):
    return next(d for d in q["decisions"] if d["decision"] == named)


# --- the recompute endpoint ---------------------------------------------------


def test_recomputing_around_an_override_marks_it_and_recomputes_the_rest(
    client, tmp_path
):
    """Changing one decision re-requests the plan: the sequence-length
    override lands in the recomputed quote marked overridden, and the
    untouched decisions stay unmarked -- nothing stale beside the change."""
    ds = valid_dataset(client, tmp_path)
    r = quote_post(
        client, ds, [{"decision": "sequence length", "value": "4096"}]
    )
    assert r.status_code == 200
    q = r.json()
    seq = decision_of(q, "sequence length")
    assert seq["chosen"] == "4096"
    assert seq["overridden"] is True
    for named in ("method", "hardware", "device count", "disk", "precision"):
        assert decision_of(q, named)["overridden"] is False


def test_overriding_precision_moves_the_method_and_marks_precision(
    client, tmp_path
):
    """Precision and method are one coupled decision: naming bf16 recomputes
    the method to lora and shows bf16 as the precision decision's own
    (overridden) choice."""
    ds = valid_dataset(client, tmp_path)
    r = quote_post(
        client,
        ds,
        [{"decision": "precision", "value": "bf16 (no quantisation)"}],
    )
    assert r.status_code == 200
    q = r.json()
    assert decision_of(q, "method")["chosen"] == "lora"
    precision = decision_of(q, "precision")
    assert precision["chosen"] == "bf16 (no quantisation)"
    assert precision["overridden"] is True


def test_the_legal_control_values_are_published_with_the_quote(
    client, tmp_path
):
    """The interface generates its controls from the quote's `override_options`
    rather than hand-listing the vocabulary -- a value the server accepts is a
    value the plan offers and vice versa."""
    ds = valid_dataset(client, tmp_path)
    q = quote_post(client, ds, []).json()
    assert q["override_options"]["method"] == ["qlora", "lora", "full"]
    assert "L4" in q["override_options"]["hardware"]
    assert q["override_options"]["precision"] == [
        "nf4 (4-bit)",
        "bf16 (no quantisation)",
    ]
    # The free-form decisions (any positive integer / any 'N GB') carry no list.
    assert "device count" not in q["override_options"]
    assert "disk" not in q["override_options"]
    assert "sequence length" not in q["override_options"]


# --- refusals on the recompute -------------------------------------------------


def _single_l4_provider(monkeypatch):
    """A fake with exactly one free L4: sharding is not an escape hatch, so a
    full fine-tune of the 4B model cannot fit anything at any device count --
    the clean way to exercise the memory refusal."""
    from temper_control_plane import quote as quote_mod
    from temper_control_plane.fake_provider import FakeProvider

    provider = FakeProvider(
        availability=[GpuAvailability("L4", 41.31, 1)],
    )
    monkeypatch.setattr(quote_mod, "QUOTE_PROVIDER", provider)
    return provider


def test_a_memory_infeasible_override_is_refused_with_the_same_arithmetic(
    client, tmp_path, monkeypatch
):
    """The memory half blocks (spec 005), and an override must not lose that
    safety property by taking control: with only one L4 free, a full
    fine-tune of the 4B model cannot fit at any device count, and the refusal
    names the peak against the capacity that refused it."""
    _single_l4_provider(monkeypatch)
    ds = valid_dataset(client, tmp_path)
    r = quote_post(client, ds, [{"decision": "method", "value": "full"}])
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "configuration_does_not_fit"
    arith = detail["arithmetic"]
    assert arith["gpu_type"] == "L4"
    assert arith["capacity_gb"] == 24.0
    assert arith["peak_gb"] > arith["capacity_gb"]
    assert "peak" in detail["message"]


def test_an_unknown_decision_is_refused_on_the_recompute(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    r = quote_post(client, ds, [{"decision": "quantisation", "value": "nf4"}])
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "unknown_decision"
    assert detail["decision"] == "quantisation"


def test_an_inconsistent_precision_and_method_pair_is_refused(
    client, tmp_path
):
    ds = valid_dataset(client, tmp_path)
    r = quote_post(
        client,
        ds,
        [
            {"decision": "method", "value": "qlora"},
            {"decision": "precision", "value": "bf16 (no quantisation)"},
        ],
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "inconsistent_overrides"


def test_a_disk_override_below_the_platform_minimum_is_refused(
    client, tmp_path
):
    ds = valid_dataset(client, tmp_path)
    r = quote_post(client, ds, [{"decision": "disk", "value": "50 GB"}])
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "disk_below_minimum"
    assert detail["minimum_gb"] == 100


def test_a_disk_override_above_the_need_is_honoured(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    r = quote_post(client, ds, [{"decision": "disk", "value": "500 GB"}])
    assert r.status_code == 200
    assert decision_of(r.json(), "disk")["chosen"] == "500 GB"


# --- refusals on the launch ---------------------------------------------------


def test_launching_a_non_executable_override_is_refused(client, tmp_path):
    """The plan may describe a configuration the trainer cannot run; the
    launch is where that description stops being a quote (spec 005's 'running
    them is Spec 009')."""
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "overrides": [{"decision": "method", "value": "lora"}],
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "not_executable"
    assert detail["method"] == "lora"


def test_launching_a_multi_device_override_is_refused(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "overrides": [{"decision": "device count", "value": "2"}],
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "not_executable"


def test_launching_an_unknown_decision_override_is_refused(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "overrides": [{"decision": "bogus", "value": "x"}],
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unknown_decision"
    # Nothing was launched.
    assert client.get("/v1/jobs").json()["jobs"] == []


def test_launching_a_memory_infeasible_override_is_refused_with_arithmetic(
    client, tmp_path, monkeypatch
):
    _single_l4_provider(monkeypatch)
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "overrides": [{"decision": "method", "value": "full"}],
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "configuration_does_not_fit"
    assert detail["arithmetic"]["gpu_type"] == "L4"
    assert client.get("/v1/jobs").json()["jobs"] == []


def test_launching_a_disk_override_below_the_minimum_is_refused(
    client, tmp_path
):
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "overrides": [{"decision": "disk", "value": "50 GB"}],
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "disk_below_minimum"
    assert client.get("/v1/jobs").json()["jobs"] == []


# --- launching freezes the overrides -----------------------------------------


def test_launching_freezes_overrides_and_the_recomputed_quote(
    client, tmp_path
):
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "overrides": [{"decision": "sequence length", "value": "4096"}],
        },
    )
    assert r.status_code == 201
    job = r.json()
    # The run says what it actually used: the sequence-length override is
    # frozen beside the hyperparameters it folded into, and the quote's
    # decision carries the mark.
    assert job["overrides"] == [
        {"decision": "sequence length", "value": "4096"}
    ]
    assert job["hyperparameters"]["sequence_len"] == 4096
    seq = decision_of(job["quote"], "sequence length")
    assert seq["chosen"] == "4096"
    assert seq["overridden"] is True


def test_launching_without_overrides_freezes_an_empty_list(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    job = client.post("/v1/jobs", json={"dataset_id": ds}).json()
    assert job["overrides"] == []
    assert all(not d["overridden"] for d in job["quote"]["decisions"])


def test_the_frozen_overrides_are_part_of_the_immutable_job_spec(
    client, tmp_path
):
    from temper_control_plane import db

    ds = valid_dataset(client, tmp_path)
    job = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "overrides": [{"decision": "disk", "value": "200 GB"}],
        },
    ).json()
    stored = db.get_job(job["id"])
    assert stored["overrides"] == [{"decision": "disk", "value": "200 GB"}]
    db.set_state(job["id"], "training", "running")
    assert db.get_job(job["id"])["overrides"] == stored["overrides"]


# --- the orchestrator provisions against the override -------------------------


def test_provisioning_honours_a_frozen_hardware_override(
    client, monkeypatch, tmp_path
):
    """A hardware override must survive to the machine: the orchestrator
    provisions the card the plan froze, not the cheapest card it could re-pick
    -- a run that provisioned something else would be lying about what it
    did."""
    from temper_control_plane import fake_models, orchestrator
    from temper_control_plane import quote as quote_mod
    from temper_control_plane.fake_provider import FakeProvider

    provider = FakeProvider(
        availability=[
            GpuAvailability("L4", 41.31, 8),
            GpuAvailability("H100", 250.0, 2),
        ],
        lines=[
            "[00:00:00] building trainer image",
            "{'loss': 0.6931, 'step': 10, 'epoch': 0.5}",
        ],
        result={
            "ok": True,
            "stage": "train",
            "adapter_path": "run/adapter_model.safetensors",
            "adapter_sha256": "abc",
            "adapter_config": {"r": 16, "lora_alpha": 32},
        },
    )
    monkeypatch.setattr(quote_mod, "QUOTE_PROVIDER", provider)
    monkeypatch.setattr(
        orchestrator,
        "launch",
        lambda job_id: orchestrator.run_job(
            job_id,
            provider=provider,
            models=fake_models.catalog_models(),
        ),
    )

    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "overrides": [{"decision": "hardware", "value": "H100"}],
        },
    )
    assert r.status_code == 201
    job = r.json()
    assert job["gpu_type"] == "H100"
    assert job["device_count"] == 1
    # The quote it was shown also reflects the pinned card.
    assert decision_of(job["quote"], "hardware")["chosen"] == "H100"
    assert decision_of(job["quote"], "hardware")["overridden"] is True
