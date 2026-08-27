"""The advanced surface over HTTP -- spec 009, issue #80.

The correctness settings pass every obvious health check and surface only as
bad output, so the product exposes them behind an explicit disclosure with the
specific thing that goes wrong beside each one. What this file asserts,
through the seam a user meets:

* `GET /v1/surface` publishes the generated surface -- the overrideable set,
  each exposed field with its reason and failure mode, and the trainer's
  known-but-unsupported settings with their reasons -- so the interface
  renders from data rather than a hand-written list.
* A hyperparameter override that makes the job infeasible (a `micro_batch_size`
  that blows through VRAM) is refused before launch with the same arithmetic
  the predictor used, on the recompute and on the launch.
* A hyperparameter the surface refuses is refused through the same seams --
  an unknown key echoed back, a known-but-unsupported key with its reason --
  before anything is priced or launched.
* A launch freezes the user's hyperparameter overrides, coerced to the
  schema's type, into the job spec.
* The two dataset-level refusals that must never become overridable -- a
  mixed thinking-mode dataset and one below the row floor -- stay refused with
  their own codes even when a launch carries hyperparameter overrides: an
  absent control, not a hidden one.
"""

from __future__ import annotations

import json

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


def valid_dataset(client, tmp_path):
    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    with open(p, "rb") as f:
        r = client.post("/v1/datasets", files={"file": (p.name, f)})
    assert r.status_code == 202
    return wait_validated(client, r.json()["id"])["id"]


def _single_l4_provider(monkeypatch):
    """A fake with exactly one free L4: sharding is not an escape hatch, so an
    override that pushes the 4B model past the L4's 24 GB cannot fit at any
    device count -- the clean way to exercise the memory refusal."""
    from temper_control_plane import quote as quote_mod
    from temper_control_plane.fake_provider import FakeProvider
    from temper_core.selection import GpuAvailability

    provider = FakeProvider(
        availability=[GpuAvailability("L4", 41.31, 1)],
    )
    monkeypatch.setattr(quote_mod, "QUOTE_PROVIDER", provider)
    return provider


# --- the published surface ---------------------------------------------------


def test_the_surface_is_published_with_every_tier_and_its_reasons(client):
    """The interface generates its advanced panel from `GET /v1/surface`: the
    overrideable set, each exposed field's reason and failure mode, and the
    trainer's known-but-unsupported settings visible with their reasons rather
    than absent."""
    r = client.get("/v1/surface")
    assert r.status_code == 200
    s = r.json()
    assert "lora_r" in s["overrideable_keys"]
    assert "sequence_len" in s["overrideable_keys"]
    exposed = s["tiers"]["exposed_with_named_failure_mode"]
    assert exposed["learning_rate"]["reason"]
    assert "diverges" in exposed["learning_rate"]["failure_mode"]
    unsupported = s["tiers"]["known_but_unsupported"]
    assert "wandb_project" in unsupported
    assert unsupported["wandb_project"]["reason"]
    assert "calculated" in s["tiers"]
    assert s["counts"]["exposed_with_named_failure_mode"] == 10
    # The schema snapshot's identity travels with it (serialised under its
    # plain name), so the panel can say what it was generated from.
    assert s["source_image"].startswith("axolotlai/axolotl:")


# --- an infeasible hyperparameter override is refused with the arithmetic ----


def test_a_memory_infeasible_hyperparameter_override_is_refused_on_the_recompute(
    client, tmp_path, monkeypatch
):
    """A `micro_batch_size` that pushes the 4B model past the L4's 24 GB is a
    demand, not an absent estimate: the recompute refuses it with the same
    peak-vs-capacity arithmetic the predictor used, instead of silently
    showing no quote."""
    _single_l4_provider(monkeypatch)
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/quotes",
        json={
            "dataset_id": ds,
            "base_model": "qwen3-4b",
            "hyperparameters": {"micro_batch_size": "64"},
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "configuration_does_not_fit"
    arith = detail["arithmetic"]
    assert arith["gpu_type"] == "L4"
    assert arith["capacity_gb"] == 24.0
    assert arith["peak_gb"] > arith["capacity_gb"]


def test_a_memory_infeasible_hyperparameter_override_is_refused_before_launch(
    client, tmp_path, monkeypatch
):
    """The launch refuses the same way, and nothing is launched: taking control
    of the correctness surface must not cost a provisioned machine that OOMs
    minutes in."""
    _single_l4_provider(monkeypatch)
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "base_model": "qwen3-4b",
            "hyperparameters": {"micro_batch_size": "64"},
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "configuration_does_not_fit"
    assert client.get("/v1/jobs").json()["jobs"] == []


# --- the surface's own refusals, through the same seams ----------------------


def test_a_hyperparameter_override_the_surface_refuses_is_refused_on_the_recompute(
    client, tmp_path
):
    """An unknown key is refused and echoed back on the recompute, before the
    quote is computed: a key the user believes is in effect but is not is
    worse than a refusal."""
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/quotes",
        json={
            "dataset_id": ds,
            "base_model": "qwen3-4b",
            "hyperparameters": {"maxSteps": "5"},
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "unknown_hyperparameter"
    assert detail["unknown"] == ["maxSteps"]


def test_a_known_but_unsupported_key_is_refused_with_its_reason_on_launch(
    client, tmp_path
):
    """A setting the trainer supports but this product does not is refused
    with its reason (the same reason the surface publishes), never silently
    dropped or accepted."""
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "base_model": "qwen3-4b",
            "hyperparameters": {"wandb_project": "x"},
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "unsupported_hyperparameter"
    assert detail["name"] == "wandb_project"
    assert "reason" in detail
    assert client.get("/v1/jobs").json()["jobs"] == []


# --- a launch freezes the overrides, coerced --------------------------------


def test_a_launch_freezes_hyperparameter_overrides_coerced_to_the_schema_type(
    client, tmp_path
):
    """The advanced overrides are recorded in the job specification, coerced
    to the schema's types: the run says what it actually used, typed the way
    the trainer reads it."""
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds,
            "base_model": "qwen3-4b",
            "hyperparameters": {"learning_rate": "0.0001", "num_epochs": "5"},
        },
    )
    assert r.status_code == 201
    job = r.json()
    assert job["hyperparameters"] == {
        "learning_rate": 0.0001,
        "num_epochs": 5.0,
    }


# --- the two refusals that must never become overridable ---------------------


def test_a_mixed_thinking_dataset_stays_refused_despite_overrides(
    client, tmp_path
):
    """A dataset mixing reasoning traces with plain answers is ambiguous by
    construction: no control resolves it, so a launch carrying hyperparameter
    overrides is refused with the dataset's own code rather than let through."""
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {
                    "role": "assistant",
                    "content": (
                        f"<think>think {i}</think>answer {i}"
                        if i % 2
                        else f"plain answer {i}"
                    ),
                },
            ]
        }
        for i in range(12)
    ]
    p = jsonl(tmp_path, rows)
    with open(p, "rb") as f:
        client.post("/v1/datasets", files={"file": (p.name, f)})
    # Find the dataset id and wait for the verdict.
    ds = wait_validated(
        client, client.get("/v1/datasets").json()["datasets"][-1]["id"]
    )
    assert ds["status"] != "valid"
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds["id"],
            "base_model": "qwen3-4b",
            "hyperparameters": {"learning_rate": "0.0001"},
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "dataset_invalid"


def test_a_dataset_below_the_row_floor_stays_refused_despite_overrides(
    client, tmp_path
):
    """A dataset below the minimum usable row count has no correct value to
    override to: the launch is refused with `dataset_invalid` before any
    hyperparameter is considered."""
    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(3)])
    with open(p, "rb") as f:
        client.post("/v1/datasets", files={"file": (p.name, f)})
    ds = wait_validated(
        client, client.get("/v1/datasets").json()["datasets"][-1]["id"]
    )
    assert ds["status"] != "valid"
    r = client.post(
        "/v1/jobs",
        json={
            "dataset_id": ds["id"],
            "base_model": "qwen3-4b",
            "hyperparameters": {"micro_batch_size": "2"},
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "dataset_invalid"
