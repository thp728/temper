"""Admitting a model from outside the catalog (Spec 009 / issue #58).

The probe is tested without network access: facts arrive through the fake
model-facts resolver the predictor's seam is pointed at, with a fixture for
each failure path -- no chat template, padding colliding with end-of-sequence,
mixture-of-experts, an unresolvable revision, an unresolvable licence, too
large for anything provisionable. Each drives exactly one finding, and the
suite asserts on the verdict and its reason, not on the mechanism.

The persistence-and-gate half is exercised over HTTP: a probe result is
stored, shown in `/v1/models`, and a job can be created against a model only
after a probe that passes -- a blocked probe refuses the launch with its
findings, and an unadmitted id refuses as unknown.
"""

from __future__ import annotations

import json

import pytest

from temper_control_plane import admission, fake_models, quote
from temper_core.models import ModelFacts

PINNED = "a" * 40


def facts(**overrides) -> ModelFacts:
    """A `ModelFacts` variant of the clean Qwen3-4B facts, overriding one
    field so each test drives exactly one finding."""
    base = dict(fake_models.QWEN3_4B_FACTS.__dict__)
    base.update(overrides)
    return ModelFacts(**base)


@pytest.fixture()
def resolver(monkeypatch):
    """Point the predictor's model-facts seam -- the one the probe reads
    through -- at a fake answering only the (repo, revision) pairs a test
    gives it, so no test reaches the network."""

    def set_(mapping: dict[tuple[str, str], ModelFacts]):
        monkeypatch.setattr(
            quote, "QUOTE_MODELS", fake_models.FakeModels(mapping)
        )

    return set_


def chat_rows(n=12):
    for i in range(n):
        yield {
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ]
        }


def upload(client, rows=12):
    data = ("\n".join(json.dumps(r) for r in chat_rows(rows))).encode("utf-8")
    r = client.post("/v1/datasets", files={"file": ("d.jsonl", data)})
    assert r.status_code == 202
    from helpers import wait_validated

    return wait_validated(client, r.json()["id"])["id"]


def probe(client, repo="org/model", revision=PINNED):
    r = client.post(
        "/v1/models/probe", json={"repo": repo, "revision": revision}
    )
    assert r.status_code == 201, r.text
    return r.json()


# --- the probe reads the same facts the predictor does -----------------------


def test_the_probe_reads_facts_through_the_predictors_seam():
    """The criterion verbatim: the probe and the predictor resolve through one
    seam, so the two cannot drift. This is the seam itself, not a copy."""
    assert admission._resolver() is quote.models_for_quote()


def test_probing_a_pinned_revision_persists_and_shows_the_verdict(
    client, resolver
):
    resolver({("org/model", PINNED): facts()})
    record = probe(client)

    assert record["id"].startswith("m_")
    assert record["repo"] == "org/model"
    assert record["revision"] == PINNED
    assert record["probe"]["ok"] is True
    assert record["probe"]["verdict"] == "usable"
    assert record["probe"]["findings"] == []
    assert record["probe"]["memory"]["fits"] is True

    # The result is shown wherever the models are listed, persisted not
    # recomputed on read.
    listing = client.get("/v1/models").json()
    admitted_ids = [a["id"] for a in listing["admitted"]]
    assert record["id"] in admitted_ids


def test_reprobing_the_same_pinned_reference_is_idempotent(client, resolver):
    resolver({("org/model", PINNED): facts()})
    first = probe(client)
    second = probe(client)
    assert second["id"] == first["id"]


def test_an_unpinned_revision_is_refused(client, resolver):
    r = client.post(
        "/v1/models/probe", json={"repo": "org/model", "revision": "main"}
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unpinned_revision"


# --- each failure path gets its own fixture ----------------------------------


def test_a_missing_chat_template_blocks_and_refuses_job_creation(
    client, resolver
):
    resolver({("org/model", PINNED): facts(has_chat_template=False)})
    record = probe(client)
    assert record["probe"]["ok"] is False
    codes = {f["code"] for f in record["probe"]["findings"]}
    assert "missing_chat_template" in codes

    ds_id = upload(client)
    r = client.post(
        "/v1/jobs", json={"dataset_id": ds_id, "base_model": record["id"]}
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "model_probe_blocked"
    assert any(
        f["code"] == "missing_chat_template" for f in detail["findings"]
    )


def test_padding_colliding_with_eos_blocks(client, resolver):
    resolver({("org/model", PINNED): facts(pad_eos_distinct=False)})
    record = probe(client)
    assert record["probe"]["ok"] is False
    codes = {f["code"] for f in record["probe"]["findings"]}
    assert "padding_collides_with_eos" in codes


def test_a_mixture_of_experts_model_warns_but_is_launchable(client, resolver):
    resolver({("org/model", PINNED): facts(is_moe=True)})
    record = probe(client)
    assert record["probe"]["ok"] is True
    assert record["probe"]["verdict"] == "usable_with_warnings"
    codes = {f["code"] for f in record["probe"]["findings"]}
    assert "untested_architecture" in codes

    ds_id = upload(client)
    r = client.post(
        "/v1/jobs", json={"dataset_id": ds_id, "base_model": record["id"]}
    )
    assert r.status_code == 201, r.text


def test_an_unresolvable_licence_warns(client, resolver):
    resolver({("org/model", PINNED): facts(license="")})
    record = probe(client)
    assert record["probe"]["ok"] is True
    codes = {f["code"] for f in record["probe"]["findings"]}
    assert "license_unresolvable" in codes


def test_a_model_too_large_for_anything_provisionable_warns_with_arithmetic(
    client, resolver
):
    resolver(
        {
            ("org/model", PINNED): facts(
                hidden_size=16384,
                num_hidden_layers=128,
                num_attention_heads=128,
                intermediate_size=65536,
            )
        }
    )
    record = probe(client)
    assert record["probe"]["ok"] is True
    memory = record["probe"]["memory"]
    assert memory["fits"] is False
    assert memory["peak_gb"] > memory["capacity_gb"]


def test_a_revision_that_does_not_resolve_is_persisted_as_blocked(
    client, resolver
):
    # The resolver knows no (repo, revision) pair, so resolution itself fails
    # -- the seam's refusal, surfaced as a blocking probe result.
    resolver({})
    record = probe(client, repo="org/ghost", revision=PINNED)
    assert record["probe"]["ok"] is False
    codes = {f["code"] for f in record["probe"]["findings"]}
    assert "revision_unresolvable" in codes

    ds_id = upload(client)
    r = client.post(
        "/v1/jobs", json={"dataset_id": ds_id, "base_model": record["id"]}
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "model_probe_blocked"


# --- the persistence-and-gate half -------------------------------------------


def test_an_admitted_model_can_be_quoted_and_launched(client, resolver):
    resolver({("org/model", PINNED): facts()})
    record = probe(client)
    ds_id = upload(client)

    quote_r = client.get(
        "/v1/quotes", params={"dataset_id": ds_id, "base_model": record["id"]}
    )
    assert quote_r.status_code == 200
    assert quote_r.json() is not None

    r = client.post(
        "/v1/jobs", json={"dataset_id": ds_id, "base_model": record["id"]}
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["base_model"] == record["id"]
    assert body["base_revision"] == PINNED


def test_an_unadmitted_id_is_refused_as_unknown(client):
    ds_id = upload(client)
    r = client.post(
        "/v1/jobs", json={"dataset_id": ds_id, "base_model": "m_nope"}
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unknown_model"


def test_a_catalog_model_needs_no_probe_and_still_launches(client):
    """The catalog stays the recommended, tested path: it needs no probe,
    because the probe is the testing done in front of the user for anything
    else."""
    ds_id = upload(client)
    r = client.post("/v1/jobs", json={"dataset_id": ds_id})
    assert r.status_code == 201
