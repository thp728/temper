"""Provenance manifest: generated from the job record (issue #70).

Spec 011 / issue #70. Nearly everything the manifest records already exists
on the job record; the job is to assemble, not to gather. The manifest
records base model and pinned revision, dataset fingerprint with counts, the
full configuration including overrides, the evaluation summary, the checkpoint
the result came from, and the licence obligations that propagate. It is
generated from the job record rather than assembled by hand, ships with the
artifact, fails loudly on a missing required field, and is readable by a
person as well as a machine.
"""

from __future__ import annotations

import pytest

from temper_core import artifacts
from temper_core.manifest import (
    MissingField,
    generate,
    render_text,
    to_pretty_json,
)

# Caller-supplied timestamp so the pure generate never reads the clock.
FIXED_TS = 1724330000.0


def gen(job, dataset=None):
    return generate(job, dataset, generated_at=FIXED_TS)


def _valid_job(**overrides):
    """A complete job record as the control plane stores it, shaped for manifest generation."""
    base = {
        "id": "job_test123",
        "created_at": 1000.0,
        "finished_at": 2000.0,
        "status": "complete",
        "base_model": "qwen3-4b",
        "base_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "method": "qlora",
        "hyperparameters": {
            "lora_r": 16,
            "learning_rate": 0.0002,
            "val_set_size": 0.05,
        },
        "overrides": [],
        "result": {
            "held_out_split": {
                "rows_in": 12,
                "rows_removed_duplicates": 0,
                "train_rows": 11,
                "held_out_rows": 1,
                "fraction": 0.05,
                "seed": 42,
            },
            "template_probe": {
                "ok": True,
                "training_ids": 27,
                "serialised_ids": 27,
            },
        },
        "best_checkpoint": {
            "step": 20,
            "basis": "best_held_out_loss",
            "reason": "Step 20 has the lowest held-out loss (0.39) of 3 retained checkpoint(s).",
            "held_out_loss": 0.39,
        },
        "artifact_record": {
            "members": [
                {
                    "name": "adapter_model.safetensors",
                    "key": "artifacts/job_test123/adapter_model.safetensors",
                },
                {
                    "name": "adapter_config.json",
                    "key": "artifacts/job_test123/adapter_config.json",
                },
            ],
            "bytes": 12345,
            "sha256": "abc123",
        },
    }
    base.update(overrides)
    return base


def _valid_dataset(**overrides):
    base = {
        "id": "ds_test1",
        "filename": "data.jsonl",
        "report": {"row_count": 12, "usable_rows": 12, "schema_type": "chat"},
    }
    base.update(overrides)
    return base


# --- required fields ----------------------------------------------------


def test_manifest_records_base_model_and_pinned_revision():
    job = _valid_job()
    ds = _valid_dataset()
    m = gen(job, ds)
    # Base model and pinned revision travel together; the manifest must carry both.
    assert m["base_model"] == "qwen3-4b"
    assert m["base_revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
    # The detailed info mirrors the same revision and licence.
    assert m["base_model_info"]["repo"] == "Qwen/Qwen3-4B"
    assert m["base_model_info"]["revision"] == m["base_revision"]
    assert m["job"]["id"] == "job_test123"


def test_manifest_records_dataset_fingerprint_with_counts():
    job = _valid_job()
    ds = _valid_dataset()
    m = gen(job, ds)
    # The held-out split is the fingerprint: dedup first, deterministic under seed.
    assert m["dataset"]["rows_in"] == 12
    assert m["dataset"]["train_rows"] == 11
    assert m["dataset"]["held_out_rows"] == 1
    assert m["dataset"]["seed"] == 42
    assert m["dataset"]["fraction"] == 0.05
    # Counts from the validation report travel too.
    assert m["dataset"]["row_count"] == 12
    assert m["dataset"]["filename"] == "data.jsonl"


def test_manifest_records_full_configuration_including_overrides():
    job = _valid_job(
        hyperparameters={"lora_r": 32, "learning_rate": 0.0001},
        overrides=[{"decision": "sequence length", "value": "2048"}],
    )
    m = gen(job, _valid_dataset())
    assert m["configuration"]["hyperparameters"]["lora_r"] == 32
    assert m["configuration"]["overrides"][0]["decision"] == "sequence length"
    assert m["configuration"]["method"] == "qlora"
    # The seed that fixes the split is part of the configuration's reproducibility.
    assert m["configuration"]["seed"] == 42


def test_manifest_records_evaluation_summary_and_checkpoint_the_result_came_from():
    job = _valid_job()
    m = gen(job, _valid_dataset())
    # Evaluation summary: the held-out split and the template probe.
    assert m["evaluation"]["held_out_split"]["held_out_rows"] == 1
    assert m["evaluation"]["template_probe"]["ok"] is True
    # The checkpoint the result came from is the stored choice, not a derivation.
    assert m["checkpoint"]["step"] == 20
    assert m["checkpoint"]["basis"] == "best_held_out_loss"
    assert "Step 20" in m["checkpoint"]["reason"]
    # The evaluation's best_checkpoint mirrors the checkpoint.
    assert m["evaluation"]["best_checkpoint"]["step"] == 20


def test_manifest_states_licence_obligations_explicitly():
    job = _valid_job()
    m = gen(job, _valid_dataset())
    obligations = m["base_model_info"]["license_obligations"]
    # Apache-2.0 obligations are stated explicitly, not left to a reader to guess.
    assert "Apache-2.0" in obligations
    assert (
        "retain" in obligations.lower() or "copyright" in obligations.lower()
    )
    # The licence itself is recorded, never guessed.
    assert m["base_model_info"]["license"] == "Apache-2.0"
    assert (
        m["base_model_info"]["license_url"]
        == "https://huggingface.co/Qwen/Qwen3-4B"
    )


def test_manifest_is_generated_from_the_run_record_rather_than_assembled_by_hand():
    # Changing the recorded held-out loss changes the manifest's checkpoint reason.
    job1 = _valid_job()
    job2 = _valid_job(
        best_checkpoint={
            "step": 30,
            "basis": "best_held_out_loss",
            "reason": "Step 30 is best",
            "held_out_loss": 0.3,
        }
    )
    assert gen(job1, _valid_dataset())["checkpoint"]["step"] == 20
    assert gen(job2, _valid_dataset())["checkpoint"]["step"] == 30
    # If a field can be supplied two ways, the recorded one wins: the job
    # carries a base_revision and the catalog carries the same; the job's
    # recorded revision is what the manifest publishes.
    job3 = _valid_job(base_revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    m3 = gen(job3, _valid_dataset())
    assert m3["base_revision"] == "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def test_manifest_includes_artifact_kind_and_loading():
    job = _valid_job(method="full")
    # Full fine-tuning produces a fully trained model, not an adapter.
    job["artifact_record"] = {
        "members": [
            {
                "name": "model.tar.gz",
                "key": "artifacts/job_test123/model.tar.gz",
            }
        ],
        "bytes": 999,
        "sha256": "xyz",
    }
    m = gen(job, _valid_dataset())
    assert m["artifact"]["kind"] == artifacts.ARTIFACT_KIND_FULL_MODEL
    assert (
        "full" in m["artifact"]["loading"].lower()
        or "model.tar.gz" in m["artifact"]["loading"]
        or "from_pretrained" in m["artifact"]["loading"]
    )
    assert m["kind"] == artifacts.ARTIFACT_KIND_FULL_MODEL


def test_a_missing_required_field_fails_rather_than_producing_a_placeholder():
    # base_revision missing -> MissingField, not "unknown"
    job = _valid_job()
    del job["base_revision"]
    with pytest.raises(MissingField) as exc:
        gen(job, _valid_dataset())
    assert exc.value.field == "base_revision"
    assert "unknown" not in str(exc.value).lower()

    # held_out_split missing -> MissingField
    job = _valid_job()
    del job["result"]["held_out_split"]
    with pytest.raises(MissingField) as exc:
        gen(job, _valid_dataset())
    assert "held_out_split" in exc.value.field

    # best_checkpoint missing -> MissingField
    job = _valid_job()
    del job["best_checkpoint"]
    with pytest.raises(MissingField) as exc:
        gen(job, _valid_dataset())
    assert "best_checkpoint" in exc.value.field

    # A field that is an empty string is as missing as one that is absent.
    job = _valid_job(base_revision="")
    with pytest.raises(MissingField) as exc:
        gen(job, _valid_dataset())
    assert exc.value.field == "base_revision"

    # A placeholder "unknown" is refused, not accepted as a licence.
    job = _valid_job()
    job["base_model"] = "unknown-model"
    # Force licence lookup to empty by using an unknown model id and no fallback
    with pytest.raises(MissingField) as exc:
        gen(job, _valid_dataset())
    assert exc.value.field in ("license", "base_model_repo", "base_model")


def test_manifest_is_readable_by_a_person_not_only_a_machine():
    m = gen(_valid_job(), _valid_dataset())
    text = render_text(m)
    # The Markdown contains headings a reviewer can skim, not just JSON.
    assert "# Provenance Manifest" in text
    assert "## Base model" in text
    assert "## Dataset" in text
    assert "## Configuration" in text
    assert "## Evaluation" in text
    assert "## Checkpoint" in text
    assert "## Artifact" in text
    # Human sentences, not just keys.
    assert "licence" in text.lower() or "license" in text.lower()
    assert "held-out" in text.lower()
    # The machine-readable JSON is pretty-printed.
    pretty = to_pretty_json(m)
    assert pretty.startswith("{\n")
    assert '  "base_model"' in pretty or '"base_model"' in pretty
    # But the pretty JSON alone is not the human document: the Markdown is.
    assert len(text.splitlines()) > 30


# --- delivery formats (issue #74) --------------------------------------------


def _job_with_delivery(**overrides):
    """A complete job whose `delivery` list records one produced format."""
    job = _valid_job()
    job["delivery"] = [
        {
            "format": "merged",
            "kind": "merged_model",
            "members": [
                {
                    "name": "model.safetensors",
                    "key": "artifacts/job_test123/model.safetensors",
                }
            ],
            "bytes": 2222,
            "sha256": "deadbeef",
        }
    ]
    job.update(overrides)
    return job


def test_manifest_describes_a_delivery_format_from_its_own_record():
    """The manifest for a delivery-format download describes that format's
    verified record -- the same generator, not a second, parallel description
    of the artifact (ADR-0054 / issue #74)."""
    job = _job_with_delivery()
    m = generate(
        job, _valid_dataset(), generated_at=FIXED_TS, delivery_format="merged"
    )
    assert m["artifact"]["kind"] == "merged_model"
    assert m["artifact"]["members"] == ["model.safetensors"]
    assert m["artifact"]["bytes"] == 2222
    assert m["artifact"]["sha256"] == "deadbeef"
    # Loading instruction is the merged model's, not the adapter's.
    assert "merged" in m["artifact"]["loading"].lower()
    # Top-level aliases follow the format's kind too.
    assert m["kind"] == "merged_model"


def test_the_canonical_manifest_still_describes_the_canonical_artifact():
    """Without a delivery_format, generation describes the canonical artifact
    exactly as before -- the delivery parameter is an addition, not a change."""
    m = gen(_valid_job(), _valid_dataset())
    assert m["artifact"]["kind"] == "adapter"
    assert m["kind"] == "adapter"


def test_a_delivery_format_without_a_record_is_a_missing_field():
    """The same rule as every manifest field: a format whose record is absent
    fails generation rather than producing a placeholder."""
    job = _valid_job()
    with pytest.raises(MissingField) as exc:
        generate(
            job,
            _valid_dataset(),
            generated_at=FIXED_TS,
            delivery_format="merged",
        )
    assert "delivery" in exc.value.field


def test_an_unknown_delivery_format_is_refused():
    from temper_core.delivery import UnknownDeliveryFormat

    with pytest.raises(UnknownDeliveryFormat):
        generate(
            _job_with_delivery(),
            _valid_dataset(),
            generated_at=FIXED_TS,
            delivery_format="nonsense",
        )
