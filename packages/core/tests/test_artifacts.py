"""The deliverable and its kinds -- spec 006, issue #32.

The glossary change made executable: the artifact is the canonical deliverable
and the adapter is one kind of it. This module owns the vocabulary -- the
kinds, the method-to-kind derivation, and the per-kind loading instructions --
so a good test here asserts on what the derivation returns for each method,
and that the load path differs by kind. The one place the issue says the
answer must be defended: whether kind is stored or derived for existing rows.
The tests assert the derivation is total over the methods that exist and reads
a pre-method row (method None) as an adapter, which is the "no migration
surprises" clause.
"""

from __future__ import annotations

import pytest

from temper_core import artifacts
from temper_core.artifacts import (
    ARTIFACT_KIND_ADAPTER,
    ARTIFACT_KIND_FULL_MODEL,
    ARTIFACT_KIND_MERGED_MODEL,
    UnknownArtifactKind,
    UnknownMethod,
    kind_for,
    loading_instructions,
)


def test_every_method_maps_to_a_kind():
    # The three methods the predictor can describe, and the kind each produces.
    # qlora and lora both write a PEFT adapter; full fine-tuning rewrites the
    # whole model. This is the single definition of the derivation, so the
    # test reads it as the source of truth rather than re-spelling it.
    assert kind_for("qlora") == ARTIFACT_KIND_ADAPTER
    assert kind_for("lora") == ARTIFACT_KIND_ADAPTER
    assert kind_for("full") == ARTIFACT_KIND_FULL_MODEL


def test_a_pre_method_row_reads_as_an_adapter():
    # "Existing adapter artifacts are described correctly under the new model
    # without migration surprises": the rows written before the method column
    # existed are all QLoRA adapters (the trainer has only ever executed
    # qlora), so None must derive to adapter, never crash or guess.
    assert kind_for(None) == ARTIFACT_KIND_ADAPTER


def test_an_unknown_method_is_refused_not_guessed():
    # A method no kind is defined for must not be handed a fabricated kind: a
    # made-up kind would describe a download whose load path was invented.
    with pytest.raises(UnknownMethod):
        kind_for("sparkles")


def test_loading_instructions_differ_by_kind():
    # The concrete point of the issue: an adapter has to be applied to a base
    # model, a fully trained or merged model is complete on its own. If two
    # kinds ever shared instructions, the classification would be cosmetic.
    adapter = loading_instructions(ARTIFACT_KIND_ADAPTER)
    full = loading_instructions(ARTIFACT_KIND_FULL_MODEL)
    merged = loading_instructions(ARTIFACT_KIND_MERGED_MODEL)
    assert len({adapter, full, merged}) == 3
    assert "PeftModel" in adapter
    assert "base model" in adapter.lower()
    assert "from_pretrained" in full and "PeftModel" not in full
    assert "base model" not in full.lower() or "no base model" in full.lower()


def test_unknown_kind_has_no_loading_instructions():
    with pytest.raises(UnknownArtifactKind):
        loading_instructions("surprise")


def test_the_kind_vocabulary_is_stable():
    # The API, the download manifest and the interface share these strings;
    # renaming one is a deliberate act, and the tuple pins the set.
    assert set(artifacts.ARTIFACT_KINDS) == {
        ARTIFACT_KIND_ADAPTER,
        ARTIFACT_KIND_FULL_MODEL,
        ARTIFACT_KIND_MERGED_MODEL,
    }
    # A merged model is named but produced by no method yet (merging is an
    # artifact-time step, spec 011's territory): the vocabulary precedes it.
    assert ARTIFACT_KIND_MERGED_MODEL not in artifacts.KIND_BY_METHOD.values()


def test_the_adapter_member_names_are_the_canonical_pair():
    # What an adapter download consists of, defined once. The control plane
    # records these, the download serves them and the legacy-row fallback
    # derives them from this tuple -- one definition, read everywhere.
    assert artifacts.ADAPTER_MEMBER_NAMES == (
        "adapter_model.safetensors",
        "adapter_config.json",
    )
