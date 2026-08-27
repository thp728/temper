"""The model-facts seam -- spec 005, issue #48.

`FakeModels` is exercised the same way `FakeProvider` is: constructed with
the facts a case needs, refusing anything it was not given. `HuggingFaceModels`
is exercised with `urllib.request.urlopen` monkeypatched locally -- the one
place in this file that undoes `conftest.no_real_models`'s narrower guard,
because this is the file that owns proving the parsing is correct.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from temper_control_plane import models
from temper_control_plane.fake_models import CATALOG_MODELS, FakeModels
from temper_core.models import ModelFacts

# --- FakeModels ---------------------------------------------------------------


def test_fake_models_resolves_exactly_what_it_was_given():
    facts = ModelFacts(
        architecture="qwen3",
        hidden_size=1,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=1,
        intermediate_size=1,
        vocab_size=1,
        tie_word_embeddings=True,
        is_moe=False,
        has_chat_template=False,
        pad_eos_distinct=False,
        context_length=1,
        license="",
    )
    fake = FakeModels({("org/repo", "deadbeef"): facts})
    assert fake.resolve("org/repo", "deadbeef") is facts


def test_fake_models_refuses_an_unknown_reference():
    fake = FakeModels({})
    with pytest.raises(ValueError, match="no facts for"):
        fake.resolve("org/repo", "deadbeef")


def test_catalog_models_covers_every_catalog_entry():
    """The double the test suite runs against by default (`conftest.py`)
    knows about both curated models -- if the catalog grows a third and
    nobody seeds it here, every test touching `/v1/models` should fail
    loudly rather than silently resolving nothing for it."""
    from temper_core import catalog

    for m in catalog.CATALOG.values():
        assert (m.repo, m.revision) in CATALOG_MODELS


# --- HuggingFaceModels ----------------------------------------------------


def _responses(config: dict, tokenizer_config: dict | None, api: dict | None):
    """A stand-in for `urllib.request.urlopen` that answers by URL suffix."""

    def opener(url, timeout=10):
        if url.endswith("tokenizer_config.json"):
            if tokenizer_config is None:
                import urllib.error

                raise urllib.error.HTTPError(url, 404, "not found", {}, None)
            body = tokenizer_config
        elif url.endswith("config.json"):
            body = config
        else:
            body = api or {}
        resp = Mock()
        resp.read.return_value = json.dumps(body).encode()
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        return resp

    return opener


def test_resolves_facts_from_a_published_config(monkeypatch):
    monkeypatch.setattr(
        models.urllib.request,
        "urlopen",
        _responses(
            config={
                "model_type": "qwen3",
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 9728,
                "vocab_size": 151936,
                "tie_word_embeddings": True,
                "max_position_embeddings": 40960,
            },
            tokenizer_config={
                "chat_template": "{{ messages }}",
                "pad_token": "<|endoftext|>",
                "eos_token": "<|im_end|>",
            },
            api={"cardData": {"license": "apache-2.0"}},
        ),
    )
    facts = models.HuggingFaceModels().resolve("Qwen/Qwen3-4B", "deadbeef")
    assert facts.architecture == "qwen3"
    assert facts.hidden_size == 2560
    assert facts.is_moe is False
    assert facts.has_chat_template is True
    assert facts.pad_eos_distinct is True
    assert facts.context_length == 40960
    assert facts.license == "apache-2.0"


def test_missing_pad_token_reads_as_not_distinct(monkeypatch):
    monkeypatch.setattr(
        models.urllib.request,
        "urlopen",
        _responses(
            config={
                "model_type": "qwen3",
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 9728,
                "vocab_size": 151936,
            },
            tokenizer_config={"eos_token": "<|im_end|>"},
            api=None,
        ),
    )
    facts = models.HuggingFaceModels().resolve("Qwen/Qwen3-4B", "deadbeef")
    assert facts.pad_eos_distinct is False
    assert facts.license == ""


def test_moe_config_key_is_detected(monkeypatch):
    monkeypatch.setattr(
        models.urllib.request,
        "urlopen",
        _responses(
            config={
                "model_type": "some-moe",
                "hidden_size": 1,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 1,
                "intermediate_size": 1,
                "vocab_size": 1,
                "num_experts": 8,
            },
            tokenizer_config={},
            api=None,
        ),
    )
    facts = models.HuggingFaceModels().resolve("org/moe-model", "deadbeef")
    assert facts.is_moe is True


def test_missing_config_is_refused(monkeypatch):
    import urllib.error

    def opener(url, timeout=10):
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)

    monkeypatch.setattr(models.urllib.request, "urlopen", opener)
    with pytest.raises(ValueError, match="no config.json"):
        models.HuggingFaceModels().resolve("org/ghost", "deadbeef")
