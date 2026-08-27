"""The model-facts seam: resolves a repository reference to `ModelFacts`.

Mirrors why the compute-provider seam (`provider.py`) exists and where it
lives. The predictor and a later compatibility probe (spec 009) must read
the *same* facts about a model -- two implementations would drift, and the
drift would surface as a job the probe passed and the predictor mispriced.
Putting the seam here rather than in `temper_core` is what
`packages/core/pyproject.toml` requires: resolving an arbitrary repository's
facts means reading its published `config.json` and `tokenizer_config.json`
over the network, and the core package carries zero I/O.

Two published files answer everything `ModelFacts` asks for except licence,
which Hugging Face reports separately, keyed to the same pinned revision:

* `config.json` -- architecture, dimensions, mixture-of-experts, context
  length.
* `tokenizer_config.json` -- chat-template presence, whether the pad and
  end-of-sequence tokens are distinct.
* the models API, pinned to the revision -- licence.

All three are public, unauthenticated GET requests. Unlike the compute
provider, resolving a model's facts costs nothing and cannot reach a billing
account, so there is no fake-by-default switch here -- only the double
`fake_models.py` supplies for tests that must not depend on network
reachability.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from temper_core.models import ModelFacts, Models

_TIMEOUT_S = 10

_RAW_URL = "https://huggingface.co/{repo}/raw/{revision}/{path}"
_API_URL = "https://huggingface.co/api/models/{repo}/revision/{revision}"

# Config keys that name a mixture-of-experts model across the families this
# platform is likely to see. Presence of any, not its value, is what decides
# `is_moe` -- a MoE config with zero experts is not a shape any real repo
# publishes.
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


def _get_json(url: str) -> dict[str, Any] | None:
    """`url`'s JSON body, or None if the resource does not exist.

    Any other HTTP failure (auth, rate limit, a 5xx) is raised rather than
    read as "absent" -- absence and failure call for different messages, and
    conflating them is the same mistake the provider seam's readiness check
    exists to avoid.
    """
    try:
        # url is built from _RAW_URL/_API_URL's own fixed https:// scheme,
        # never from user input.
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _token_text(value: Any) -> str | None:
    """A tokenizer special token's text, whether the field is a bare string
    or the `{"content": ...}` form some tokenizers use."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return content if isinstance(content, str) else None
    return None


def _license(repo: str, revision: str) -> str:
    """Best-effort licence lookup. Empty string, not a guess, when the API
    carries none -- a wrong licence is worse than a visibly missing one."""
    info = _get_json(_API_URL.format(repo=repo, revision=revision)) or {}
    card = info.get("cardData") or {}
    if isinstance(card.get("license"), str):
        return card["license"]
    for tag in info.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag[len("license:") :]
    return ""


class HuggingFaceModels:
    """Resolves facts by reading a repository's published config at its
    pinned revision. Requires no credential: every file it reads is public."""

    def resolve(self, repo: str, revision: str) -> ModelFacts:
        config = _get_json(
            _RAW_URL.format(repo=repo, revision=revision, path="config.json")
        )
        if config is None:
            raise ValueError(
                f"'{repo}' at revision '{revision}' has no config.json; "
                f"cannot resolve its facts."
            )
        tokenizer_config = (
            _get_json(
                _RAW_URL.format(
                    repo=repo, revision=revision, path="tokenizer_config.json"
                )
            )
            or {}
        )

        pad = _token_text(tokenizer_config.get("pad_token"))
        eos = _token_text(tokenizer_config.get("eos_token"))

        return ModelFacts(
            architecture=config.get("model_type", "unknown"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config.get(
                "num_key_value_heads", config["num_attention_heads"]
            ),
            head_dim=config.get(
                "head_dim",
                config["hidden_size"] // config["num_attention_heads"],
            ),
            intermediate_size=config["intermediate_size"],
            vocab_size=config["vocab_size"],
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            is_moe=any(config.get(k) for k in _MOE_KEYS),
            has_chat_template="chat_template" in tokenizer_config,
            # Absence reads as "not distinct" rather than "unknown": a
            # tokenizer with no declared pad token commonly reuses EOS for
            # padding, which is exactly the collapsed case this field names.
            pad_eos_distinct=bool(pad) and bool(eos) and pad != eos,
            context_length=config.get("max_position_embeddings", 0),
            license=_license(repo, revision),
        )


def new_models() -> Models:
    """The default resolver.

    Real unless `TEMPER_FAKE_PROVIDER` is set. Resolving a model's facts
    carries no billing risk the way provisioning does, but `/v1/models` is
    called on every load of the model-choice screen, and the browser
    journeys (`apps/web/e2e`) exist to run with no dependency on network
    reachability at all -- the same reason they set that flag for the
    provider. Reusing it here rather than adding a second flag keeps "this
    process cannot reach anything external" a single switch. The test suite
    goes further still and replaces `main.MODELS` outright with
    `fake_models.catalog_models()` (`conftest.no_real_models`), so pytest
    never even routes through this function.
    """
    from . import config

    if config.FAKE_PROVIDER:
        from .fake_models import catalog_models

        return catalog_models()
    return HuggingFaceModels()
