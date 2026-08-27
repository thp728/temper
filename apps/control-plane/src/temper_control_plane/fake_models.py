"""A model-facts resolver that reaches no network.

Kept in the package rather than in a test file, matching `fake_provider.py`:
every ticket that touches the predictor tests against the same double
instead of growing its own. Constructed with the facts a case needs, per
spec 005's testing decisions -- there is no canned single scenario here
because, unlike the provider, there is no "one demo job" for facts to
describe.

`CATALOG_MODELS` seeds it with both curated models' real facts, each
independently confirmed against the repository's own published
`config.json` / `tokenizer_config.json` at the pinned revision -- the same
files `HuggingFaceModels` reads live. Two of these numbers are anchors, not
just plausible data: Qwen3-4B's facts predict 33,030,144 trainable
parameters at LoRA r=16, exactly what the real run
(`apps/trainer/README.md`, 2026-08-19) returned, and its predicted peak
lands within `temper_core.memory.PEAK_TOLERANCE` of the 5.31 GB that run
measured. `test_memory.py` pins both.
"""

from __future__ import annotations

from temper_core import catalog
from temper_core.models import ModelFacts

# Qwen3-4B, confirmed against https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json
# and tokenizer_config.json at the pinned revision, 2026-08-27.
QWEN3_4B_FACTS = ModelFacts(
    architecture="qwen3",
    hidden_size=2560,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=9728,
    vocab_size=151936,
    tie_word_embeddings=True,
    is_moe=False,
    has_chat_template=True,
    pad_eos_distinct=True,  # pad "<|endoftext|>" vs eos "<|im_end|>"
    context_length=40960,
    license="apache-2.0",
)

# Qwen3-8B, confirmed the same way. Reuses the wiki's by-hand arithmetic
# (`wiki/foundations.md`): the parameter count these facts derive is
# 8,190,427,136 -- exactly the model's advertised 8.19B.
QWEN3_8B_FACTS = ModelFacts(
    architecture="qwen3",
    hidden_size=4096,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=12288,
    vocab_size=151936,
    tie_word_embeddings=False,
    is_moe=False,
    has_chat_template=True,
    pad_eos_distinct=True,
    context_length=40960,
    license="apache-2.0",
)

# Keyed by (repo, revision) exactly as `Models.resolve` is called, so a
# pinned-revision change that nobody updates here fails loudly instead of
# silently answering with a stale model's facts.
CATALOG_MODELS: dict[tuple[str, str], ModelFacts] = {
    (catalog.CATALOG["qwen3-4b"].repo, catalog.CATALOG["qwen3-4b"].revision): (
        QWEN3_4B_FACTS
    ),
    (catalog.CATALOG["qwen3-8b"].repo, catalog.CATALOG["qwen3-8b"].revision): (
        QWEN3_8B_FACTS
    ),
}


class FakeModels:
    """Resolves exactly the `(repo, revision)` pairs it was given. Refuses
    anything else with a message naming what it does know, rather than
    guessing or reaching for a network no test should depend on."""

    def __init__(self, facts: dict[tuple[str, str], ModelFacts]) -> None:
        self._facts = dict(facts)

    def resolve(self, repo: str, revision: str) -> ModelFacts:
        try:
            return self._facts[(repo, revision)]
        except KeyError:
            raise ValueError(
                f"FakeModels has no facts for ('{repo}', '{revision}'). "
                f"Known: {sorted(self._facts)}"
            ) from None


def catalog_models() -> FakeModels:
    """The double seeded with every curated model's real facts -- what
    `main.MODELS` is replaced with for the whole test suite, so `/v1/models`
    behaves identically to production without reaching the network."""
    return FakeModels(CATALOG_MODELS)
