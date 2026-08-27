"""`ModelFacts` -- the model-facts seam's data contract.

The seam's I/O half (resolving an arbitrary repo) lives in
`apps/control-plane`, per `packages/core/pyproject.toml`'s no-I/O rule; what
belongs here is the shape itself and the one derivation it can do
unassisted -- turning dimensions into a parameter count.
"""

from __future__ import annotations

from temper_core.models import ModelFacts

QWEN3_4B = ModelFacts(
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
    pad_eos_distinct=True,
    context_length=40960,
    license="apache-2.0",
)


def test_params_matches_the_catalogs_advertised_size():
    # 4,022,331,400, computed -- close enough to the catalog's rounded 4.0B
    # that a wildly wrong dimension would visibly miss it.
    assert abs(QWEN3_4B.params - 4_000_000_000) / 4_000_000_000 < 0.05


def test_tied_embeddings_are_counted_once():
    untied = ModelFacts(**{**QWEN3_4B.__dict__, "tie_word_embeddings": False})
    assert untied.params > QWEN3_4B.params
    assert untied.params - QWEN3_4B.params == QWEN3_4B.vocab_size * (
        QWEN3_4B.hidden_size
    )
