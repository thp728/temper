"""The model-facts data contract.

Spec 005's `models` seam resolves a repository reference and a pinned
revision to the facts that decide whether a job fits a card: parameter
count, dimensions, architecture family, mixture-of-experts, chat-template
presence, whether padding and end-of-sequence are distinct tokens, context
length and licence. `temper_core.memory` -- the peak-memory arithmetic -- and
a later compatibility probe both read `ModelFacts`, so the shape lives here
rather than beside either reader.

Resolving the facts of an arbitrary repository is I/O: it means reading a
published `config.json` and `tokenizer_config.json` over the network, which
this package's own rule forbids (see `packages/core/pyproject.toml`). That
work -- the real implementation and its double -- lives in
`apps/control-plane/src/temper_control_plane/models.py`, matching where the
compute-provider seam already lives for the same reason. What belongs here
is only the shape both sides agree on, plus the one piece of arithmetic that
needs nothing but the shape: turning dimensions into a parameter count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelFacts:
    """Everything the predictor and a compatibility probe need to know about
    one model, at one pinned revision.

    Dimensions name the seven linear layers a transformer block holds, per
    `docs/research-reports/report-b.md` and the wiki's from-`config.json`
    arithmetic: `q_proj`/`o_proj` are `hidden_size <-> num_attention_heads *
    head_dim`, `k_proj`/`v_proj` are `hidden_size <-> num_key_value_heads *
    head_dim` (grouped-query attention means these differ from the query
    heads), and `gate_proj`/`up_proj`/`down_proj` are `hidden_size <->
    intermediate_size`. Together with `num_hidden_layers` and
    `tie_word_embeddings`, that is everything `total_params` and
    `temper_core.memory.trainable_params` need -- and nothing else, because a
    field neither reads is a field this contract does not need to carry.

    Mixture-of-experts models carry `num_experts` and an optional
    `moe_intermediate_size`: the MLP (gate/up/down) is replicated per expert
    and memory scales with **total** rather than active parameters
    (ADR-0056). The predictor must price the model as it is stored, not as it
    is routed -- predicting on active would under-price and hand the user an
    OOM minutes into a paid machine (ADR-0043).
    """

    architecture: str  # config.json's model_type, e.g. "qwen3"
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool
    is_moe: bool
    has_chat_template: bool
    pad_eos_distinct: bool
    context_length: int
    license: str
    num_experts: int = 0
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None

    @property
    def params(self) -> int:
        """Total parameter count, from dimensions alone.

        Verified against two real configs: Qwen3-4B's arithmetic lands at
        4.02B against a catalog figure of 4.0B, and Qwen3-8B's lands on
        8,190,427,136 -- exactly the model's advertised 8.19B. See
        `wiki/foundations.md` in the private vault for the by-hand version
        this reproduces.

        For mixture-of-experts models the MLP is replicated per expert and
        the count scales with **total** parameters (all experts stored), not
        active parameters per token -- a MoE with 8 experts holds 8× the MLP
        weights even when only 2 are routed per token. When `is_moe` is set
        but no expert count was supplied (legacy fixtures), the count
        defaults to 8 (Mixtral-like) so the total is visibly larger than the
        dense equivalent and the memory predictor cannot silently under-price
        it.
        """
        q_out = self.num_attention_heads * self.head_dim
        kv_out = self.num_key_value_heads * self.head_dim
        attn_per_layer = (
            self.hidden_size * q_out  # q_proj
            + self.hidden_size * kv_out  # k_proj
            + self.hidden_size * kv_out  # v_proj
            + q_out * self.hidden_size  # o_proj
        )
        mlp_intermediate = (
            self.moe_intermediate_size
            if self.moe_intermediate_size is not None
            else self.intermediate_size
        )
        mlp_per_expert = (
            self.hidden_size * mlp_intermediate  # gate_proj
            + self.hidden_size * mlp_intermediate  # up_proj
            + mlp_intermediate * self.hidden_size  # down_proj
        )
        if self.is_moe:
            n_exp = self.num_experts if self.num_experts > 0 else 8
            mlp_per_layer = n_exp * mlp_per_expert
        else:
            mlp_per_layer = mlp_per_expert
        per_layer = attn_per_layer + mlp_per_layer
        embeddings = self.vocab_size * self.hidden_size
        if not self.tie_word_embeddings:
            embeddings *= 2  # separate input and output tables
        return per_layer * self.num_hidden_layers + embeddings

    @property
    def active_params(self) -> int:
        """Parameters active per forward pass.

        For dense models this equals `params`. For MoE it is the
        routing-selected subset: attention plus `num_experts_per_tok` experts'
        MLP. The predictor never prices on this -- memory scales with total --
        but it is the honest figure to show why predicting on active would be
        wrong, and the probe's message names it.
        """
        if not self.is_moe:
            return self.params
        q_out = self.num_attention_heads * self.head_dim
        kv_out = self.num_key_value_heads * self.head_dim
        attn_per_layer = (
            self.hidden_size * q_out
            + self.hidden_size * kv_out
            + self.hidden_size * kv_out
            + q_out * self.hidden_size
        )
        mlp_intermediate = (
            self.moe_intermediate_size
            if self.moe_intermediate_size is not None
            else self.intermediate_size
        )
        mlp_per_expert = (
            self.hidden_size * mlp_intermediate
            + self.hidden_size * mlp_intermediate
            + mlp_intermediate * self.hidden_size
        )
        n_active = (
            self.num_experts_per_tok
            if self.num_experts_per_tok is not None
            and self.num_experts_per_tok > 0
            else 2
        )
        per_layer = attn_per_layer + n_active * mlp_per_expert
        embeddings = self.vocab_size * self.hidden_size
        if not self.tie_word_embeddings:
            embeddings *= 2
        return per_layer * self.num_hidden_layers + embeddings


class Models(Protocol):
    """Resolves a model reference to its facts. Implemented in
    `apps/control-plane`, where the network I/O that answering it requires is
    allowed to live."""

    def resolve(self, repo: str, revision: str) -> ModelFacts:
        """`repo` and `revision` name a Hugging Face repository and a pinned
        commit. Raises when the reference cannot be resolved; never guesses."""
        ...
