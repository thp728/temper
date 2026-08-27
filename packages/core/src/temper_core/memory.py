"""The peak-VRAM predictor: arithmetic over a model's own facts.

Replaces the per-catalog-entry `est_peak_vram_gb` that spec 005 deletes. A
number measured for one model and extrapolated by parameter ratio for
another cannot describe a model the catalog has never seen; this computes
the same figure from `temper_core.models.ModelFacts` for any model,
including the two catalog ones, which now resolve through the same seam
everything else does.

**The model is the published one, not an invention** -- four memory pools
plus a calibrated fixed cost, per `docs/research-reports/report-b.md` §4.1:

* **Weights** = total params x bytes/param, by dtype: 0.5 for NF4 (QLoRA's
  frozen base), 2.0 for bf16 (LoRA and full fine-tune).
* **Gradients** = trainable params x 2 bytes (bf16). LoRA/QLoRA gradients
  flow only to the adapter; full fine-tuning's flow to every weight, so
  "trainable" there is every parameter.
* **Optimizer state** = trainable params x 12 bytes -- AdamW's fp32 master
  copy plus its two running-average moments, mixed precision.
* **Activations** = `2 x seq_len x micro_batch x hidden_size x
  num_hidden_layers` bytes, the Korthikanti et al. (arXiv 2205.05198) formula
  collapsed toward its checkpointed limit -- correct here because
  `gradient_checkpointing: true` is a default-locked entrypoint setting
  (`apps/trainer/entrypoint.py`), never a choice this predictor has to infer.

Trainable-parameter counting is exact, not approximate: LoRA at rank `r`
targets all seven linear layers (report-b.md's `all-linear`), each
contributing `r x (in_features + out_features)` per layer. Verified against
both real Qwen3 configs -- Qwen3-4B gives 33,030,144 at r=16, Qwen3-8B gives
43,646,976, and both match the by-hand arithmetic in the private vault's
`wiki/lora-and-peft.md` exactly.

**The fixed cost is calibrated, not derived.** The four pools above sum to
roughly 2.85 GB for the one configuration this project has actually measured
-- Qwen3-4B, QLoRA, r=16, all-linear, sequence length 2048, single L4 -- and
the run's own peak (`torch.cuda.max_memory_allocated()`) was 5.31 GB
(2026-08-19, `apps/trainer/README.md`). The residual, roughly 2.46 GB, is
CUDA context, cuBLAS/cuDNN workspace, bitsandbytes' NF4 quantisation state
and allocator fragmentation -- real costs the four-pool arithmetic does not
model, none of them broken out further because there is exactly one real run
to attribute them against. `FIXED_OVERHEAD_GB` rounds that residual up
slightly rather than fitting it exactly, so the anchor is inside
`PEAK_TOLERANCE` rather than reproduced to the decimal -- the honest
position for a constant with a sample size of one. It should shrink, and the
tolerance should tighten, as more runs are recorded (spec 005's calibration
clause).
"""

from __future__ import annotations

from dataclasses import dataclass

from temper_core.models import ModelFacts

# The seven linear layers `all-linear` LoRA targets -- the complete set of
# learned matrices in a transformer block, per report-b.md and
# wiki/foundations.md. Not configurable: `apps/trainer/entrypoint.py`
# hard-codes the same seven, and a predictor that targeted a different set
# would price a job the trainer does not run.
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Bytes stored per parameter, by training method. NF4 4-bit packs roughly
# half a byte; bf16 (LoRA's unquantised base, and full fine-tuning) is two.
WEIGHT_BYTES_PER_PARAM: dict[str, float] = {
    "qlora": 0.5,
    "lora": 2.0,
    "full": 2.0,
}

GRADIENT_BYTES_PER_PARAM = 2.0  # bf16
OPTIMIZER_BYTES_PER_PARAM = 12.0  # AdamW: fp32 master (4) + m (4) + v (4)

# See the module docstring: the residual between the four-pool sum and the
# one measured anchor (5.31 GB, Qwen3-4B QLoRA). Rounded up slightly rather
# than fitted exactly.
FIXED_OVERHEAD_GB = 2.5

# How far a prediction may sit from a measured anchor and still be called
# correct. Generous on purpose: the fixed overhead above is calibrated from a
# single run, and a tolerance narrow enough to look precise would misstate
# how much evidence backs it. Tightens as more anchors accumulate.
PEAK_TOLERANCE = 0.15

BYTES_PER_GB = 1e9  # decimal, matching torch.cuda.max_memory_allocated() / 1e9


def trainable_params(facts: ModelFacts, lora_r: int) -> int:
    """Exact trainable-parameter count for all-linear LoRA/QLoRA at rank
    `lora_r` -- the only targeting this product trains with.

    `r x (in_features + out_features)` per targeted matrix, summed over the
    seven in `TARGET_MODULES` and every layer. Grouped-query attention means
    `k_proj`/`v_proj` use `num_key_value_heads`, not `num_attention_heads`,
    for their output width -- getting that wrong is the usual way this
    arithmetic goes quietly off by the GQA ratio.
    """
    q_out = facts.num_attention_heads * facts.head_dim
    kv_out = facts.num_key_value_heads * facts.head_dim
    per_layer = lora_r * (
        (facts.hidden_size + q_out)  # q_proj
        + (facts.hidden_size + kv_out)  # k_proj
        + (facts.hidden_size + kv_out)  # v_proj
        + (q_out + facts.hidden_size)  # o_proj
        + (facts.hidden_size + facts.intermediate_size)  # gate_proj
        + (facts.hidden_size + facts.intermediate_size)  # up_proj
        + (facts.intermediate_size + facts.hidden_size)  # down_proj
    )
    return per_layer * facts.num_hidden_layers


@dataclass(frozen=True)
class PeakMemory:
    """The predicted peak, broken into the pools it was summed from.

    The breakdown is not an implementation detail hidden from callers: spec
    005's whole premise is that a prediction shows the arithmetic that
    produced it, not just the answer.
    """

    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    overhead_gb: float
    total_gb: float
    trainable_params: int


def predict_peak(
    facts: ModelFacts,
    *,
    method: str,
    lora_r: int,
    sequence_len: int,
    micro_batch_size: int,
    device_count: int = 1,
) -> PeakMemory:
    """The predicted **per-device** peak VRAM for training `facts` with
    `method` across `device_count` devices.

    A pure function of the model's facts and the run's shape -- no I/O, so
    it is exercised the same way whether `facts` came from a live resolve or
    a test's double. `method` selects both the weight dtype and what
    "trainable" means: full fine-tuning trains every parameter, so its
    gradients and optimizer state are sized on `facts.params` rather than
    the LoRA adapter.

    `device_count` only changes the arithmetic for `method="full"`: spike 6
    proved FSDP FULL_SHARD divides weights, gradients and optimizer state
    evenly across ranks for a full fine-tune, so the per-device share of
    those three pools shrinks as devices are added. LoRA and QLoRA have never
    run sharded, and their trainable set is small enough already that
    sharding it would not plausibly change whether a job fits -- so for them
    `device_count` is accepted (the selection search calls every method the
    same way) but changes nothing: more devices simply replicate the same
    per-device footprint. Activations and the fixed overhead never shard
    either way, because each device still runs its own micro-batch.
    """
    if method not in WEIGHT_BYTES_PER_PARAM:
        raise ValueError(
            f"unknown method {method!r}; expected one of "
            f"{sorted(WEIGHT_BYTES_PER_PARAM)}"
        )
    if device_count < 1:
        raise ValueError(f"device_count must be >= 1, got {device_count}")
    trainable = (
        facts.params if method == "full" else trainable_params(facts, lora_r)
    )
    weights_gb = facts.params * WEIGHT_BYTES_PER_PARAM[method] / BYTES_PER_GB
    gradients_gb = trainable * GRADIENT_BYTES_PER_PARAM / BYTES_PER_GB
    optimizer_gb = trainable * OPTIMIZER_BYTES_PER_PARAM / BYTES_PER_GB
    if method == "full" and device_count > 1:
        weights_gb /= device_count
        gradients_gb /= device_count
        optimizer_gb /= device_count
    activations_gb = (
        2
        * sequence_len
        * micro_batch_size
        * facts.hidden_size
        * facts.num_hidden_layers
        / BYTES_PER_GB
    )
    total_gb = (
        weights_gb
        + gradients_gb
        + optimizer_gb
        + activations_gb
        + FIXED_OVERHEAD_GB
    )
    return PeakMemory(
        weights_gb=weights_gb,
        gradients_gb=gradients_gb,
        optimizer_gb=optimizer_gb,
        activations_gb=activations_gb,
        overhead_gb=FIXED_OVERHEAD_GB,
        total_gb=total_gb,
        trainable_params=trainable,
    )


def headroom_gb(peak: PeakMemory, card_capacity_gb: float) -> float:
    """How much VRAM is left on a card of `card_capacity_gb` after `peak`.

    Negative means the job is predicted not to fit -- shown, not hidden: spec
    005's memory half blocks on exactly this number being negative, and a
    caller here decides what to do with the sign rather than this function
    silently clamping it away.
    """
    return card_capacity_gb - peak.total_gb
