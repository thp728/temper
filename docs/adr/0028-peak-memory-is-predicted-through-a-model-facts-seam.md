# ADR-0028 — Peak memory is predicted through a model-facts seam

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/005-predictor-and-quote.md`
- **Issue:** [#48](https://github.com/thp728/temper/issues/48)

## Context

`temper_core.catalog.BaseModel` carried `est_peak_vram_gb` by hand: 5.3 GB for
Qwen3-4B, measured on the one real QLoRA run this project has, and 9.5 GB for
Qwen3-8B, extrapolated from the 4B figure by parameter ratio and documented in
the source comment as unverified. A number typed against one catalog entry
cannot describe a model the catalog has never seen, and the extrapolation for
the *second* entry was already a guess — the catalog was one addition away
from either guessing again or leaving a new model's peak blank.

Spec 005 names the fix directly: a `models` seam that resolves a repository
reference and pinned revision to the facts that decide memory — parameter
count, dimensions, architecture family, mixture-of-experts, chat-template
presence, whether padding and end-of-sequence are distinct tokens, context
length, licence — and a predictor that computes peak VRAM from those facts for
any model, catalog or not.

## Decision

**The facts' shape lives in `packages/core`; resolving them lives in
`apps/control-plane`.** `temper_core.models.ModelFacts` is the data contract
both the predictor and a later compatibility probe (spec 009) read, plus one
pure derivation neither reader should duplicate: total parameter count from
dimensions alone. Resolving an arbitrary repository's facts is I/O — reading
its published `config.json` and `tokenizer_config.json` over the network —
which `packages/core/pyproject.toml`'s zero-I/O rule forbids. That work,
`temper_control_plane.models.HuggingFaceModels`, sits beside the compute
provider seam (`provider.py`) for the same reason and the same shape: a
`Protocol`, a real implementation, and a double kept in the package rather
than grown per test file (`fake_models.py`, matching `fake_provider.py`).

**The double is seeded with real facts, not synthetic ones.** Unlike
`FakeProvider`'s one canned demo job, `fake_models.CATALOG_MODELS` carries
both curated models' facts as independently confirmed against their published
config at the pinned revision — the same files `HuggingFaceModels` reads live.
This is what lets `test_memory.py` assert the two anchors spec 005 names:
Qwen3-4B's facts predict exactly 33,030,144 trainable parameters at LoRA
r=16, matching the real run precisely, and its predicted peak lands inside a
stated tolerance of the 5.31 GB that run measured.

**The memory model is four published pools plus one calibrated constant,**
per `docs/research-reports/report-b.md` §4.1: weights (parameter count ×
bytes/param, by dtype), gradients and optimizer state (sized on *trainable*
parameters, exact via the same LoRA formula the anchor confirms), and
activations (Korthikanti et al.'s formula, collapsed to its checkpointed limit
because `gradient_checkpointing: true` is a trainer default this predictor
never has to infer). Summing the four for the one measured configuration
lands at roughly 2.85 GB against a measured 5.31 GB — the residual is real
cost (CUDA context, cuBLAS/cuDNN workspace, bitsandbytes' NF4 state, allocator
fragmentation) that a four-pool sum does not model and one anchor cannot
decompose further. `temper_core.memory.FIXED_OVERHEAD_GB` names it as a
calibrated constant rather than folding it silently into one of the four pools
or pretending the arithmetic alone explains the number.

**The tolerance is stated and generous on purpose.** `PEAK_TOLERANCE = 0.15`
is wide because the overhead constant it is checked against is calibrated from
a sample size of one. A narrow tolerance here would claim more confidence than
the evidence supports; both the constant and the tolerance are expected to
move as spec 005's calibration clause records more real runs.

**GPU capacity is data, not a provider call.** `temper_core.gpus.CAPACITY_GB`
holds VRAM by GPU type name — a physical constant of the card, unlike price
and live availability, which stay behind the provider seam
(`provider.GpuChoice`). `catalog.BaseModel` gained `min_gpu_type` (e.g. `"L4"`)
as the machine-readable key into that table, kept separate from `min_gpu`
(e.g. `"L4 (24 GB)"`, the display string) rather than parsed out of it.

**`/v1/models` computes every entry's peak fresh, on every request.** No
`lru_cache`: nothing has yet measured that resolving two catalog models per
request costs enough to justify the staleness a cache would introduce, and
adding one is cheap the day something does.

**`TEMPER_FAKE_PROVIDER` also switches the models resolver.** The browser
journeys already boot the control plane with that flag to guarantee no
external reachability; `/v1/models` is called on every load of the
model-choice screen, so leaving its resolver real would trade that guarantee
for a network dependency spec 005 gives no reason to keep. Reusing the
existing flag keeps "this process reaches nothing external" one switch
instead of two.

## Alternatives considered

**Store facts on the catalog entry, populated by a one-time script instead of
resolved live.** Rejected: this is what `est_peak_vram_gb` already was, and
the defect being fixed is exactly that a stored number goes stale the moment
the catalog grows past what curated it.

**Put `HuggingFaceModels` in `packages/core` since `ModelFacts` lives there
too.** Rejected: `packages/core/pyproject.toml`'s dependency list is the
enforced boundary the Phase B migration depends on — "no web framework, no
ORM, no cloud SDK" — and network I/O is exactly the category that boundary
exists to keep out, regardless of how small the client is.

**Derive `is_moe`/`pad_eos_distinct` from a hard-coded per-architecture table
instead of reading each repository's own config.** Rejected: a table keyed by
model family is the same shape of stored, un-updating knowledge the seam
exists to replace, and both facts are already present in every repository's
own published files.

## Consequences

- `BaseModel.est_peak_vram_gb` is deleted; `min_gpu_type` is added.
  `CatalogEntry.est_peak_vram_gb` is replaced by `CatalogEntry.peak_memory`
  (`PeakMemoryEstimate`), published with its full breakdown so the API, not
  just an internal module, can show the arithmetic behind the number.
- `LaunchForm` shows predicted peak VRAM and headroom against each model's
  recommended card before launch, labelled as an estimate with its stated
  tolerance.
- GPU *selection* (today's hard-coded `GPU_PREFERENCE` list in
  `orchestrator.py`) and *blocking* a job whose predicted peak does not fit
  are unchanged by this record — spec 005 assigns both to later tickets. This
  ADR covers only the seam and the number it produces.
