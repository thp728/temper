# Report B — The Training Loop: Hyperparameters, Compute Planning, and Orchestration

*Three-part engineering briefing on building an LLM fine-tuning platform, as of August 2026. Report A (dataset ingestion, tokenization, base-model support, method selection) is assumed read; method choice (SFT vs LoRA vs QLoRA vs DPO) and tokenization are referenced here, not re-derived.*

*Currency note: costs in USD with INR at ₹95.5/USD (USD/INR traded 95.38–95.69 on 14 Aug 2026 per Investing.com; ~95.4 per MTFX/Federal Reserve H.10 the same week). GPU prices carry individual verification dates. Confidence labels: High/Medium/Low.*

---

## TL;DR

- **Build the platform around QLoRA-on-a-single-card as the default path**: a 7B model fine-tunes in ~6–12 GB VRAM (fits an RTX 4090/A100), the hyperparameters have converged across every major framework to a small set of defaults you can hard-code (rank 16, α=32, lr 2e-4, cosine, warmup ratio 0.1, 3 epochs, effective batch 16–32), and a pre-flight estimator built on `FLOPs ≈ 6·N·D` plus a realized-MFU band of 35–50% predicts wall-clock and dollar cost to within a small multiple before launch.
- **Your cost/OOM predictor is the product moat, not the trainer.** VRAM is deterministic from five inputs (params, dtype, method, sequence length, batch); wall-clock is `6·N·D / (peak_FLOPS · MFU)`; dollar cost is wall-clock × GPU rate × safety margin. For a 7B QLoRA run over 30M training tokens this is roughly ~1 GPU-hour on an H100 (≈$3 / ₹285) or ~3 GPU-hours on an A100 (≈$4–5 / ₹380–475).
- **70–80% of commercial capability = SFT + LoRA/QLoRA + DPO on popular ≤14B open models, auto-configured, with a pre-flight quote, checkpoint/resume, spot-preemption recovery, and basic eval.** Defer full multi-node pretraining, RLHF/GRPO/RFT, 70B+ full fine-tuning, and multimodal to v2.

---

## 1. What training actually does — minimum viable mechanics

A fine-tuning step is a four-phase loop repeated once per micro-batch:

1. **Forward pass.** The batch of tokenized sequences flows through the frozen/trainable weights and produces, for each position, a probability distribution over the vocabulary. Cost ≈ 2·N FLOPs per token (N = parameter count).
2. **Loss.** The model's predicted distribution is compared to the actual next token via cross-entropy. Loss is a single scalar per batch; **this is the number you stream to the user.** Lower = the model assigns higher probability to the correct tokens. Typical SFT starting loss for a 7B model is ~1.5–2.5 and should decline to ~0.5–1.2.
3. **Backward pass (gradient).** Automatic differentiation computes, for every trainable weight, the derivative of the loss w.r.t. that weight — the direction that would reduce loss. Cost ≈ 4·N FLOPs per token (backward is ~2× forward). This is why the training constant is **6·N** per token, not 2·N.
4. **Optimizer update.** AdamW nudges each weight a small step (scaled by the learning rate) in the negative-gradient direction, using running averages of past gradients (momentum) and squared gradients (variance).

**Gradient accumulation** runs phases 1–3 several times, summing gradients, before one phase-4 update — this simulates a larger batch without the memory. **The effective batch size is what the optimizer actually "sees" per update.**

Reading a live log, an engineer needs exactly three signals: **loss** (should trend down, noisily), **grad_norm** (magnitude of the gradient vector — should be stable, typically 0.1–2.0), and **learning_rate** (should follow the schedule: rise during warmup, then decay). Everything in §6 is derived from these three.

---

## 2. Decision inventory (training)

Verdict key: **Expose** = user-facing knob; **Infer** = platform computes from inputs; **Hard-code** = fixed, not surfaced. "Coupled to" lists decisions that must be recomputed when this one changes.

| # | Decision | Recommended default | Valid range | Verdict | Failure symptom | Coupled to |
|---|---|---|---|---|---|---|
| 1 | Learning rate (LoRA/QLoRA) | 2e-4 | 1e-5 – 5e-4 | Infer (from method) | Too high: loss spikes/NaN. Too low: flat loss | eff. batch, rank, schedule |
| 2 | Learning rate (full FT) | 1e-5 | 5e-6 – 5e-5 | Infer | as above | eff. batch |
| 3 | Learning rate (DPO/pref) | 5e-6 | 1e-7 – 1e-5 | Infer | reward accuracy stuck at 0.5 | beta |
| 4 | LR schedule | cosine | cosine/linear/constant/wsd | Hard-code (cosine) | early plateau if wrong length | epochs/max_steps |
| 5 | Warmup ratio | 0.1 | 0.0 – 0.2 | Hard-code | early spike if 0 | LR, total steps |
| 6 | Epochs | 3 | 1 – 5 | Expose | >3 overfits; <1 undertrains | dataset size, LR schedule |
| 7 | Max steps | (derived) | — | Infer | — | epochs, eff. batch, dataset |
| 8 | Micro-batch size | (fit to VRAM) | 1 – 32 | Infer | OOM if too big | grad-accum, seq len, VRAM |
| 9 | Gradient accumulation | (derived for EBS 16–32) | 1 – 64 | Infer | noisy loss if EBS too small | micro-batch, world size, LR |
| 10 | Max sequence length | 2048 | 512 – 32768 | Expose | truncated data; OOM (quadratic) | micro-batch, VRAM |
| 11 | Weight decay | 0.01 | 0.0 – 0.1 | Hard-code | mild over/under-reg | LR |
| 12 | Gradient clipping (max_grad_norm) | 1.0 | 0.5 – 1.0 | Hard-code | NaN if disabled | LR |
| 13 | Optimizer | adamw_8bit | adamw_torch/8bit/paged | Infer (from VRAM) | OOM (full AdamW) | precision, VRAM |
| 14 | Precision | bf16 | bf16/fp16/fp8 | Infer (from GPU) | NaN (fp16 overflow) | GPU arch, gradient scaler |
| 15 | Gradient checkpointing | on | on/off | Infer (from VRAM headroom) | OOM if off; 20–30% slower if on | micro-batch, seq len |
| 16 | LoRA rank (r) | 16 | 4 – 256 | Expose | underfit if too low; VRAM/overfit if high | alpha, LR |
| 17 | LoRA alpha (α) | 32 (=2r) | r – 4r | Infer (=2r) | weak adaptation if α<r | rank |
| 18 | LoRA dropout | 0.0 | 0.0 – 0.1 | Hard-code | mild overfit | epochs |
| 19 | Target modules | all-linear | attn-only / all | Hard-code (all) | underfit (attn-only) | rank, VRAM |
| 20 | LoRA variant | rsLoRA if r≥32 else LoRA | LoRA/rsLoRA/DoRA/LoftQ | Infer (from rank) | instability at high r w/o rsLoRA | rank, alpha, LR |
| 21 | Random seed | 42 | any int | Hard-code (expose optionally) | non-reproducibility | none |
| 22 | DPO/pref beta | 0.1 | 0.05 – 0.5 | Expose | model drifts (low) / won't learn (high) | LR |
| 23 | Quantization (QLoRA) | 4-bit NF4 + double-quant | 4/8-bit | Infer (from VRAM) | OOM if disabled on small GPU | base dtype, VRAM |
| 24 | Distributed strategy | single-GPU → FSDP | DDP/FSDP/ZeRO | Infer (from model+VRAM) | OOM if model won't fit | GPU count, model size |
| 25 | Sample packing | on | on/off | Hard-code (on) | slow training / wasted compute if off | seq len |
| 26 | Eval/checkpoint cadence | every 500 steps or 10%/epoch | — | Hard-code | can't resume / detect overfit | total steps |
| 27 | Early-stopping patience | 3 evals | 2 – 5 | Infer | wasted compute / overfit | eval cadence |
| 28 | Per-job timeout & budget cap | user budget → max GPU-hours | — | Expose | runaway cost | cost estimator |

---

## 3. Hyperparameters, exhaustively

Each entry follows the six-point format plus a platform verdict. Framework defaults cited are ground truth as of the versions current in mid-2026.

### 3.1 Learning rate
1. **What/where:** Scalar multiplier on each optimizer step; set at config time, active every step.
2. **Controls:** Step size in weight space. `new_weight = old_weight − lr × (adjusted_gradient)`.
3. **Default:** **2e-4 for LoRA/QLoRA; 1e-5 for full FT; 5e-6 for DPO.**
4. **Why:** These are the *converged* defaults across independent frameworks, not convention. Unsloth's hyperparameter guide names 2e-4 as the explicit LoRA starting recommendation (range 2e-4→5e-6); Axolotl's LoRA example ships `learning_rate: 0.0002`; LLaMA-Factory ships `learning_rate: 1.0e-4`. Full FT uses ~10–20× lower because all weights move (not a small adapter), and higher LR overwrites pretrained knowledge (see §6 forgetting). DPO uses ~40× lower than SFT because it operates on an already-tuned model and large steps cause reward collapse (TRL DPOConfig documents 5e-6-class rates).
5. **Range/boundaries:** Below 1e-5 for LoRA: loss barely moves (undertraining). Above 5e-4: instability, loss spikes, divergence. The "LoRA Learns Less and Forgets Less" paper (arXiv 2405.09673) recommends sweeping [1e-5, 5e-4] and "picking the highest value that enables stable training."
6. **Symptom when wrong:** Too high → loss spikes upward or to NaN in the first 20–50 steps, grad_norm balloons. Too low → loss curve nearly flat, minimal improvement across an epoch.
**Verdict: Infer** from method. Hard-coding a single value blocks power users doing full FT or preference tuning, so key off the selected method rather than a global constant.

### 3.2 LR schedule
1–2. Shapes the LR over time. **Cosine**: linear warmup to peak, then half-cosine decay to a floor (~10% of peak).
3. **Default: cosine.** 4. **Why:** It is the de-facto standard validated from millions to hundreds of billions of parameters (GPT-2/3, Llama), and every framework example uses it (`lr_scheduler_type: cosine` in LLaMA-Factory, `lr_scheduler: cosine` in Axolotl). Unsloth Studio defaults to `linear` for very short runs — acceptable, because for runs under a few hundred steps the schedule shape barely matters.
5. **Range:** constant (fine for <200-step LoRA runs), linear (short fine-tunes), WSD (warmup-stable-decay; lets you extend a run without committing a token budget upfront — the most consequential alternative for long/branchable runs, per ZeroEntropy's scheduler writeup). Cosine's weakness: it commits the token budget upfront; you cannot stop early without leaving the LR mid-decay.
6. **Symptom:** LR decaying to floor long before data is exhausted (schedule length set wrong) → loss plateaus early.
**Verdict: Hard-code cosine.** WSD is a v2 feature for users who want to resume/extend.

### 3.3 Warmup ratio
1–2. Fraction of total steps spent ramping LR from 0 to peak.
3. **Default: 0.1 (10%).** 4. **Why:** LLaMA-Factory and Axolotl examples ship `warmup_ratio: 0.1`. Warmup prevents early large steps from destabilizing the model before AdamW's variance estimate has calibrated: "the initial destabilization from high learning rates can cause loss spikes that occasionally grow so large that the model never recovers" (Brenndoerfer, on transformer warmup). Unsloth uses a small fixed `warmup_steps: 5` for tiny runs — equivalent when total steps are few.
5. **Range:** 0.0 (risky; only safe at very low LR) to ~0.2 (wastes steps at low LR).
6. **Symptom:** With warmup 0, a loss spike in the first handful of steps; grad_norm huge at step 1.
**Verdict: Hard-code 0.1** (with a floor of ~5–10 absolute steps for tiny datasets).

### 3.4 Epochs vs max steps
1–2. Epochs = passes over the dataset; max_steps = optimizer updates. `max_steps = ceil(num_examples / effective_batch_size) × epochs`.
3. **Default: 3 epochs.** 4. **Why:** Unsloth (epochs 3), Axolotl (num_epochs 4), LLaMA-Factory (num_train_epochs 3.0) all cluster here. Unsloth's guidance: "training for more than 3 epochs offers diminishing returns and increases the risk of overfitting"; code data overfits faster (1 epoch).
5. **Range:** 1 (large datasets / continued pretraining) to 5 (tiny datasets). Above 5 almost always overfits.
6. **Symptom:** Overfitting — training loss keeps dropping while eval loss rises (see §6).
**Verdict: Expose epochs** (users have strong intuitions about "train longer"); **infer max_steps.**

### 3.5 Micro-batch size & 3.6 Gradient accumulation — the effective-batch identity

**Why (derivation):** `effective_batch_size (EBS) = micro_batch_size × gradient_accumulation_steps × world_size (number of GPUs)`. The optimizer updates once per EBS worth of examples. **Consequence:** two configs with the same EBS but different micro-batch/grad-accum splits produce *nearly identical loss curves* (only floating-point summation order differs); but changing EBS itself changes the curve — a larger EBS gives smoother, less noisy gradients and *requires a higher LR to converge in the same number of tokens* (§3.7).

1–2. **Micro-batch** = sequences processed at once on one GPU (bounded by VRAM); **grad-accum** = how many micro-batches to sum before an update.
3. **Defaults: micro-batch fit to VRAM (start 2), grad-accum set to reach EBS 16–32.** Axolotl ships micro-batch 2 / grad-accum 4 (EBS 8×world); LLaMA-Factory 1/8; Unsloth 4/8 (EBS 32).
4. **Why:** EBS 16–32 is the empirical sweet spot for instruction fine-tuning — large enough to denoise gradients, small enough to retain useful stochasticity and fit memory.
5. **Range:** micro-batch 1 (long sequences / small GPU) to 32; grad-accum 1 to 64. Micro-batch too high → OOM. EBS too small (<8) → noisy, unstable loss; too large (>64 for small datasets) → too few update steps to learn.
6. **Symptom:** OOM at step 1 (micro-batch too big); jagged, high-variance loss (EBS too small).
**Verdict: Infer both** from VRAM + target EBS. This is the single most important thing to get right automatically for a no-ML-background user.

### 3.7 LR ↔ effective-batch scaling
**Why (derivation, ≤6 lines):** Gradient noise falls as `1/√EBS` (law of large numbers, not `1/EBS`). For adaptive optimizers (Adam/AdamW), the matching rule is **square-root scaling**: `lr_new = lr_ref × √(EBS_new / EBS_ref)` — a result derived for adaptive optimizers via random matrix theory (arXiv 2006.09092), distinct from the linear rule used for SGD. **Consequence for the platform:** if you change EBS away from the reference the defaults were tuned at, scale LR by the square root of the ratio — do *not* scale linearly (linear scaling "easily diverges" at large batch, per the same literature). Too-high LR after scaling → loss spikes; too-low → training stalls.

### 3.8 Weight decay
3. **Default: 0.01.** 4. **Why:** Unsloth, Axolotl, LLaMA-Factory examples all ship `weight_decay: 0.01`; pretraining uses 0.1 (Llama/AdamW protocol) but fine-tuning uses the lighter 0.01 because runs are short. 5. **Range:** 0.0 (fine for very short LoRA) to 0.1 (heavier regularization). 6. **Symptom:** subtle; excessive decay slows learning, too little marginally worsens generalization. **Verdict: Hard-code 0.01.**

### 3.9 Gradient clipping (max_grad_norm)
1–2. Caps the L2 norm of the gradient vector before the update. 3. **Default: 1.0.** 4. **Why:** Standard across nanoGPT/Llama protocols ("gradient clipping is applied by norm with threshold 1.0"); it is the primary defense against a single bad batch producing an exploding update. 5. **Range:** 0.5 (aggressive) to 1.0. Disabling it invites NaNs. 6. **Symptom:** Without clipping, occasional grad_norm spikes to 100s → loss NaN. **Verdict: Hard-code 1.0.**

### 3.10 Optimizer
3. **Default: adamw_8bit** (8-bit AdamW via bitsandbytes); **paged_adamw_8bit** on the smallest GPUs. 4. **Why:** AdamW is universal for transformers (betas 0.9/0.95 for LLMs). The 8-bit variant quantizes optimizer state, cutting its VRAM ~4× with negligible quality loss — Axolotl (`adamw_bnb_8bit`), Unsloth (AdamW 8-bit) both default to it. 5. **Range:** `adamw_torch` (full precision, needs the most VRAM), `adamw_8bit`, `paged_adamw_8bit` (offloads to CPU on OOM spikes). Newer options (Muon) exist but are v2. 6. **Symptom:** OOM during the optimizer step specifically (full AdamW chosen on a tight GPU). **Verdict: Infer from VRAM budget.**

### 3.11 Precision
3. **Default: bf16.** 4. **Why:** bf16 has the same exponent range as fp32, so it does not overflow the way fp16 does; every modern config uses `bf16: true`/`bf16: auto`. fp16 requires a loss scaler and still risks NaN. 5. **Range:** bf16 (Ampere/Hopper/Blackwell — A100, H100, RTX 30/40/50, L40S), fp16 (older cards like V100/T4 without bf16), fp8 (H100/Blackwell, experimental for training). 6. **Symptom:** fp16 without proper scaling → NaN loss mid-run; fp8 → instability. **Verdict: Infer from GPU arch** (bf16 if supported, else fp16).

### 3.12 Gradient checkpointing
**Why (quantified):** In a normal forward pass every intermediate activation is stored for the backward pass. Checkpointing stores only a sparse subset (segment boundaries) and *recomputes* the rest during backward. This shifts activation memory from O(n) to O(√n) in layer count — roughly a **5× reduction in activation memory** at the cost of **~20–30% slower training** (one extra forward pass). Chen et al. (2016) showed √n is near-optimal for uniform compute graphs, which transformer stacks match. A measured example: activation memory 48 GB → 7 GB with checkpointing enabled (Lyceum). With FlashAttention (which already recomputes the attention matrix internally), only FFN/norm activations get rematerialized, so the real overhead is often at the low end (~20%).
3. **Default: on.** 5. **Range:** off (only if activation memory isn't the bottleneck — the 20–30% throughput hit is then pure waste). 6. **Symptom:** off → OOM during backward on longer sequences; on but unnecessary → training slower than expected. **Verdict: Infer** — enable whenever predicted VRAM without it exceeds ~80% of the card.

### 3.13 LoRA rank (r)
1–2. Dimension of the low-rank update matrices A (r×d_in) and B (d_out×r); the adapter learns `ΔW = B·A`. Controls trainable-parameter count and adapter capacity.
3. **Default: 16.** 4. **Why:** Unsloth Studio and the DEV 2026 guide default to 16; an independent ablation (UniMaia, arXiv 2605.27767) found "benchmark performance peaks at rank 16, with rank 8 performing similarly… increasing capacity beyond a moderate rank provides comparatively small improvements." (Note: the original QLoRA recipe used r=8/α=32; the modern default has drifted to r=16.) At r=16 a 7B model trains <0.6% of its parameters (~40M).
5. **Range:** 4 (minimal capacity; may underfit) to 256. "LoRA Learns Less" found ranks 16–64 insufficient for *code* tasks, recommending up to 256 with all-modules — a genuine exception. Higher rank = more VRAM + overfitting risk.
6. **Symptom:** Too low → task metric plateaus below target while loss looks fine. Too high → overfitting + higher VRAM.
**Verdict: Expose** (users doing code/domain shift need to raise it), default 16.

### 3.14 LoRA alpha (α)
1–2. Scaling factor: the adapter output is multiplied by α/r (or α/√r under rsLoRA). Controls *how strongly* the adapter influences the model.
3. **Default: 32 (= 2r).** 4. **Why:** The α=2r heuristic is near-universal (Unsloth α=32 at r=16; the "LoRA Learns Less" paper: "the choice α=2r is crucial for high ranks… high LR cannot compensate for a fixed low α"). An independent replication (arXiv 2410.01532) likewise uses "r=8, alpha=32, and dropout=0.1" following Dettmers et al. 5. **Range:** α=r (conservative) to 4r. 6. **Symptom:** α<r → adaptation too weak, model barely changes. **Verdict: Infer α=2r from rank.** Do not expose independently; it is mechanically tied to rank.

### 3.15 LoRA dropout
3. **Default: 0.0.** 4. **Why:** Unsloth defaults dropout to 0 ("Not that useful"); short fine-tunes rarely need it. Axolotl examples use 0.05. 5. **Range:** 0.0–0.1. 6. **Symptom:** mild overfitting reduction only. **Verdict: Hard-code 0.0** (or 0.05 for many-epoch runs).

### 3.16 Target modules
3. **Default: all-linear** (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`). 4. **Why:** "Earlier guidance recommended q_proj/v_proj only, but current benchmarks show including all linear layers consistently produces better results with minimal VRAM overhead" (DEV 2026); Axolotl exposes `lora_target_linear: true`, LLaMA-Factory `lora_target: all`. 5. **Range:** attention-only (less VRAM, underfits) → all. 6. **Symptom:** attention-only → underfitting on tasks needing FFN adaptation. **Verdict: Hard-code all-linear.**

### 3.17 LoRA variant (rsLoRA / DoRA / LoftQ)
3. **Default: plain LoRA at r≤16; rsLoRA at r≥32.** 4. **Why:** Standard LoRA scales by α/r, which over-shrinks high ranks; rsLoRA scales by α/√r, stabilizing high-rank training (arXiv 2312.03732; practitioner reports of rank-64 runs crashing under plain LoRA but training stably under rsLoRA). At low rank the difference is negligible. DoRA (weight-decomposed) can improve convergence via `use_dora=True` but adds overhead. 5. **Range:** LoRA/rsLoRA/DoRA/LoftQ. 6. **Symptom:** high-rank plain LoRA → instability/poor results. **Verdict: Infer from rank.**

### 3.18 Random seed
3. **Default: 42.** 4. **Why:** Reproducibility; fixed seed makes runs comparable and debuggable. 5. **Range:** any int. 6. **Symptom:** irreproducible results across identical configs. **Verdict: Hard-code 42, optionally expose** for users running seed sweeps.

### 3.19 Method-specific: DPO/preference beta
1–2. β controls the KL penalty keeping the tuned model near the reference: higher β = less divergence.
3. **Default: 0.1.** 4. **Why:** TRL's DPOConfig ships `beta=0.1` as the documented default ("Higher beta means less divergence from the initial policy"); it is the value used across the DPO literature. 5. **Range:** 0.05–0.5. Low β → model drifts far from reference (may degrade); high β → model won't move toward preferences. 6. **Symptom:** `rewards/accuracies` stuck at 0.5 (β too high) or reward margins exploding with quality collapse (β too low). **Verdict: Expose** for preference-tuning users; default 0.1, paired with lr 5e-6.

### 3.20 Interdependency map — what to recompute when X changes

- **Change method (LoRA→full FT→DPO):** recompute LR (2e-4→1e-5→5e-6), optimizer (may need paged), distributed strategy, VRAM budget, quantization.
- **Change effective batch size:** recompute LR by √-scaling (§3.7); recompute max_steps.
- **Change micro-batch or sequence length:** recompute VRAM → possibly toggle gradient checkpointing, change grad-accum to hold EBS, possibly change GPU/strategy.
- **Change LoRA rank:** recompute alpha (=2r), select rsLoRA if r≥32, recompute VRAM.
- **Change epochs:** recompute max_steps → LR schedule length, eval/checkpoint cadence, early-stopping patience.
- **Change GPU/VRAM:** recompute micro-batch, checkpointing, optimizer (8-bit vs paged), strategy.

### 3.21 Auto-configuration function (implementable spec)

```
def auto_config(dataset_num_examples, dataset_token_count, base_model_params,
                available_vram_gb, user_budget_usd, method, gpu_type):
    cfg = {}
    # --- method-driven core ---
    if method in ("lora", "qlora"):
        cfg.lr = 2e-4; cfg.rank = 16; cfg.alpha = 32; cfg.dropout = 0.0
        cfg.target_modules = "all-linear"
        cfg.variant = "rslora" if cfg.rank >= 32 else "lora"
        cfg.optimizer = "adamw_8bit"
    elif method == "full_ft":
        cfg.lr = 1e-5; cfg.optimizer = "paged_adamw_8bit"
    elif method == "dpo":
        cfg.lr = 5e-6; cfg.beta = 0.1; cfg.rank = 16; cfg.alpha = 32
    cfg.schedule = "cosine"; cfg.warmup_ratio = 0.1
    cfg.weight_decay = 0.01; cfg.max_grad_norm = 1.0; cfg.seed = 42
    cfg.precision = "bf16" if gpu_supports_bf16(gpu_type) else "fp16"
    cfg.quant = "nf4_double" if method == "qlora" else None

    # --- epochs from dataset size ---
    if dataset_num_examples < 1000:   cfg.epochs = 5
    elif dataset_num_examples < 50000: cfg.epochs = 3
    else:                              cfg.epochs = 1

    # --- sequence length: default 2048, or 95th-percentile of data (capped) ---
    cfg.seq_len = min(2048, ceil_pow2(p95_token_length(dataset)))

    # --- fit micro-batch to VRAM via the §4 predictor, then hold EBS≈16-32 ---
    cfg.grad_checkpointing = True
    mb = largest_microbatch_that_fits(base_model_params, cfg.seq_len,
                                      method, available_vram_gb, safety=0.85)
    cfg.micro_batch = max(1, mb)
    world = num_gpus(gpu_type)
    target_ebs = 32
    cfg.grad_accum = max(1, round(target_ebs / (cfg.micro_batch * world)))

    # --- LR sqrt-rescale if realized EBS deviates from reference (32) ---
    ebs = cfg.micro_batch * cfg.grad_accum * world
    cfg.lr *= sqrt(ebs / 32)

    # --- steps, cadence, stopping ---
    cfg.max_steps = ceil(dataset_num_examples / ebs) * cfg.epochs
    cfg.save_steps = max(50, cfg.max_steps // 10)
    cfg.eval_steps = cfg.save_steps
    cfg.early_stop_patience = 3

    # --- distributed strategy (see §4) ---
    cfg.strategy = choose_strategy(base_model_params, method, available_vram_gb, world)

    # --- budget gate ---
    est = estimate_cost(base_model_params, dataset_token_count, cfg, gpu_type)  # §4
    cfg.max_gpu_hours = user_budget_usd / gpu_hourly_rate(gpu_type)
    if est.gpu_hours > cfg.max_gpu_hours:
        raise BudgetExceeded(est, suggestion=cheaper_gpu_or_fewer_epochs())
    return cfg, est
```

---

## 4. Compute planning and cost prediction

### 4.1 Full training-time VRAM budget (build the OOM predictor from this)

Training uses five distinct memory pools. **Every OOM sizing mistake comes from counting only one or two** (Spheron, 2026). For a parameter count N and dtype byte-width `b`:

- **Weights** = N × b. bf16: 2 B/param. NF4 (QLoRA base): 0.5 B/param.
- **Gradients** = (trainable params) × 2 B (bf16). Full FT: all N. LoRA/QLoRA: only adapter params (~0.5–2% of N).
- **Optimizer state (AdamW, mixed precision)** = (trainable params) × 12 B (fp32 master copy 4 + momentum 4 + variance 4). With 8-bit AdamW the two moments drop to ~1 B each → ~6 B. LoRA/QLoRA: only on adapter params.
- **Activations** = `≈ seq × micro_batch × hidden × (34 + 5·heads·seq/hidden) × layers` bytes without checkpointing (Korthikanti et al., arXiv 2205.05198); with full checkpointing this collapses toward `2·seq·micro_batch·hidden·layers`. Scales **linearly with batch, quadratically with sequence length**.
- **KV cache**: negligible during *training* (only matters at inference); the attention intermediates are counted in activations.

**Worked estimates — 7B model (N=7×10⁹; hidden 4096, 32 layers, 32 heads), seq 2048, micro-batch 1, gradient checkpointing on:**

| Regime | Weights | Grads | Optim | Activations (ckpt on) | **Total** |
|---|---|---|---|---|---|
| **Full FT, bf16, 8-bit AdamW** | 14 GB | 14 GB | ~42 GB (8-bit) / 84 GB (fp32) | ~2–4 GB | **~72 GB (8-bit) / ~112 GB (fp32)** |
| **Full FT, int8 weights** | 7 GB | 14 GB | ~42 GB | ~2–4 GB | **~65 GB** |
| **LoRA, bf16 base** | 14 GB | ~0.1 GB | ~0.5 GB | ~2–4 GB | **~17–20 GB** |
| **QLoRA, NF4 base (int4)** | 3.5 GB | ~0.1 GB | ~0.5 GB | ~2–4 GB | **~6–12 GB** |

These match reported figures: full FT 7B "100–120 GB" (fp32 AdamW), LoRA "~28 GB" (with overhead), QLoRA "~6–12 GB" / "12 GB" (llmhardware.io, May 2026; Spheron 2026). The QLoRA paper (Dettmers et al., NeurIPS 2023) confirms the extreme end: a 65B model fits fine-tuning "on a single 48GB GPU while preserving full 16-bit finetuning task performance." **Confidence: High** — corroborated across ≥4 independent sources and the primary activation-memory paper.

**OOM predictor logic:** `predicted_peak = weights + grads + optim + activations(micro_batch, seq_len)`, then require `predicted_peak ≤ 0.85 × card_VRAM` (see §4.6 margin). If it fails: reduce micro-batch → enable checkpointing → shorten seq_len → quantize (LoRA→QLoRA) → shard (FSDP) → bigger GPU, in that order.

### 4.2 Compute cost: FLOPs → GPU-hours → dollars

**The constant.** Training compute `C ≈ 6·N·D` FLOPs, where N = parameters, D = tokens processed (= dataset_tokens × epochs). The 6 decomposes as 2·N (forward) + 4·N (backward, ~2× forward). This is the Kaplan et al. (2020) / Hoffmann et al. (2022) "6ND" law, used to estimate compute for GPT-3, PaLM, and Llama. **Confidence: High** — it is the standard across the entire scaling-law literature. For LoRA/QLoRA the *trainable*-parameter FLOPs are tiny, but the base model still requires a full forward and a backward pass to propagate gradients to the adapters, so **6ND on the full N remains the right wall-clock anchor** (NVIDIA's DGXC applies a ×2/3 correction for LoRA; use 6ND as a conservative upper bound).

**Conversion to time:** `wall_clock_seconds = (6·N·D) / (peak_FLOPS × MFU)`.
- **Peak dense bf16 throughput** (NVIDIA datasheets): **H100 SXM = 989 TFLOP/s; A100 80GB = 312 TFLOP/s.** (Use the *dense*, non-sparsity number — using the 2:4-sparsity or FP8 figure inflates the denominator and understates time by ~2–4×.)
- **Realized MFU** (fraction of peak actually achieved): plan on **35–50% for well-optimized full/LoRA fine-tuning on H100/A100.** Concrete anchors: IBM Research/PyTorch measured **57% MFU** on Llama-2-7B ("a rapid training speed of 3,700 tokens/sec/GPU, or 40B tokens/day on 128 A100 GPUs. This translates to a model FLOPS utilization (MFU) and hardware FLOPS utilization (HFU) of 57%"); HuggingFace's own FSDP fine-tuning run reports **45.5% end-to-end MFU** ("this is based on end-to-end training time involving overheads other than training like loading the pre-trained model, preparing the dataloaders, intermediate checkpointing and final model saving"). Single-GPU small-batch 7B often realizes only **~30%** (memory-bandwidth-bound attention); unoptimized eager-mode HuggingFace Trainer measures as low as **7.7% MFU** on A100. **QLoRA runs below bf16 LoRA** because 4-bit weights are dequantized to bf16 on the fly. **Confidence: Medium** — MFU is workload-dependent; treat 40% as a planning midpoint, not a guarantee, and recalibrate from your own measured tokens/sec.

**Worked quote — 7B, 10M-token dataset, 3 epochs (D = 30M tokens), LoRA, H100:**
`C = 6 × 7e9 × 30e6 = 1.26e18 FLOPs`. At 989 TFLOP/s × 0.40 MFU = 396 TFLOP/s → `1.26e18 / 3.96e14 = ~3,180 s ≈ 0.9 GPU-hours`. On A100 (312 × 0.40 = 125 TFLOP/s): `~2.8 GPU-hours`. **Budget QLoRA at ~1.2–1.3× the bf16-LoRA wall-clock** for optimized stacks (Unsloth/bitsandbytes); poorly-configured small-batch QLoRA can run far slower (one A100 study saw QLoRA ~10× a LoRA run). Cross-check against reported wall-clocks: the QLoRA paper fine-tuned a 65B model to 99.3% of ChatGPT's performance level "while only requiring 24 hours of finetuning on a single GPU"; practitioner guides report 7B QLoRA, 50K samples, 3 epochs ≈ 4–6 h on a single A100 40GB.

**Cost:** `dollars = wall_clock_hours × gpu_hourly_rate × num_gpus × (1 + margin)`. The 7B LoRA run above: H100 ≈ 0.9 h × $2.89 (RunPod Secure Cloud H100 PCIe, verified Aug 2026) ≈ **$2.6 / ₹250**; A100 ≈ 2.8 h × $1.39 ≈ **$3.9 / ₹370**. Add the §4.6 margin and provisioning/idle overhead → quote ~$5 / ₹475 to be safe.

### 4.3 GPU selection logic

| Model + method | Fits on | Recommended default card |
|---|---|---|
| ≤7–8B QLoRA | 12–16 GB | RTX 4090 24 GB (cheap) or A100 40/80 GB |
| ≤7–8B LoRA (bf16) | 20–24 GB | RTX 4090 24 GB / A100 80 GB / L40S 48 GB |
| 7–8B full FT | 72–120 GB | 1× H100 80 GB (8-bit optim) or 2× via FSDP |
| 13–14B QLoRA | ~20 GB | A100 80 GB / L40S |
| 13–14B LoRA | ~40 GB | A100 80 GB |
| 70B QLoRA | ~46 GB | 1× A100/H100 80 GB |
| 70B LoRA/full FT | 140 GB+ | multi-GPU FSDP/ZeRO-3 (8× H100) |

Rule: **pick the cheapest card whose VRAM ≥ predicted_peak / 0.85.** Prefer single-GPU wherever it fits — multi-GPU adds 10–20% throughput loss and orchestration complexity.

### 4.4 Multi-GPU strategies and the thresholds that trigger each

- **Single GPU** — use whenever the whole job (weights+grads+optim+activations) fits one card. Right for essentially all ≤14B LoRA/QLoRA. **Default.**
- **DDP (Distributed Data Parallel)** — replicates the full model on each GPU, splits the batch. Use when the model fits one GPU but you want to train faster / use a bigger EBS across cards. No memory saving. **Threshold: model fits one GPU AND you have spare GPUs for speed.**
- **FSDP FULL_SHARD (PyTorch native)** — shards parameters, gradients, and optimizer state across GPUs; memory scales ~linearly with GPU count. **This is the right default for 7B–70B when the model does *not* fit one card** (ML Journey, 2026). Configure via HuggingFace Accelerate. **Threshold: predicted_peak > single-card VRAM.**
- **DeepSpeed ZeRO** — Stage 1 shards optimizer state; Stage 2 adds gradient sharding; Stage 3 adds parameter sharding (memory-equivalent to FSDP FULL_SHARD). Choose DeepSpeed over FSDP when you need CPU/NVMe offload (ZeRO-Infinity) or its fused Adam. **Threshold: Stage 2 when optimizer+grad state is the pressure; Stage 3 / offload when even that won't fit.**

Decision function: `single-GPU if fits; elif fits with DDP speedup wanted → DDP; elif model too big → FSDP FULL_SHARD; elif still OOM → ZeRO-3 + CPU/NVMe offload.` **Confidence: High** (HuggingFace Accelerate docs; DeepSpeed docs; corroborating 2026 analysis).

### 4.5 Current GPU pricing (verify dates as shown)

**Global providers** (per GPU-hour, on-demand unless noted):

| Provider | H100 | A100 80GB | L40S / 48GB | RTX 4090 | Verified |
|---|---|---|---|---|---|
| **RunPod** (Secure Cloud) | $2.89 PCIe / $3.29 SXM (Community ~$1.99 PCIe) | $1.39 PCIe / $1.59 SXM | $0.99 L40S | $0.74 | pricing page dated 27 Jul 2026; read 15 Aug 2026 |
| **Lambda** | $3.29 PCIe / $4.29 SXM (8-node only) | $1.99 (40GB) / ~$2.79 | — | — | secondary, Apr–Jul 2026 |
| **Vast.ai** (marketplace) | ~$1.53–2.50 | as low as $0.67 | — | many <$0.70 | Jul 2026 |
| **Together AI** (dedicated endpoint) | $6.49/hr | — | — | — | Jun–Jul 2026 |

RunPod also lists B200 $6.79, H200 $4.59, RTX 5090 $0.99, A40 $0.44, RTX A6000 $0.53, RTX Pro 6000 $2.09 (read 15 Aug 2026). **Spot/interruptible** runs 50–90% cheaper across providers (AWS spot A100 $0.41–1.23 vs on-demand). **Confidence: High for RunPod** (fetched live); **Medium for others** (secondary aggregators).

**India-accessible providers** (INR at ₹95.5/USD):

| Provider | H100 | A100 80GB | Notes | Verified |
|---|---|---|---|---|
| **JarvisLabs.ai** | ₹217.89/hr ($2.69) SXM, India region | $1.49/hr (₹142); spot from $0.89 | per-minute billing, no commitment; 8×H100 ₹1,743/hr; RTX Pro 6000 Blackwell $1.89; L4/A5000 ₹41/hr | 2026 |
| **E2E Networks** | from $1.80/hr (~₹172); ~₹249/hr on-demand cited | A100 instances (INR-billed) | NSE-listed, India data residency, INR billing (no forex/GST friction); H200 spot from ₹88/hr; B200 $4.90 | May 2026 |
| **Cyfuture AI** | ₹329/hr on-demand → ₹219/hr reserved (12-mo) | — | per-second billing, DPDP/SOC2/ISO | 2026 |
| **Yotta (Shakti), Tata (Vayu)** | quote-based enterprise | quote-based | Tier IV+ / MPLS; India-resident SLA | 2026 |

**Recommendation:** default global jobs to **RunPod** (cheapest transparent on-demand + per-second billing + spot); default India-data-residency jobs to **JarvisLabs** (transparent INR, per-minute, India H100 region) or **E2E Networks** (INR billing, no forex/GST). **Confidence: High** for JarvisLabs/RunPod (transparent live pages); **Medium** for E2E/Cyfuture (portal rates change with demand).

### 4.6 Safety margin

Reserve **~15% VRAM headroom** (target ≤85% of card capacity) to absorb: allocator fragmentation, the transient backward-pass peak (gradients coexist with activations), and sequence-length variance across batches. For time/cost quotes, add a **~20–30% wall-clock buffer** over the MFU-based estimate (real MFU underperforms the planning midpoint; checkpointing, data loading, and eval steps add overhead — the HuggingFace 45.5% figure is explicitly end-to-end including these). Quote the buffered number.

---

## 5. Training execution and orchestration

### 5.1 Job queuing and GPU scheduling
Model each job as: `{config, dataset_ref, gpu_type, max_gpu_hours, budget_cap}`. Use a queue (Redis/Celery or a managed queue) with one worker per GPU. Because your workloads are single-GPU-dominant, a simple **bin-packing scheduler** (place job on the smallest free GPU whose VRAM ≥ requirement) is sufficient for v1; defer gang-scheduling for multi-node to v2. **SkyPilot** is the recommended orchestration layer for provisioning across RunPod/Lambda/clouds with automatic failover and spot management — it handles multi-cloud placement and preemption recovery so you don't hand-roll it.

### 5.2 Container and dependency pinning — the real version traps
Pin **exact** versions in a locked image; this ecosystem breaks constantly across minor releases. The specific conflict traps as of 2026:
- **PyTorch ↔ CUDA ↔ driver**: the training image's CUDA must match the host driver; a bf16/fp8 kernel mismatch surfaces as silent NaNs or `CUDA error: no kernel image`.
- **bitsandbytes ↔ CUDA**: QLoRA's 4-bit kernels are compiled per-CUDA-version; a mismatch fails at model load.
- **transformers ↔ peft ↔ trl ↔ accelerate**: these four move together. TRL v1.0 (2026) unified the post-training API but broke older `DPOConfig`/`SFTTrainer` call sites. Unsloth "requires specific PEFT versions — always install Unsloth's recommended dependency versions to avoid conflicts."
- **flash-attention**: must be built against the exact torch/CUDA; a wheel mismatch disables it silently and training runs 2–3× slower with no error.
**Verdict:** ship one immutable, fully-locked Docker image per (framework, CUDA) combination; never `pip install` at job runtime.

### 5.3 Checkpointing cadence and resume
Checkpoint every **500 steps or 10% of total steps** (whichever is smaller), and always at end-of-epoch. Save adapter weights (LoRA/QLoRA — tiny, MBs) plus optimizer state, scheduler state, and step counter to resume exactly. For FSDP use `SHARDED_STATE_DICT`; for DeepSpeed ZeRO-3 post-convert with `zero_to_fp32.py`. **Resume-from-checkpoint** must restore RNG state and dataloader position, not just weights, or the run diverges from a clean one.

### 5.4 Spot preemption handling
Spot is 50–90% cheaper but interruptible. Strategy: **checkpoint frequently (§5.3) + auto-resume on a new spot instance from the last checkpoint.** SkyPilot automates this (detect preemption → reprovision → resume). Only offer spot for jobs that are checkpoint-tolerant (all fine-tuning is); surface an estimated completion-time range because preemptions extend wall-clock. AWS spot A100 interruption rates run ~5%.

### 5.5 OOM detection and automatic retry
Catch `torch.cuda.OutOfMemoryError`. **Auto-retry with reduced micro-batch** (halve it, double grad-accum to preserve EBS, so the loss curve is unchanged). If micro-batch already 1: enable gradient checkpointing if off → shorten sequence length → escalate to a larger GPU or FSDP. Cap at 2–3 automatic retries, then surface to the user. This is the single most valuable automatic recovery for a no-ML-background audience.

### 5.6 Metrics to stream, and frequency
Stream every **1–10 steps** (LLaMA-Factory logs every 10; Unsloth guidance is to watch the first ~50 steps closely): **loss, grad_norm, learning_rate, tokens/sec, GPU memory used, % complete, ETA, running cost**. Stream **eval loss** at each eval step. For DPO also stream `rewards/accuracies` and `rewards/margins`. Downsample for the UI chart but retain full resolution server-side.

### 5.7 Divergence and NaN detection — automatic abort
Abort automatically if any of: **loss becomes NaN/Inf** (immediate abort — unrecoverable without rollback); **loss increases >2× its trailing 50-step average for >20 consecutive steps** (divergence); **grad_norm exceeds ~100 repeatedly** despite clipping (instability). On NaN, the sophisticated recovery (used in PaLM/OPT training) is to roll back ~100 steps and skip the triggering batch — a v2 feature; for v1, abort and surface "training diverged — try a lower learning rate," optionally auto-retrying once at half LR.

### 5.8 Early stopping
Enable when a validation split exists (carve ~5–10% via `val_set_size`). Stop if eval loss fails to improve for **3 consecutive evals** (patience 3), and keep the best checkpoint, not the last. This both saves compute and prevents overfitting.

### 5.9 Per-job timeout and budget caps
Enforce `max_gpu_hours = budget_usd / gpu_rate` from §3.21. Hard-kill at the cap. Also enforce a wall-clock timeout (e.g., 2× the estimate) to catch hangs. **Surface to user; never silently exceed budget.**

### 5.10 Retry-vs-surface matrix
- **Auto-retry:** OOM (reduce micro-batch), spot preemption (resume from checkpoint), transient infra errors (node failure, network blip, image pull failure), single NaN at start (retry once at half LR).
- **Surface to user:** repeated divergence after LR reduction, budget/timeout cap hit, dataset/format errors (should be caught in Report A's ingestion stage), persistent OOM at micro-batch 1 on the largest available GPU, dependency/version errors (a platform bug — alert ops, not user).

---

## 6. Reading a training run — signals the platform computes and surfaces

The platform should compute these automatically and display a health badge (Healthy / Overfitting / Diverging / Under-trained / Misconfigured):

- **Healthy:** training loss declines and flattens into a noisy plateau; eval loss tracks it down then flattens; grad_norm stable (0.1–2.0); LR follows the schedule. A curve that "drops fast then flattens is healthy."
- **Diverging:** loss rising, spiking, or NaN; grad_norm spiking to 10s–100s. **Warn threshold:** loss > 2× trailing-average for 20 steps, or any NaN. **Cause:** LR too high / fp16 overflow / no clipping. **Action shown:** "lower learning rate."
- **Overfitting:** training loss keeps dropping while **eval loss rises**. **Warn threshold:** eval loss increases for 2 consecutive evals while train loss decreases. **Cause:** too many epochs / rank too high / dataset too small. **Action:** "reduce epochs or enable early stopping" (which the platform already does).
- **Under-trained:** training loss still declining steeply at end of run; eval loss still falling. **Warn threshold:** loss slope over final 20% of steps still steeply negative. **Action:** "increase epochs or learning rate."
- **Misconfigured:** loss essentially flat from step 1 (LR too low, or LR decayed to floor too early — schedule length wrong), or loss oscillating wildly in first 20–30 steps (LR too high — "cut it in half and restart"). **Warn threshold:** <5% loss reduction over the first full epoch.

Concrete numeric defaults to surface: **loss NaN → red/abort; eval-loss-up-2-evals → amber/overfit; final-20%-slope steeply negative → amber/undertrained; first-30-step loss variance high → amber/LR-too-high; grad_norm > 100 → amber/instability.** For DPO, `rewards/accuracies` climbing toward 0.7–0.9 = healthy; stuck at 0.5 = β too high or LR too low.

**Catastrophic forgetting (mechanical):** standard gradient descent moves weights to minimize loss on the *new* data distribution, overwriting weights that encoded old capabilities — "neural networks have no inductive bias toward preserving knowledge they're not currently being optimized on." Per-step forgetting is bounded by `learning_rate × √(current_loss)` (arXiv 2605.20005), which is *why* lower LR and warmup reduce it. **Mitigations the platform should default to:** (1) **LoRA/QLoRA** — freezes the base model so base capability is physically preserved; the strongest practical mitigation and a core reason LoRA dominates production fine-tuning (though not universal — high-magnitude adapter directions can still induce forgetting, and one 2026 study found LoRA did *not* alleviate forgetting when matched to full-FT learning performance). (2) **Lower LR.** (3) **Rehearsal:** mix 5–20% general/instruction data into the fine-tuning set. (4) **Early stopping on a held-out general benchmark (e.g., MMLU delta).** Surface an MMLU-delta check post-training so users see if general capability regressed.

**Overfitting (mechanical):** with too many passes over a small dataset, the model memorizes specific examples rather than learning the pattern; capacity (high rank) accelerates it. Detected by the train/eval-loss divergence above; mitigated by fewer epochs, dropout, weight decay, lower rank, and early stopping.

---

## 7. Minimum viable training stack

One named choice per layer; alternatives are conditional.

- **Training framework: Axolotl** (YAML-driven, broad model + method support: SFT/LoRA/QLoRA/DPO/GRPO, FSDP+DeepSpeed integration, active 2025–2026 development). *Conditional alternative:* **Unsloth** when the job is single-GPU ≤14B on a supported architecture (2× faster, ~70% less VRAM via Triton kernels — mathematically equivalent output); **LLaMA-Factory** if you want a broader model zoo or its GUI. Under all of them sits **HuggingFace TRL/PEFT/transformers** — the ground-truth trainers.
- **Launcher: HuggingFace Accelerate** (unifies single-GPU, DDP, FSDP, and DeepSpeed behind one launch config; both Axolotl and LLaMA-Factory build on it).
- **Distributed strategy: single-GPU by default → FSDP FULL_SHARD when a model won't fit one card** (§4.4). *Conditional:* DeepSpeed ZeRO-3 + offload only when FSDP still OOMs.
- **Orchestration/provisioning: SkyPilot** (multi-cloud placement, spot management, auto-recovery across RunPod/Lambda/India providers).
- **Monitoring: Weights & Biases** (or self-hosted MLflow/TensorBoard) for loss/grad_norm/LR streaming; wire the §6 health checks on top.

**What v1 supports:** SFT + LoRA + QLoRA + DPO on popular open models up to ~14B (and 70B via QLoRA on one 80GB card); single-GPU and FSDP multi-GPU; pre-flight VRAM/time/cost quote; checkpoint/resume; spot recovery; OOM auto-retry; NaN/divergence abort; early stopping; budget caps; basic eval-loss + optional MMLU-delta.

**What v1 defers:** full multi-node pretraining; RLHF/PPO/GRPO/RFT; full fine-tuning of 70B+; multimodal; advanced eval harnesses; WSD schedules and checkpoint branching; automatic hyperparameter sweeps; NaN rollback-and-skip.

**Calibration to the 70–80% target:** commercial platforms (Together AI: per-token LoRA/full FT/DPO on 200+ models, hosted inference; Predibase: LoRA + serving + RFT; Fireworks; Google Vertex AI) differentiate on (a) breadth of methods including RFT/GRPO, (b) integrated hosted inference, and (c) large-scale distributed training. Notably, **OpenAI is winding down its self-serve fine-tuning API** — per OpenAI's deprecations notice, developers were notified on 7 May 2026 of availability changes; existing active users can create fine-tuning jobs only until 6 Jan 2027 (inference on already-tuned models continues until the base model is deprecated), with OpenAI citing newer models being "much better at following instructions and formats." This signals the market consolidating around open-weight fine-tuning, which validates this build. Matching SFT+LoRA+QLoRA+DPO on ≤14B open models with auto-config, a pre-flight quote, and reliable orchestration reaches the 70–80% band; the remaining 20–30% (RFT, multimodal, integrated serving, 100B+) is the explicit v2 scope.

---

## Caveats and confidence

- **MFU is the softest number in the cost model (Medium confidence).** 35–50% is a planning band, not a guarantee; measured anchors span 45.5% (HuggingFace FSDP, end-to-end) and 57% (IBM Llama-2-7B) at the high end, ~30% for single-GPU small-batch 7B, and as low as 7.7% for unoptimized eager-mode HF Trainer. Calibrate against your own first runs and update the estimator's MFU constant per (GPU, method) from measured tokens/sec.
- **QLoRA slowdown vs LoRA is config-dependent (Medium).** Sources range from ~1.2–1.3× (optimized Unsloth/bitsandbytes) to ~10× (a small-batch A100 study). Budget 1.3× and measure.
- **GPU pricing churns weekly (High for RunPod/JarvisLabs, Medium for aggregator-sourced rows).** Re-fetch provider pricing pages at quote time; the rates here carry the verification dates shown. India portal rates (E2E, Cyfuture) move with demand and are quote-based at the enterprise end (Yotta, Tata).
- **Framework defaults are stable and cross-corroborated (High).** LoRA r=16/α=32, lr 2e-4, cosine, warmup 0.1, EBS 16–32, weight decay 0.01, grad clip 1.0, DPO β=0.1 appear consistently across Axolotl, LLaMA-Factory, Unsloth, and TRL — safe to hard-code/infer.
- **Anything older than ~18 months is flagged inline.** The 6ND law (2020), Chen et al. checkpointing (2016), and Korthikanti activation-memory formula (2023) are foundational and still current; the QLoRA paper (2023) wall-clocks are hardware-relative and should be re-measured on your GPUs.

*Cross-references: method selection (SFT/LoRA/QLoRA/DPO), tokenization, and base-model support are covered in Report A; serving/deployment of the trained adapter is Report C's scope.*