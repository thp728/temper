# Report C: Evaluation, Artifacts, Platform Concerns, and a Teardown of Commercial Fine-Tuning Products (2026)

**Scope note and staleness policy.** This report is written as of 15 August 2026. Product interfaces, pricing, and model dropdowns change monthly; every pricing figure carries a verification date. Any product detail I could not re-verify inside 18 months is flagged inline. Reports A (data, tokenization, models, method selection) and B (hyperparameters, compute, orchestration) are assumed read; I reference their conclusions rather than re-deriving them.

---

## TL;DR
- **Build the "boring 75%": a 3-screen create-job flow (SFT + LoRA only) on 6–10 open-weight models (Llama, Qwen, Mistral, Gemma), with dataset validation, a live loss/eval-loss chart, a held-out generation diff against the base model, and one-click serving.** That is what every credible competitor (OpenAI's now-closing dashboard, Together, Fireworks, Predibase, Kiln) actually exposes; the deep-config knobs are a differentiator, not table stakes.
- **The single most important 2026 fact for this build: OpenAI is winding down its self-serve fine-tuning platform** — closed to organizations that had never previously fine-tuned as of 7 May 2026, and per OpenAI's own developer post, "existing active customers can continue running fine-tuning training jobs through January 6, 2027, after which creating new training jobs will no longer be possible." The frontier-lab "just fine-tune GPT" option is disappearing, which shifts demand toward open-weight fine-tuning platforms — exactly the segment a small team can serve.
- **Self-hosting a fine-tuned open model only beats a frontier API when you have sustained, high utilization.** A dedicated H100 at roughly $1.99/hr (RunPod, verified Jul 2026) must be kept busy: at 24/7 it is about $1,440/month, so unless you push enough tokens to amortize that below the frontier per-token price, calling GPT-5.6 Luna ($0.20/$1.20 per 1M tokens) or Gemini 2.5 Flash-Lite ($0.10/$0.40) is cheaper. Be blunt with users about this.

## Key Findings
1. **Eval loss is a necessary tripwire, not a quality metric.** It detects "did training do anything / did it diverge," but LLM-as-judge and held-out generation diffs are what tell a user whether the model is actually better. Ship all three; do not ship eval loss alone.
2. **Quantization is nearly free at 8-bit and cheap-but-real at 4-bit.** Kurtić et al. ("Give Me BF16 or Give Me Death?", arXiv:2411.02355) find "W8A8-FP (FP8) weight and activation quantization is lossless across all model scales," with the lowest per-task 8-bit recovery still 98.44%. INT4 weight-only costs more and hits small models harder: Song et al. (arXiv:2505.20276) report "Llama-3.1 8B loses an average of 10% (excluding FP8), while Llama-3.1 70B loses only 4.5%." Offer INT4 GGUF as the default local-serving export, FP16 safetensors as the "quality" export.
3. **LoRA is the correct default method and the adapter-vs-merged choice is the core artifact decision.** Adapters are tiny (megabytes), multiplexable (Predibase LoRAX; Fireworks docs: "up to 100 LoRA adaptations to run simultaneously on a dedicated deployment without extra cost"), and merge into base weights with negligible precision loss when merged in fp16/bf16. Merging into a quantized base is where quality is silently lost.
4. **License propagation is the biggest under-appreciated legal exposure.** Llama and Gemma licenses flow through to fine-tuned derivatives: Llama requires "Llama" in the derivative name and "Built with Llama," and the Llama 3.1/4 Community License requires that any licensee exceeding "700 million monthly active users in the preceding calendar month" request a separate license from Meta "in its sole discretion." Gemma prohibits training competing foundation models. Apache-2.0 models (Qwen, Mistral open weights) are clean. The platform must surface the base license on every artifact.
5. **The competitive field splits into five tiers**, and the sweet spot for a 70–80% product is between the "too-managed, closing" frontier APIs and the "too-raw" GPU clouds: an opinionated, open-weight, LoRA-first platform with real evals and one-click serving — the Together/Fireworks/Predibase/Kiln quadrant.

---

## 1. Evaluation

The platform's job is to answer one user question after a run: *did this work, and is it safe to deploy?* That decomposes into five measurable signals.

### 1.1 Eval (validation) loss and its limits
Compute cross-entropy loss on a held-out split the model never trained on, at the end of each epoch. It is the cheapest signal and the only one that is essentially free (you already have the forward pass).

**Why (marked aside):** Training loss is measured on the same examples whose gradients update the weights, so it can be driven arbitrarily low by memorization. Validation loss is measured on data outside the gradient path. When the model stops learning generalizable structure and starts memorizing token sequences, the two curves diverge: train loss keeps falling while eval loss flattens then rises. Mechanically, the parameters are moving to reduce error on the training set in directions that *increase* error on the held-out distribution — the generalization gap. **Engineering consequence:** you do not need a fancy overfitting detector; you need both curves on one chart and a rule that fires when eval loss rises for N consecutive evals while train loss falls (see 1.4).

Limits: loss is a proxy for next-token probability, not task success. A model can have lower eval loss and worse instruction-following (e.g., it became more confident on stylistic tokens but regressed on format). **Verdict: expose eval loss as a chart, but never as the sole pass/fail signal.**

### 1.2 Held-out generation quality
Generate completions on a fixed, versioned held-out prompt set and show them side-by-side with the base model's completions on the same prompts. This is the single most persuasive artifact for a non-ML user: they can *read* the difference. **Verdict: expose. This is table stakes.**

### 1.3 Regression testing against the base model
Fine-tuning on a narrow dataset causes catastrophic forgetting of general capability. Run a small fixed suite (a few hundred items from general benchmarks) on both base and tuned model and show deltas. If the tuned model gained 15 points on the target task but lost 20 on general reasoning, the user must see that trade. **Verdict: expose a compact regression panel; auto-run it.**

### 1.4 Overfitting detection (the train/eval gap)
Concrete displayed rule: flag a warning when validation loss increases across 2+ consecutive evaluation points while training loss continues to decrease, OR when the gap (eval_loss − train_loss) exceeds a task-dependent band. Recommended default action: surface the best checkpoint (lowest eval loss), not the last. OpenAI's platform saves per-epoch checkpoints and reports `full_valid_loss` and `full_valid_mean_token_accuracy` per checkpoint — mirror this. **Verdict: infer and surface automatically; recommend the best checkpoint by default.**

### 1.5 LLM-as-judge and its documented failure modes
Use a strong model to score tuned-vs-base outputs pairwise. It scales, but has quantified, systematic biases you must mitigate:
- **Position bias** — judges favor an answer by its slot. Studied across 15 judges and ~150,000 instances (Shi et al., "Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge," IJCNLP 2025), which found the bias "varies significantly across judges and tasks and is not due to random chance." Mitigation: evaluate both orderings and average/require consistency.
- **Verbosity bias** — longer answers score higher regardless of quality (Saito et al. 2023, "Verbosity Bias in Preference Labeling," documented in GPT-4/GPT-3.5). Mitigation: penalize length in the rubric.
- **Self-preference bias** — a judge inflates win-rates for outputs from its own family (Wataoka, Takahashi & Ri, "Self-Preference Bias in LLM-as-a-Judge," 2024/2025; EMNLP 2025 follow-ups). Mitigation: judge with a *different* model family than the one being tuned/served.
The frequently quoted "~80% judge/human agreement" (MT-Bench, Zheng et al., measured across 3,000 expert votes) is an average and a misleading floor for production go/no-go decisions. **Verdict: offer LLM-as-judge as an optional eval, default the judge to a different family, always run both position orderings, and label results "indicative, not authoritative."**

### 1.6 Benchmark suites worth auto-running, and cost
The de-facto open harness is EleutherAI's **lm-evaluation-harness** (widely used, supports vLLM backends for fast local eval). **Inspect AI** (UK AI Security Institute) is the better choice if you ever add tool-use/agent evals and want sandboxing and a log viewer; it is heavier and its agent tasks can be expensive. HELM is comprehensive but too heavy for a per-job auto-eval. Run a small curated slice (e.g., a few hundred items spanning MMLU-style knowledge, a reasoning task, and a task-relevant set) on the tuned model via vLLM on the same GPU you just trained on — marginal cost is a few minutes of GPU time. **Verdict: bundle a lm-evaluation-harness slice, auto-run it, show base-vs-tuned deltas. Skip full HELM.**

### 1.7 Pass/fail signals to display (with thresholds and reasoning)
| Signal | Green | Yellow | Red | Reasoning |
|---|---|---|---|---|
| Eval loss trend | Monotone down, plateau | Noisy/flat | Rising 2+ evals | Rising = overfit/diverge |
| Train/eval gap | Small, stable | Widening | Gap large + eval rising | Generalization failure |
| Held-out judge win-rate vs base | ≥60% (both orderings) | 45–60% | <45% | <45% means tuned is worse |
| General regression delta | ≥ −2 pts | −2 to −8 | < −8 pts | Catastrophic forgetting |
| Safety/moderation | Pass | — | Any category fails | Block deploy |

### 1.8 What a 70–80% product implements vs skips
**Implement:** eval-loss chart, held-out generation diff, compact base-vs-tuned regression, optional cross-family LLM-as-judge, best-checkpoint selection, a safety check gate. **Skip (with stated cost):** full HELM (cost: you cannot publish leaderboard-grade numbers — acceptable, users don't need them); human-annotation eval workflows (cost: no gold-standard for subjective tasks — acceptable early, add later for enterprise); custom RFT graders as an eval surface (cost: power users leave — acceptable for v1).

---

## 2. Output Artifacts and Delivery

### 2.1 Adapter weights vs merged weights
LoRA training produces a small adapter (the A and B low-rank matrices), typically megabytes to low hundreds of MB, vs a full merged model of many GB. Offer **adapter** when the user wants to (a) keep storage/transfer small, (b) serve many fine-tunes on one base (multi-LoRA), or (c) swap behaviors cheaply. Offer **merged** when the user wants a single self-contained artifact to run anywhere, export to GGUF, or hand to a serving stack that does not support adapter hot-loading. **Default: keep the adapter as the canonical artifact and offer merged/GGUF as export options.**

### 2.2 Merge mechanics and where precision is lost
**Why (aside):** Merging computes W' = W₀ + (α/r)·(B·A) and writes W' back into the base tensor. If W₀ is held in fp16/bf16 during the merge, the added low-rank term is representable with negligible error and the merged model matches the adapter-on-base model. **Precision is lost in two places:** (1) if you merge into a *quantized* (INT4/INT8) base, the base weights were already rounded, and adding the adapter cannot recover that lost information — you get a doubly-approximated model; (2) if you then quantize the merged model for GGUF export, you round again. **Engineering consequence:** always merge in fp16/bf16, *then* quantize once for export. Never merge into an already-4-bit base and call it "the fine-tuned model." (Note: QLoRA trains against a 4-bit *frozen* base but keeps adapters in bf16 for exactly this reason; the merge for serving should still target a fp16 base.)

### 2.3 Export formats and who needs each
| Format | Who needs it | Serving stacks |
|---|---|---|
| safetensors (fp16/bf16) | GPU serving, further training, distribution | vLLM, TGI, Transformers |
| safetensors LoRA adapter | multi-LoRA GPU serving | vLLM `--lora-modules`, LoRAX |
| GGUF (Q4_K_M default, up to Q8) | laptop/CPU/edge local inference | llama.cpp, Ollama, LM Studio |
| INT8/INT4 quantized safetensors | tight-VRAM GPU serving | vLLM (GPTQ/AWQ) |

**Why (quant accuracy, aside):** Going fp16 → INT8 is near-lossless — Song et al. (arXiv:2505.20276) find "FP8 and GPTQ-int8 have the smallest average accuracy drops (Δaccuracy 0.8% and 0.2%, respectively) relative to BF16 across tasks," and Kurtić et al. (arXiv:2411.02355) call FP8 W8A8 "lossless across all model scales." fp16 → INT4 weight-only costs more: Song et al. report "AWQ-int4 and GPTQ-int4 suffer average drops of 1-3%, while BNB-nf4 loses an average of 6.9%," and the small-vs-large split noted above (8B ~10% vs 70B ~4.5% for Llama-3.1). Group-wise quantization (e.g., 128-element groups, AWQ/GPTQ) is needed to avoid the catastrophic collapse seen with naive per-column INT4. **Engineering consequence:** default local export to Q4_K_M (best size/quality knee, ~95% quality at ~4GB for a 7B), but warn small-model users that INT4 hurts them more and offer Q5/Q8.

### 2.4 Model card and provenance metadata
Auto-generate a card with: base model + exact version, base license and its propagated obligations, training method (SFT/DPO/LoRA), hyperparameters used, dataset fingerprint (hash, row count, token count), eval results, checkpoint lineage, and timestamps. This is both a UX nicety and a compliance artifact (see §3). **Verdict: expose; auto-generate; non-negotiable for any regulated user.**

### 2.5 Artifact sizes and storage cost
A LoRA adapter for a 7–8B model is typically tens of MB; a merged fp16 7–8B model is ~14–16GB; Q4 GGUF ~4–5GB. At commodity object-storage rates (~$0.02/GB-month), storing a merged 8B model is trivial (~$0.30/month); the cost problem is *many* merged models per tenant. **Default: store adapters by default, generate merged/GGUF on demand and expire them, keep fp16 base models cached platform-side (shared), not per-tenant.**

### 2.6 The serving path and cost-per-million-tokens
Three paths: (a) platform-hosted dedicated endpoint, (b) platform multi-LoRA shared endpoint, (c) user downloads and self-hosts.

**Why (inference cost model, aside):** Cost per 1M output tokens on a self-hosted GPU ≈ (GPU $/hr) ÷ (tokens/sec × 3600 ÷ 1e6) ÷ utilization. Example: an H100 at ~$2/hr serving an 8B model at ~2,500 tok/s aggregate throughput = 9M tok/hr; at 100% utilization that's ~$0.22/1M tokens, but at a realistic 20% utilization it's ~$1.10/1M. **Engineering consequence:** self-hosting is dominated by *utilization*, not sticker GPU price. A dedicated H100 kept warm 24/7 is ~$1,440–$4,700/month depending on provider (RunPod H100 PCIe ~$1.99/hr community, Together dedicated H100 $6.49/hr — verified Jul 2026). Compare to frontier APIs (verified late Jul 2026): GPT-5.6 Luna $0.20/$1.20, Gemini 2.5 Flash-Lite $0.10/$0.40, Claude Haiku 4.5 $1/$5, DeepSeek V4 Flash $0.14/$0.28 per 1M in/out.

**Blunt conclusion on when fine-tune + self-host LOSES on cost:** if your total throughput is low or bursty (the endpoint sits idle much of the day), a fine-tuned 8B on a dedicated H100 can cost *more per token* than calling a frontier mini/flash model that is already better at instruction-following. Fine-tuning + self-hosting wins when: (1) sustained high utilization, (2) the frontier per-token price × your volume exceeds the warm-GPU monthly cost, (3) data residency/privacy forbids the API, or (4) latency/throughput SLAs require dedicated capacity. Multi-LoRA serving (many tenants' adapters on one warm base) is the only way small players make the economics work at low per-tenant volume.

---

## 3. Platform Concerns

*This section is engineering and product guidance, not legal advice. Consult qualified counsel for your jurisdiction and use case.*

### 3.1 License propagation
The fine-tuned model and LoRA adapter carry the base model's license forward, often with *added* obligations triggered by the fine-tune. Concretely (verify current text per version):
- **Llama (Community License, e.g., Llama 4, Apr 2025):** commercial use allowed, but the Llama 3.1/4 Community License states that if a licensee exceeds "700 million monthly active users in the preceding calendar month, you must request a license from Meta, which Meta may grant to you in its sole discretion." Derivative models must include "Llama" at the start of the name and display "Built with Llama"; the AUP is incorporated by reference; multimodal Llama variants exclude EU-domiciled individuals from using the weights directly. Not OSI-open — it is a bilateral contract under California law.
- **Gemma (Google Terms of Use):** commercial use allowed but Google can revoke; prohibits using the weights to train *competing* foundation models; requires notice/flow-down of the use policy to downstream users.
- **Qwen, Mistral open-weight models:** Apache-2.0 — clean, relicensable derivatives, no naming/AUP flow-down. (Some Mistral models, e.g., Codestral MNPL, are non-production-only; check per model.)
- **Unsettled law:** whether a fine-tuned model or LoRA adapter is legally a "derivative work," and who owns new weights, is not resolved in case law. Running behind an API can avoid *distribution*-triggered clauses but not use-restriction/competitor clauses.

**What the platform must surface:** the base license and its concrete obligations on the model card, at export, and (for Llama) an automated naming/attribution reminder. Block or warn on license-incompatible actions (e.g., exporting a Gemma derivative to train a competitor). **Verdict: expose license lineage everywhere; hard-code per-base obligation checklists.**

### 3.2 Tenant isolation
Datasets and weights are among the most sensitive assets a customer has. Enforce per-tenant storage isolation (separate buckets/prefixes with per-tenant KMS keys), network isolation for training jobs, and strict authz on artifact download. Shared base-model weights are fine (public); everything tenant-derived must be isolated. **Verdict: hard-code strict isolation; SOC-2-style controls are a sales requirement (Together, Predibase, OpenPipe all advertise SOC-2 / HIPAA-aligned options).**

### 3.3 Retention and deletion
Offer explicit retention windows and hard-delete of datasets, checkpoints, merged artifacts, and logs. This is both good practice and, under India's DPDP regime, increasingly an obligation.

### 3.4 India-specific data-protection considerations (not legal advice)
India's **Digital Personal Data Protection (DPDP) Act, 2023** was operationalized by the **DPDP Rules, 2025, notified 14 November 2025**, with a phased ~12–18 month rollout. It is consent-first, applies extraterritorially to processing connected to offering goods/services to individuals in India, mandates reasonable security safeguards (encryption/masking), breach reporting, and enhanced obligations (annual DPIA + audit) for "Significant Data Fiduciaries." For this platform: if users upload datasets containing Indian residents' personal data, you are likely a Data Processor (and possibly Fiduciary), so consent provenance, deletion rights, and breach workflows matter. **Verdict: build consent/retention/deletion and breach-notification hooks; document data flows; get counsel.**

### 3.5 Abuse vectors and what platforms actually implement
Two main vectors: (1) training on harmful/illegal data, (2) fine-tuning to strip safety alignment ("un-alignment"). What existing platforms actually do: OpenAI runs a **post-training safety assessment across 13 categories** and *blocks deployment* if too many examples fail — the most concrete implemented mitigation observed. Most open-weight platforms (Together, Fireworks, Predibase) rely primarily on AUP terms + abuse review rather than automated content gating of training data. **Verdict for a small team:** enforce an AUP, run a lightweight input-data screen and a post-train safety eval gate (mirror OpenAI's block-on-fail), log everything, and do not advertise "we remove safety filters."

### 3.6 Audit logging
Log who trained what, on which dataset hash, with which hyperparameters, producing which artifact, downloaded by whom, and when. This underwrites tenant trust, incident response, and DPDP/enterprise audits. **Verdict: expose immutable audit logs per tenant.**

---

## 4. Commercial Fine-Tuning Products: Catalogue (2026)

Confidence is High unless noted. Prices verified Jul–Aug 2026 where given; treat all as directional and re-verify before quoting to users.

**Tier 1 — Frontier-lab managed APIs**
1. **OpenAI fine-tuning** — platform.openai.com — SFT/DPO/RFT on gpt-4.1 family (RFT: o4-mini). Target: app devs. Pricing: e.g., GPT-4.1 training $25/1M tokens. Positioning: *the anchor incumbent — but winding down (closed to new orgs 7 May 2026; no new jobs after 6 Jan 2027).* **Worth deep study (see §5) as the UX benchmark, but not as a durable competitor.**
2. **Google Vertex AI / Gemini tuning** — cloud.google.com/vertex-ai — SFT + preference tuning on gemini-2.5-pro/flash/flash-lite; also open Gemma tuning. Target: GCP enterprises. Pricing: per training-token + per node-hour; fine-tuned inference carries a premium burndown from Gemini 3 onward. Positioning: enterprise, GCP-locked. Out of scope to compete with; study its console flow.
3. **Amazon Bedrock custom models** — aws.amazon.com/bedrock — SFT/continued-pretraining/RFT/distillation on Titan/Nova/Llama (e.g., Llama 3.2 fine-tuning, Mar 2025); fine-tuned models require Provisioned Throughput. Nova Pro fine-tune ~$0.008/1K tokens; storage ~$1.95–5/model/mo. Target: AWS enterprises. Available in Asia Pacific (Mumbai) for some workflows. Positioning: AWS-native, provisioned-throughput cost trap.
4. **Azure AI Foundry (OpenAI) fine-tuning** — learn.microsoft.com — SFT/DPO on gpt-4.1 family with regional data-residency tiers. Target: Azure enterprises. Positioning: OpenAI models with enterprise governance; likely outlives OpenAI's own self-serve platform.
5. **Mistral La Plateforme** — mistral.ai — managed SFT on Mistral models; API fine-tune ~$1–2/1M training tokens (12–25× cheaper than GPT-4.1); also `mistral-finetune` OSS + Forge enterprise. Target: EU/open-weight devs. Positioning: cheap, European, open-weight-friendly.
6. **Cohere** — cohere.com — fine-tuning for enterprise RAG/search/classify. Target: enterprise NLP. *Pricing not re-verified here — flag as possibly stale.*
7. **AI21** — listed on Bedrock; niche. Out of scope.
8. **Anthropic** — no self-serve public fine-tuning of Claude; limited custom offerings via Bedrock historically. **Confidence Medium; treat "no fine-tuning" as the practical answer for a small team.**

**Tier 2 — Independent managed platforms**
9. **Together AI** — together.ai — LoRA (default) + full SFT + DPO + RL on Llama/Qwen/Mistral/DeepSeek/Kimi etc. LoRA from $0.48/1M tokens (16B), up to specialized tiers ($10–60/1M for DeepSeek-R1/GLM-5); $4 min/job; dedicated H100 endpoint $6.49/hr. **Worth deep study (§5).**
10. **Fireworks AI** — fireworks.ai — SFT/LoRA/qLoRA via FireOptimizer; LoRA from $0.50/1M (≤16B), full SFT from $1.00/1M; multi-LoRA up to 100 adapters/deployment. **Worth deep study.**
11. **Predibase (now Rubrik)** — predibase.com — LoRA/Turbo-LoRA + RFT (GRPO); LoRAX multi-adapter serving; VPC option; RFT up to ~$20/1M tokens. Acquired by Rubrik (Jun 2025). **Worth deep study — deepest config + RFT.**
12. **OpenPipe (now CoreWeave)** — openpipe.ai — capture prod data → fine-tune small models; ART RL library; training from $4/1M tokens. **Legacy OpenPipe platform stops new training/inference 30 July 2026; migrating to W&B.** Study its data-capture flow; flag migration.
13. **Modal** — modal.com — serverless GPU (`@app.function(gpu="H100")`); not a fine-tuning product per se but a common substrate. Target: Python devs. Reference-stack candidate.
14. **Replicate** — replicate.com — hosted training/inference of open models via a simple API. Target: devs/prototypers.
15. **Baseten, Anyscale, Nebius AI Studio, Lamini** — managed training/serving; enterprise/infra-heavy. Mostly out of scope for direct competition.
16. **Hugging Face AutoTrain** — huggingface.co — no/low-code training on HF Hub models. Target: HF ecosystem users. Study as a low-code reference.
17. **Databricks Mosaic AI / Snowflake Cortex** — data-platform-native fine-tuning. Target: existing Databricks/Snowflake customers. Out of scope to compete; strong lock-in.

**Tier 3 — GPU clouds with fine-tuning docs/templates**
18. **RunPod** — runpod.io — H100 PCIe ~$1.99/hr, SXM ~$2.69/hr, A100 80GB ~$1.19–1.39/hr (verified 2026); serverless per-second. **Reference-stack GPU candidate.**
19. **Lambda Labs** — H100 PCIe ~$3.29/hr; research-friendly; 1-click clusters.
20. **Vast.ai** — marketplace, H100 from ~$1.49/hr; cheapest, variable reliability.
21. **CoreWeave** — enterprise; 8×H100 nodes (~$6.16/GPU-hr); now owns OpenPipe.
22. **Paperspace, JarvisLabs** — smaller GPU clouds with notebooks/templates.

**Tier 4 — No-code / low-code**
23. **Kiln AI** — kiln.tech — zero-code desktop app; fine-tunes Llama/GPT-4o via Together/Fireworks/OpenAI; synthetic data + evals; local-first (data stays on machine); free/open-source. **Worth deep study (§5) as the no-code benchmark.**
24. **Entry Point AI** — entrypointai.com — no-code SFT UI over provider back-ends; strong docs.
25. **FinetuneDB, Haven, Oxen.ai, Arcee AI, Axolotl Cloud, Unsloth (Unsloth Studio, incl. GGUF/safetensors export)** — assorted low-code/hosted tools around the OSS training stack.

**Tier 5 — Self-hostable OSS with UI**
26. **LLaMA-Factory** — github.com/hiyouga/LLaMA-Factory — Apache-2.0; Gradio "LlamaBoard" web UI; 100+ models; full/LoRA/QLoRA/DPO/KTO/ORPO/PPO; merge+export+vLLM serve. **Worth deep study (§5) as the self-host benchmark.**
27. **Axolotl** — config-driven YAML fine-tuning; the practitioner standard for reproducible configs.
28. **H2O LLM Studio, Ludwig** — GUI/declarative OSS training.

**Out of scope for a small team to compete with directly:** the hyperscaler-locked platforms (Vertex, Bedrock, Azure, Databricks, Snowflake) and pure GPU clouds — but several are *reference-stack* components. **Worth deep study:** OpenAI (UX benchmark), Together, Fireworks, Predibase, Kiln, LLaMA-Factory.

---

## 5. Interface and User-Flow Teardown (six products)

### 5.1 OpenAI fine-tuning (frontier-lab managed API — the UX anchor)
**Status flag (High confidence):** OpenAI is winding down self-serve fine-tuning. Per its deprecations page, developers were notified 7 May 2026; from 7 May 2026 orgs that never previously fine-tuned cannot create jobs; from 2 July 2026 orgs that haven't run inference on a fine-tuned model in 60 days are cut off; and per OpenAI's developer post, "existing active customers can continue running fine-tuning training jobs through January 6, 2027, after which creating new training jobs will no longer be possible" (inference persists until the base model is deprecated). The dashboard is login-gated and closed to new users, so the flow below is reconstructed from OpenAI's 2026 docs.

- **Upload:** Dashboard → fine-tuning → **+ Create** → upload JSONL under Training data. JSONL must be chat-format, ≥10 lines. Docs push token-counting to the tokenizer/cookbook rather than an in-UI counter (older gpt-4o-era UI showed counts; not verifiable in 2026).
- **Base model choice:** dropdown lists SFT-eligible models only — **gpt-4.1, gpt-4.1-mini, gpt-4.1-nano** (RFT: o4-mini). (Correcting a common assumption: gpt-4o/4o-mini are historical, not current SFT targets.)
- **Hyperparameters exposed:** `n_epochs`, `batch_size`, `learning_rate_multiplier`, all defaulting to "auto" (platform picks from dataset size). Recommendation is to leave them auto and only bump epochs ±1–2 if under/overfitting.
- **Cost shown before run:** no documented pre-run estimator; `trained_tokens` reported after. Users estimate manually.
- **Progress view:** job status, checkpoints (one per epoch, last 3 kept), and per-checkpoint metrics `full_valid_loss`, `full_valid_mean_token_accuracy`, `step_number`.
- **Eval surfaced:** validation metrics per checkpoint + a mandatory 13-category safety assessment that can *block* deployment. (Note: OpenAI's separate Evals product is itself deprecating — read-only 31 Oct 2026, shutdown 30 Nov 2026.)
- **Delivery:** result is `ft:gpt-4.1-...:org::id`; copy from Output model pane, call in Playground/API like any model.
- **Failure states/complaints:** the dominant "failure state" in 2026 is existential — the platform is closing. OpenAI's stated rationale (from its developer email, quoted by Tessl): "Newer base models like GPT-5.5 are much better at following instructions and formats than prior models... Prompt-based approaches are now cheaper and faster — as such, we're seeing fewer use cases that require fine-tuning."

### 5.2 Together AI (independent managed, LoRA-first)
- **Upload/validation:** web dashboard drag-and-drop JSONL (or CLI/SDK). Two-stage validation: local structural check on upload (catches non-UTF-8, malformed JSON) and server-side ingestion returning a `validation_report` — success shows `{valid:true, dataset_format:"conversation", nlines:7199}`; failure gives a line-specific error, e.g., *"Line 7: messages[1] must contain a role field."* Four dataset formats (conversational, instruction, preference, generic).
- **Base model choice:** dropdown of Llama/Qwen/Mistral/DeepSeek/Kimi etc.; LoRA vs full via a toggle (`--lora` default true).
- **Hyperparameters exposed with documented defaults:** `lora_r=8` (max 64), `lora_alpha=8` (1:1 with rank — notably *not* the 2×r community heuristic), `lora_dropout=0.0`, `learning_rate=1e-5`, `n_epochs=1`, `batch_size="max"`, `warmup_ratio=0.0`, `weight_decay=0.0`, `max_grad_norm=1.0`, `train_on_inputs="auto"`, `lora_trainable_modules="all-linear"`.
- **Cost:** billed per training token (dataset tokens × epochs) + optional eval tokens, $4 job minimum; hosting is a *separate* recurring charge (dedicated endpoint metered per minute, H100 $6.49/hr).
- **Progress/eval:** job dashboard at api.together.ai/jobs; `n_evals` runs validation during training; `n_checkpoints` controls saved checkpoints.
- **Delivery:** completed job is already a private model (`ml_...` id); deploy via `tg beta endpoints deploy` to a dedicated endpoint, then call `your-project/endpoint` as the model. LoRA can be served merged (recommended) or as swappable adapters.
- **Documented gotcha:** many fine-tunable base models are **not serverless** — you must stand up a dedicated endpoint to call them, and that warm endpoint is the real recurring bill (~$4,700/mo for a 24/7 H100), not the training. Community complaint: hosting cost dwarfs training cost.

### 5.3 Predibase (deep-config + RFT)
- **What/who:** engineers wanting deep control, RFT (GRPO), and multi-adapter serving; managed cloud or VPC (AWS/GCP/Azure) for data control; SOC-2.
- **Flow:** declarative config (Ludwig lineage) or SDK; supports quantization, LoRA, distributed training, hyperparameter tuning, and (since Aug 2025) direct classification-head training. Rich changelog cadence (releases roughly monthly through 2025).
- **Hyperparameters:** among the most exposed of managed platforms (adapter rank, quantization, target modules, RFT reward functions).
- **Serving/cost:** LoRAX packs many adapters on one GPU (claimed ~80% infra cost cut, up to 4× faster); Turbo LoRA adds speculative decoding. Two cost traps: dedicated serving is the recurring spend, and **RFT (up to ~$20/1M tokens) can be ~40× a comparable LoRA SFT run.**
- **Status flag:** acquired by Rubrik (Jun 2025); live rate card became harder to capture post-acquisition — **flag pricing as possibly stale; verify.**

### 5.4 Kiln AI (no-code, local-first)
- **What/who:** non-ML builders, PMs, subject-matter experts; free open-source desktop app (Win/Mac/Linux) + Python library.
- **Flow (zero-code, ~6 steps, one optional code step):** create task → generate/import data (interactive synthetic data with a "ladder" strategy) → pick model → one-click fine-tune → auto serverless deploy → evaluate. Builds "9 fine-tuned models in ~18 minutes of active work" per its own guide.
- **Back-ends:** dispatches to OpenAI/Together/Fireworks under the hood; **it is an orchestration UX, not its own trainer.** W&B metrics for Fireworks/Together jobs; OpenAI metrics via OpenAI's dashboard. For deterministic (classification) tasks it passes a validation set and surfaces `val_loss`; for generative tasks it uses its own eval tools.
- **Privacy:** local-first — datasets/keys never leave the machine (strong selling point vs cloud platforms, and relevant to DPDP-conscious users).
- **Limitation:** depth is capped by the underlying provider's exposed knobs; not for users who need raw config or self-hosted training.

### 5.5 LLaMA-Factory (self-hostable OSS with web UI)
- **What/who:** engineers who want to self-host training; Apache-2.0; 100+ models (Llama, Qwen3, Mistral, Gemma, DeepSeek, Phi, GPT-OSS, multimodal).
- **Flow:** launch Gradio **LlamaBoard** web UI (`llamafactory-cli webui`); pick model + template, choose method (full / LoRA / 2–8-bit QLoRA / DPO/KTO/ORPO/PPO / DoRA / GaLore / OFT), point at a dataset registered in `dataset_info.json`, set knobs, train; then Chat tab to test a checkpoint, and Export to merge + push to HF or serve via vLLM/SGLang.
- **Exposed knobs:** effectively *everything* — this is the opposite end of the spectrum from OpenAI; the UI exposes learning rate, epochs, batch size, gradient accumulation, LoRA rank/alpha/dropout/target modules, quantization, scheduler, etc.
- **Cost:** free software; you pay only for your own GPU (e.g., RunPod H100 $1.99/hr).
- **Common complaints (r/LocalLLaMA-style):** "zero-code is great right up until the loss won't converge and you're back reading the configs"; environment/CUDA/driver/VRAM compatibility is the main friction; no managed serving or multi-tenant isolation — it is a toolkit, not a product.

### 5.6 Fireworks AI (independent managed, serving-optimized)
- **What/who:** teams wanting fast serving + fine-tuning; OpenAI-compatible API.
- **Flow:** upload JSONL (strict format), select a tunable base (Llama 3, Qwen 2/2.5, Qwen3 non-MoE, DeepSeek V3/R1, Gemma 3/4, Phi 3/4, or custom same-architecture uploads), run SFT/LoRA/qLoRA via FireOptimizer.
- **Deployment (three options):** live-merge (merge LoRA into base, best latency, recommended), multi-LoRA (100s of adapters as add-ons on one deployment), or dedicated. **Documented limitation:** *serverless LoRA is not currently supported (as of Feb 2026)* — you need a dedicated deployment.
- **Cost:** LoRA from $0.50/1M (≤16B), full SFT from $1.00/1M; inference ~$0.20/1M (8B) to $0.90/1M (70B). Multi-LoRA is free of extra per-adapter cost on a dedicated deployment (Fireworks docs: "up to 100 LoRA adaptations to run simultaneously on a dedicated deployment without extra cost").
- **Complaint themes:** per-token cost at scale, model-catalog constraints for the newest checkpoints, and fine-tune portability (getting adapters in/out).

---

## 6. Cross-Product Comparison

### 6a. Knob exposure matrix (Report B hyperparameters × product)
E=exposed, A=auto/hidden-with-default, H=hidden/not-user-settable.

| Knob | OpenAI | Together | Fireworks | Predibase | Kiln | LLaMA-Factory |
|---|---|---|---|---|---|---|
| Method (SFT/DPO/RFT) | E (SFT/DPO/RFT) | E (SFT/DPO/RL) | E (SFT/LoRA/qLoRA) | E (+RFT/GRPO) | A (SFT) | E (all) |
| Epochs | A | E (def 1) | E | E | A | E |
| Learning rate | A (multiplier) | E (1e-5) | E | E | H | E |
| Batch size | A | E ("max") | E | E | H | E |
| LoRA rank r | H | E (8) | E | E | H | E |
| LoRA alpha | H | E (8) | E | E | H | E |
| LoRA dropout | H | E (0.0) | E | E | H | E |
| Target modules | H | E (all-linear) | E | E | H | E |
| Warmup / weight decay | H | E (0.0/0.0) | E | E | H | E |
| Quantization (QLoRA) | H | A | E | E | H | E |
| Full vs LoRA | H (LoRA-only) | E (toggle) | E | E | H | E |

**Read:** managed frontier APIs (OpenAI) hide almost everything behind "auto"; independent platforms expose LoRA internals; Predibase/LLaMA-Factory expose everything. **To be credible you must expose at minimum: method, epochs, learning rate, and LoRA rank/alpha — behind an "advanced" toggle — while defaulting everything.**

### 6b. Default values matrix (where published)
| Param | OpenAI | Together | Unsloth (recommended) | Common heuristic |
|---|---|---|---|---|
| Epochs | auto (~3–4 historically) | 1 | 1–3 | 1–3 |
| LoRA r | n/a (hidden) | 8 | 16 | 16 |
| LoRA alpha | n/a | 8 (=r) | 16–32 (2×r) | 2×r |
| LoRA dropout | n/a | 0.0 | 0–0.1 | 0.0–0.1 |
| Learning rate | auto | 1e-5 | 2e-4 (LoRA) | 1e-4–3e-4 |

**Disagreement flagged:** Together's `alpha=r` (1:1) contradicts the widely-taught `alpha=2r` and Unsloth/Microsoft convention. **Adjudication:** for a general default, use **r=16, alpha=32 (2×r), dropout=0.0–0.1, LR 2e-4 for LoRA**, applied to all linear layers — this matches the weight of practitioner evidence (QLoRA paper: LoRA on all linear layers is what matters most; diminishing returns past r=64). Together's lower defaults are conservative and optimized for their cost model; do not blindly copy them.

### 6c. Feature matrix
| Feature | OpenAI | Together | Fireworks | Predibase | Kiln | LLaMA-Factory |
|---|---|---|---|---|---|---|
| Dataset formats | chat JSONL | 4 (conv/instr/pref/text) | JSONL | declarative/CSV/JSONL | CSV/JSONL/synthetic | many + custom |
| Methods | SFT/DPO/RFT | SFT/DPO/RL, LoRA+full | SFT/LoRA/qLoRA | LoRA/RFT | SFT | full/LoRA/QLoRA/DPO/KTO/ORPO/PPO/DoRA |
| Model families | GPT-4.1 only | Llama/Qwen/Mistral/DeepSeek/Kimi | Llama/Qwen/DeepSeek/Gemma/Phi | open-weight | via providers | 100+ |
| Eval features | val metrics + safety gate | in-train evals | metrics | eval + HPO | evals + judge | chat test |
| Export | none (hosted only) | download (full) / adapter | merge/multi-LoRA | adapter/LoRAX | serverless deploy | safetensors/GGUF/merge |
| Deploy | ft: model id | dedicated endpoint | live-merge/multi-LoRA/dedicated | LoRAX autoscale | serverless | self / vLLM |

### 6d. Pricing model comparison (verified Jul–Aug 2026; re-verify)
| Product | Model | Representative job cost |
|---|---|---|
| OpenAI | per token | GPT-4.1 $25/1M train tokens |
| Together | per token + hosting | LoRA $0.48/1M (16B), $4 min; hosting H100 $6.49/hr |
| Fireworks | per token | LoRA $0.50/1M (≤16B); full $1.00/1M |
| Predibase | per token + GPU-hr | RFT up to $20/1M; LoRAX serving per-GPU |
| Mistral | per token | $1–2/1M train tokens |
| Bedrock | per token + provisioned throughput | Nova ~$0.008/1K; storage ~$2–5/mo; PT required for inference |
| RunPod (self-host) | per GPU-hr | H100 $1.99/hr → ~$1,440/mo 24/7 |
| Kiln / LLaMA-Factory | free software | pay only underlying provider / your GPU |

**Representative concrete example (verified late Jul 2026):** 500M training tokens LoRA-SFT on a mid-size open model on Together's specialized tier ≈ $1,500 training; but a warm H100 dedicated endpoint = $6.49 × 24 × 30 ≈ $4,673/month recurring — the recurring hosting bill, not training, is the real cost. (In INR at ~₹87.5/USD, ~₹1.31 lakh training, ~₹4.09 lakh/month hosting.)

---

## 7. The 70–80% Specification

### 7a. Table stakes (absence disqualifies)
1. **Dataset upload with real validation feedback** — line-level errors, format detection, row/token counts. Every serious competitor (OpenAI, Together, Fireworks) does line-specific validation; without it users can't debug.
2. **A curated base-model dropdown of open-weight models** (Llama, Qwen, Mistral, Gemma) with license shown. Frontier labs offer few models; your edge is open weights — but the *list must be curated, not overwhelming*.
3. **SFT + LoRA with sensible auto-defaults** and an advanced toggle exposing epochs/LR/rank/alpha. This is the universal baseline.
4. **A training-progress view with a live loss + eval-loss chart** and checkpoints. Users expect to watch the run.
5. **Held-out generation diff vs base + eval-loss + a safety gate.** OpenAI's block-on-fail safety assessment is the bar; skipping it is a legal/abuse liability.
6. **One-click serving OR clean export (safetensors + GGUF).** The result must be usable — either hosted endpoint or downloadable artifact.
7. **Model card with base license + provenance.** Compliance table stakes.
8. **Tenant isolation + delete.** Enterprise/DPDP requirement.

### 7b. Differentiators available to a small team
- **Serve the users OpenAI just abandoned.** With OpenAI self-serve closing (7 May 2026 / 6 Jan 2027), there is a concrete migration segment: teams with existing SFT datasets and no frontier home. A clean "import your OpenAI JSONL, fine-tune an open model, get an endpoint" flow is a wedge.
- **Transparent, predictable pricing.** Incumbents are opaque (Predibase post-Rubrik; Bedrock provisioned-throughput; Together's hidden hosting bill). A flat, published per-job + per-hour price with an upfront estimator (which OpenAI/Together do *not* show pre-run) is a differentiator.
- **Local-first / data-residency for India (DPDP).** Kiln proves demand for local-first. An India-hosted (Mumbai region) option with DPDP-aligned consent/retention/deletion targets a real, underserved segment.
- **Honest cost guidance.** A built-in "should you even fine-tune, or just call Gemini Flash?" calculator builds trust and differentiates from platforms incentivized to sell training.
- **Multi-LoRA serving for many small tenants** (the Predibase/Fireworks trick) is the only way to make hosting economics work for low-volume users — and a genuine cost advantage to pass on.

### 7c. Safe omissions (the 20–30% to skip), with cost of skipping
- **DPO/RFT/GRPO at launch.** Cost: power users and preference-data holders go elsewhere. Acceptable — SFT+LoRA covers the large majority of jobs; add DPO in v2.
- **Full fine-tuning.** Cost: users needing maximal quality on large models leave. Acceptable — LoRA on all linear layers matches full FT for most tasks (QLoRA evidence).
- **Frontier/proprietary base models.** Cost: can't fine-tune GPT/Claude/Gemini. Acceptable and increasingly moot (OpenAI closing; Anthropic has none).
- **Full HELM / leaderboard evals.** Cost: no publishable benchmark numbers. Acceptable — a lm-eval-harness slice suffices for go/no-go.
- **Human-annotation eval workflows, custom graders, distillation, continued pretraining.** Cost: enterprise/research users. Acceptable for v1.
- **MoE expert-targeted LoRA and 100+ model catalogs.** Cost: breadth-seekers. Acceptable — curation is a feature.

### 7d. Interface recommendation (prescriptive)
**Three screens.**
- **Screen 1 — Data:** drag-drop JSONL; immediate validation panel (format detected, row count, token count, per-line errors); a "10 examples minimum" gate; a sample-preview table. Pre-fill nothing; block "Next" until valid.
- **Screen 2 — Model & Method:** curated base-model dropdown (6–10 models, each with a one-line "good for" + license badge); method fixed to **SFT+LoRA** (v1); an **"Advanced" collapsed toggle** revealing epochs (default 3), learning rate (default 2e-4), LoRA rank (default 16), alpha (default 32), dropout (default 0.05). Show a **live cost + time estimate** (tokens × epochs × rate) — the thing incumbents don't. Pre-fill all defaults.
- **Screen 3 — Review & Launch:** summary + estimated cost + a plain-language "what you'll get (adapter + optional merged/GGUF export, an endpoint)" + AUP/license acknowledgment checkbox. One "Start" button.
- **Post-launch — Run view:** live loss/eval-loss chart, checkpoint list (best flagged), and on completion: held-out generation diff vs base, regression deltas, optional judge win-rate, safety gate status, then Deploy/Export buttons.

### 7e. Pricing recommendation
**Adopt a transparent two-part price: per-training-token + per-hour serving, with an upfront estimator and a hard job minimum.** Concretely: price LoRA-SFT at roughly **$0.50–$1.00 per 1M training tokens** (competitive with Fireworks/Together's $0.48–$1.00) with a **$4–5 job minimum**, and serving via **multi-LoRA shared endpoints billed per-hour** with scale-to-zero (pass through your RunPod ~$2/hr H100 economics with margin). Offer a **free tier of a few small jobs** (mirrors Predibase's $25 credit / Fireworks' free credits) to convert. Reasoning: this undercuts frontier fine-tuning (OpenAI $25/1M) by ~25×, matches the independent-platform going rate, and — crucially — the upfront estimator plus scale-to-zero multi-LoRA serving directly attacks the "hidden hosting bill" that is the top complaint against Together. (In INR: ~₹44–88 per 1M training tokens; ₹350–440 job minimum.)

---

## 8. Reference Stack and Build Sequence

### 8.1 One named choice per layer (conditional alternatives)
- **Training framework: Axolotl** (config-driven, reproducible, production-proven). *Alternative if you want a built-in web UI to fork: LLaMA-Factory. Alternative if VRAM/speed is the binding constraint: Unsloth.*
- **Orchestration: your existing job-queue + Docker on the GPU provider's API** (you have CI/CD skills; keep it boring). *Alternative if you want serverless GPU with per-second billing and sub-second cold starts: Modal.*
- **GPU provider: RunPod** (cheapest reliable self-serve H100 at ~$1.99/hr, per-second serverless, templates). *Alternative for guaranteed capacity/multi-GPU clusters: Lambda Labs; for absolute cheapest interruptible: Vast.ai.*
- **Storage: S3-compatible object storage with per-tenant prefixes + per-tenant KMS keys.** *Alternative for India residency: a Mumbai-region bucket.*
- **Monitoring: Weights & Biases** (both Together and Fireworks already emit W&B metrics; free tier is enough early). *Alternative self-hosted: MLflow.*
- **Frontend: a standard React/Next.js app** calling your API (you're a full-stack engineer; no reason to adopt Gradio in production).
- **Serving: vLLM with multi-LoRA (`--lora-modules`)** for shared endpoints; llama.cpp/Ollama path for GGUF exports.

### 8.2 Build sequence
- **v1 (narrowest viable):** SFT + LoRA only; 3–4 base models (Llama-3.x-8B, Qwen-2.5-7B, Mistral-7B, Gemma-2-9B); the 3-screen flow; Axolotl on RunPod; dataset validation; live loss + eval-loss; held-out diff + safety gate; adapter + merged + Q4 GGUF export; single dedicated-endpoint serving; model card with license. Knobs exposed: method (fixed SFT), epochs, LR, rank, alpha, dropout (behind Advanced).
- **v2 (defer):** DPO; multi-LoRA shared serving with scale-to-zero; more models; LLM-as-judge eval; OpenAI-JSONL import wizard; India/Mumbai residency + DPDP consent/retention tooling; cost estimator refinements.
- **v3+:** RFT/GRPO; full fine-tuning; VPC/on-prem; human-eval workflows.

**Hard problems, in the order they will surface:**
1. **Dataset validation edge cases** — malformed JSONL, wrong chat schema, encoding, token overflow. This is where users churn first; invest here before anything fancy.
2. **GPU reliability and OOM** — a job that dies at hour 3 of 4 with a cryptic CUDA error is your worst UX. Checkpoint aggressively; auto-retry; surface human-readable failure states (LLaMA-Factory's top complaint is exactly this).
3. **Serving economics** — a warm GPU per tenant bankrupts you; multi-LoRA is not optional at scale, and it's genuinely hard to operate.
4. **Merge/quantization correctness** — silently merging into a quantized base or double-quantizing ships degraded models; get the fp16-merge-then-quantize pipeline right.
5. **License/abuse gating** — the first time a user fine-tunes on something they shouldn't, you need logs, an AUP, and a safety gate already in place.

### 8.3 Honest failure modes for a small team
- **You out-manage the managed players and out-flexibility the OSS tools — pleasing neither.** Pick the "opinionated open-weight LoRA product for people leaving OpenAI" position and hold it.
- **Serving costs eat the margin** before you build multi-LoRA. Do not offer always-on dedicated endpoints on a flat price.
- **You compete on model breadth** (100+ models) and drown in per-model breakage. Curate.
- **You skip the boring reliability work** (validation, checkpointing, failure messaging) to build DPO/RFT nobody asked for yet.
- **Regulatory drift:** DPDP Rules (Nov 2025) phase in over 12–18 months; if you serve Indian users' data without consent/deletion hooks, you inherit real exposure.

## Caveats
- All pricing is verified Jul–Aug 2026 and volatile; treat every figure as directional and re-verify at quote time. Frontier model names/prices (GPT-5.6 tiers, Claude Fable 5, etc.) shift monthly. INR conversions use ~₹87.5/USD.
- OpenAI's dashboard is login-gated and closed to new users; its UI-flow details are reconstructed from 2026 docs, not first-hand screenshots. The wind-down dates (7 May 2026 / 2 Jul 2026 / 6 Jan 2027) are from OpenAI's deprecations page and developer post; a one-day discrepancy (7 vs 8 May) exists between the deprecations page and the blog banner.
- Predibase pricing became harder to verify post-Rubrik acquisition; OpenPipe is migrating to W&B (legacy platform stops new training 30 Jul 2026); Cohere fine-tuning pricing was not re-verified — all flagged as possibly stale.
- The legal status of fine-tuned models/LoRA adapters as "derivative works," and weight ownership, is genuinely unsettled in 2026 case law. This report is not legal advice.
- Quantization accuracy figures are drawn from multiple 2024–2026 studies (notably Kurtić et al. arXiv:2411.02355 and Song et al. arXiv:2505.20276) with different models/methods; exact deltas vary by architecture and quantization algorithm — use them as ranges, not guarantees.