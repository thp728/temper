# Report A — Upstream of the Training Loop: Data, Tokenization, Base Models, and Method Selection

*Three-part engineering briefing on building an LLM fine-tuning platform, as of August 2026. This is Report A. Reports B (training loop) and C (platform/market) are referenced but not covered here. Currency conversions use USD/INR = ₹87.5, the approximate rate as of August 2026; treat INR figures as indicative.*

Confidence labels (High/Medium/Low) are attached to non-obvious claims. Sources older than ~February 2025 are flagged as possibly stale.

---

## TL;DR

- **Build a LoRA/QLoRA-first platform on a deliberately small model matrix (Qwen3, Gemma, Llama, gpt-oss families), and treat tokenization/chat-template correctness — not hyperparameters — as your highest-risk surface.** The single most common way these platforms ship broken models is a training/inference chat-template mismatch or an untrained EOS token, both of which pass all obvious health checks (loss looks fine) and only surface as garbage generations. Hard-code the template resolution and EOS handling; do not expose them.
- **Auto-select the method from four inputs — dataset shape, dataset size, base-model size, and budget — defaulting to QLoRA (rank 16, α=32, all-linear targets).** Full fine-tuning and preference tuning (DPO/ORPO) are v1.5 features; ship SFT-only first. The 2025 "LoRA Without Regret" result (Thinking Machines, Sep 2025) means a well-configured LoRA is not a quality compromise for the small-to-medium datasets your users will upload, so full FT buys you little at large cost.
- **Block jobs on a small set of hard failures (unparseable file, <10 usable examples, all-empty completions, encoding corruption); warn on everything else.** Ruthless scoping: skip semantic dedup, PII auto-redaction, and MoE base models in v1. These cost real coverage only for a minority of users, and each is a large engineering surface relative to its payoff.

---

## 1. Pipeline map

The pipeline is a linear state machine with a validation gate near the front. Stages marked **[A]** are covered in depth here; **[B]** and **[C]** belong to later reports.

| # | Stage | Inputs | Outputs | Failure states | Typical duration | Report |
|---|-------|--------|---------|----------------|------------------|--------|
| 0 | **Upload & storage** | User file(s) (JSONL/CSV/Parquet), base-model choice, budget | Stored blob, job record | Corrupt upload, unsupported MIME, oversize | seconds–minutes | A |
| 1 | **Format & schema detection** | Raw file | Detected schema (instruction / chat / completion / preference), field map | Ambiguous/unrecognized schema | seconds | A |
| 2 | **Validation & normalization** | Parsed records | Cleaned dataset, validation report | Too small, empty fields, encoding errors, all-duplicate | seconds–minutes | A |
| 3 | **Dedup & length analysis** | Cleaned dataset | Deduped dataset, length histogram, max_len recommendation | Dataset collapses below minimum after dedup | seconds–minutes | A |
| 4 | **Train/eval split** | Deduped dataset | Train + eval shards | Too small to split | seconds | A |
| 5 | **Tokenizer & template resolution** | Base model, dataset schema | Tokenizer, chat template, special-token config, loss mask spec | Missing template, double BOS, untrained EOS | seconds | A |
| 6 | **Method selection** | Dataset shape/size, model size, budget | Chosen method (SFT LoRA/QLoRA/full/DPO), adapter config | No feasible method within budget | seconds | A |
| 7 | **Tokenization & packing** | Dataset, tokenizer, template, max_len | Tokenized, masked, packed tensors | Truncation loss, packing contamination | minutes | A/B |
| 8 | **Hardware provisioning** | Method, model size | GPU instance | No capacity, OOM at load | seconds–minutes | B |
| 9 | **Training loop** | Tensors, config | Checkpoints, loss/eval curves | OOM, divergence, NaN loss | minutes–days | B |
| 10 | **Evaluation** | Checkpoint, eval shard | Metrics, sample generations | Eval worse than base | minutes | B/C |
| 11 | **Merge/export & delivery** | Adapter/checkpoint | Merged weights, GGUF/adapter artifact | Merge dtype mismatch, broken template in export | minutes | B/C |
| 12 | **Serving/registry** | Artifact | Deployed endpoint or download | Template not propagated to inference | minutes | C |

**Decisions by stage (upstream only):** format detection (1); required-field policy, encoding normalization, min-size gate, dedup algorithm & threshold, length-outlier policy (2–3); split ratio & method (4); tokenizer resolution, template resolution, BOS/EOS handling, loss masking, packing vs padding, truncation, max_len (5, 7); method & adapter hyperparameters (6). This report specifies each.

**Spine principle:** the chat template and special-token configuration resolved at stage 5 must be the *same object* that is serialized into the delivered artifact at stage 11 and served at stage 12. Most silent failures are a break in this propagation.

---

## 2. Decision inventory (upstream)

Verdict key: **E** = expose to user, **I** = infer from inputs, **H** = hard-code. Defaults are single values, not ranges.

| # | Decision | Stage | Recommended default | Valid range | Verdict | Failure symptom |
|---|----------|-------|---------------------|-------------|---------|-----------------|
| 1 | Accepted file formats | 1 | JSONL (+ CSV, Parquet) | — | H | Parser error at upload |
| 2 | Schema auto-detection | 1 | Key-signature heuristic | 4 schema types | I | Wrong loss mask; garbage training |
| 3 | Required fields per schema | 2 | Non-empty completion/last-assistant turn | — | H | Empty-target rows silently no-op |
| 4 | Encoding normalization | 2 | UTF-8, NFC, strip control chars | — | H | Mojibake tokens; vocab drift |
| 5 | Min viable dataset size (block) | 2 | 10 usable examples | ≥10 | H | Job blocked / overfit |
| 6 | Recommended dataset size (warn) | 2 | <50 warns | — | I | Underfit, no measurable gain |
| 7 | Exact-dedup | 3 | SHA-256 on normalized text | on/off | H | Memorization, wasted compute |
| 8 | Near-dedup algorithm | 3 | MinHash+LSH, 128 hashes, 5-grams | — | I | Overfit to repeated cluster |
| 9 | Near-dedup Jaccard threshold | 3 | 0.8 | 0.7–0.9 | I (expose advanced) | Too aggressive: data loss; too loose: dupes remain |
| 10 | Length-outlier policy | 3 | Warn >p99; cap max_len at p95–p99 | — | I | Truncated targets; VRAM spikes |
| 11 | max_sequence_length | 5/7 | p95 of token lengths, capped at model limit & VRAM | 512–model max | I | Truncated answers; OOM |
| 12 | Train/eval split ratio | 4 | 95/5, eval capped at 1000 | 90/10–99/1 | I | No overfit signal / too little train |
| 13 | Tokenizer resolution | 5 | From base-model repo, exact revision | — | H | Loss incomparable; wrong ids |
| 14 | Chat-template resolution | 5 | Model's own `chat_template`; fail closed | — | H | Train/infer mismatch → subtle wrongness |
| 15 | BOS handling | 5 | `add_special_tokens=False` when template adds BOS | — | H | Double BOS → quality drop |
| 16 | EOS-in-target | 5 | Always include & train on EOS | — | H | Model never stops generating |
| 17 | pad_token | 5 | Distinct pad; never pad==eos unmasked | — | H | EOS masked → infinite generation |
| 18 | Loss masking | 7 | Completion/assistant-only | full/completion | I | Learns to parrot prompts |
| 19 | Packing vs padding | 7 | Packing **with** varlen masking | on/off | I | Cross-example attention contamination |
| 20 | Truncation policy | 7 | Right-truncate; drop if target lost | — | H | Trains on promptless/answerless rows |
| 21 | Base-model allow-list | 5/6 | Curated matrix (§5) | — | H | Unsupported arch breaks stack |
| 22 | Base vs instruct checkpoint | 6 | Instruct for chat data, base for completion/continued-pretrain | — | I | Template/format mismatch |
| 23 | Method selection | 6 | QLoRA (r=16, α=32, all-linear) | SFT/LoRA/QLoRA/full/DPO | I (expose override) | OOM, cost overrun, underfit |
| 24 | LoRA rank | 6 | 16 | 8–256 | I (expose) | Underfit (low) / wasted memory (high) |
| 25 | LoRA target modules | 6 | all-linear | attn-only / all | I | Underfit if attn-only |
| 26 | Quantization for QLoRA base | 6/7 | NF4 double-quant | nf4/int8/none | I | Quality loss (aggressive) / OOM (none) |
| 27 | License gate & naming propagation | 6/11 | Enforce per-family (§5) | — | H | Legal exposure downstream |
| 28 | Gated-repo access | 5 | Pre-provision tokens; block if no access | — | H | Job fails at model download |
| 29 | PII/safety screening | 2 | Warn-only regex scan (v1) | off/warn/block | E | Privacy leakage into weights |

---

## 3. Dataset ingestion and validation

### 3.1 Accepted formats and detection

**Default: accept JSONL as the first-class format; also accept CSV and Parquet, converting both to JSONL internally.** JSONL is the lingua franca of every major framework (TRL, Axolotl, LLaMA-Factory, and the OpenAI/Together/Fireworks APIs all consume JSONL). Detect by extension first, then sniff: attempt line-delimited JSON parse; on failure try a JSON array; on failure try CSV dialect detection; then Parquet magic bytes. Verdict: **hard-code** — no user benefits from a custom loader in v1, and each extra format is attack/parse surface.

### 3.2 Schema variants and auto-detection

Four schemas cover essentially all uploads (High confidence; corroborated by LLaMA-Factory docs, Unsloth docs, and Red Hat's SFT-format taxonomy, 2025):

1. **Instruction/response (Alpaca)** — keys `instruction`, optional `input`, `output`. Single-turn.
2. **Chat turns (ShareGPT/messages)** — a `conversations`/`messages` list of `{role/from, content/value}` objects. Multi-turn. Supports `system`, `human/user`, `gpt/assistant`, and tool roles.
3. **Completion-only** — `text`, or `prompt`+`completion`. Used for continued pretraining or raw completion tuning.
4. **Preference pairs** — `prompt`, `chosen`, `rejected` (DPO/ORPO), or the KTO variant with a binary label.

**Auto-detect by key signature, in priority order:** presence of `chosen`+`rejected` → preference; presence of `messages`/`conversations` list → chat; presence of `instruction` → Alpaca; presence of lone `text`/`prompt`+`completion` → completion. This is a deterministic function of the JSON keys — verdict **infer**, and surface the detected schema back to the user for one-click confirmation. Do not silently proceed on ambiguous keys (e.g., a file with both `instruction` and `messages`); that is a **warn** requiring confirmation.

**Why this matters mechanically:** the schema determines the *loss mask*. TRL's `SFTConfig` defaults `completion_only_loss=None`, which means "compute loss on the completion for prompt-completion datasets, and on the full sequence for language-modeling datasets" (trl `sft_config.py`, v0.27.x, 2025). If you misclassify a prompt-completion dataset as language-modeling, the model trains on the prompt tokens too — a different objective. *Engineering consequence:* detection errors are not cosmetic; they change what the model learns. Confirm schema before tokenizing.

### 3.3 Required vs optional fields

- Alpaca: `instruction` and `output` required; `input` optional. A row with empty `output` is a no-op target — **block the job if the great majority of rows have empty targets; drop-and-warn otherwise.**
- Chat: at least one user turn and one assistant turn; the final assistant turn must be non-empty (it is the training target under assistant-only loss).
- Completion: non-empty `text`/`completion`.
- Preference: all of `prompt`, `chosen`, `rejected` non-empty; `chosen != rejected`.

Rows failing these are dropped with a per-row reason logged; if drops push the dataset under the size gate, the job blocks.

### 3.4 Encoding and normalization

**Default: decode as UTF-8 (with BOM stripping), apply Unicode NFC normalization, strip C0/C1 control characters except `\n`/`\t`, and normalize newlines to `\n`.** Verdict **hard-code**. Rationale: inconsistent encodings produce replacement characters and rare byte sequences that tokenize into long strings of fallback bytes, distorting length statistics and wasting sequence budget. NFC prevents the same visual string existing as two different token sequences (which also defeats dedup). Do **not** lowercase or strip punctuation for training text (unlike dedup shingling, §3.5) — that destroys signal.

### 3.5 Deduplication and near-duplicate detection

Run two passes:

1. **Exact dedup** — SHA-256 over the normalized full record; drop collisions. Cheap, always on.
2. **Near-dedup** — **MinHash + LSH with a Jaccard-similarity threshold of 0.80**, 128 permutations/hashes, over lowercased n-gram shingles.

**Why 0.80 and MinHash+LSH:** this is the default that large-scale curation pipelines converged on. SlimPajama's team reported verbatim: "we were able to prune 49.6% of bytes from RedPajama, leaving us with the 627B token SlimPajama dataset. To perform deduplication we used MinHashLSH … with a Jaccard similarity threshold of 0.8. We construct document signatures on top of pre-processed lower-cased 13-grams" (Cerebras, 2023 — flagged >18 months old but still the reference implementation). NVIDIA NeMo Curator's `MinHashDeduplicator` defaults to `num_hashes=128, jaccard_threshold=0.8` (2025–2026). Some clinical/curated corpora tighten to 0.85 with 256 hashes over 5-grams. Threshold semantics: at 0.7 you catch heavy paraphrases but risk deleting legitimately similar-but-distinct examples; at 0.9 you catch only near-verbatim copies (a WildChat study found MinHash at Jaccard 0.9 caught 31.3% of near-dupes vs 5.8% for byte-exact — arXiv 2605.09611, 2026). For instruction datasets, which are shorter than pretraining documents, **use word-level 3-grams rather than 13-grams**, because 13-grams rarely overlap in short texts. Verdict: **infer** the pipeline; **expose the threshold as an advanced knob** with 0.8 default.

**Why duplicates and near-duplicates distort training (Why-aside):** SGD weights each example by how often it appears. If a cluster of 500 near-identical examples sits in a 5,000-row dataset, the gradient signal for that cluster is effectively 10× any unique example, so the model over-memorizes that pattern and its held-out loss on everything else worsens. Deduplication is not primarily a storage optimization; it is a *reweighting correction*. *Engineering consequence:* report the duplicate rate in the validation summary and dedup before computing the size gate, because a "5,000-row" dataset that is 80% duplicates is really a 1,000-row dataset.

Skip **semantic/embedding dedup** in v1 (embedding every row + ANN clustering is a GPU-bearing service for marginal gain over MinHash on typical uploads). Note the gap explicitly to users.

### 3.6 Length distribution analysis

Tokenize a sample (or all) rows, build a token-length histogram, and compute p50/p95/p99/max. Use this to (a) set `max_sequence_length` (§4.6) and (b) flag outliers. **Default: warn on any example above the 99th percentile, and set max_len at p95–p99 capped by the model's context limit and the VRAM budget.**

**Why length outliers distort training (Why-aside):** a handful of very long examples force either truncation (losing the target) or a large `max_len` that quadratically inflates attention memory/time and forces a tiny batch size, which raises gradient variance. One 32k-token row in a corpus whose p95 is 800 tokens can single-handedly set your VRAM ceiling. *Engineering consequence:* cap `max_len` from the distribution, not from the single longest row; surface how many rows will be truncated at the chosen cap.

### 3.7 Minimum viable dataset size

**Block below 10 usable examples; warn below 50; recommend 1,000+ for style/behavior tasks.** Evidence, most-to-least authoritative:

- **OpenAI fine-tuning docs (2025):** the enforced minimum is **10 examples**, verbatim: "The minimum number of examples you can provide for fine-tuning is 10 … We see improvements from fine-tuning on 50–100 examples … We recommend starting with 50 well-crafted demonstrations and evaluating the results."
- **Fireworks AI:** minimum **3 examples** (max 3M) (docs.fireworks.ai, 2026) — the lowest floor in the market.
- **Predibase:** **10 rows** as the seed minimum for their augmentation pipeline; meaningful results shown from ~300–800 rows (docs.predibase.com, 2026).
- **Together AI:** enforces a minimum sample count client-side (exact number not published in accessible docs — verify at integration); default epochs 1, default LoRA rank 8.
- **Google Vertex AI (Gemini SFT):** "provide at least 100 to 500 examples" (cloud.google.com, 2025).
- **AWS Bedrock:** no single universal minimum — record counts follow per-model quotas (docs.aws.amazon.com, 2025/2026).
- **LIMA (Zhou et al., "LIMA: Less Is More for Alignment," NeurIPS 2023, arXiv 2305.11206):** a 65B LLaMA "fine-tuned with the standard supervised loss on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling" (750 mined from Stack Exchange/wikiHow/WritingPrompts + 250 manually written; ~750k tokens) produced a competitive instruction-follower — the canonical "less is more" result. But replications show 1,000 LIMA examples *underperform* on knowledge benchmarks (MMLU) versus larger sets (Databricks LIMIT, arXiv 2311.13133), and translation tasks needed 200k+ segments to beat baseline (Vieira et al., 2024, via Latitude).

*Interpretation:* the right minimum depends on the task, and it is computable. For **style/format/tone** tasks, 50–1,000 examples suffice. For **new knowledge or hard reasoning**, tens of thousands are needed. **Infer** a task-type hint from schema and content and set the warning threshold accordingly; **hard-code** the 10-example hard floor.

### 3.8 Train/eval split

**Default 95/5, with the eval set capped at 1,000 examples and floored so that datasets under ~200 rows use a fixed held-out count (e.g., 10–20) rather than 5%.** Axolotl's example configs use `val_set_size: 0.05` (5%); Together's quickstart uses a 90/10 split (`split_ratio=0.9`); Fireworks auto-carves a validation slice from training data by default. The eval set exists to produce an overfitting signal (train loss ↓ while eval loss ↑) and sample generations; beyond ~1,000 examples the marginal statistical value is low and you are just spending training data. Verdict **infer**, expose as advanced. Always split **after** dedup and **stratify** on schema/length buckets where possible so eval is representative.

### 3.9 PII and safety screening

**v1 default: run a regex/pattern scan (emails, phone numbers, credit-card/Aadhaar-like patterns, API keys) and *warn* with counts; do not auto-redact and do not block.** Verdict **expose** (let the user choose off/warn/block). Auto-redaction (e.g., Microsoft Presidio) risks corrupting legitimate training targets and is a large surface; defer to v2. Be explicit: fine-tuning *memorizes* training data, so PII in the dataset can resurface in generations — state this in the UI. Safety-content screening (CSAM, etc.) is a legal/ToS matter for Report C; at minimum, log and retain the ability to block.

### 3.10 Block vs warn — the exact policy

**Block (job cannot start):** unparseable file; unrecognized schema after all heuristics; fewer than 10 usable examples after dedup/validation; a schema whose required target field is empty in essentially all rows; encoding that cannot be decoded as UTF-8 after fallback; requested base model not on the allow-list or gated without access.

**Warn (job proceeds, flagged in report):** 10–49 examples; high duplicate rate; length outliers/expected truncation; PII matches; ambiguous-but-resolvable schema; heavily imbalanced roles; missing system prompts.

---

## 4. Tokenization and chat templates

This is the highest-frequency silent-failure surface. A job completes, the loss curve looks healthy, and the model is subtly or completely wrong at inference. Treat everything in this section as **hard-coded correctness**, not user configuration.

### 4.1 Tokenizer resolution

**Resolve the tokenizer from the exact base-model repository and pin the revision/commit.** Never substitute a "compatible" tokenizer from a sibling model. The tokenizer defines the integer id space the model's embedding matrix was trained against; a mismatch silently maps text to the wrong rows of the embedding table.

**Why loss is only comparable under fixed tokenization (Why-aside):** next-token cross-entropy is `L = −ln p(actual_next_token)`, averaged over target tokens. It is measured *per token*, and the token boundaries are defined by the tokenizer. A tokenizer that splits text into more, shorter tokens makes each token more predictable (lower per-token loss) without the model being better. So a loss of 1.9 with tokenizer X and 2.1 with tokenizer Y say nothing about relative quality. *Engineering consequence:* never compare loss across runs with different base models/tokenizers, and never show users a cross-model loss leaderboard. Only compare loss for the same base model + tokenizer + masking scheme.

**What a loss of 2.1 vs 0.8 physically means (Why-aside):** perplexity = `exp(L)`. `exp(2.1) ≈ 8.2`; `exp(0.8) ≈ 2.2`. A loss of 2.1 means the model is, on average, as uncertain as if choosing uniformly among ~8 equally likely next tokens; at 0.8 it is effectively choosing among ~2 (perplexity as "effective branching factor" — The Gradient; Comet, 2025). For instruction tuning on constrained outputs, converged completion-only loss commonly lands roughly in the 0.5–1.5 band; a training loss stuck near 2+ that will not fall usually means the target is genuinely high-entropy (creative text) **or** the loss mask/template is wrong (you are asking the model to predict unpredictable prompt tokens). *Engineering consequence:* use absolute loss bands only as a coarse alarm, and always pair them with sample generations — loss alone cannot distinguish "hard task" from "broken pipeline."

### 4.2 Chat-template resolution and why families differ

Each instruct model ships a Jinja `chat_template` in its tokenizer config defining exactly how roles are wrapped: Llama 3 uses `<|begin_of_text|>` and `<|start_header_id|>role<|end_header_id|>` … `<|eot_id|>`; Qwen uses `<|im_start|>role` … `<|im_end|>`; Mistral uses `[INST] … [/INST]`; Gemma uses `<start_of_turn>role` … `<end_of_turn>`. These are not interchangeable — they were the exact byte patterns the model saw during its own instruction tuning.

**Default: always call `tokenizer.apply_chat_template(...)`, never hand-concatenate role strings.** Resolve the template from the model's own tokenizer. If a base (non-instruct) model has no template and the user uploaded chat data, either apply a standard template via TRL's `clone_chat_template()`/`setup_chat_format` (and train the new special tokens) or fall closed with a clear error. Verdict **hard-code**.

**What breaks on a train/infer template mismatch:** if you train with template A and the serving stack applies template B, the model sees role-delimiter tokens at inference it never learned to condition on, and the very first generated tokens are off-distribution — outputs degrade from "subtly wrong tone" to "ignores the system prompt" to "gibberish." This is the modal production bug (High confidence; documented repeatedly across r/LocalLLaMA and framework issue trackers, 2024–2026, e.g. the LLM post-training cheatsheet: "using the wrong model's template … causes a sharp performance drop"). **Detection:** at export, re-tokenize a fixed probe conversation through both the training template and the artifact's serialized template and assert byte-for-byte equality of the input ids.

### 4.3 Special tokens: BOS, EOS, pad

**Double-BOS:** many chat templates already emit the BOS token, and calling `tokenizer(text)` with `add_special_tokens=True` on top adds a second one. Daniel Han (Unsloth) documented double-BOS on Llama-3 and Gemma-it (X/Twitter, May 2024) as a measurable quality bug; the same class of bug appears in llama-cpp-python (issue #1501). **Default: when the template injects BOS, tokenize with `add_special_tokens=False`.** Detection: scan the first two token ids of each tokenized example for a repeated BOS id; assert count of BOS ≤ 1.

**Untrained tokens → NaN:** Llama-3 *base* shipped with untrained reserved/special tokens (`<|eot_id|>`, header tokens); training on them (e.g., applying the instruct template to the base model) produces NaN gradients. Unsloth's fix sets these embeddings to the mean of the trained embeddings (they found 287 untrained tokens; Unsloth Llama-3 blog, 2024). **Detection:** check for NaN in the loss on step 1; if using instruct-template tokens on a base model, initialize those embeddings before training.

**EOS must be trained:** the model only learns to stop if the EOS token appears in the (unmasked) training target. **Default: append EOS to every target and include it in the loss.**

**pad_token must not silently equal EOS:** the classic bug — setting `pad_token = eos_token` and using a collator that masks pad tokens causes *every* EOS in the labels to be masked to −100, so the model never learns to stop and generates until `max_new_tokens` at inference (documented across HF transformers issue #23530, mlabonne/llm-course #33, the Falcon QLoRA tutorial, and multiple HF forum threads, 2023–2024; still a live footgun in 2026). **Default: use a distinct pad token; if none exists, add one (and resize embeddings) or reuse an unused reserved token — never an unmasked EOS.** Detection: assert `pad_token_id != eos_token_id`, or, if they must be equal, assert the label mask masks *only* padding positions and not in-sequence EOS.

### 4.4 Loss masking (completion-only vs full-sequence)

**Default: completion-only (prompt-completion) / assistant-only (chat).** In TRL, `completion_only_loss` defaults to `None` (loss on completion for prompt-completion data), and `assistant_only_loss=True` restricts loss to assistant turns but requires the template to contain `{% generation %}`/`{% endgeneration %}` markers (TRL auto-patches known families like Qwen3; SmolLM3 ships them). Axolotl expresses the same via `train_on_inputs: false`.

**Measured effect:** masking the prompt/user tokens means the model is graded only on what it should *produce*, not on reciting the instruction. Training on the full sequence teaches the model to also model the prompt distribution, which for instruction data wastes capacity and can make the model parrot or continue prompts. For continued pretraining or completion tasks, full-sequence loss is correct. Verdict: **infer** from schema (completion-only for instruction/chat, full for completion/continued-pretrain). **Symptom when wrong:** if you accidentally train full-sequence on instruction data, the model tends to echo/restate instructions and its completion-only eval loss is deceptively low because prompt tokens (which it now models) are easy.

### 4.5 Packing vs padding and attention contamination

**Default: packing ON, but only with variable-length ("varlen") attention that respects example boundaries.** Packing concatenates short examples to fill the sequence length (Axolotl `sample_packing: true`; TRL packing with the `bfd` strategy enables padding-free), giving a 3–5× throughput gain by eliminating pad waste.

**The contamination risk:** if you naively pack and run dense attention, tokens in example 2 attend to example 1, corrupting the training signal. The fix is to pass `position_ids`/`cu_seqlens` so flash-attention uses `flash_attn_varlen_func` and computes attention only within each example (Kundu et al., arXiv 2407.09105, 2024; TRL `padding_free=True` with `DataCollatorWithFlattening`; IBM Research, 2024). **Only enable packing for models/stacks that support this.** Note a real 2026 breakage: Qwen3.5's 3D `position_ids` were misinterpreted as packed-sequence boundaries and caused illegal-memory-access crashes in flash-attention (transformers issue #44910, 2026) — a reason to keep packing behind a per-model capability flag. Verdict **infer** (per-model). **Symptom when wrong:** eval loss plateaus higher than expected and generations show odd cross-topic bleed; hard to spot without the varlen assertion.

### 4.6 Truncation and max sequence length

**Default: set `max_sequence_length` to the p95 of the dataset's token-length distribution, capped by the model's context window and the VRAM budget; right-truncate; and drop (with a warning) any example where truncation would remove the target.** Truncating the *answer* trains the model on incomplete targets and can teach non-termination. Detection: after tokenization, assert that the EOS/last-assistant token survives truncation for every retained row; count and report truncations.

---

## 5. Base model support

### 5.1 What differs operationally between architecture families

For a fine-tuning platform, the operationally relevant differences are: (1) chat-template format and special tokens (§4); (2) tokenizer implementation quirks; (3) dense vs Mixture-of-Experts (MoE); (4) attention variant (standard vs sliding-window vs the linear/hybrid attention in newer Qwen); (5) license. Architectural internals beyond these do not change your pipeline.

**MoE is the big operational fault line.** MoE models (many DeepSeek, Qwen, Llama-4, GLM, and the gpt-oss-120B variant) route each token through a subset of "expert" MLPs. They break common training stacks in specific ways: LoRA target-module selection must account for expert layers (PEFT added `target_parameters`/per-expert rank handling for exactly this — its docs show `effective_r = max(1, r // num_experts)`), memory footprint is set by *total* not *active* parameters, and router/load-balancing behavior interacts badly with small-batch LoRA. **Recommendation: exclude MoE base models from v1** and state the cost (you lose the very largest and some best coding models). Dense models are the safe default surface.

### 5.2 Base vs instruct checkpoints

- **Chat/instruction/preference data → instruct checkpoint.** The instruct model already knows the chat template and role structure; you are adapting behavior.
- **Completion data or continued pretraining → base checkpoint.** Base models have no chat template; imposing one is wrong.

Verdict **infer** from schema, expose an override. **Symptom when wrong:** fine-tuning a base model with an instruct template but untrained special tokens → NaN/garbage (§4.3); fine-tuning an instruct model on raw completion data → it fights its own chat formatting.

### 5.3 Size tiers and the VRAM arithmetic (weight storage & inference only)

**Why: bytes-per-parameter (Why-aside).** Weight memory = `params × bytes_per_param`. Bytes per parameter by precision: fp32 = 4; fp16/bf16 = 2; int8 = 1; int4/NF4 ≈ 0.5 (NF4 with double-quantization adds ~0.4 bits/param, so ~0.5–0.6 in practice). So a 7B model is ~14 GB in bf16, ~7 GB in int8, ~4 GB in NF4; a 70B model is ~140 GB in bf16 but ~35–43 GB in NF4 (Spheron, 2026; Lyceum, 2026). *Engineering consequence:* the quantization choice, not the parameter count alone, decides which GPU a model fits on — this is the gate that determines your supported matrix.

**Why: KV cache for inference (Why-aside).** At inference you also pay for the KV cache, which grows linearly with context length and batch: roughly ~0.25 MB/token for a 7B model and ~2.5 MB/token for 70B in fp16 (VRLA Tech, 2026). At 32k context a 70B model adds ~10 GB of KV cache on top of weights. *Engineering consequence:* budget weights + ~15–20% overhead + KV cache when sizing serving hardware; long-context serving can cost more in KV cache than in weights.

Approximate **inference/weight-storage** footprint (weights + ~15% overhead, short context):

| Model size | bf16 (2 B/p) | int8 (1 B/p) | NF4/int4 (~0.55 B/p) | Smallest single GPU (NF4) |
|-----------|--------------|--------------|----------------------|---------------------------|
| 1–2B | 2–4 GB | 1–2 GB | ~1 GB | any 8 GB card |
| 3–4B | 6–8 GB | 3–4 GB | ~2 GB | any 8 GB card |
| 7–8B | ~14–16 GB | ~7–8 GB | ~4–5 GB | RTX 4090 24 GB |
| 12–14B | ~28 GB | ~14 GB | ~8–9 GB | RTX 4090 24 GB |
| 27–32B | ~60 GB | ~30 GB | ~18–20 GB | RTX 4090/5090, L40S 48 GB |
| 70B | ~140 GB | ~70 GB | ~38–43 GB | L40S 48 GB / A100 80 GB |
| 120B+ | 240 GB+ | 120 GB+ | 65 GB+ | H100 80 GB / H200 141 GB |

(Full *training* VRAM — optimizer states, gradients, activations — is Report B. As a preview multiplier over inference: QLoRA ≈ 1.2–1.5×, LoRA ≈ 1.5–2×, full FT ≈ 3–4× per VRLA Tech, 2026.)

Common datacenter/consumer GPUs and VRAM: RTX 4090 (24 GB), RTX 5090 (32 GB), L40S (48 GB), A100 (40/80 GB), H100 (80 GB), H200 (141 GB).

### 5.4 Licenses and propagation to derived models

This is a hard gate because the base-model license *flows through to your users' fine-tuned outputs*.

- **Apache 2.0 (Qwen3/3.x, Gemma 4, gpt-oss 20B/120B, OLMo):** most permissive. No naming or MAU restrictions; fine-tuned derivatives are the user's to license freely. **Prefer these as your default surface.** Note: Google moved Gemma to Apache 2.0 with Gemma 4 (2026); earlier Gemma releases used the more restrictive Gemma Terms — check the specific checkpoint.
- **Llama Community License (Llama 3.x/4):** commercial use is free *below 700M monthly active users*, above which a separate Meta license is required (Llama 4 Community License, Meta, April 2025). It also requires "Built with Llama" attribution and that derivative model names begin with "Llama," and requires derivatives to carry the same license. These obligations **propagate to every downstream user of a Llama-derived fine-tune** — a Llama derivative "must itself be distributed under the Llama 3 Community License" (wcr.legal; promise.legal, 2025). Llama 4's Acceptable Use Policy additionally excludes multimodal rights for EU-based entities. *Platform consequence:* if you support Llama, you must auto-prepend "Llama" to output model names and surface the attribution/MAU terms to the user.
- **Gemma Terms (older Gemma):** permissive commercially but with use restrictions and a downstream flow-through; Google retains a remote-restriction right.
- **Others:** some Mistral models are Apache 2.0 (permissive) while others (e.g., Codestral) require a paid license; certain Mistral models carry a $20M monthly-revenue threshold; Cohere/CC-BY-NC weights are non-commercial. **Block non-commercial-licensed weights from a commercial platform.**

Verdict **hard-code** the license gate and the naming/attribution propagation per family.

### 5.5 Gated repos

Several checkpoints (notably Meta Llama, some Gemma) are *gated* on Hugging Face and require access approval before download. **Pre-provision access tokens for every model on your allow-list and verify download access at allow-list-build time, not at job time.** If a user selects a model your service account cannot fetch, block with a clear message rather than failing mid-provision.

### 5.6 Recommended launch matrix (small team, 70–80% coverage)

Reasoning: cover the most-fine-tuned families across three size tiers, prefer Apache-2.0 licenses, dense-only, and pick sizes that fit single consumer/datacenter GPUs under QLoRA.

| Tier | Model (instruct + base) | Why included | License |
|------|-------------------------|--------------|---------|
| Small (1–4B) | **Qwen3 ~1.7–4B**, **Gemma small (E-series/4B)** | Cheap, fit any GPU, fast iteration, on-device targets | Apache 2.0 |
| Small-mid (7–9B) | **Llama 3.1 8B**, **Qwen3 ~8B** | The most-fine-tuned tier in the world; huge ecosystem | Llama Community / Apache 2.0 |
| Mid (12–14B) | **Qwen3 ~14B**, **Gemma ~12B**, **Phi-4 (~14B)** | Best quality that still QLoRAs on a 24 GB card | Apache 2.0 / MIT |
| Large (27–32B) | **Qwen3 ~32B**, **Gemma ~27B** | Quality ceiling for single-GPU QLoRA on 48 GB | Apache 2.0 |
| Optional edge | **gpt-oss-20B** | OpenAI-lineage reasoning at Apache 2.0, 16 GB-class | Apache 2.0 |

**What to leave out and the cost:**
- **MoE models (gpt-oss-120B, large DeepSeek/Qwen/GLM MoE):** you lose top-end coding/reasoning and the "run a 120B on one 80 GB GPU" story. Cost: the handful of power users who want frontier open models. Acceptable for v1.
- **70B+ dense:** requires multi-GPU or 80 GB cards and long jobs; defer. Cost: users chasing maximum single-model quality.
- **Brand-new releases (<2 weeks old):** they routinely ship with broken chat templates/tokenizers (Unsloth repeatedly patches these on launch weekends, upstreaming fixes for gpt-oss, Qwen3, Llama 4, Mistral, and Gemma per Pinggy, 2026). Cost: hype-cycle users. Add models only after templates stabilize.
- **Non-Apache/Community-licensed niche models (Mistral paid, Cohere NC):** legal surface > payoff.

This matrix gives roughly 70–80% of the practical fine-tuning demand (instruction/chat tuning of 4–32B dense models) at a fraction of the engineering surface.

---

## 6. Method selection

### 6.1 The methods, with data/hardware/cost/quality

**Full fine-tuning (Full FT).** Updates all weights. Solves: maximal capacity change, new knowledge/large domain shift, when you have lots of data. Data: typically tens of thousands+ examples to justify it. Hardware floor: ~3–4× inference VRAM (for 70B full FT you're at ~400–600 GB — roughly eleven 80 GB cards, per Spheron 2026). Quality: the historical gold standard. Cost/wall-clock: highest.

**LoRA.** Freezes base weights; trains two small low-rank matrices per targeted layer.

**Why: LoRA parameter arithmetic (Why-aside).** A weight matrix `W` of shape `d×k` normally has `d·k` parameters. LoRA replaces the *update* `ΔW` with `B·A`, where `B` is `d×r` and `A` is `r×k`, for `r ≪ min(d,k)`. Trainable params drop from `d·k` to `r·(d+k)`. Concrete: a Llama-class attention projection with `d=k=4096` has `4096² ≈ 16.8M` params; at `r=16`, LoRA trains `16·(4096+4096) ≈ 131k` — a **~128× reduction** per matrix. Rank `r` bounds the *rank* (expressive dimensionality) of the update, so a low `r` cannot represent an update whose true rank is higher, but it costs proportionally less memory. *Engineering consequence:* `r` trades representable capacity against memory linearly; you tune it to the task's complexity, not the model's size.

Data: works from ~1,000 examples up. Hardware: ~1.5–2× inference VRAM. Quality: with the right configuration, matches full FT for most post-training (see 6.2).

**QLoRA.** LoRA on top of a base model quantized to 4-bit NF4; adapters stay in bf16. Solves: fitting large models on one GPU. The original QLoRA paper (Dettmers et al., arXiv 2305.14314, 2023) "reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance," and reports that "NF4 with double quantization fully recovers the 16-bit LoRA … performance" (FP4 lagged by ~1 point); their Guanaco model reached "99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU." In practice QLoRA runs ~1–3% below full FT and ~0.5–1% below 16-bit LoRA (Spheron, 2026) — usually not meaningful for production tasks. **This is the platform default.**

**Other PEFT variants:**
- **DoRA** (magnitude+direction decomposition): closes ~half the LoRA→full-FT gap for ~5–10% more VRAM (Spheron PEFT guide, 2026). Worth exposing as an advanced quality option once v1 is stable. (Caveat: DoRA does not reduce catastrophic forgetting relative to LoRA — arXiv 2604.04516, 2026.)
- **rsLoRA**: rescales the adapter by `α/√r` instead of `α/r`, stabilizing training at high rank (PEFT built-in). Enable automatically when `r` is large.
- **LoRA+**: different learning rates for `A` and `B`; small, cheap gain.
- **GaLore**: projects gradients to low rank; achieves near-full-FT quality at reduced memory but with 1.6–13× time overhead (arXiv 2506.16500; GaLore paper arXiv 2403.03507, 2024) — poor throughput economics for a platform.
- **Spectrum/ReFT/PiSSA/VeRA**: niche; defer.

**Preference / RL-based tuning (needs preference or reward data, runs *after* SFT):**
- **DPO**: the baseline; needs `{prompt, chosen, rejected}` pairs. Strong, stable.
- **ORPO**: folds preference into SFT in one stage, no reference model — attractive operationally.
- **KTO**: needs only binary good/bad labels, not pairs — lowest data-collection burden.
- **SimPO**: reference-free; simpler and cheaper than DPO but reported ~1 point lower on average (Tülu-3 ablations, arXiv 2507.06187, 2025).
- **GRPO**: RL with a reward function/verifier; for reasoning/verifiable tasks — heaviest to operate; defer to Report B/C.

**Continued pretraining:** full or LoRA on raw text (completion schema) to shift domain/style before SFT; needs large unlabeled corpora.

### 6.2 The 2025 LoRA-vs-full-FT resolution

"LoRA Without Regret" (Schulman & Thinking Machines Lab, Sep 29 2025; DOI 10.64434/tml.20250929) swept LoRA rank 1–512 against full FT on Llama-3 and Qwen3 (incl. an MoE) on Tülu3/OpenThoughts3. Findings (High confidence, reproduced by HuggingFace TRL): LoRA **matches full FT** across most post-training scenarios *when applied to all weight matrices — especially the MLP/FFN layers, not just attention —* and given adequate rank; the TRL reproduction states it does so "while using only ~67% of the compute." The paper characterizes a "low-regret regime": a rank-32 adapter on a 7B model matched full fine-tuning on datasets up to ~50,000 examples, with rank 64/128 restoring parity beyond that. Increasing rank does not compensate for restricting LoRA to attention-only. *Platform consequence:* default LoRA target modules to **all-linear** (attn + MLP), not the historical q/v-only. This is why full FT is a low-priority v1 feature: for the datasets your users actually upload, a correctly-configured LoRA is not a quality compromise.

### 6.3 Framework/platform default anchors (for calibrating your own defaults)

| Source | LoRA rank | α | dropout | epochs | LR | targets |
|--------|-----------|---|---------|--------|-----|---------|
| PEFT `LoraConfig` | 8 | 8/16 | 0.0 | — | — | user-set |
| Axolotl QLoRA examples | 32 | 16 | 0.05 | 1–4 | 2e-4 | all-linear |
| Together API | 8 (max 64) | 8 | 0.0 | 1 (max 20) | — | — |
| Predibase SDK | 16 | — | 0.0 | 3 | 2e-4 | inferred |
| Fireworks | 8 (pow2 ≤32) | — | — | 1 | auto | — |
| Unsloth guidance | 16 | 2×r=32 | 0 | 1–3 | 2e-4 | all-linear |

Note the disagreement: rank ranges 8–32 across serious platforms, and α convention differs (some use α=rank, others α=2r). **Present the disagreement rather than averaging.** The "LoRA Without Regret" result argues capacity should scale with dataset size, not be a fixed constant. **Recommended platform default: r=16, α=32 (α=2r), dropout 0.0, all-linear targets, cosine schedule, LR 2e-4, 1–3 epochs by dataset size.**

### 6.4 The selection function (implementable)

```python
def select_method(
    schema,
    n_examples,
    model_params_B,
    budget_usd,
    vram_gb,
    user_opts_full_ft=False,
):
    # 1. Preference data → preference tuning (assumes a prior/base SFT)
    if schema == "preference":
        return (
            "ORPO" if n_examples < 20_000 else "DPO"
        )  # ORPO = single-stage, cheaper

    # 2. Completion / continued-pretrain
    if schema == "completion" and n_examples > 200_000:
        return "continued_pretrain_lora"

    # 3. SFT path (instruction/chat) — choose LoRA flavor by memory fit
    bf16_gb = model_params_B * 2  # inference weight size in bf16
    # Full FT only if the user explicitly opts in AND has the memory + data:
    if user_opts_full_ft and vram_gb >= 3.5 * bf16_gb and n_examples >= 50_000:
        return "full_ft"
    # QLoRA if a 16-bit base won't fit training comfortably; else 16-bit LoRA.
    if vram_gb < 2.0 * bf16_gb:  # base too big for 16-bit LoRA training
        return "qlora"  # NF4 base + bf16 adapters
    return "lora"


def select_rank(n_examples):
    # "LoRA Without Regret": rank 32 matches full FT up to ~50k examples on a 7B
    if n_examples < 1_000:
        return 8
    if n_examples < 50_000:
        return 16  # 16–32 both defensible here
    return 64  # scale capacity with data past ~50k
```

Additional inferred settings: `alpha = 2*rank`; `epochs = 3 if n<1000 else 2 if n<10000 else 1`; `max_len = p95_token_length` (capped); targets = all-linear; enable rsLoRA scaling when `rank >= 64`. Budget acts as a hard filter: estimate GPU-hours (Report B) × the cheapest qualifying GPU rate (§6.5) and, if it exceeds `budget_usd`, step down model size or switch LoRA→QLoRA before failing.

### 6.5 GPU cost anchors (for the budget filter)

International (on-demand, mid-2026): RunPod H100 PCIe **$1.99/hr** (SXM $2.69), A100 80 GB **$1.19–1.39/hr**, L40S **$0.79/hr**, RTX 4090 **$0.69/hr**; Lambda H100 PCIe **$3.29/hr**; H100 spot as low as ~$1.49–1.66/hr (Vast.ai/Spheron). India (INR, on-demand): E2E Networks A100 40 GB **₹170/hr** (~$1.94), A100 80 GB **₹220/hr**, H100 from **₹249–362/hr**; JarvisLabs H100 SXM **₹255/hr** (~$2.91), A100 40 GB **₹84/hr** (~$0.96), H200 SXM **₹378/hr**, L4 **₹41/hr**. The IndiaAI Mission subsidized pool undercuts commercial rates for eligible teams (~₹67–92/GPU-hr per ecorpit.com, 2026). *Consequence:* for the recommended matrix (4–32B, QLoRA), a single A100 40 GB or RTX 4090/L40S handles almost every job; a typical small/medium SFT run costs single-digit dollars.

### 6.6 What v1 should support vs defer

**v1: SFT via QLoRA (default) and 16-bit LoRA**, all-linear targets, rank auto-selected. **v1.5: full fine-tuning** (opt-in, size-gated) and **DPO/ORPO** preference tuning. **Defer: GRPO/RL, GaLore, MoE bases, DoRA** (add DoRA first among these as a quality toggle). This ordering matches where user demand and reliability are highest.

---

## 7. Open questions for Reports B and C

- **Full training-time VRAM budget** (optimizer states, gradients, activations, gradient checkpointing, offload) — needed to make §6.4's budget filter exact. (B)
- **Learning-rate, scheduler, warmup, epochs, batch/grad-accum** defaults and the gradient-accumulation loss-normalization bug (Unsloth/HF, Oct 2024) — do current versions still require the fix? (B)
- **Convergence/early-stopping and overfitting detection** from loss/eval curves — the observable thresholds that should auto-stop a job. (B)
- **Eval methodology** beyond held-out loss (LLM-as-judge, task metrics) and how to tell users their model actually improved. (B/C)
- **Merge/export correctness** (adapter merge dtype, GGUF/AWQ conversion) and guaranteed template propagation to serving. (B/C)
- **Multi-GPU / large-model orchestration** (FSDP/DeepSpeed ZeRO) for the 70B+ tier deferred in §5.6. (B)
- **Pricing/packaging, quota, abuse, and safety-content policy** — the business and ToS layer around PII/safety screening flagged in §3.9. (C)

---

## Caveats

- **Field velocity:** model names, prices, and framework defaults turn over in months. Anything here should be re-verified at build time; the framework config values (TRL, PEFT, Axolotl) are the most durable, marketing-sourced model rankings the least.
- **2026 model specifics** (Qwen3.5/3.6, Gemma 4, GLM-5.x, DeepSeek V4, Kimi K3) come substantially from vendor/aggregator blogs and should be treated as **Medium confidence** on exact benchmark numbers; the *licensing* and *architecture-family* claims are higher confidence.
- **VRAM and cost figures** are approximate and provider/region-dependent; the bytes-per-parameter arithmetic is exact but real footprints carry 15–20% overhead plus KV cache.
- **The LoRA≈full-FT claim** is well-supported for post-training on small/medium datasets but is *not* a claim that LoRA replaces large-scale pretraining or very large domain shifts.
- **Together AI's exact minimum example count** and **Fireworks/Vertex validation-split percentages** were not fully documented publicly; verify at integration.
- Sources older than ~February 2025 (SlimPajama 2023, QLoRA 2023, LIMA 2023, GaLore 2024, the double-BOS 2024 posts) are flagged inline; they remain the canonical references but predate the 18-month freshness window.