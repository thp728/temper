# Trainer Image

The pinned training container. Job spec in, adapter out.

⚠️ **This is product code living in the vault temporarily.** It belongs in the take-home's own repository — which the take-home needs anyway, since it is the publishable portfolio artifact and this vault is private and holds unrelated personal data. Per the architecture's §6 it lands at `gpu/trainer/`. **Moving it is an open decision.**

## Why Axolotl is the base

Spike 3 measured the cost of owning the version matrix. Installing *latest* gave `transformers 5.15.0` + `trl 1.10.0` on `torch 2.5.1`, and three things chained:

1. TRL 1.x's `SFTConfig` **no longer inherits `TrainingArguments`** → `warmup_ratio` raised `TypeError`
2. `save_safetensors` was rejected → checkpoints written as torch `.bin`
3. transformers 5.x **refused to load them** — requires `torch >= 2.6`

**Result: checkpoint/resume was impossible.** Not a design flaw — four packages moving independently.

Axolotl's maintainers resolve and test that matrix on every build, and **Axolotl's own config surface still exposes `warmup_ratio`**, so the defaults the research settled on stay expressible regardless of what TRL does to its API. Delegating the matrix to people who test it beats pinning six packages by hand and re-resolving whenever one moves.

The base carries **torch 2.12** — far past the 2.6 floor that blocked resume — with CUDA 13.0 and Python 3.12.

## Pin discipline

**The digest is the contract. The tag is a comment.**

```
axolotlai/axolotl:main-20260817-py3.12-cu130-2.12.0
@sha256:29327e75d7ae0348e001809df5c11151887fdeec63fe4bc5c4330917777c701e
```

Changing either is a deliberate act that re-runs the GPU smoke test. Our layer adds **two files** — `entrypoint.py` and `thinking.py` — and installs nothing; a `pip install` here would reintroduce the exact resolution problem this base exists to avoid.

⚠️ **`thinking.py` was missing from the COPY until 2026-08-19.** It was added to `entrypoint.py` as a top-level import after this image was last built, so nothing caught it until the first assembled run was read line by line. **Anything the entrypoint imports has to be listed here**, and the failure mode is the one below.

## Contract

| Path | Direction | Contents |
| --- | --- | --- |
| `/job/job.json` | in | job spec — see [job.example.json](job.example.json) |
| `/job/dataset.jsonl` | in | one JSON object per line |
| `/out/config.yaml` | out | the rendered Axolotl config, for auditability |
| `/out/run/` | out | checkpoints and the adapter |
| `/out/result.json` | out | **always written, including on failure** — see the boundary below |
| `/out/train.log` | out | full training output |

⚠️ **The boundary of "always written".** The guarantee comes from a `try/finally` inside `main()`, so it holds only from the moment `main()` is entered. **An import-time failure escapes it entirely** — the container exits with no `result.json`, and the orchestrator reports `training_failed: "Trainer produced no result.json"`, an error that points at training and says nothing about the image. That is exactly what a missing COPY produces. A guarantee whose boundary is undocumented is one you will over-trust.

```bash
docker run --rm --gpus all \
  -v /path/to/job:/job:ro \
  -v /path/to/out:/out \
  ghcr.io/<owner>/finetune-trainer@sha256:<digest>
```

## What is settable, and what is not

**Overridable** (`ALLOWED_OVERRIDES`): `lora_r`, `lora_alpha`, `learning_rate`, `num_epochs`, `max_steps`, `sequence_len`, `micro_batch_size`, `gradient_accumulation_steps`, `val_set_size`, `save_steps`.

**Default-locked, not hidden** — chat-template resolution, `train_on_inputs: false`, EOS handling, NF4 double-quant, `lora_target_linear`, bf16, seed. These are the highest-frequency silent-failure surface: they pass every obvious health check and only surface as garbage generations, so the common path never touches them.

⚠️ **This section used to say exposing them "buys a user nothing and costs correctness." That was overturned on 2026-08-19** — every serious platform in this space exposes these, and hiding them permanently is a limitation dressed as a safety feature. **Phase B adds an Advanced mode** that exposes each one with its failure mode named inline and records overrides in the run spec; the export-time template probe is what makes that safe. Today they are simply not wired to the job spec, which is a state of the build, not a principle.

Two behaviours worth knowing:

- **α tracks r.** Move `lora_r` without `lora_alpha` and α is recomputed as `2r` rather than pairing a new rank with a stale scale.
- **Unknown keys are refused loudly.** They come back in `result.json` as `rejected_overrides` — an override the caller believes is in effect but isn't is worse than a refusal.

## Verified 2026-08-18 (spike 4)

Built and run on an L4. **The image works, and resume works.**

| | |
| --- | --- |
| Build from pinned digest | 183s, 8.5 GB |
| Job through `/job` → `/out` | 120.9s, 64 rows, exit 0 |
| `axolotl train` accepts `warmup_ratio` | ✅ — the argument TRL 1.x removed |
| **Resume from `checkpoint-2` → step 6** | ✅ 54s |
| Adapter | **safetensors**, 132.2 MB, `r=16 alpha=32 rslora=False` |
| `target_modules` | all seven — `q,k,v,o,gate,up,down` |

**C14 closed.** The version chain that made resume impossible on an unpinned stack is gone.

Spike 4 also caught a real bug here: unknown keys at the **top level** of the job spec were not refused, only unknown keys inside `hyperparameters`. A caller misspelling `max_steps` as `maxSteps` would have got a full-length run with no warning. Both levels are now validated.

## Verified 2026-08-19 (first run through the product)

Same image, driven by `api/orchestrator.py` rather than a spike script. **L4, 336s job wall clock; the VM lived 363s, ≈₹4.8 derived from provision-to-teardown at ₹41.31/hr.**

| | |
| --- | --- |
| Build from pinned digest | **87s** — pull throughput varies; 183s on 08-18 |
| `axolotl train`, 64 rows × 3 epochs | 161.4s, 23 steps, exit 0 |
| Checkpoints written | `checkpoint-8`, `checkpoint-16`, `checkpoint-23` |
| Adapter shipped | `adapter_source: "final"`, 132.2 MB, verified by SHA-256 end to end |
| Thinking mode | detected `false` from the data, not configured |

⚠️ **`collect_artifacts()` picked the wrong adapter until this run.** `sorted(rglob(...))[-1]` sorts lexicographically, so `checkpoint-8` sorted last of those three — and every `run/checkpoint-N/…` sorts after `run/adapter_model.safetensors` anyway. It would have shipped **step 8 of 23** as the finished model, silently, with a valid hash. Selection is explicit now and `result.json` records which via `adapter_source`. **A hash check proves the bytes arrived intact; it says nothing about whether they were the right bytes.**

## Still open

- **fp32 adapters.** 132.2 MB is 33M × 4 bytes; bf16 would halve it. Undecided — it doubles artifact size and transfer cost. Re-confirmed 2026-08-19 from the safetensors header: **504 tensors, every one `F32`**. Note this is a size policy, not a defect — `bf16` is the *compute* dtype, and fp32 trainable parameters under 4-bit quantisation are standard.
- ~~**Qwen3 thinking mode.** Unhandled.~~ ✅ **Closed 2026-08-18.** Detected from the dataset in `thinking.py`, applied identically at training and serving; mixed datasets block with a line-numbered error. Confirmed on the 08-19 product run: `0/64 assistant turns have <think>` → `enable_thinking: false`.
- **`chat_template: tokenizer_default` ran without error, but nothing asserts it resolved to the *right* template.** The export-time probe from report A §4.2 — re-tokenise a fixed conversation through both the training template and the artifact's, assert identical ids — is the check, and it is not written yet.
- **`sample_packing`** stays off pending a per-model varlen-attention check.
- **Not pushed to GHCR.** The image builds per-VM today. Pushing needs a token and belongs in CI. At 87–183s per run it is also the largest remaining slice of cold start.
- **No assertion that the shipped adapter matches the run's final step.** `adapter_source` records the choice but nothing cross-checks it against the trainer's last step — that is the check that would have caught the selection bug above, and it is not written.
- **Training output never leaves the container.** `axolotl` writes to `/out/train.log`, which dies with the VM, so **no loss value is retrievable during or after a run.** A channel for it is the prerequisite for the loss curve.
