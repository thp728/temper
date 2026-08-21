# Temper

A fine-tuning platform. Upload a dataset, pick a base model, get a trained adapter you can actually use.

> You don't reforge steel to change its properties — you temper it. Controlled, and the base material survives.
> That is QLoRA: the base weights stay frozen, a small adapter carries the change. Full fine-tuning is reforging.

**Status: in development.** Built as a take-home for [JarvisLabs.ai](https://jarvislabs.ai), August 2026.

> ⚠️ **This README is a holding version** — accurate, but not the pass it gets before the repo goes public.

## What it does

Fine-tunes open-weight LLMs on user-supplied instruction data, on real GPUs, end to end — dataset in, adapter out.

**The whole journey runs from a browser**, on server-rendered pages: upload and validate a dataset, choose a base model, watch the run as it provisions and trains, and download the adapter — with the machine destroyed afterwards and confirmed gone. The same journey is available over the API. Cancellation and the runaway-job limits are in place.

**Deliberately absent:** auth and billing, which the brief sanctions cutting. **Not built yet:** a pre-run cost quote, an inference endpoint, and imports from Hugging Face.

- **Method:** supervised fine-tuning via QLoRA — NF4 double-quant base, bf16 compute, rank 16, α=32, **all linear layers**. Adapter weights save as **fp32**, which is what `prepare_model_for_kbit_training` does and is why the artifact is 132 MB rather than ~66 MB
- **Models:** curated and pinned — `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`
- **Compute:** JarvisLabs VMs, provisioned and destroyed per job
- **Trainer:** Axolotl in a digest-pinned container
- **Dataset limit:** 1 GB per upload (`TEMPER_MAX_DATASET_MB`). This is a limit of the current in-memory validation path, which holds about 4.8× the file size — not a product rule. Streaming validation removes it; until then, uploads over the limit are refused immediately with both sizes named.
- **Duration warning:** a dataset that plainly cannot finish inside the 24-hour job ceiling (`TEMPER_MAX_JOB_DURATION_S`) gets a warning at job creation — an **estimate** from measured throughput on one real run (~1.19 row-passes/s, L4, Qwen3-4B), not a quote. The job launches anyway; the estimate is crude and only the user should decide whether the run is worth attempting.

## Why these choices

Every one of them is written down with alternatives and tradeoffs, because a decision you cannot explain is not a decision you made.

**All linear layers, not attention-only.** In Qwen3-8B the MLP is **78.3%** of every transformer block's parameters. Attention-only LoRA reaches about 11% of each block — no rank compensates for the rest simply not being in the optimisation.

**Axolotl pinned by digest, rather than pinning six packages by hand.** Installing "latest" produced `transformers 5.15` + `trl 1.10` on `torch 2.5.1`: TRL 1.x dropped `warmup_ratio`, rejected `save_safetensors` so checkpoints wrote as torch `.bin`, and transformers 5.x then refused to load them without `torch >= 2.6`. **Training worked; resume was impossible.** Delegating that matrix to people who test it daily is the fix.

**Correctness settings are locked by default, not hidden.** Chat-template resolution, EOS handling and loss masking are the highest-frequency silent failure in this category — they pass every obvious health check and surface only as bad output — so they ship with correct values and no way to fumble them by accident. **They are not permanently sealed:** an Advanced mode exposes each with its failure mode named inline and records the override in the run spec, guarded by an export-time template probe. ⚠️ *This reverses an earlier position that they should never be user-settable — every serious platform in this space exposes them, and permanent hiding is a limitation wearing the costume of a safety feature.*

## Measured, not estimated

On an NVIDIA L4 (24 GB), Qwen3-4B:

| | |
| --- | --- |
| Peak VRAM | **5.31 GB** |
| Trainable parameters | **33,030,144** (predicted from `config.json`, confirmed by the run) |
| Cold start | **2–4 min** — 10–13s provision, 40–47s to SSH, **87–183s image pull** (same digest; the spread is registry throughput) |
| Checkpoint resume | verified |
| End-to-end job | **336s**, VM alive 363s, ≈**₹4.8** derived from provision-to-teardown |

⚠️ **Derived is not measured.** The cost line is computed from the event log against the stored hourly price; nothing here reads an invoice, and nothing in the product computes a job cost yet.

## Layout

```
api/        control plane — upload, validate, catalog, launch, watch, download
trainer/    the pinned training container and its /job -> /out contract
spike/      infrastructure probes against the live JarvisLabs account
```

## Known gaps

Named here rather than left for a reader to find.

- **The loss curve is verified, not observed.** The classifier promotes the loss and epoch from a real run's training output, checked line by line against one — but no run has yet been watched rendering the chart live.
- **A finished job's page truncates its log** before the end, so it does not show its own final events. Live watching is unaffected.
- **The job log is mostly build noise.** A real run writes several hundred events, the large majority of them container-build progress, which buries the trainer's own output.
- **Costs are derived, never invoiced.** See the note above.

## License

TBD before publication.
