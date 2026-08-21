# Temper

A fine-tuning platform. Upload a dataset, pick a base model, get a trained adapter you can actually use.

> You don't reforge steel to change its properties — you temper it. Controlled, and the base material survives.
> That is QLoRA: the base weights stay frozen, a small adapter carries the change. Full fine-tuning is reforging.

**Status: in development.** Built as a take-home for [JarvisLabs.ai](https://jarvislabs.ai), August 2026.

## What it does

Fine-tunes open-weight LLMs on user-supplied instruction data, on real GPUs, end to end — dataset in, adapter out.

**Working as of 2026-08-21:** upload and validate a dataset, launch a job, watch it provision a VM, train, and return a downloadable adapter, with the machine destroyed and confirmed gone. Upload and its validation report are usable from a browser (server-rendered pages, no JavaScript); the rest of the journey — model choice, live watch, download — is still API-only. **Not working yet:** any visibility into a running job (see below), cancellation, an enforced spend cap.

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
| End-to-end job through the API | **336s**, VM alive 363s, ≈**₹4.8** derived from provision-to-teardown |

⚠️ **Derived is not measured.** The cost line is computed from the event log against the stored hourly price; nothing here reads an invoice, and nothing in the product computes a job cost yet.

## Layout

```
api/        control plane — upload, validate, catalog, launch, watch, download
trainer/    the pinned training container and its /job -> /out contract
spike/      infrastructure probes against the live JarvisLabs account
```

## The largest known gap

**A training job is 251 seconds of total silence.** The orchestrator makes one blocking call over SSH for the whole build-and-train phase and turns its output into events only after it returns — every log event from the first real run carries an identical timestamp. Axolotl's own output never leaves the machine at all: the entrypoint redirects it to a file inside the container, which is destroyed with the VM.

**So there is no loss value retrievable anywhere, during or after a run.** Streaming the loss curve is therefore not a parsing task; the channel has to be built first. It is the next thing being built, and it is named here rather than discovered by a reader because *"no visibility into the run"* is the hello-world version this was explicitly not meant to be.

## License

TBD before publication.
