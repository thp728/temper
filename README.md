# Temper

A fine-tuning platform. Upload a dataset, pick a base model, get a trained adapter you can actually use.

> You don't reforge steel to change its properties — you temper it. Controlled, and the base material survives.
> That is QLoRA: the base weights stay frozen, a small adapter carries the change. Full fine-tuning is reforging.

**Status: in development.** Built as a take-home for [JarvisLabs.ai](https://jarvislabs.ai), August 2026.

## What it does

Fine-tunes open-weight LLMs on user-supplied instruction data, on real GPUs, end to end — dataset in, adapter out, with the run visible while it happens.

- **Method:** supervised fine-tuning via QLoRA — NF4 double-quant base, bf16 adapters, rank 16, α=32, **all linear layers**
- **Models:** curated and pinned — `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`
- **Compute:** JarvisLabs VMs, provisioned and destroyed per job
- **Trainer:** Axolotl in a digest-pinned container

## Why these choices

Every one of them is written down with alternatives and tradeoffs, because a decision you cannot explain is not a decision you made.

**All linear layers, not attention-only.** In Qwen3-8B the MLP is **78.3%** of every transformer block's parameters. Attention-only LoRA reaches about 11% of each block — no rank compensates for the rest simply not being in the optimisation.

**Axolotl pinned by digest, rather than pinning six packages by hand.** Installing "latest" produced `transformers 5.15` + `trl 1.10` on `torch 2.5.1`: TRL 1.x dropped `warmup_ratio`, rejected `save_safetensors` so checkpoints wrote as torch `.bin`, and transformers 5.x then refused to load them without `torch >= 2.6`. **Training worked; resume was impossible.** Delegating that matrix to people who test it daily is the fix.

**Correctness is not configurable.** Chat-template resolution, EOS handling and loss masking are hard-coded. They are the highest-frequency silent failure in this category — they pass every obvious health check and surface only as bad output. Exposing them buys a user nothing.

## Measured, not estimated

On an NVIDIA L4 (24 GB), Qwen3-4B:

| | |
| --- | --- |
| Peak VRAM | **5.31 GB** |
| Trainable parameters | **33,030,144** (predicted from `config.json`, confirmed by the run) |
| Cold start | ~4 min — 13s provision, 40s to SSH, 183s image pull |
| Checkpoint resume | verified |

## Layout

```
trainer/    the pinned training container and its /job -> /out contract
spike/      infrastructure probes against the live JarvisLabs account
```

## License

TBD before publication.
