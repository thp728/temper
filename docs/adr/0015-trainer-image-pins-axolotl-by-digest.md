# ADR-0015 — The trainer image pins Axolotl by digest rather than resolving the stack

- **Status:** accepted
- **Date:** 2026-08-18

> **A note on the number and the date.** This decision was made on
> 2026-08-18 and filed in this directory on 2026-08-26
> ([#26](https://github.com/thp728/temper/issues/26)). Its file number reflects
> when the record landed here, not when the decision was made.

## Context

Spike 3 measured what owning the version matrix costs. Installing *latest* produced `transformers 5.15.0` + `trl 1.10.0` + `peft 0.20.0` + `bitsandbytes 0.50.1` on `torch 2.5.1`, and three consequences chained: TRL 1.x's `SFTConfig` stopped inheriting `TrainingArguments` so `warmup_ratio` raised `TypeError`; `save_safetensors` was likewise rejected so checkpoints wrote as torch `.bin`; and transformers 5.x then refused to load them because `torch < 2.6`. **Checkpoint/resume was impossible** — not from a design flaw, but from four packages moving independently. That is exactly the dependency-instability risk identified in the research phase.

Axolotl's maintainers resolve and test that matrix on every build, its config surface still exposes `warmup_ratio`, and its images carry `torch 2.12` — far past the 2.6 floor that blocked resume. Spike 4 confirmed it end to end: image built in 183s, a Qwen3-4B QLoRA job ran to completion, and **the run resumed from `checkpoint-2` to step 6**. The adapter came back as safetensors with `r=16, alpha=32, rslora=False` and all seven linear target modules — matching the arithmetic derived from `config.json` before any of it ran.

## Decision

The trainer container is a thin layer over `axolotlai/axolotl`, pinned by digest, and training is invoked as `axolotl train <config.yaml>` rather than by calling TRL/PEFT directly. Our layer adds one file and installs nothing.

## Alternatives considered

*Build from a PyTorch base and pin six packages by hand* (rejected — that is the thing that broke, and every upstream release re-opens it); *pin to older pre-1.0 TRL* (rejected — freezes the stack against security fixes and still leaves us owning the matrix); *use Axolotl's image directly with code mounted at runtime* (rejected — forfeits the immutable-image guarantee).

## Consequences

The image is **8.5 GB**, so a cold pull costs ~183s on every fresh VM — against ~72s for a 3.3 GB PyTorch base. We inherit Axolotl's choices, including a CUDA and Python version we did not pick. And we are now coupled to Axolotl's release cadence: a broken upstream build is our problem, mitigated only by the digest pin, which means upgrades are deliberate and gated on re-running spike 4.

## Rollback

The entrypoint's contract (`/job` → `/out`) is independent of the base. Swapping to a hand-pinned image means rewriting config generation to emit whatever that stack accepts; the job spec, the artifact layout and the orchestrator are untouched.
