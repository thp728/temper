# ADR-0013 — The architecture's own assumptions are verified before writing code

- **Status:** accepted
- **Date:** 2026-08-17

> **A note on the number and the date.** This decision was recorded on
> 2026-08-17 in the private working vault where the first thirteen decisions
> were logged, and copied into this directory on 2026-08-26
> ([#26](https://github.com/thp728/temper/issues/26)). Its file number reflects
> when the record landed here, not when the decision was made; the wording is
> the original, with private-vault links replaced by descriptions of what they
> pointed at.

## Context

The research reports and the reference architecture were written in a single research block, largely from model-generated synthesis, and the architecture's own risk section names JarvisLabs VM automation as the highest technical risk in the build. An assumption that is wrong at the platform layer does not fail cheaply — it fails on Friday night, on the checkpoint. The check cost under an hour and found that **the SDK does not accept a custom Docker image**, which invalidates the bootstrap sequence the architecture described as written.

## Decision

Before starting the Phase 0 spike, checked the load-bearing claims in the reference technical architecture and the three research reports against live vendor documentation. Three corrections recorded in that document's new corrections section rather than silently edited into its body.

## Alternatives considered

Start the spike and discover the Docker constraint empirically (rejected — same discovery, later, with less time to react, and the spike would have been debugged without knowing whether the failure was configuration or capability); trust the reports (rejected — three of the checked claims were wrong, so the base rate does not support it).

## Consequences

An evening hour spent verifying rather than building, in a week that has very few. Against that: one of the three corrections would otherwise have surfaced mid-spike, and the Qwen3 thinking-mode issue would plausibly not have surfaced until a fine-tuned model produced visibly wrong output — the class of bug Report A describes as passing every obvious health check.

**Worth keeping for the review:** the corrections are an asset, not an embarrassment. The project standard — *"here's what I got wrong and how I caught it" lands better than a clean chart* — is applied here, and one of the three corrections **vindicated** the original choice (Qwen3 dense over Qwen3.5) for a reason better than the one it was originally made for.

## Rollback

Not applicable; verification is not reversible work. The corrections stand until re-checked, and each carries its source and check date so staleness is visible.
