# ADR-0018 — Thinking mode is detected from the dataset, not fixed or exposed

- **Status:** accepted
- **Date:** 2026-08-18

> **A note on the number and the date.** This decision was made on
> 2026-08-18 and filed in this directory on 2026-08-26
> ([#26](https://github.com/thp728/temper/issues/26)). Its file number reflects
> when the record landed here, not when the decision was made.

## Context

Qwen3 dense models emit `<think>` blocks through their chat template by default. A dataset without them, trained under the default template, is a train/infer mismatch — the class of bug that passes every obvious health check and surfaces only as bad output. Something had to be chosen; the default was not safe.

Detection was chosen over hard-coding `false` because **the platform should not silently discard a capability the base model has.** A user who brings reasoning-trace data gets a correct fine-tune rather than a quietly degraded one, and detection is a deterministic function of the data rather than a policy imposed on it.

## Decision

Temper **detects** whether a dataset carries reasoning traces and sets Qwen3's `enable_thinking` to match, identically at training and serving. **Mixed datasets are a blocking validation error, not a guess.**

- Every assistant turn contains `<think>` → `enable_thinking: true`
- No assistant turn contains `<think>` → `enable_thinking: false`
- **Some but not all** → **block the job** with a line-level error naming the offending rows

## Alternatives considered

*Hard-code `enable_thinking: false` and state it as a v1 boundary* — the safer engineering answer, one line, no detection logic to get wrong, and consistent with reasoning fine-tuning already being OUT at flow #17. **Rejected because it silently narrows what the product can do for a user whose data is fine.** *Expose it as a job option* — rejected outright: it contradicts the trainer's own stated principle that chat-template handling is hard-coded precisely because it fails silently, and it would mean defending the exposure of the one setting documented as too dangerous to expose.

## Consequences

This adds new logic to the **highest-frequency silent-failure surface in the system**, which is exactly where the reports say not to be clever. The mitigations are that the ambiguous case becomes a *loud* failure rather than a guess, and that the export-time template probe from research report A §4.2 — re-tokenise a fixed conversation through both the training and artifact templates, assert identical ids — becomes **mandatory rather than nice-to-have**, because it is what catches a wrong detection. It also needs its own unit test with all three cases.

## Rollback

The detector is one function with one output. Forcing it to return `False` unconditionally restores the hard-coded alternative in a single line.
