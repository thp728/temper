# ADR-0016 — Together AI is the reference baseline

- **Status:** accepted
- **Date:** 2026-08-18

> **A note on the number and the date.** This decision was made on
> 2026-08-18 and filed in this directory on 2026-08-26
> ([#26](https://github.com/thp728/temper/issues/26)). Its file number reflects
> when the record landed here, not when the decision was made; the wording is
> condensed from the original decision note.

## Context

Judging this product's scope requires a reference that is like-for-like. **Together AI is structurally the same product** — LoRA-first, open-weight, per-token pricing — so a comparison against it means something.

## Decision

**Together AI is the reference baseline** against which product scope is measured, with OpenAI's dashboard as UX anchor only.

## Alternatives considered

*OpenAI* (rejected — closing to new jobs and hides nearly every knob); *Predibase* (rejected — its RFT surface makes any comparison enormous); *AutoTrain* (rejected — too low a bar to evidence anything); *Axolotl* (rejected — it is a component of this build, so comparing against it is circular).

## Consequences

Scope is judged against what a third party can count and check at the reference product, rather than against capability claims.

## Rollback

Choosing a different baseline changes the checklist, not the product; re-ratify against the new reference.
