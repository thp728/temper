# ADR-0016 — The parity boundary is ratified at 75%, with Together AI as the baseline

- **Status:** accepted
- **Date:** 2026-08-18

> **A note on the number and the date.** This decision was recorded on
> 2026-08-18 in the private working vault where the earlier decisions were
> logged, and copied into this directory on 2026-08-26
> ([#26](https://github.com/thp728/temper/issues/26)). Its file number reflects
> when the record landed here, not when the decision was made; the wording is
> condensed from the original, which stays in the vault.

## Context

The draft scope table computed to 66% against Together AI — four points under a bar stated up front. Arguing that the flows Together lacks are worth more than the gap is a *claim about value* offered immediately after missing a *countable* target, and reads as moving goalposts. Two flows cut for time rather than principle could be restored for roughly an evening's work, and the argument disappears.

Together AI is the baseline because it is structurally the same product — LoRA-first, open-weight, per-token pricing — so a percentage against it means something. Coverage is defined deliberately as share of **user-facing flows**, not capability or effort: it is the only one of the three a third party can count and check.

## Decision

Ratified the scope flow table (a private working document whose verdicts are reflected in `docs/specs/`). **Together AI is the parity baseline**; two time-cut flows restored — **#39 general-capability regression check** and **#37 GGUF export**. Result: 38 baseline-comparable flows, **71% coverage, 75% excluding the two cuts the brief sanctions**.

## Alternatives considered

Argue that the eight flows Together lacks are worth more than the gap (rejected — see above); accept 66% (rejected — below the stated bar).

## Consequences

Two more flows to build in a short week, and regression eval needs a fixed, versioned prompt set.

## Rollback

If the checkpoint slips, #37 GGUF goes first — it is a convenience where #39 catches a real failure mode; coverage drops to ~72%, still inside the band.
