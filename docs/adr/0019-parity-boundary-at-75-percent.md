# ADR-0019 — The parity boundary is ratified at 75%, with Together AI as the baseline

- **Status:** accepted
- **Date:** 2026-08-18

> **A note on the number and the date.** This decision was recorded on
> 2026-08-18 in the private working vault where the first thirteen decisions
> were logged, and copied into this directory on 2026-08-26
> ([#26](https://github.com/thp728/temper/issues/26)). Its file number reflects
> when the record landed here, not when the decision was made; the wording is
> the original, with private-vault links replaced by descriptions of what they
> pointed at.

## Context

The draft scope table computed to 66% coverage against Together AI — four points under a bar the founder stated up front. The alternative was to argue that the eight flows Together lacks are worth more than the gap — which may be true, but it is a *claim about value* offered immediately after missing a *countable* target, and that reads as moving the goalposts even when it isn't. Two flows had been cut for time rather than principle: **#39 general-capability regression check** and **#37 GGUF export**. **"I ran out of time" is the weakest available answer to a number you were given in advance.** The regression check is a small base-vs-tuned slice on a GPU already warm with the model loaded; GGUF is one conversion step off the correctly-merged bf16 model. Roughly an evening, and the argument disappears.

**Why Together and not the others:** it is structurally the same product — LoRA-first, open-weight, per-token pricing — so a percentage against it means something. OpenAI is closing to new jobs after 6 Jan 2027 and hides nearly every knob; Predibase's RFT surface makes the cut list enormous; AutoTrain is too low a bar to evidence anything; Axolotl is a *component of this build*, so comparing against it is circular.

**How the percentage is defined, deliberately:** share of **user-facing flows**, not capability. Capability parity against a platform that fine-tunes 397B models across six objectives would be dishonest, and effort parity is unfalsifiable. **Flow coverage is the only one of the three a third party can count and check.** State the definition before the number.

## Decision

Ratified the scope flow table (a private working document whose verdicts are reflected in `docs/specs/`). **Together AI** is the parity baseline, with OpenAI's closing dashboard as UX anchor only. 38 baseline-comparable flows: 14 IN, 13 IN-REDUCED, 11 OUT — **71% coverage, or 75% excluding the two flows the brief sanctions cutting.** Two flows restored from the draft: **#39 general-capability regression check** and **#37 GGUF export**.

## Alternatives considered

Argue that the eight flows Together lacks are worth more than the gap (rejected — see above: a value claim offered right after missing a countable target); restore nothing and accept 66% (rejected — below the stated bar).

## Consequences

Two more flows to build in a week that has three evenings and a day. Regression eval also needs a fixed, versioned prompt set, which is a small piece of curation nobody has done yet.

## Rollback

If Friday's checkpoint slips, #37 GGUF is the first thing out again — it is a convenience, where #39 catches a real failure mode. That drops coverage to ~72% excluding sanctioned cuts, still inside the band.

**Correction found while ratifying:** the draft's arithmetic block was labelled *"46 baseline-comparable flows"* while its own counts summed to 38 — 46 total less the 8 BEYOND rows. The percentages were computed off 38 and were right; only the label was wrong. Fixed rather than left.
