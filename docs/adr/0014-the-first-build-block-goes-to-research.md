# ADR-0014 — The first build block goes to research instead of code

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

The brief sets two bars that are not satisfiable by code alone — *"cannot be the hello-world level of fine-tuning"* and *"I should be able to explain every decision that's done."* Starting from no ML background, building first would have produced defaults copied from a tutorial, which is exactly the hello-world outcome the brief disqualifies by name. The research also surfaced things that are invisible until they have already ruined a run: chat-template and EOS-token mismatches pass every obvious health check and only appear as garbage generations, and `pad_token == eos_token` silently teaches a model never to stop.

## Decision

Used the 08-15 evening and the 08-16 full day — the first of only two good contiguous blocks — to close the ML knowledge gap rather than to start the core loop. Output: a fine-tuning crash course with notes, three commissioned research reports covering data/tokenization/method selection, the training loop and cost model, and evaluation/artifacts/commercial teardown, plus a full reference technical architecture. Those research documents stayed in the private vault; their conclusions are what this repository's records cite.

## Alternatives considered

Build the vertical spike first and read alongside it (rejected — the spike is the highest technical risk in the architecture and would have been debugged without the vocabulary to read the errors); read only Report A and start (rejected once the reports were commissioned together, though this would have preserved the 08-16 block).

## Consequences

The core loop now starts from zero with four office evenings and one full day before the 08-21 checkpoint, instead of four evenings, one full day, and a working spike. The checkpoint is materially at risk and that risk is self-inflicted. A second cost is that the research produced a reference architecture sized for a fundable product rather than a fourteen-day take-home, so an explicit cutting pass is now required before any code is written.

## Rollback

Not reversible — the time is spent. The mitigation is that the 08-21 checkpoint stands unchanged and unweakened: if dataset in → job runs → adapter out has not happened once on real infrastructure by that evening, scope gets cut that day rather than in the final week.
