# Spec 009 — Models, methods, and the advanced surface

**Status:** ready for tickets
**Phase:** B, band 2 (the defensibility path)
**Depends on:** Spec 004 spikes 6 and 7 (multi-device execution, and the trainer's schema shape), Spec 005 (the `models` seam and the predictor), Spec 006 (artifact kinds)
**Produces:** ADR-0017 (the advanced surface is generated from the trainer's own configuration schema), ADR-0018 (a mixture-of-experts model warns and is labelled untested rather than refused), ADR-0019 (a locked setting and a refused input are different things)
**Assumes:** the rule that unknown job keys are refused loudly and echoed back; the thinking-mode decision, which this spec deliberately does not reopen

## Problem Statement

The product currently supports two models, one method, and five adjustable
values. Each of those numbers was chosen well and defended in writing, and all
three are now the wrong shape.

**Two models.** The catalog is curated and pinned, with a good argument behind
it: detection is easy and support is not, every family brings its own template
quirks and tokenizer edge cases, and the catalog is a promise about what has
been tested. That argument justifies a *recommended set*. It does not justify a
*boundary*, and the code's own docstring already calls curation a default rather
than a limitation while the catalog behaves as a limitation.

**One method.** Supervised fine-tuning through a quantised low-rank adapter,
always. The research supports it as a default — a well-configured adapter
matches full fine-tuning across most post-training work — but a platform that
can only do the cheap method cannot be described as covering what commercial
products offer. Real workloads include full fine-tuning, and real models are
large enough to need more than one card.

**Five knobs.** The exposed surface is a hand-picked list, with everything else
refused. The refusal is right and the list is arbitrary: it is what the first
version needed, not what the trainer supports. Meanwhile the settings that
matter most — template resolution, end-of-sequence handling, loss masking,
quantisation, target modules, precision, seed — are locked with no way to reach
them. That was defended as safety and is better described as a limitation
wearing a safety costume, because every serious tool in this space exposes them.

And there are two model revisions pinned to a moving reference. The catalog's
own documentation says a revision is pinned and never a branch name, and both
entries carry a branch name with a note to fix it. **A model can change under a
completed run**, which breaks the reproducibility claim the pinning exists to
support.

## Solution

**Models: unbounded, gated by a probe whose result is shown.**

Any model repository may be used, at a pinned revision, once it has passed a
compatibility probe. The catalog remains as the recommended, tested path — the
answer to *"why should I use one of yours?"* is that they have been run, not that
the others are forbidden.

The probe checks what actually breaks: that the repository resolves at a fixed
revision; that a chat template is present, without which chat data cannot be
formatted and hand-writing role delimiters becomes the modal silent failure;
that the tokenizer loads and that padding and end-of-sequence are distinct,
because an unmasked shared token teaches a model never to stop; that the licence
resolves and propagates; and that the predicted memory fits something available.

**Mixture-of-experts models warn rather than block**, and this reverses an
earlier position. The earlier reasoning was sound about the mechanics — expert
routing changes target-module selection, memory scales with total rather than
active parameters, and routing interacts poorly with small-batch adapters — but
blocking on it makes *"any model, including large ones"* untrue in practice,
because the open models above a certain size are largely of this kind. The probe
reports what changes and labels the configuration untested; the user decides,
knowing.

**Methods: chosen by the predictor, overridable by the user.**

Supervised fine-tuning through a quantised adapter, an unquantised adapter, or
full fine-tuning — selected on predicted memory and dataset size, with the
reasoning shown. Multi-device execution is a provisioning choice rather than a
different architecture: the same single-machine, one-container path asks for
more cards and shards across them.

**The advanced surface is generated from the trainer's own configuration
schema.**

This is what reconciles *expose every dial* with *refuse unknown keys loudly*.
The exposed set is derived from the schema of the pinned trainer rather than
hand-curated, which means it is exactly as wide as the trainer and no wider; it
cannot claim support for something the trainer lacks; it cannot silently drop
something the trainer has; and it cannot drift when the pinned image moves,
because it is derived from that image. Refusal is inherited rather than
invented — an unknown key is one the trainer does not know either.

Each field lands in one of three tiers, and the tier is data rather than code so
that a classification is reviewable in a change:

- **Calculated** — the predictor sets it, the common path never sees it.
- **Exposed with a named failure mode** — reachable, with the specific thing that
  goes wrong written beside it, not a general warning.
- **Known but unsupported here** — visible, with the reason, rather than absent.

**And two refusals stay refusals**, because a locked setting and a refused input
are different things. A locked setting is paternalism: there is a correct value
and the product chose it, and an informed user may choose otherwise. A refused
input has no correct interpretation at all — a dataset mixing reasoning traces
with plain answers is ambiguous by construction, and no value of any control
resolves it. That is not a hidden control; it is an absent one. The same holds
for a dataset below the minimum usable row count.

**The export-time template probe** is what makes all of this safe. A fixed probe
conversation is tokenised through the template used in training and through the
template serialised into the artifact, and the resulting identifiers must be
identical. It was already required by detecting thinking mode rather than fixing
it; exposing the template controls makes it the only thing standing between an
advanced user and a model that is silently wrong. It runs on every export
regardless of whether anything was overridden.

## User Stories

1. As a user, I want to fine-tune a model that is not in the catalog, so that the product is not limited to what it shipped with.
2. As a user importing a model, I want to see the compatibility result before I can launch, so that I learn what I am taking on rather than discovering it during a paid run.
3. As a user importing a model, I want a clear blocking reason when it cannot work, so that I do not retry the same thing.
4. As a user importing a model, I want warnings distinguished from blocks, so that I can proceed with something untested if I choose to.
5. As a user importing a model, I want its licence surfaced, so that I know what obligations flow through to what I produce.
6. As a user importing a model, I want the revision pinned, so that my completed run describes a model that cannot change afterwards.
7. As a user, I want catalog models to be pinned to fixed revisions too, so that the reproducibility claim is true of every path, not just the new one.
8. As a user importing a mixture-of-experts model, I want to be told what differs and that it is untested here, so that I can decide with the facts.
9. As a user, I want to import a dataset from a public repository, so that I can start without preparing a file.
10. As a user importing a dataset, I want it validated exactly as an uploaded one is, so that nothing gets a shortcut for arriving over a network.
11. As a user importing a dataset that fails validation, I want it kept with its report, so that I can see which rows to fix.
12. As a user, I want the product to choose a training method, so that I do not have to know the difference to get a good result.
13. As a user, I want to see why a method was chosen over the alternatives, so that the choice is legible.
14. As a user with a large model, I want the product to use more than one device when one will not hold it, so that model size is not a wall.
15. As a user, I want the product not to use more devices than the job needs, so that I am not billed for hardware sitting idle.
16. As a user, I want full fine-tuning available when it is the right choice, so that the product is not limited to the cheap method.
17. As a user running full fine-tuning, I want to download what it produced, so that a different method does not mean a broken delivery path.
18. As an experienced user, I want to reach every setting the underlying trainer supports, so that I am not limited by what the interface chose to show.
19. As an experienced user, I want each reachable setting to name what goes wrong if I get it wrong, so that the warning is specific rather than decorative.
20. As an experienced user, I want a setting the trainer supports but this product does not to be visible with its reason, so that I do not wonder whether it was overlooked.
21. As an experienced user, I want my overrides recorded in the job spec, so that the run says what it actually used.
22. As an experienced user, I want an override that breaks the job caught before it launches, so that taking control does not cost me a provisioned machine.
23. As an experienced user, I want an unknown setting refused and named back to me, so that I never believe an override is in effect when it is not.
24. As a user, I want the product to verify that the template used in training is the one shipped with my result, so that a model that trains cleanly and answers wrongly is caught here rather than by me.
25. As a user whose dataset mixes reasoning traces with plain answers, I want it refused with the lines named, so that an ambiguous file is not silently interpreted one way.
26. As a user, I want to understand why some things are adjustable and others are refused, so that the boundary reads as judgment rather than as inconsistency.
27. As a reviewer, I want the exposed surface to be derived from the trainer rather than hand-listed, so that I can see it cannot drift.

## Implementation Decisions

**The `models` seam gains the probe.** The probe and the predictor read the same
facts through the same seam, because two implementations of *how many parameters
does this model have* would drift, and the drift would show up as a model the
probe accepted and the predictor mispriced.

**The probe's result is persisted and displayed**, not merely enforced. A model
that passes with warnings is usable and the user knows what they took on. That is
the honest form of curation: the catalog is a promise about what was tested, and
importing is possible with the testing done in front of you.

**Catalog revisions are pinned to fixed commit identifiers** as part of this
spec, closing the gap between what the catalog documents and what it does.

**Imported datasets go through the identical validation path.** Not a parallel
one, not a relaxed one. The streaming validation from Spec 006 takes an iterator
of rows; where those rows came from is the import's problem and nothing below it
knows or cares.

**Method selection is the predictor's, not a separate rule.** Memory blocks and
the rest is preference, exactly as Spec 005 established, and multi-device
selection is part of the same decision rather than a mode.

**Multi-device execution is a provisioning parameter plus a sharding
configuration**, expressed through the trainer's own support for it rather than
hand-written. The path is the same single-machine, one-container path; it asks
for more cards.

**Full fine-tuning changes the artifact, and that change belongs to Spec 006.**
This spec produces the method; the kinds of artifact and the transport that
carries them are already handled.

**Sharded checkpoints are a distinct format and are treated as one.** Resume is
proven for a single device and unproven for many. If spike 6 shows sharded
resume does not work, multi-device jobs ship without resume and say so, rather
than claiming a recovery path that has never run.

**The exposed surface is generated at build time from the pinned trainer's
schema, and the tier classification is a checked-in data file.** Generated
because it must not drift; checked in because a classification that changes
should be visible in a diff and defensible in review.

**Constraints that the schema does not express are the risk, and are handled
explicitly.** If a combination is only rejected by the trainer at runtime, a
generated form will happily accept it and the job fails minutes into a paid
machine — precisely the failure the predictor exists to prevent. Combinations
known to be invalid are validated ahead of launch; the ones that cannot be known
are named as such.

**Unknown keys keep being refused loudly and echoed back**, at every level. The
rule is unchanged; what changes is that *unknown* now means unknown to the
trainer rather than absent from a hand-written list.

**The template probe runs on every export.** Not only when something was
overridden — a wrong thinking-mode detection produces a wrong template with no
override involved, and that is the case it was originally required for.

## Testing Decisions

**A good test here asserts on the decision and its reason, not on the mechanism
that produced it.** A probe result is a verdict plus findings; that is what
callers see and that is what to assert. Whether the check ran before or after
another is implementation.

**The probe is tested without network access** through the `models` seam's
double, with fixtures covering: a clean model; one with no chat template; one
where padding and end-of-sequence collide; one that is mixture-of-experts; one
whose revision does not resolve; one whose licence is unresolvable; one too large
for anything available.

**Generation is tested for stability.** The generated surface must be
deterministic for a given pinned image, and every field must land in exactly one
tier. A field appearing in no tier is a hole through which an unclassified
control reaches a user, and the test for that is a completeness assertion rather
than a sample.

**The template probe is tested by making it fail.** A deliberately mismatched
template must be caught. A probe that has only ever passed is a probe with no
evidence of working, and this one is load-bearing twice over.

**Refusals are tested as behaviour, not as messages.** A mixed-thinking dataset
is refused with its lines named; a dataset below the row floor is refused; an
unknown key is refused and echoed. What matters is the refusal and the
reference, not the wording.

**Prior art:** the existing hyperparameter tests define the override-and-refuse
contract that the generated surface must continue to satisfy; the thinking-mode
tests define the shape of a line-referenced refusal; the catalog tests define
what a pinned entry must carry.

**The verification clause.** Three real runs, and none of them is optional:
a multi-device job that genuinely does not fit one card, executed through the
interface; a full fine-tuning job whose artifact is downloaded and loaded; and an
imported model outside the catalog, probed and then trained. Until those have run
on hardware, this spec is designed rather than done, and the documentation says
so in those words. A multi-device job on hardware that did not need it proves
nothing and should not be run in place of one that does.

## Out of Scope

- **Preference-based objectives.** They need a different dataset schema, a
  different learning-rate regime, different streamed measurements, and a
  reference model in memory. Worth doing and second in line; not in this spec.
- **Reinforcement-style objectives with verifiers.** The heaviest to operate and
  the furthest from this product's shape.
- **Vision-language and multimodal.** A different data pipeline end to end with
  no shared surface here.
- **Continued pretraining on raw text.** A different data contract.
- **Serving an imported model**, and anything about endpoints. Spec 011.
- **Automatic recovery from a failed configuration.** Spec 010.

## Further Notes

The tension worth naming, because it will be asked about directly: this spec
simultaneously argues that curation is valuable and that the boundary should come
down. Both are true and the resolution is that they answer different questions.
*Which model should I use?* is answered by a tested, pinned, licensed
recommendation. *May I use another?* is answered by a probe that does the testing
in front of the user and shows its result. A product that only answers the first
is a demonstration; one that only answers the second is a wrapper.

The second thing to say plainly: reversing the position on mixture-of-experts
models is a real reversal, not a clarification, and the earlier reasoning was not
wrong about the mechanics. It was wrong about the consequence — refusing them
made a claim about supporting large models untrue, and the fix is to surface what
differs rather than to pretend the difference does not exist. The record for that
is a new decision entry that supersedes, not an edit.
