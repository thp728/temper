# Spec 005 — The predictor and the quote

**Status:** ready for tickets
**Phase:** B, band 1 (the demo path)
**Depends on:** Spec 004 spikes 5, 7 and 9 — they supply the disk ceiling, the download rate, and the token-counting cost this spec turns into numbers
**Produces:** an ADR that the predictor blocks on memory and warns on time, and one that a calculated default carries its reason
**Assumes:** ADR-0007 (the feasibility warning is an estimate and warns rather than blocks) — this spec inherits that posture and narrows where it does not apply

## Problem Statement

A user arrives with a dataset and a model in mind. Everything that happens next
is a decision someone has to make: which method, whether to quantise, which GPU,
how many, how much disk, how long it will take, and what it will cost. Today the
product makes almost none of those decisions and hides the ones it does.

The current state, precisely:

- **Duration** is estimated from a single constant measured on one run — one
  model, one GPU, one method — and applied to everything. It warns and does not
  block, which is right, but it is the only prediction the product makes.
- **Peak memory** is a number stored by hand against each catalog entry. It was
  measured for the 4B and extrapolated for the 8B, and the extrapolation is
  documented as unverified.
- **GPU choice** is a hard-coded preference list, taken in order, first one with
  free capacity wins. It does not consider what the job needs.
- **GPU count** is not a concept. **Disk** is a constant equal to the platform
  minimum.
- **Cost** is not computed at all before a run. It is derived afterwards, from
  the event log against a stored hourly price, and the README says plainly that
  nothing in the product computes a job cost.

None of that survives the catalog becoming unbounded. A user who selects a 70B
model gets a hard-coded 100 GB disk that cannot hold its weights, a GPU chosen
from a list of three, and no warning until the job fails minutes into a machine
they are paying for.

**And the deeper problem is not the missing numbers — it is the missing
reasons.** A fine-tuning platform's value is that it decides well on the user's
behalf and shows its work. A product that silently picks an L4 is not obviously
better than one that asks. A product that says *"an L4, because your predicted
peak is 5.3 GB and it is the cheapest card that holds it with headroom; two
would finish 1.7× faster at twice the cost"* has taught the user something and
earned the right to decide.

## Solution

One component computes an entire job configuration from what the user asked
for, and returns every decision with the constraint that forced it and the
alternative that lost. That output is persisted as a **quote**, shown before
launch, and frozen into the job spec at launch.

The user sees a single page. A first-time user reads it as an explanation and
presses go. An experienced user reads the same page, disagrees with one line,
and overrides it. There is no beginner mode and no expert mode — there is one
surface with progressive disclosure, and the explanation is the product.

The predictor answers in two layers, and the difference between them is
load-bearing:

- **Memory is arithmetic, and it blocks.** Whether a configuration fits a card
  is deterministic — parameters, gradients, optimizer state, activations. Being
  wrong means an out-of-memory failure minutes into a paid machine, so a
  configuration predicted not to fit is refused with the arithmetic shown.
- **Time and cost are estimates, and they warn.** They rest on a throughput
  constant that the research calls the softest number in the whole model, with
  measured anchors spanning an order of magnitude. So they are quoted as a
  range, labelled as an estimate, calibrated against every run the platform
  performs, and they never block. This is exactly the posture ADR-0007
  established, and naming which half of the predictor inherits it — and which
  half deliberately does not — is the point of this spec.

## User Stories

1. As a user, I want the product to choose a training method for me, so that I do not have to know what QLoRA is to fine-tune a model.
2. As a user, I want to see which method was chosen and why, so that I learn something rather than being handed a black box.
3. As a user, I want the product to choose a GPU for me, so that I do not have to reason about VRAM.
4. As a user, I want to see which GPU was chosen and what it costs per hour, so that I understand where the money goes.
5. As a user, I want to see the alternatives that were considered and rejected, so that I can tell whether the choice was made or merely defaulted to.
6. As a user, I want to see how much faster a more expensive configuration would be, so that I can decide whether the speed is worth the money.
7. As a user, I want the product to choose how many GPUs to use, so that a large job is not silently attempted on a card that cannot hold it.
8. As a user, I want the product to refuse to use more GPUs than a job needs, so that I am not billed for hardware that sits idle.
9. As a user, I want the product to calculate the disk my job needs, so that a large model does not fail after downloading half its weights.
10. As a user, I want to see the disk size and what it contributes to the cost, so that storage is not an invisible line on a bill.
11. As a user, I want the predicted peak memory shown, so that I can see the headroom my job is running with.
12. As a user, I want a job that cannot fit any available hardware to be refused before I spend anything, so that I do not pay to discover it.
13. As a user whose job is refused for memory, I want to see the arithmetic, so that I can tell what to change — a smaller model, a shorter sequence, or a different method.
14. As a user, I want a duration range rather than a single number, so that I am not misled by a precision the estimate does not have.
15. As a user, I want the duration estimate labelled as an estimate, so that I know which numbers are measured and which are predicted.
16. As a user, I want a cost range in the currency my account is billed in, so that the figure means something to me.
17. As a user, I want the cost broken down by phase — provisioning, downloading, training — so that I understand what a long cold start costs me.
18. As a user, I want to see the token count of my dataset, so that I understand the size of what I am training on.
19. As a user, I want to see how sequence length was chosen, so that I can tell whether my longest rows are being truncated.
20. As a user, I want a quote to expire, so that a price I was shown yesterday is not silently honoured against hardware that has changed.
21. As a user, I want the quote frozen into the job when I launch, so that a completed job describes what it actually did.
22. As an experienced user, I want to override any decision the predictor made, so that I am not limited to what it inferred.
23. As an experienced user, I want the predictor to recompute the rest of the configuration when I override one part, so that my change does not leave a stale value beside it.
24. As an experienced user, I want an override that makes the job infeasible to be refused with the same arithmetic, so that the safety property is not lost by taking control.
25. As an experienced user, I want my overrides recorded in the job spec, so that a run says what it actually used rather than what it would have defaulted to.
26. As a user, I want the same explanation available after the job has finished, so that I can understand a completed run as well as a planned one.
27. As an operator, I want every prediction compared against what actually happened, so that the estimate improves with each run rather than staying as accurate as it was on day one.
28. As an operator, I want to see where the predictor was wrong and by how much, so that a systematically bad estimate is visible rather than merely disappointing.
29. As a reviewer of this repository, I want to see which numbers are measured and which are derived, so that I can tell what has been proven from what has been calculated.

## Implementation Decisions

**A new `models` seam resolves a base model to facts.** Given a repository
reference and a pinned revision, it returns parameter count, layer and hidden
dimensions, architecture family, dense-or-MoE, chat-template presence, whether
pad and EOS are distinct, context length, and licence. It is a seam because the
predictor and the compatibility probe must read the *same* facts — two
implementations would drift, and the drift would surface as a job that the
probe passed and the predictor mispriced. It is also a seam so that tests can
supply a 70B model without network access.

**`feasibility` becomes the predictor** rather than gaining a sibling. Its
existing contract — take the dataset and the choices, return something the
user is shown before launching — is the right one. What changes is that it
returns a configuration rather than a sentence, and that half of it blocks.

**Two hard-coded values are deleted, not parameterised.** The per-catalog-entry
peak-VRAM figure goes, because a stored number cannot describe a model the
catalog has never seen; peak memory is computed from model facts for every
model, including catalog ones. The GPU preference list goes, because ordering
three names by hand is not a selection rule; selection reads live availability
and price from the provider and picks the cheapest configuration that fits.

**The memory model is the published one, not an invention.** Weights,
gradients, optimizer state and activations, summed per method and dtype, with
a fixed headroom fraction against card capacity. Its first anchor already
exists and is exact: the trainable-parameter count for a 4B QLoRA run was
predicted from the model's own config and the run returned precisely that
figure. The peak-VRAM extension is arithmetic of the same kind, and the 5.31 GB
measured on that run is its first calibration point.

**Selection prefers the cheapest configuration that fits, then fewer larger
cards over more smaller ones.** Multi-GPU costs interconnect overhead, so two
cards are not twice one card, and a configuration that asks for hardware the
job does not need is worse than a slower one — particularly in front of a
reader who sells the hardware.

**Disk is computed, not constant.** Weights at their download dtype, plus
checkpoints at their retained count, plus any merged output, plus the trainer
image and working space, floored at the platform minimum. Spike 5 supplies the
ceiling and the price per unit.

**Cost is composed per phase, not as one multiplication.** Provisioning,
readiness, image pull, model download, training, evaluation and teardown have
different durations and the first four are largely independent of the dataset.
A cold start that is four minutes on an 8 GB model and forty on a 140 GB one is
the difference between a rounding error and half the bill, and quoting a single
blended rate hides exactly the number a user most wants.

**Every decision is returned as a structured record, not prose.** Each carries
the decision made, the value chosen, the constraint that forced it, and the
alternatives considered with what each would have cost. This is deliberately
the same shape as a decision record — the discipline the project has applied to
its own choices, turned into a product surface — and it is structured rather
than rendered so that the API, the interface and the artifact manifest all read
from one source.

**The quote is persisted, immutable, and expires.** It references the dataset
version and the model revision it was computed against, so a quote cannot
outlive its inputs. Launching copies it into the job spec, which is written
once and never updated.

**Currency is read from the account on every quote.** This account bills in
rupees and assuming otherwise misreports every figure by nearly two orders of
magnitude. The currency travels with the amount as a unit, never as a
formatting choice.

**Every attempt records predicted against actual** — duration, peak memory,
cost — so that over time the estimate is calibrated against a countable
number of runs. The sentence *"calibrated against N real runs"* is worth more
than a better constant, and it is only available if recording starts with the
first run of the week rather than the last.

## Testing Decisions

**A good test here asserts on what the predictor returns, never on how it got
there.** The memory model is arithmetic and its intermediate terms are
implementation; the assertion is that a named configuration fits or does not,
and that the reason names the right constraint. Tests that pin the internal
breakdown will fail every time the model is refined, which is exactly when they
should stay green.

**The predictor is tested without a network and without a GPU.** The `models`
seam is faked, following the pattern the provider seam already established:
one double in the package rather than one per test file, constructed with the
model facts a case needs. The provider seam supplies availability and pricing
the same way.

**Anchored against the two measurements that exist.** The 4B QLoRA
configuration must predict the trainable-parameter count that the real run
returned exactly, and must predict a peak within a stated tolerance of the
measured figure. These are the only two hard anchors the project has, and a
change that breaks either is a regression in the literal sense.

**Prior art:** the existing feasibility tests are the closest model — pure
functions over a dataset record and a hyperparameter map, no I/O, asserting on
the returned warning rather than on how it was phrased. The orchestrator tests
show the seam-and-double pattern this spec extends to `models`.

**Property-shaped tests for selection**, because the rules are comparative
rather than absolute: a configuration that fits must never be passed over for
a more expensive one that also fits; adding GPUs must never reduce predicted
total memory; a larger dataset must never predict a shorter duration.

**The verification clause.** This spec is not done when its tests pass. It is
done when a real job has been quoted before launch and its actuals recorded
against the prediction, through the interface, on real hardware. It belongs to
the healthy-run cluster and rides the first end-to-end run of the week — and
every subsequent run this week feeds its calibration, which is why the
recording path ships before anything else in this spec.

## Out of Scope

- **Importing models from outside the catalog**, and the compatibility probe
  that gates them. The predictor consumes model facts through the `models`
  seam; where a model came from is Spec 009's problem.
- **Exposing the full hyperparameter surface.** This spec's overrides are the
  decisions the predictor itself makes. The Axolotl-derived advanced surface is
  Spec 009.
- **Full fine-tuning and multi-GPU execution.** The predictor computes
  configurations for both because a predictor that cannot describe them cannot
  refuse them honestly. Running them is Spec 009, gated on spike 6.
- **Token counting's implementation.** This spec consumes a token count;
  producing one from a streamed dataset is Spec 006.
- **Cost reconciliation against an invoice.** Costs remain derived from the
  event log against a stored rate. Nothing in the product reads a bill, and the
  interface says so.

## Further Notes

The riskiest number in this spec is throughput, and it is worth stating the
risk rather than burying it. The product has one measurement, from one model on
one card with one method, and the published range for the underlying efficiency
figure spans nearly an order of magnitude across workloads. Extrapolating from
that single point to a 70B job on different silicon crosses model size, GPU
generation, method and sharding simultaneously. The mitigations are structural
rather than clever: quote a range, never a point; label it an estimate; never
block on it; and calibrate from every run.

The second-order effect worth watching is that this spec makes the interface's
central page the *plan*, not the form. That is a deliberate inversion — the
existing flow asks the user to choose and then launches. After this spec the
user asks for an outcome, the product proposes a plan with reasons, and the
user accepts or edits it. Every later spec inherits that shape, which is why
this one is written first.
