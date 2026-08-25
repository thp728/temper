# Spec 011 — Evaluation and delivery

**Status:** ready for tickets
**Phase:** B, band 3 (the parity path — what raises the number)
**Depends on:** Spec 006 (artifact kinds and object storage), Spec 009 (methods, and the export-time template probe)
**Produces:** ADRs that the best checkpoint is chosen by held-out loss with the choice recorded, that a served endpoint stops itself, and that a general-capability check is a small slice whose limits are stated
**Assumes:** ADR-0008 (artifacts ship at the precision they were trained in)

## Problem Statement

A job finishes and hands back a file. Nothing in the product answers the question
the user actually has, which is not *did it train* but **did it work, and is it
better than what I started with.**

The gaps, in the order a user meets them:

- **There is no held-out split**, so there is no eval loss, so nothing
  distinguishes a model that learned from one that memorised. Overfitting is
  invisible.
- **The last checkpoint is the result**, by default rather than by choice. The
  last checkpoint is frequently not the best one, and selecting it silently is a
  quality decision made by omission.
- **Nothing compares the tuned model to the base model.** The comparison a user
  without a background in this can actually read — the same prompt, before and
  after — does not exist.
- **Nothing checks what was lost.** Fine-tuning degrades general capability, the
  effect is real, and no other signal in the product catches it.
- **The artifact ships without provenance.** No record of which base model at
  which revision, which dataset, which settings, which licence obligations
  travel with it, or which checkpoint it came from.
- **The artifact ships in one form only.** A user who wants to run it locally,
  or to serve it as a single merged model, does the conversion themselves.
- **There is nothing to try it on.** Downloading weights is not the same as
  seeing the model answer.

The product currently proves that training happened. It does not help anyone
decide whether the training was any good, and that judgment is the reason people
fine-tune.

## Solution

**Everything the platform learns about a finished job travels with it.**

A held-out portion of the dataset, split automatically after deduplication, gives
an eval loss plotted against the training loss on one chart — the single clearest
overfitting signal available and the reason the split exists at all. The
checkpoint with the best held-out loss is selected as the result, and the choice
is recorded rather than implied.

**A comparison a non-specialist can read.** The same held-out prompts run through
the base model and through the tuned model, shown side by side. Nothing about
this is a benchmark and the interface should not pretend otherwise; it is the
most directly useful evidence the platform can produce, and it is produced on a
machine that is already warm with both models loaded.

**A small general-capability check.** A fixed slice, run before and after,
reported as a delta. It is a smoke test for catastrophic forgetting, not an
evaluation harness, and the interface says so — a small slice with a stated
margin of error is honest; the same slice presented as a benchmark score is not.

**Provenance travels with the artifact.** Base model and pinned revision, dataset
fingerprint with counts, the full configuration including any overrides, the
evaluation summary, the checkpoint it came from, and the licence obligations that
propagate to whatever the user does next. This is a compliance artifact as much
as a convenience: the base model's licence terms flow through to the tuned
result, and a user who cannot see them cannot comply with them.

**More than one form of the artifact.** The trained result as it is; a merged
model for serving; and a quantised local-inference format, which the baseline
this product is measured against does not offer at all and which is the
"run it on your own machine" story.

**And somewhere to try it.** A temporary authenticated endpoint that stops
itself after inactivity. Endpoints that outlive their usefulness are the top
complaint against the commercial baseline — the training is cheap and the
forgotten warm machine is the bill — so stopping itself is the feature, not a
convenience.

## User Stories

1. As a user, I want part of my dataset held out automatically, so that I get an honest signal without preparing a second file.
2. As a user, I want the held-out portion excluded from training, so that the signal means something.
3. As a user, I want to see training and held-out loss on one chart, so that I can see them diverge.
4. As a user, I want to be told when held-out loss stops improving, so that I understand what overfitting looks like in my own run.
5. As a user, I want the best checkpoint chosen rather than the last, so that I get the best model my run produced.
6. As a user, I want to see which checkpoint was chosen and why, so that the choice is legible.
7. As a user, I want to download a checkpoint other than the chosen one, so that I am not locked out of my own run's history.
8. As a user, I want to see my tuned model and the base model answer the same prompt, so that I can judge the difference myself.
9. As a user, I want that comparison to use held-out examples, so that I am not shown answers the model memorised.
10. As a user, I want to know whether the model got worse at general tasks, so that I can catch a model that learned my task and forgot everything else.
11. As a user, I want the general check labelled as a small sample, so that I do not read more into it than it supports.
12. As a user, I want a record of exactly what produced my model, so that I can reproduce or explain it later.
13. As a user, I want the base model's licence obligations shown with my result, so that I know what applies to what I do next.
14. As a user, I want a fingerprint of the dataset recorded, so that I can tell which data produced which model.
15. As a user, I want my overrides recorded with the result, so that the record describes what actually ran.
16. As a user, I want a merged single-file model, so that I can serve it without composing pieces at load time.
17. As a user, I want a quantised local format, so that I can run my model on my own machine.
18. As a user, I want to know what each format is for, so that I can choose without knowing the internals.
19. As a user, I want the merge performed at full precision before any quantisation, so that my result is not degraded twice.
20. As a user, I want to send a prompt to my model without downloading anything, so that I can try it immediately.
21. As a user, I want that endpoint to require a key, so that it is not open to anyone who finds the address.
22. As a user, I want the endpoint to stop itself when I stop using it, so that I do not pay for something I forgot.
23. As a user, I want to know before starting an endpoint what it costs per hour and when it will stop, so that there is no surprise.
24. As a user, I want to stop the endpoint immediately, so that I am not waiting on a timer.
25. As an operator, I want the endpoint unreachable from outside except through the intended path, verified rather than assumed, so that a served model is not an open door.
26. As an operator, I want endpoint keys stored hashed, so that a leak of the store is not a leak of the keys.
27. As a user, I want evaluation failure not to destroy my run, so that a problem in the extra step does not cost me the training.

## Implementation Decisions

**The split happens after deduplication and before training**, and its size is
recorded. Splitting before deduplication lets a duplicated row appear on both
sides, which makes the held-out loss optimistic in a way nothing downstream can
detect.

**Best-checkpoint selection is by held-out loss, and the decision is stored** on
the run rather than inferred at download time. A user who returns a week later
must see the same answer, and a selection rule that is re-evaluated is a selection
rule that can change.

**Evaluation runs on the machine that is already warm.** Both the base and tuned
models are loaded there; provisioning a second machine to compare them would cost
another cold start for work that takes moments where it already is.

**Generation comparison uses fixed decoding settings recorded with the result**,
because a comparison run at a different temperature is not a comparison.

**The general-capability slice is small, fixed, versioned, and reported with its
uncertainty.** Its purpose is to catch a large regression, not to rank models. The
interface states the sample size beside the number; a small slice honestly
labelled is useful, and the same slice dressed as a benchmark is misleading.

**Evaluation failure never fails the run.** The training result is the
deliverable; the evaluation is what the platform adds. A failed evaluation is
recorded with its reason and the artifact is still delivered.

**Merging happens at full precision, and quantisation happens once, afterwards.**
Merging into an already-quantised base compounds error, and the correct order is
the difference between a usable local model and a subtly degraded one.

**Format conversion is a step off the correctly merged model**, not a parallel
pipeline. Each additional format is one more step from the same source, which is
why offering them is cheap and why getting the merge right matters more than the
formats do.

**The manifest is generated, never hand-written**, from what the run recorded. A
provenance document that can drift from the run it describes is worse than none,
because it will be believed.

**The endpoint is temporary by construction.** It carries an expiry from the
moment it starts, extends on use, and stops itself. It is authenticated with a
key hashed at rest, and its reachability from outside is verified rather than
assumed — the platform's firewall does not filter published container ports the
way it appears to, which is why the trainer publishes nothing and why anything
that does publish is checked from outside before it is handed a key.

## Testing Decisions

**A good test here asserts on what the user receives.** That the artifact loads;
that the manifest describes the run that produced it; that the chosen checkpoint
is the one with the best held-out loss; that a comparison shows both sides. How
the evaluation was scheduled is implementation.

**Selection, splitting and manifest generation are pure and tested without
hardware.** Given a set of checkpoints with held-out losses, the best is chosen;
given a dataset, the split is proportional, deterministic under a seed, and
disjoint; given a run record, the manifest contains every required field and no
placeholder.

**The merge-then-quantise order is asserted as a property**, because the failure
it prevents is silent: a model merged in the wrong order still loads and answers,
and only produces slightly worse output.

**Format conversions are verified by loading the result**, not by checking a file
exists. A conversion that produces an unloadable file is the failure mode, and
file existence does not detect it.

**Endpoint reachability is tested from outside the machine**, not from on it.
Testing from inside proves the service is running and proves nothing about who
else can reach it, and that distinction has already been the source of a
correction in this project.

**Prior art:** the artifact verification in the existing collection path — a
checksum computed on the machine and verified against what arrived — is the
model for how every produced file should be treated. The catalog tests define
what licence information an entry must carry.

**The verification clause.** Done means one real run whose held-out loss is
charted, whose best checkpoint is selected, whose comparison renders with real
generations, whose manifest is generated, whose merged and converted formats are
downloaded and loaded, and whose endpoint answers a prompt and then stops itself.
This is the healthy-run cluster: **all of it rides the tail of a single job**,
which is why these flows are specified together rather than separately.

## Out of Scope

- **A general evaluation harness.** Task suites, leaderboards, custom metrics. The
  slice here is a smoke test and is labelled as one.
- **Human preference collection.** No annotation surface.
- **Automatic promotion or gating on evaluation results.** The user decides; the
  platform reports.
- **Persistent hosted endpoints**, replica management and resumption semantics.
  Temporary and self-stopping only.
- **Serving many adapters from one warm base.** It pays off at volumes that do
  not exist here — cut on economics rather than on time, and that distinction is
  worth stating.
- **Comparing two tuned models to each other.** Base against tuned only.

## Further Notes

This spec is in band 3, and the ordering consequence should be stated plainly:
if the days run out, these are the flows that become open issues. That is a
deliberate ranking rather than an accident, and the reasoning is that a user who
cannot see whether their model improved has still received a model, whereas a
user whose job cannot recover from a failure has received nothing.

Two of these flows — the local-inference format and the general-capability check
— were cut earlier for time and then restored, because the bar was stated in
advance and running out of time is the weakest available answer to missing a
number you were given at the start. That reasoning still holds, and if they slip
again the honest framing is not that they were cut but that they were ranked
last and the ranking was published beforehand.
