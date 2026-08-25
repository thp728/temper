# ADR-0012: The repository ships under MIT

Date: 2026-08-25
Status: accepted

## Context

The repository becomes public at submission (2026-08-31). Until a licence
exists, a public repository grants nobody anything: default copyright means
"all rights reserved", so a reviewer cloning it to evaluate may read it only by
toleration, not by right.

The awkwardness is sharper than usual because this product is in the licence
business whether it wants to be or not. It fine-tunes open-weight models whose
licence terms flow through to everything the user produces — the catalog
surfaces each base model's licence and pinned revision at job creation for
exactly that reason (the curated models, `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B`,
are Apache-2.0, which is part of why they were chosen). A platform that tells
its users what obligations attach to their artifacts, while attaching no terms
of its own, is an awkward thing to be grilled about.

The constraint set:

- The artifact is a portfolio piece, read and run by a company whose own
  product is GPU infrastructure. The licence must allow reading, running,
  modifying and reusing the code without friction.
- It must be recognisable. An evaluator should not have to study the licence to
  know what they may do.
- The ecosystem it sits on is permissive: FastAPI, Pydantic, uv, Axolotl and
  the Qwen weights are all Apache-2.0 or MIT. Nothing here imposes copyleft on
  the whole.
- The author keeps copyright; the take-home relationship with JarvisLabs does
  not change who owns the code.

## Decision

MIT, as `LICENSE` at the repository root, copyright "the Temper authors".

Roughly 170 words that an evaluator can read in full in under a minute, and the
default licence for a portfolio project. Nothing an evaluator has to reason
about before cloning.

## Alternatives considered

**No licence until someone asks.** Rejected: it is the prior state and it is
the problem. Default copyright grants nothing, and the gap reads as an
oversight rather than a decision — which is precisely what this repository
tries never to ship.

**Apache-2.0.** Rejected, and this record's first draft chose it, so the
reasoning is worth stating rather than deleted. The argument for it was the
explicit patent grant, on the reasoning that this repository's value is
substantially in orchestration methods — teardown-in-`finally`, stall and
duration circuit breakers, the event channel over SSH — and that a grant is
cheap insurance.

That argument does not survive inspection. **None of those methods is
patentable subject matter anyone is plausibly asserting**; they are ordinary
operational practice, and treating them as patent-adjacent overstates what this
repository is. The grant insures against a risk that does not exist here, and
it costs about ten times MIT's length plus a paragraph on patent retaliation
that an evaluator has to skim past. Apache-2.0 earns its place when code is
expected to be vendored into a product whose legal review looks for the grant.
A take-home is not that, and choosing the heavier licence for a benefit nothing
in this situation needs is exactly the kind of unexamined default this
repository is supposed to catch.

**GPL or AGPL.** Rejected: copyleft serves a project building a commons it
wants to stay open. This is a work sample; its reach should be as wide as
possible, and AGPL's network clause would give an evaluator one more thing to
check before running it at all.

**A proprietary or source-available licence** ("look, don't reuse"). Rejected:
it would contradict the README's own argument. A repository that publishes its
decision records, its rejected alternatives and its measured findings because
explainability is the deliverable does not then restrict the explaining.

## Consequences

- Anyone may use, modify and redistribute this code, commercially, with
  attribution and the licence text preserved.
- **No express patent grant.** Accepted knowingly, on the reasoning above. If
  this codebase ever becomes something a company vendors rather than reads,
  that is the trigger to revisit — and relicensing permissively to Apache-2.0
  later is a move MIT permits.
- Base-model licences remain a separate, user-facing matter: an adapter
  produced from Qwen3 weights carries Qwen's Apache-2.0 terms regardless of
  what licences this repository's own code. The catalog already surfaces this;
  nothing about this decision changes what flows through to users' artifacts.
- The licence lands before publication, so no history rewrite or relicensing
  pass is needed.

## Rollback

Relicensing requires the consent of every copyright holder, which while the
author is the only one is a single decision. Replace `LICENSE`, write a
superseding record, and note the change in the README. After any external
contribution, the same move needs that contributor's agreement — which is the
reason the decision was made before the repository went public rather than
after.
