# ADR-0012: The repository ships under Apache-2.0

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

Apache-2.0, as `LICENSE` at the repository root, copyright "the Temper
authors".

## Alternatives considered

**No licence until someone asks.** Rejected: it is the current state and it is
the problem. Default copyright grants nothing, and the gap reads as an
oversight rather than a decision — which is precisely what this repository
tries never to ship.

**MIT.** Rejected, narrowly. MIT is equally recognisable and equally
permissive, but it carries no patent grant. This repository's value is
substantially in orchestration methods — teardown-in-`finally`, stall and
duration circuit breakers, the event channel over SSH — and a patent grant is
cheap insurance for a codebase built on techniques learned from infrastructure
software that itself relies on them. Apache-2.0 costs nothing extra and states
the grant explicitly.

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
- The NOTICE mechanics of Apache-2.0 are available but unused while there are
  no NOTICE-worthy components of our own.
- Base-model licences remain a separate, user-facing matter: an adapter
  produced from Qwen3 weights carries Qwen's Apache-2.0 terms regardless of
  what licences this repository's own code. The catalog already surfaces this;
  nothing about this decision changes what flows through to users' artifacts.
- The licence lands before publication, so no history rewrite or relicensing
  pass is needed. Contributors between now and submission are covered by the
  standard Apache-2.0 grant-back unless stated otherwise.

## Rollback

Relicensing requires the consent of every copyright holder, which while the
author is the only one is a single decision. Replace `LICENSE`, update this
record with a superseding entry, and note the change in the README. After any
external contribution, the same move needs that contributor's agreement — which
is the reason the decision was made before the repository went public rather
than after.
