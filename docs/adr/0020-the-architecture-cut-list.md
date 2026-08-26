# ADR-0020 — The architecture cut list is ratified as drafted

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

The reference architecture produced by the research phase describes a fundable product — seven build phases, a modular monolith, WCAG 2.2 AA browser tests — and it cannot be built by 08-31. The brief sanctions the two largest cuts outright: *"auth and billing are optional."* Everything else on the list is either a scaling argument with no scale behind it (the separate gateway process), a cost trap the product is positioned against (persistent warm endpoints — research report C names the hidden hosting bill as Together's top user complaint), or breadth nobody asked for (a third model is another template and tokenizer surface that can break independently, and report A argues curation *is* the feature).

**Why the GPU-minute cap survives billing:** it is a **safety control**, not a billing feature. A runaway job is a failure path, and the brief says non-auth flows must be perfect. It is also enforced outside the training process, so it holds when the worker crashes.

## Decision

Ratified the cut list unchanged. **Out:** Clerk auth, workspaces and roles; Stripe, the credit ledger, reservations and settlement; the separate inference-gateway process; persistent hosted endpoints with min-replica semantics; the third base model (14B); multi-LoRA, spot and the reconciler. **Kept:** a hard GPU-minute cap.

## Alternatives considered

Keep three base models (rejected — 14B needs an A100-80GB or better, where L4 handles 4B and 8B, and each model multiplies the silent-failure surface); keep a persistent endpoint for a smoother demo (rejected — it discards auto-stop, which is currently a differentiator, in exchange for exactly the cost trap the positioning attacks).

## Consequences

The demo has a cold start in front of whoever is watching. Single-tenant is assumed throughout, so nothing about the data model is proven under multi-tenancy. Spot pricing stays unreachable, which is separately forced by the VM execution model ([ADR-0016](0016-training-runs-on-jarvislabs-vms-over-ssh.md)).

## Rollback

Each cut is additive rather than structural — auth, billing and a persistent endpoint can be layered on later without touching the training path. They are the first entries on the "what I'd do with another month" list.
