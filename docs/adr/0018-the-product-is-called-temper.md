# ADR-0018 — The product is called Temper, in its own repository

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

A name needs to survive the question *"why that name?"*, and a portfolio artifact needs a home that can be published without dragging private material with it.

**Why this name:** you do not reforge steel to change its properties — you temper it, a controlled process that alters behaviour while the base material survives. That is precisely QLoRA: the base weights stay frozen and a small adapter carries the change, where full fine-tuning is reforging. The brief says every decision gets questioned, and *"why that name?"* is a cheap question with a real answer here. It also matches the shape of earlier project names — a single evocative word rather than a description.

**Why a separate repo:** the take-home is the publishable portfolio artifact. It cannot live in the private personal vault, which holds finance, health and career material that is not going anywhere near a public reveal.

**Why private first:** a repository with three commits and no product, visible on the Wednesday of a build week, reads worse than the same history revealed at once. Commit history still proves incremental work when the switch is flipped — nothing is lost by waiting, and the reveal moment stays controlled.

## Decision

Named the product **Temper**. Created `github.com/thp728/temper` as a **private** repository, seeded with `trainer/` and `spike/`, to be **made public at submission**.

## Alternatives considered

*Hone* (rejected — cleaner on collisions, but the metaphor is weaker: honing sharpens an edge without changing the material); *Whetstone* (rejected — `sandialabs/Whetstone` is an ML training framework, a direct collision); *Lathe* (rejected — a lathe removes material, so the analogy inverts under one follow-up question, and `devenjarvis/lathe` is an LLM tool); *a descriptive name* like `qwen-tuner` (rejected — unambiguous but forgettable, and distinctiveness is part of this artifact's job); *keeping it in the private vault* (rejected, see above); *public from commit one* (rejected, see above).

## Consequences

`temperlang/temper` is an existing programming language — a different domain, but it exists and will compete for the bare word in search. Code lived in two places while the vault copies were retired, which was a real duplication risk: **the repo is authoritative from here, and the vault copies must not be edited.** Nothing public existed to point at if the founder asked early.

## Rollback

Renaming a GitHub repository preserves history and redirects the old URL, so the name is cheap to change until it is public. Visibility flips in one click.
