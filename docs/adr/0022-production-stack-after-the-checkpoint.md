# ADR-0022 — The product ships production-ready; scrappy only until the checkpoint

- **Status:** accepted
- **Date:** 2026-08-19

> **A note on the number and the date.** This decision was recorded on
> 2026-08-19 in the private working vault where the first thirteen decisions
> were logged, and copied into this directory on 2026-08-26
> ([#26](https://github.com/thp728/temper/issues/26)). Its file number reflects
> when the record landed here, not when the decision was made; the wording is
> the original, with private-vault links replaced by descriptions of what they
> pointed at.

## Context

I had argued for staying scrappy throughout, on the grounds that parity is measured in user-facing flows and stack work moves that number by zero. Two errors in that.

**The time argument was anchored on the wrong date.** It treated Friday's checkpoint as the budget — two evenings and a day — when the checkpoint is a *scope tripwire* and the deadline is 08-31. There are eleven build days after it, including seven away from the desk.

**And it judged the artifact only against the flow table.** The repository is also the public portfolio piece, and it will be read by a company whose own product is GPU infrastructure. **The orchestration layer is the work sample.** Handing an infrastructure company SQLite and a thread pool is a poor demonstration of orchestration ability regardless of what the flow count says, and *"cannot be the hello-world level"* plausibly reads at the infrastructure layer too, not only the fine-tuning one.

The cost side also moved: with AI-assisted scaffolding, standing up Postgres, Celery, object storage and a Next.js front end is no longer the multi-day exercise the original reasoning priced in.

## Decision

Temper ships on a production stack — PostgreSQL with SQLAlchemy and Alembic, Celery and Redis, S3-compatible object storage, SSE over Redis pub/sub, Next.js with shadcn/ui, Ruff and mypy, GitHub Actions, Docker Compose, structured logging. Split in two phases: **the current scrappy stack stands until the Friday 08-21 checkpoint**, and every build day after it goes to migration and hardening. The Phase B architecture document (private) supersedes the research-phase reference document.

## Alternatives considered

*Scrappy throughout* (rejected — see above, this was my recommendation and it was wrong on both the date and the audience); *production stack starting immediately* (rejected — it puts the checkpoint at risk to buy days that already exist after it); *production stack but keep SQLite* (rejected — a durable job queue needs `SELECT … FOR UPDATE SKIP LOCKED`, and a thread that dies with its process is not an acceptable owner of a paid GPU).

## Consequences

A migration in the middle of the build, which is real work and real risk. Mitigated by four constraints on how Phase A is written — validation stays pure, the orchestrator stays a function over a job record, no SQL in request handlers, domain logic carries over untouched. What changes in Phase B is what the code persists to and what runs it, not the rules it encodes. Also: Phase B is now the larger half of the build and sits mostly in travel time, which is interruptible time.

## Rollback

If Phase B runs out of road, the slice still works and still demonstrates the loop. The honest fallback is to ship it with the migration partially done and say exactly which parts landed — better than a half-migrated system presented as finished.
