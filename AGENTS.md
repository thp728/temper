# Temper

A fine-tuning platform: upload a JSONL dataset, pick a base model, get a trained LoRA adapter.

Built as a take-home for [JarvisLabs.ai](https://jarvislabs.ai). **Submission 2026-08-31.** The bar set by the brief is *"70–80% of what commercial products offer"*, *"cannot be the hello-world level of fine-tuning"*, and — the one that shapes everything here — **"I should be able to explain every decision that's done. Expect to be grilled."**

Auth and billing are the only sanctioned gaps. Every other flow is meant to be complete.

---

## Environment — read before running anything

### ⚠️ Anything that SSHes must run via PowerShell, not Bash

Git Bash ships its own `ssh` and **cannot see the Windows `ssh-agent` service**. The JarvisLabs key is passphrase-protected, so auth fails — and it surfaces as `"ssh ready — no answer within 240s"`, which looks exactly like a dead VM. That misdiagnosis already cost an evening and produced a wrongly-filed platform bug.

```powershell
& "d:\Dev\life-os\.venv\Scripts\python.exe" -u spike/spike4.py
```

Before any GPU work: `ssh-add -l` must list one ED25519 key. If it doesn't, `ssh-add ~/.ssh/id_ed25519`.

### Other things that have bitten

- **`python -u`** for backgrounded runs — stdout buffers otherwise and you see nothing until exit.
- **Prefer the Write tool over bash heredocs** for files over ~50 lines. Large heredocs have silently failed here, producing no file and no error.
- **`git commit -F -` with a heredoc**, never `-m "…"` with backticks — backticks get command-substituted and eat part of the message.
- **`PYTHONIOENCODING=utf-8`** for any Python that writes files containing emoji; Windows defaults to cp1252 and raises.
- **Interpreter:** `d:\Dev\life-os\.venv\Scripts\python.exe` (has `jarvislabs`, `fastapi`, `pytest`).

### Secrets

`spike/.env` holds `JL_API_KEY`. It is git-ignored at two levels. **Never print it, never commit it, never paste it into a message.** This repo goes public at submission.

### Cost discipline

GPU work bills per minute against a ₹50,000 grant (~₹50 spent so far). Cheapest VM-capable GPU is **L4 at ₹41.31/hr**. The account bills in **INR** — always read `account.currency()` rather than assuming USD.

**Every code path that creates a VM must destroy it in a `finally` block and then confirm by listing instances.** A destroy call's return value is not evidence. An orphaned GPU bills until somebody notices.

---

## Design rules — decided, do not relitigate

Each of these has a decision entry with alternatives and tradeoffs. Changing one means writing a new entry, not editing the old.

- **Correctness settings are never user-settable.** Chat-template resolution, EOS handling, loss masking, NF4 double-quant, all-linear LoRA targets, bf16, seed. These are the highest-frequency silent-failure surface — they pass every obvious health check and surface only as bad output. Exposing them buys a user nothing and costs correctness.
- **α tracks r.** Change `lora_r` without `lora_alpha` and α recomputes as `2r`. Pairing a new rank with a stale scale is a silent quality bug.
- **rsLoRA is inferred at `r >= 32`**, never exposed.
- **Unknown job keys are refused loudly** — at the top level *and* inside `hyperparameters` — and echoed back as `rejected_overrides`. An override the caller believes is in effect but isn't is worse than a refusal.
- **Thinking mode is detected from the dataset**, applied identically at training and serving. Mixed datasets **block** with a line-numbered error, because they are ambiguous by construction.
- **The trainer publishes no ports.** `ufw` does not filter Docker-published ports, and a `DOCKER-USER` rule matched on the published port never fires (the packet is already DNAT'd and carries the *container* port). Not publishing is the only mitigation that holds.
- **Readiness distinguishes *unreachable* from *authentication failed*.** Opposite remedies; collapsing them into "no answer" is how the evening above was lost.
- **The trainer image is pinned by digest.** The tag is a comment. Never `pip install` inside it — that reintroduces the dependency-resolution problem the pinned base exists to avoid.
- **Axolotl owns the training loop; we own the contract.** Do not call TRL/PEFT directly — those APIs move underneath you (TRL 1.x dropped `warmup_ratio` from `SFTConfig`, which broke checkpoint resume outright).

## Two phases, split at the Friday checkpoint

**The product ships production-ready. The scrappy slice is a means to the checkpoint, not the deliverable.**

### Phase A — until Fri 2026-08-21: prove the loop

Current stack stays: FastAPI, SQLite, thread per job, plain server-rendered HTML. **Do not migrate anything during Phase A.** The checkpoint asks one question — *can a user go from dataset to adapter through the product?* — and hours spent on Postgres before the loop runs are hours not spent making the loop run. If the journey does not work by Friday evening, scope gets cut that day.

### Phase B — Sun 2026-08-23 to Mon 2026-08-31: production

Everything after the checkpoint is polish and hardening, and it is the larger half of the build. Postgres with SQLAlchemy and Alembic, Celery and Redis, S3-compatible object storage, SSE over Redis pub/sub, Next.js with shadcn/ui, Ruff and mypy, GitHub Actions, Docker Compose, structured logging with correlation IDs, Sentry.

**Target spec: `technical-architecture.md` in the vault.** Not `reference-technical-architecture.md`, which is superseded — though its §0 corrections log stays useful as the record of what was assumed versus what turned out true.

### Write Phase A so Phase B is a migration, not a rewrite

This costs nothing now and saves a rewrite later:

- **Keep validation pure.** Functions over parsed rows, no I/O, no framework imports.
- **Keep the orchestrator a function over a job record.** It should not care whether the record came from SQLite or Postgres, or whether a thread or a Celery worker called it.
- **No SQL in request handlers.** All persistence behind `db.py`-style functions.
- **Domain logic carries over unchanged** — validation rules, the state machine, the orchestration sequence, thinking-mode detection. What changes in Phase B is what it persists to and what runs it.

**Production-grade is part of the deliverable, not a stretch goal.** The brief asks for 70–80% of what commercial products offer, and none of them runs on a thread pool and a local file. This repository is also the public portfolio artifact, read by a company whose own product is GPU infrastructure — **the orchestration layer is the work sample.**

## Error handling

- Every API error carries a **stable machine-readable code**, a user-safe message, and a line or field reference where one applies.
- **Validation errors name the line.** A rejection that doesn't say *which line* leaves the user guessing at a file they can't see. This is where users churn first.
- **`result.json` is always written, including on failure.** The orchestrator should never have to parse logs to learn what happened.

---

## How to work

- **Direct and concise.** Lead with what needs action or a decision, not a status recap.
- **Label assumptions. Never present a guess as a fact.** If something is extrapolated rather than measured, say so in the code and the docs.
- **Record what turned out wrong, visibly.** The standard for this project is *"here's what I got wrong and how I caught it"* — it lands better than a clean chart. Corrections are an asset here, not an embarrassment. Do not quietly fix and move on.
- **Log decisions the day they are made.** Entries written during the build record reasoning; entries written afterwards reconstruct it, and reconstruction is what fails under questioning.
- **Prove it on real hardware before believing it.** Four spikes found eleven wrong assumptions in a document that looked authoritative. Measured numbers beat estimated ones everywhere in this repo — mark which is which.
- **Tests do not cost money.** Stub the provider. A suite that provisions a GPU per run does not get run.

## Commits

Short imperative subject, then a substantive body explaining **why** — including tradeoffs accepted and alternatives rejected. The commit log is part of the defensibility story.

---

## Reference docs (outside this repo)

Reasoning and project state live in the private vault at `d:\Dev\life-os\projects\jarvislabs-assignment\`:

| File | What it holds |
| --- | --- |
| `decisions.md` | Decisions with Why/Alternatives/Tradeoffs/Rollback. **This is the deliverable** |
| `technical-architecture.md` | **The spec.** Production stack, scoped. Phase B builds this |
| `reference-technical-architecture.md` §0 | Superseded — but §0's 17 corrections are the record of what was assumed vs true |
| `scope-flow-table.md` | Ratified parity boundary — 75% of Together AI's user-facing flows |
| `tasks.md` | Current status and full backlog |
| `wiki/` | Plain-English concept pages; the explainability gate |

In-repo: `spike/README.md` (what each spike proved, and what it cost) and `trainer/README.md` (the image contract and what is still unverified).

## Layout

```
api/       control plane — FastAPI + SQLite, one process
trainer/   the pinned training container and its /job -> /out contract
spike/     infrastructure probes against the live account
```

`python -m pytest -q` from the repo root.
