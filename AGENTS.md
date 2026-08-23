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

- **Correctness settings are default-locked, not hidden.** Chat-template resolution, EOS handling, loss masking, NF4 double-quant, all-linear LoRA targets, bf16, seed. Defaults are hard-coded and the common path never touches them — they are the highest-frequency silent-failure surface. **But Advanced mode exposes every one of them**, with the failure mode named inline next to each, and overrides recorded in the run spec. Every serious platform in this space exposes these; hiding them permanently is a limitation dressed as a safety feature. **The export-time template probe is what makes exposure safe** — it catches a wrong override before the user does.
- **α tracks r.** Change `lora_r` without `lora_alpha` and α recomputes as `2r`. Pairing a new rank with a stale scale is a silent quality bug.
- **rsLoRA is inferred at `r >= 32`**, never exposed.
- **Unknown job keys are refused loudly** — at the top level *and* inside `hyperparameters` — and echoed back as `rejected_overrides`. An override the caller believes is in effect but isn't is worse than a refusal.
- **Thinking mode is detected from the dataset**, applied identically at training and serving. Mixed datasets **block** with a line-numbered error, because they are ambiguous by construction.
- **Curated is a default, not a boundary.** Phase B adds Hugging Face import for **both** datasets and base models. Imported datasets go through the identical validation pipeline — nothing gets a shortcut for arriving over the network. Imported models must pass a **compatibility probe** (pinned revision, dense not MoE, chat template present, tokenizer loads, `pad != eos`, licence resolved, predicted VRAM fits) whose result is **shown to the user, not just enforced**.
- **Multi-GPU is provisioning, not architecture.** JarvisLabs VMs take **up to 8 GPUs** — `num_gpus` is a create parameter, and 8 devices are free on every VM-capable type. 8× H100 = 640 GB, which covers 70B full fine-tuning on one host via `accelerate` FSDP FULL_SHARD (Axolotl configures it). v1 ships single-GPU because the catalog is 4B and 8B. ⚠️ **Partly exercised as of spike 6 (2026-08-23), and the split matters.** *Proven on 2× L4:* the pinned image sees both devices, FSDP FULL_SHARD shards and steps, a `.distcp` checkpoint is written, and **sharded resume works**. *Not proven:* the loss collapses to zero with a `nan` grad_norm, so **the mechanism runs and the numerics do not** — taking steps is not training. *Still untouched:* 8 devices is a different NCCL topology from 2. Never describe multi-GPU training as working without naming the `nan`.
- **Nothing is deployed.** Docker Compose is the deployment target; everything runs on localhost. The stack stays deploy-ready — S3-compatible storage, env-driven config — but deploying is out of scope and answered in `grilling-prep.md` instead.
- **The trainer publishes no ports.** `ufw` does not filter Docker-published ports, and a `DOCKER-USER` rule matched on the published port never fires (the packet is already DNAT'd and carries the *container* port). Not publishing is the only mitigation that holds.
- **Readiness distinguishes *unreachable* from *authentication failed*.** Opposite remedies; collapsing them into "no answer" is how the evening above was lost.
- **The trainer image is pinned by digest.** The tag is a comment. Never `pip install` inside it — that reintroduces the dependency-resolution problem the pinned base exists to avoid.
- **Axolotl owns the training loop; we own the contract.** Do not call TRL/PEFT directly — those APIs move underneath you (TRL 1.x dropped `warmup_ratio` from `SFTConfig`, which broke checkpoint resume outright).

## Two phases, split at the Friday checkpoint

**The product ships production-ready. The scrappy slice is a means to the checkpoint, not the deliverable.**

### Phase A — until Fri 2026-08-21: prove the loop

Current stack stays: FastAPI, SQLite, thread per job, plain server-rendered HTML. **Do not migrate anything during Phase A.** The checkpoint asks one question — *can a user go from dataset to adapter through the product?* — and hours spent on Postgres before the loop runs are hours not spent making the loop run. If the journey does not work by Friday evening, scope gets cut that day.

### Phase B — Sun 2026-08-23 to Mon 2026-08-31: production

Everything after the checkpoint is polish and hardening, and it is the larger half of the build. Postgres with SQLAlchemy and Alembic, **Temporal** for durable job orchestration, Redis for SSE pub/sub, MinIO for object storage, Next.js with shadcn/ui, Ruff and mypy, GitHub Actions, Docker Compose, structured logging with correlation IDs.

**Temporal, not Celery.** A run is multi-step, takes hours, and holds an irreversible side effect — a billing VM — mid-workflow. That is what durable execution is for; Celery would mean hand-rolling idempotency and compensation per step. **The reconciler stays regardless** — durable execution recovers *the job*, the reconciler destroys any VM with no run that owns it and protects *the money*. Two mechanisms, two different failures.

**Target spec: `technical-architecture.md` in the vault.** Not `reference-technical-architecture.md`, which is superseded — though its §0 corrections log stays useful as the record of what was assumed versus what turned out true.

### Write Phase A so Phase B is a migration, not a rewrite

This costs nothing now and saves a rewrite later:

- **Keep validation pure.** Functions over parsed rows, no I/O, no framework imports.
- **Keep the orchestrator a function over a job record.** It should not care whether the record came from SQLite or Postgres, or whether a thread or a Temporal activity called it. In Phase B each step becomes an activity, so keep the steps separable and individually idempotent — creating a VM twice is the failure that costs money.
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
| `decisions.md` | Decisions with Why/Alternatives/Tradeoffs/Rollback, through 2026-08-19. **Closed to new entries** — it stays untouched as the record of the build to that date. New decisions are ADRs in this repo; see below |
| `technical-architecture.md` | **The spec.** Production stack, scoped. Phase B builds this |
| `grilling-prep.md` | **Answers for everything cut** — auth, billing, deployment, multi-tenancy, why Axolotl, why not Ray. Living doc; update it as decisions land |
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

---

## Agent skills

### Issue tracker

GitHub Issues on `thp728/temper`, driven through the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, each label string equal to its name — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. Both exist. `docs/adr/README.md` is the index, and it explains why the numbering starts partway through the project: the first thirteen decisions were logged in the private vault and are copied in before the repo goes public.

**ADRs in this repo are where new decisions go**, and they are public at submission — write them for that audience from the first entry. The vault's `decisions.md` is closed but not superseded: its thirteen entries get copied in before the repo goes public, so the record does not appear to start three-quarters of the way through the project.

**Decision records are written as the change lands, not afterwards.** Each Phase A spec names the ADRs its work produces, for exactly this reason: entries written during the build record reasoning, entries written after it reconstruct reasoning, and reconstruction is what fails under questioning. See `docs/agents/domain.md`.
