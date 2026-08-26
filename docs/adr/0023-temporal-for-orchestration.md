# ADR-0023 — Temporal for orchestration; multi-GPU is a provisioning choice

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

A run is multi-step, takes hours, and holds an irreversible side effect — a billing VM — mid-workflow. Durable execution exists for exactly that: workflow state persists server-side and resumes from the exact point of failure. Celery retries whole tasks, so matching the guarantee means hand-rolling idempotency and compensation per step, which is re-implementing a worse Temporal.

⚠️ **This reverses my own reversal, and the error is worth recording.** I recommended Temporal, then withdrew it after finding that production means either a self-managed cluster or Temporal Cloud from $100/month, arguing the reconciler covered the failure that actually costs money. **That conflated an operating expense with an engineering choice.** Whether to pay for Cloud or self-host is a decision for whoever deploys this; the right primitive for durable workflows is not. The reconciler is a good idea *regardless* — it is not a substitute for durable execution, and framing it as one was motivated reasoning working backwards from a line item.

**Both mechanisms stay, because they address different failures:** durable execution recovers *the job*; the reconciler independently destroys any VM with no run that owns it and protects *the money*.

**Multi-GPU, measured 2026-08-19:** every VM-capable GPU type shows **8 free devices**, and `num_gpus` is a creation parameter. So scaling is a provisioning decision on the existing single-VM path — same bootstrap, same container, more cards. 8× L4 = 192 GB at ₹330/hr; 8× RTX-PRO6000 = 768 GB at ₹1,432; 8× H100 = 640 GB at ₹2,041; 8× H200 = 1,128 GB at ₹3,026. Against the memory model that covers **70B full fine-tuning on one host**. Mechanism is `accelerate` with FSDP FULL_SHARD, configured through Axolotl.

**This corrects an earlier answer about Ray.** The threshold for adopting Ray had been stated as "the first model that does not fit a single card" — wrong, and it would have been a weak answer under questioning. The real threshold is **multi-*node***, which is far higher: roughly anything past 70B full fine-tuning. Within one host, FSDP via `accelerate` is simpler and already configured.

## Decision

**Temporal** orchestrates training runs. The reconciler stays regardless. Multi-GPU is supported by provisioning more cards on one VM, not by adopting a cluster scheduler.

## Alternatives considered

*Celery + reconciler* (rejected — see above; it approximates durable execution by hand); *Ray/KubeRay* (rejected — solves multi-node, and one VM takes eight cards; adopting it means operating a cluster on rented VMs); *hand-rolled state machine over Postgres* (rejected — that is the thing Temporal is, minus the testing).

## Consequences

One more Compose service and the workflow/activity model, including its determinism constraints. Production carries a real cost decision that has to be surfaced honestly rather than hidden. And **multi-GPU is designed and measured but unexercised** — no 8-card job had run when this was written, and it must not be presented as proven.

## Rollback

The orchestration sequence is already a function over a job record. Moving it between engines changes what invokes it, not what it does.
