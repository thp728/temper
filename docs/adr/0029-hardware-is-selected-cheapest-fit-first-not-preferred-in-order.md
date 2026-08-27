# ADR-0029 — Hardware is selected cheapest-fit-first, not preferred in order

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/005-predictor-and-quote.md`
- **Issue:** [#55](https://github.com/thp728/temper/issues/55)

## Context

`orchestrator.py` chose a GPU from `GPU_PREFERENCE = ["L4", "RTX-PRO6000",
"H100"]` — three names, taken in order, first one the provider had free
capacity for. That is not a selection rule: it does not read price, it does
not consider what the job needs, it never asks for more than one device, and
it does not exist for a model the three names were not chosen against.
ADR-0028 named this directly as the piece it left for a later ticket: "GPU
*selection*... [is] unchanged by this record — spec 005 assigns [it] to later
tickets."

The `models` and `memory` seams that record introduced now answer "does this
job fit a given card" for any model, method and shape. This record is what
turns that arithmetic into a decision: which card, how many, and which
training method — read from what the provider actually has free and what it
costs, not from a list three names long.

## Decision

**`temper_core.selection.select_hardware` searches the whole (method, GPU
type, device count) space at once and returns the cheapest triple the memory
model predicts will fit.** Method is not chosen first and reconciled after:
price never depends on it, so fixing it ahead of the search would be exactly
the kind of preference list this record deletes, and at a tied price the more
capable method that still fits is a strictly better answer, not a different
one. Device count is the same kind of decision, not a separate one — a job
that fits no single card is only satisfiable by asking `memory.predict_peak`
about more of one, and the cheapest fit is still whichever triple wins the
same comparison.

**Ties break toward fewer, larger devices.** Multi-GPU training pays
interconnect overhead a single card does not, and asking for hardware a job
does not need is worse than being slower — particularly in front of a reader
who sells the hardware. Implemented as a second comparison key: equal price
prefers the lower device count.

**`temper_core.memory.predict_peak` gained a `device_count` parameter,
modelling FSDP FULL_SHARD for `method="full"` only.** Spike 6 proved that
mechanism shards weights, gradients and optimizer state evenly across ranks,
at small scale, for a full fine-tune. LoRA and QLoRA have never run sharded
and their trainable set (tens of millions of parameters against billions of
frozen ones) is too small for sharding to plausibly change whether a job
fits, so for them `device_count` changes nothing — more devices replicate the
same per-device footprint rather than dividing it. This is also what makes
the spec's named property hold without special-casing it: adding devices
divides the *sharded* pools but never the fixed overhead or activations each
device still pays, so the cluster-wide total (`peak.total_gb × device_count`)
never decreases as devices are added, even though the per-device number
checked against card capacity can.

**Only QLoRA is offered to the live orchestrator.** `apps/trainer/entrypoint.py`
hard-codes `adapter="qlora", load_in_4bit=True` — the trainer cannot execute
LoRA, full fine-tuning, or more than one device yet; that is spec 009's job,
gated on spike 6's still-open questions about sharded resume and numerics.
`select_hardware`'s `methods` parameter defaults to
`selection.EXECUTABLE_METHODS = ("qlora",)`, so the search still ranges over
whatever a caller passes — a future quote surface computing configurations to
*show*, not run, can pass the full set — but the path that actually
provisions billing hardware only ever offers what can be launched today. A
predictor that silently picked a method the trainer ignores would launch a
QLoRA job under a label that said otherwise, which is a worse defect than the
hard-coded list it replaces.

**The provider seam now exposes live data, not a decision.** `Provider.select_gpu`
(one call, one answer) is replaced by `gpu_availability()` and `currency()`
(two live facts), and `Provider.create` gains `num_gpus`. Selection itself
moved to `packages/core`, matching the split ADR-0028 established for
`models`/`memory`: the provider seam is I/O, and the decision is arithmetic
that must run without a network or a GPU to be tested at all. Availability
rows are kept one-per-node, never merged by `gpu_type`: spike 6 found that
capacity is per node, and two free devices on two different nodes cannot be
attached to one machine — collapsing them would let the search believe
capacity exists that no single node actually offers.

## Alternatives considered

**Rank candidates by a weighted score (price, quality, speed) instead of
lexicographic price-then-device-count.** Rejected: the issue's acceptance
criterion is explicit and testable as written — a configuration that fits is
never passed over for a pricier one that also fits — and a weighted score
would trade that provable property for a tuning surface nothing has
calibrated.

**Model sharding for LoRA/QLoRA the same way as full fine-tuning.** Rejected:
nothing has run adapter training sharded, and the trainable set is small
enough that the benefit would be theoretical. Claiming a memory benefit
spike 6 never measured would be exactly the kind of unverified extrapolation
ADR-0028 replaced `est_peak_vram_gb` for.

**Let the live orchestrator choose from all three methods now, ahead of the
trainer supporting them.** Rejected: a decision the platform cannot execute
is a worse answer than a narrower one it can, and honesty about that gap is
cheaper than a job that silently trains QLoRA under a "lora" label.

## Consequences

- `orchestrator.GPU_PREFERENCE` is deleted. `Provider.select_gpu` and
  `provider.GpuChoice` are deleted; `Provider.gpu_availability`,
  `Provider.currency`, and `temper_core.selection.{GpuAvailability,
  HardwarePlan, select_hardware}` replace them.
- `jobs` rows gain `device_count` and `method`, recorded at provisioning
  alongside `gpu_type`, `price_per_hour` and `currency` — the same
  observability those three already had, extended to the two new axes of the
  decision.
- A job whose predicted peak fits nothing the provider currently has free
  fails with `provider_capacity_unavailable` before anything is provisioned,
  the same code the old list's exhaustion produced — the failure mode is
  unchanged, only the reason it can now legitimately fire (a real memory
  arithmetic refusal, not just "none of three names happened to be free").
- Choosing LoRA, full fine-tuning, or a device count above one for a *live*
  job is spec 009's job, gated on the trainer learning to execute them
  (itself gated on spike 6's open questions). This record teaches the
  decision to describe those configurations honestly; running them is not
  in scope here.
