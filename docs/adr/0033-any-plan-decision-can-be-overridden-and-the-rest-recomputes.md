# ADR-0033 — Any plan decision can be overridden, and the rest recomputes

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/005-predictor-and-quote.md`
- **Issue:** [#79](https://github.com/thp728/temper/issues/79)

## Context

ADR-0032 made every predictor decision a structured record -- the decision,
the value chosen, the constraint that forced it, and the alternatives with
what each would have cost -- shown on the plan. The product thesis is that a
first-time user reads that plan as an explanation and an experienced user
edits it, and the spec is explicit that there is one surface, not a beginner
mode and an expert mode. Until now the plan explained but could not be
changed: a user who disagreed with a line had no path from disagreement to a
different configuration except launching and hoping.

Editing is not free of new failure modes. A new choice beside a stale value is
a silent quality bug -- the same reason a rank change recomputes its scale --
and the memory half of the predictor blocks, so an override must not be able
to take control and lose that safety property. The acceptance criteria name
the shape: any decision can be changed, changing one re-requests the plan so
recomputation lives in one place, an infeasible override is refused with the
same arithmetic, overrides are recorded in the job spec, an overridden
decision is visibly marked, and the controls sit beside the explanations.

## Decision

**Every decision the predictor makes is overridable, from the surface that
explains it.** Six decisions carry a control (method, hardware, device count,
disk, precision, sequence length), each control offering the decision's own
vocabulary -- the same strings the records show as `chosen` and the server
publishes as `override_options`, so a value the server accepts is a value the
plan offers and vice versa. Free-form decisions (device count, disk, sequence
length) take a number input; select-style ones (method, hardware, precision)
take a list.

**Changing one decision re-requests the plan rather than mutating it
locally.** `POST /v1/quotes` takes the overrides and runs the same
`quote.build_quote` the unpinned plan runs, with the pinned decisions passed
into `selection.select_hardware` and `disk.required_disk` as hard constraints.
The recomputation rules are exactly the ones the predictor already uses; the
override only restricts which configuration is searched. An override is never
a local edit of the JSON on the page, because that would put the recompute
rule in the interface and let it drift from the server.

**An override that cannot be honoured is refused with the same arithmetic.**
The memory half blocks (spec 005), and an override must not lose that by
taking control: `select_hardware` with a pinned configuration that does not
fit raises `NoFittingHardwareError` carrying the peak against the capacity
that refused it, surfaced as a 400 with `configuration_does_not_fit` and the
arithmetic. A disk override below the job's need or the platform minimum is
refused the same way (`disk_below_need`, `disk_below_minimum`). An unknown
decision or value, and an inconsistent precision/method pair, are refused with
their own stable codes -- the same discipline as `unknown_hyperparameter`
(#83): a key the caller believes is in effect but is not is worse than a
refusal.

**Precision and method are one coupled decision, named twice.** QLoRA
quantises to NF4; LoRA and full fine-tuning hold bf16
(`overrides.METHOD_PRECISION`, the one definition). Overriding precision to
bf16 *means* lora; naming both and having them disagree is refused
(`inconsistent_overrides`), never silently resolved.

**Executability is separate from feasibility.** The plan may describe a
configuration the trainer cannot run (lora, full, multi-GPU) -- spec 005's
"a predictor that cannot describe them cannot refuse them honestly" -- and the
launch is where that description stops being a quote. `POST /v1/jobs` refuses
a non-executable override with `not_executable` naming the gap (spec 009's
territory), because silently running QLoRA while the plan said lora would be
the lie the platform exists to prevent.

**Overrides are recorded in the job specification.** The `{decision, value}`
pairs are frozen into the job row beside the hyperparameters, the sequence
length is folded into the frozen hyperparameters the trainer receives, and the
orchestrator re-provisions against the frozen overrides rather than
re-picking -- a run says what it actually used, and an override that is no
longer available at provisioning fails by name rather than being silently
dropped.

**An overridden decision is visibly marked** on the plan (a "you changed
this" badge) and the mark rides on the frozen quote, so a completed job
explains what it actually used.

## Alternatives considered

**Apply overrides client-side and mutate the quote locally.**
Rejected: the recomputation rule would live in the interface, a second copy of
the selection loop that this repo exists to prevent from drifting. The spec's
acceptance criterion is explicit that changing one re-requests the plan.

**Let an override fall back to the predictor's pick when it cannot be met.**
Rejected: a user who pins H100 and gets an L4 has been lied to about what ran.
A hard constraint that refuses is honest; a preference that degrades is a
surprise on the bill.

**Offer only executable values in the controls.**
Rejected: it would make the plan unable to describe lora, full or multi-GPU
configurations, which spec 005 explicitly requires ("the predictor computes
configurations for both"). The honest shape is describe-on-the-plan,
refuse-at-launch.

**Resolve an inconsistent precision/method pair silently.**
Rejected: a configuration that contradicts itself describes a job nobody
intended, and picking one side hides the contradiction instead of surfacing
it.

## Consequences

- `temper_core.overrides` is new: the six-decision vocabulary, the
  precision-method coupling, `resolve` (typed pins + effective spec), and the
  refusal errors. `selection.select_hardware` accepts pinned
  method/gpu_type/device_count and carries refusal arithmetic on
  `NoFittingHardwareError`; `disk.required_disk` accepts a `provisioned_gb`
  override and refuses below need/minimum.
- The API gains `POST /v1/quotes` (recompute) and `overrides` on
  `POST /v1/jobs`; the quote response publishes `override_options` and each
  decision's `overridden` flag; `JobRecord` carries the frozen overrides.
- The orchestrator re-provisions against the frozen overrides, so a hardware
  override survives to the machine.
- The plan screen's decision cards carry the controls; the finished-job page
  renders the same component without them (the quote is frozen).
- Non-executable overrides are describable on the plan and refused at launch;
  this is the honest boundary until spec 009 teaches the trainer more.

## Rollback

Drop the `overrides` fields and the `POST /v1/quotes` endpoint, remove the
pinning parameters from `selection`/`disk`, and the plan returns to a
read-only explanation.
