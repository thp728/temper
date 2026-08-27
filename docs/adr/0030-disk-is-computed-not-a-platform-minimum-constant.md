# ADR-0030 — Disk is computed, not a platform-minimum constant

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/005-predictor-and-quote.md`
- **Issue:** [#64](https://github.com/thp728/temper/issues/64)

## Context

`orchestrator.py` provisioned every machine with `STORAGE_GB = 100` — the
platform minimum, asked for regardless of what the job needed. A 70B model's
weights alone are larger than that on their own: a user who selected one got
a machine that reached Running, started downloading, and failed partway
through with no warning that it ever would.

Spike 5 (`spike/findings-spike5.json`) measured the two facts this record
turns into arithmetic: the provider's disk parameter has a floor of 100 GB
and a ceiling of 7200 GB, named by the API itself when it refused a request
for 8000; and the GPU-hour rate does not move when a 40x larger disk is
attached, so storage is billed on a line the quote must add explicitly or
understate a large-disk job by exactly what storage costs.

The `models` and `memory` seams ADR-0028 introduced already answer what a
model's weights and trainable parameters cost; ADR-0029 turned that into a
hardware decision. This record is the same move applied to disk: a stored
constant cannot describe a model the catalog has never seen, so disk is
computed from the same `ModelFacts` everything else reads.

## Decision

**`temper_core.disk.required_disk` sums four pools and floors and caps the
result**, mirroring `temper_core.memory.predict_peak`'s shape: pure
arithmetic over `ModelFacts`, no I/O, exercised identically whether the
facts came from a live resolve or a test's double.

- **Weights at download precision**, always bf16 (`facts.params * 2`
  bytes) regardless of training method. QLoRA quantises to NF4 on-device
  (`apps/trainer/entrypoint.py`'s `load_in_4bit=True`); it never downloads an
  already-quantised checkpoint, so pricing this pool at the *training* dtype
  — as `memory.WEIGHT_BYTES_PER_PARAM` does — would answer the wrong
  question.
- **Retained checkpoints**, at `save_total_limit` copies of the trained
  weights. Axolotl keeps the newest `save_total_limit` directories and
  deletes older ones as it goes, so disk holds that many at once, not the
  run's cumulative total. A checkpoint is the adapter's own size for
  LoRA/QLoRA — measured at fp32, `apps/trainer/README.md`'s 132.2 MB for 33M
  trainable params — and the full model's bf16 size for a full fine-tune,
  which this trainer has never executed (ADR-0029), so that estimate is
  labelled rather than measured.
- **Merged output**, zero for every method this trainer runs. LoRA and
  QLoRA ship the adapter as the deliverable; full fine-tuning's checkpoint
  already is the full model. The term stays in the arithmetic rather than
  being omitted, so a method that does merge does not need a second formula.
- **Image and working space**, a fixed allowance: 8.5 GB is the trainer
  image's own measured build size (spike 4, `apps/trainer/README.md`); the
  remaining 10 GB is unmeasured headroom for the dataset copy, HF's staging
  directory and Axolotl's scratch and logs, carried as a labelled assumption
  rather than presented as a figure any spike produced.

**The result is floored at 100 GB and capped at 7200 GB, both from spike
5.** A requirement above the cap raises `DiskExceedsCeilingError` naming the
shortfall, checked *before* the floor is applied — a job whose weights alone
exceed the ceiling is refused for that reason, not quietly rounded down to
something provisionable.

**`save_total_limit` moved into `packages/contracts/trainer-defaults.json`.**
It was a literal in `apps/trainer/entrypoint.py`'s config build, invisible to
anything outside that file. Disk needs the same number the trainer actually
configures, and CLAUDE.md's rule is explicit: a value two components must
agree on is defined once and read, never retyped. It is a default, not an
override — nothing added it to `allowed_overrides` — so a user cannot inflate
it, but both `entrypoint.py` and `temper_core.disk`'s caller now read the one
definition instead of a second copy drifting from the first.

**Storage cost is carried on the plan, in USD, not converted.** Spike 5 found
GPU-hour pricing unaffected by disk size and a documented — not measured —
rate of $0.10/GB-month for storage itself. No live per-account storage price
exists behind the provider seam to convert that figure against the account's
INR billing, so `DiskPlan.storage_cost_usd_per_hour` is exposed labelled
rather than silently presented as though it were in the account's currency.

## Alternatives considered

**Convert the documented USD storage rate to INR at some fixed exchange
rate.** Rejected: an invented exchange rate would look measured and is not;
carrying the figure in the currency spike 5 actually found it in is more
honest than a conversion nothing verified.

**Leave `save_total_limit` a private literal in `entrypoint.py` and duplicate
its value as a constant in `temper_core.disk`.** Rejected: this is exactly
the two-copies failure CLAUDE.md's rule exists to prevent, and
`test_agreement_with_the_domain.py`'s literal-scan would have had nothing to
catch a second copy drifting from the first, because a hand-typed `3` in a
new module was never one of the names it watches.

**Price merged output at the full model's size unconditionally, even though
nothing produces one today.** Rejected: it would overstate every disk figure
by the model's full weight size for no job the platform can currently run,
and the honest answer for a pool nothing populates is zero, named as such.

## Consequences

- `orchestrator.STORAGE_GB` is deleted. `provider.create`'s `storage_gb`
  argument is now `disk.required_disk(...).provisioned_gb`, computed after
  hardware selection (disk needs the chosen `method`) and before the machine
  is created.
- `jobs` rows gain `disk_gb` and `storage_cost_usd_per_hour`, recorded at
  provisioning alongside `gpu_type`, `price_per_hour`, `currency`,
  `device_count` and `method` — the same observability those five already
  had, extended to the sixth axis of the decision. Published through
  `JobRecord` the same way.
- A job whose disk requirement exceeds 7200 GB fails with
  `disk_exceeds_ceiling` before anything is provisioned, alongside the
  existing `provider_capacity_unavailable` memory refusal. On the live
  orchestrator, which only ever offers QLoRA (ADR-0029), memory constrains
  model size far more tightly than disk does — a model large enough to blow
  the disk ceiling under QLoRA's 0.5-bytes/param training footprint would
  already fail the memory check first, since no available card holds it.
  The disk refusal's live trigger is therefore an unreasonable
  `save_total_limit`, not an unreasonable model, until spec 009 offers wider
  methods to the search; `packages/core/tests/test_disk.py` exercises the
  arithmetic directly regardless of which path can reach it today.
- `apps/trainer/entrypoint.py` no longer hard-codes `save_total_limit`; it
  reads the resolved value from the job spec like every other hyperparameter,
  matching how `lora_r`, `sequence_len` and the rest already arrive.
