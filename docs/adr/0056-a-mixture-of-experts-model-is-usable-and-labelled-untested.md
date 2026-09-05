# ADR-0056 — A mixture-of-experts model is usable and labelled untested

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/009-models-methods-and-advanced-mode.md`
- **Issue:** [#65](https://github.com/thp728/temper/issues/65)
- **Supersedes:** [ADR-0050](0050-a-model-outside-the-catalog-is-usable-once-a-probe-reports-on-it.md)

## Context

ADR-0050 landed the compatibility probe that lets any model repository at a
pinned revision be used once the probe reports on it. That record treated a
mixture-of-experts architecture as **blocked**: the reasoning named three
mechanics that are specific to this architecture — expert routing changes
LoRA target-module selection, memory scales with total rather than active
parameters, and routing interacts poorly with small-batch adapters. That
reasoning was correct about the mechanics.

Blocking on it was the wrong consequence. The open models above a certain size
— the very models that make "any model, including large ones" true — are
largely of this kind (report-a.md's survey: many DeepSeek, Qwen, Llama-4,
GLM and the gpt-oss-120B variant are MoE). A platform that refuses them at
admission cannot claim to support large models in practice, and the earlier
position made that claim untrue. Spec 009's solution is explicit: a
mixture-of-experts model **warns and is labelled untested** rather than
refused, with the probe stating what differs so the user decides knowing.

This record supersedes ADR-0050 on that point only, and it does not edit
ADR-0050's text. An accepted decision is not rewritten; a new record
supersedes the earlier position rather than editing it, stating that the
mechanics reasoning was sound and the consequence was not. That sentence is
the one reviewers will press on hardest, and it is kept verbatim.

Three things this issue adds beyond the verdict change, each with its own
acceptance criterion:

- **The probe states what differs.** Not a bare "untested" flag, but the
  three facts the research names: target selection changes, memory scaling on
  total parameters, and the small-batch interaction.
- **Memory prediction uses total rather than active parameters for these
  models.** Predicting on active would under-predict badly and hand the user
  an out-of-memory failure minutes into a paid machine, which is exactly what
  ADR-0043 says memory must **block** rather than warn about at creation.
  The arithmetic must be the spine, not a second copy.
- **The label travels with the job and appears on the finished run.** A
  finished run's record says what it was trained on, even after the admission
  record changes. The value two components must agree on is defined once; the
  job's frozen `is_moe` is that definition for the finished page, not a
  re-read of the admission row.

## Decision

**A mixture-of-experts model is usable and labelled untested, not refused.**

- **The probe warns on `is_moe`, with the three differences named.** A
  `ModelFacts` where `is_moe` is true produces a single `warn` finding with
  code `untested_architecture`, whose message names: expert routing changes
  LoRA target-module selection; memory scales with total rather than active
  parameters; routing interacts poorly with small-batch adapters. It is
  `usable_with_warnings` and launchable; the curation is a default, not a
  boundary. An unknown architecture warns the same way, with the same code
  and a different message — the two share "untested here" because both are
  outside what this platform has trained.

- **Memory is predicted on total parameters for these models.** `ModelFacts`
  carries `num_experts` and an optional `moe_intermediate_size`; `params`
  derives total parameters as attention plus `num_experts × MLP` per layer
  (when `is_moe` is set but no count was supplied, the count defaults to 8,
  Mixtral-like, so the total is visibly larger than the dense equivalent and
  cannot silently under-price). `active_params` is the routed subset and is
  never used for the prediction; the predictor's weight pools are sized from
  `facts.params` (total). Because the probe, the quote and the creation-time
  refusal all call `memory.predict_peak` — one function, refined once — the
  probe and the refusal cannot drift, and a MoE model that would OOM is
  **blocked at creation** (ADR-0043) with the same arithmetic the probe
  warned with, not discovered on a billing machine.

- **The label is frozen into the job row at creation and shown on the
  finished record.** `jobs.create` derives `is_moe` from the admitted probe's
  snapshot (`probe.is_moe`) at the one creation path, stores it as `is_moe`
  on the `jobs` row alongside `hyperparameters` and `quote`, and the API
  publishes `JobRecord.is_moe`. The finished-run page (`JobRecordView`)
  renders the MoE untested banner from that frozen field, so a finished job
  says what it was trained on even if the admission is later re-probed. Two
  components that also write to the finished record (#70's provenance manifest
  and #56's outcome) are explicitly out of this record's boundary; this one
  owns the model label only.

## Alternatives considered

**Keep the block.** Rejected: the mechanics reasoning was sound and the
consequence was not. Blocking made "any model, including large ones" untrue
in practice because the open models above a certain size are largely MoE,
and spec 009 explicitly requires that the boundary come down with the facts
shown rather than hidden. The probe already names what differs; hiding the
difference behind a refusal is less honest than surfacing it.

**Block on memory at admission for MoE (a MoE that fits no card at defaults
is refused up front).** Rejected for the same reason as dense models in
ADR-0050: sequence length, batch size and rank are not known at admission,
and a lighter configuration may fit. The probe warns with the peak; the
creation path blocks on the same peak against the real configuration where it
is known.

**Predict memory on active parameters for MoE (the routed subset).**
Rejected: the model's weights are total, not active — all experts are stored
even when only two are routed per token — so the weight pool is sized on
total. Predicting on active would under-predict by the MoE factor (e.g. 4×
for 8 experts, 2 active) and hand the user an OOM minutes into a paid
machine. ADR-0043's "memory blocks" exists precisely for that class of
failure; a warning here would be the wrong severity.

**Re-derive `is_moe` at display time from the admission row.** Rejected: a
value two components must agree on is defined once and read, never retyped.
Re-reading the admission row would make a finished run's label change if the
admission was re-probed or deleted, and a finished run must not rewrite
history. Freezing `is_moe` at creation is the same freeze `hyperparameters`
and `quote` already use.

**Edit ADR-0050 in place to change the verdict paragraph.** Rejected: an
accepted decision is not rewritten on this project; #82 handled the same
situation by appending a dated findings entry rather than editing an accepted
record. This record supersedes rather than edits, and states by number that
it supersedes 0050.

## Consequences

- `temper_core.models.ModelFacts.params` is total parameters for MoE: the
  MLP is replicated per expert, with `num_experts` and an optional
  `moe_intermediate_size` surfacing the total; `active_params` is the routed
  subset and is never priced. The resolver (`apps/control-plane/models.py`)
  populates those fields from `config.json` (`num_experts` /
  `num_local_experts` / `n_routed_experts`, and the various
  `*_per_tok` / `moe_intermediate_size` keys); legacy fixtures with
  `is_moe=True` and no count default to 8 experts so the total is visibly
  larger than dense.
- `temper_core.probe` warns on `is_moe` with code `untested_architecture`
  and a message that names the three differences; the finding is `warn`,
  the verdict is `usable_with_warnings`, and the model is launchable. The
  memory line reuses `memory.predict_peak` — one function — so a MoE peak is
  the total-peak and the admission warning and the creation block are the
  same numbers.
- `temper_core.memory.predict_peak` sizes weights on `facts.params` (total)
  for MoE, so the prediction is the larger figure and a job that would OOM
  is refused at creation (`configuration_does_not_fit`), not warned.
- The `jobs` row carries frozen `is_moe`; `JobRecord.is_moe` publishes it;
  the finished-run page shows the MoE untested banner from that field, so
  the label travels with the job. No other finished-record writer is touched;
  #70's manifest and #56's outcome remain theirs, and rebasing is expected.
- The orchestrator resolves admitted models through `admission.get`, not
  `catalog.get` alone, so a MoE job prices the admitted facts, not the
  catalog default, on the provisioning path that spends money.

## Rollback

Re-introduce the block: make `probe` emit `block` for `is_moe`, drop
`is_moe` from `jobs` and `JobRecord`, and revert the orchestrator to
`catalog.get` only. A model outside the catalog that is MoE would again be
refused at admission and the label would not travel; the platform's claim
about large models would again be untrue. This record would then need to be
superseded rather than edited.
