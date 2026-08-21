# ADR-0007 — The feasibility warning is an estimate from one measured run, and warns rather than blocks

- **Status:** accepted
- **Date:** 2026-08-21
- **Spec:** `docs/specs/002-dataset-transport-and-limits.md`
- **Issue:** [#10](https://github.com/thp728/temper/issues/10)

## Context

The size limit (ADR-0005) and the duration ceiling (ADR-0002) answer different
questions and do not reconcile. One gigabyte — accepted at upload, roughly
577,000 conversational rows — is far beyond what finishes in 24 hours. A user
could walk straight into that gap: the dataset uploads cleanly, the job
launches, and the failure arrives hours later as a `gpu_max_duration_exceeded`
kill **on a machine they paid for**. Nothing in the product said otherwise
before launch.

## Decision

**Job creation attaches a warning when a dataset plainly cannot finish inside
the maximum duration. The job launches anyway.**

The estimate is deliberately crude: usable rows × epochs ÷ throughput.

- **Throughput is measured, not assumed**: 192 row-passes (64 rows × 3 epochs)
  in the measured 161.4 s training phase of the one real run through the
  product (2026-08-19), hence ~1.19 row-passes/s. The derivation is recorded
  beside the constant in `api/feasibility.py`.
- That figure includes model load and other fixed overhead, so it *understates*
  steady-state throughput and therefore *overstates* duration for large
  datasets. The warning errs toward firing — the right bias for a mechanism
  whose failure modes are "spurious warning" (mild) and "paid-for kill" (the
  thing this exists to prevent).
- A `num_epochs` override is honoured; a malformed one falls back to the
  default rather than crashing job creation (the trainer refuses it later).
- With `max_steps` set there is **no estimate at all**: run length is then
  bounded by steps rather than data volume, and no per-step throughput was
  ever measured. Estimating from volume there would be wrong by orders of
  magnitude.
- The warning fires only when the estimate **strictly exceeds** the ceiling.
  At equality the value is inside the estimate's own error bar; "plainly
  cannot" is the standard, not "might not".

The warning is labelled an estimate wherever it appears — the message says
"estimate", names its measured basis, and states what happens at the ceiling.
It is frozen onto the job row (`warnings_json`) like the hyperparameters, so
what the user was told before launching stays part of the run's record, and it
is also recorded as an event so it appears in the job's own history.

### A scope mismatch, stated rather than hidden

The estimate covers **training only**, while the ceiling counts from job start
— provisioning, SSH wait and pushes all consume it (ADR-0002). So the
denominator here is narrower than the limit being compared against. Two
countervailing biases keep this honest in practice: the throughput figure
*overstates* duration (it includes fixed overhead that amortises away on large
datasets), while excluding provisioning *understates* it — and provisioning
was measured at ~5 minutes against a 24-hour ceiling, three orders of
magnitude smaller. Both effects are noise at the scale where this warning
fires. Recorded because an unstated mismatch between two limits is exactly
the failure shape this project records.

### Why a warning and not a refusal

The estimate comes from one run of 64 rows on one GPU type. Extrapolating it
to 500,000 rows carries error bars nobody has quantified. Blocking on it would
either strand users with datasets that would actually finish (throughput
improves per-row as fixed overhead amortises) or imply a precision the system
does not have. The user can still act on a warning; only they should decide
whether the run is worth attempting. A precise predictor belongs with Phase B's
quoting work, which needs a memory and throughput model anyway.

## Alternatives considered

**Refuse jobs whose estimate exceeds the ceiling.** Rejected above: a wrong
block is worse than a wrong warning, and this estimate is wrong by design —
crude on purpose.

**Estimate from tokens or sequence length instead of rows.** More accurate in
principle — training cost tracks tokens, not rows. Rejected: nothing measured
exists to calibrate it. Row-passes are what the one real run actually counted.

**Skip the warning when `max_steps` is set, vs estimating anyway.** Chose
skip: with steps capped, data volume does not determine duration, and any
volume-based number would be fiction. Recorded because it is a silent
non-behaviour a reviewer might read as an oversight.

**A real feasibility predictor now.** Rejected for Phase A — it needs the
memory and throughput model that Phase B's quoting work builds anyway. This
warning is the interim acknowledgement of the gap, written down as such.

## Consequences

- A new stable code, `duration_feasibility`, carried on a warning — not an
  error, and never a veto.
- Jobs gain a `warnings_json` column (added via the `ADDED_COLUMNS` migration,
  so existing databases pick it up). An absent list reads as `[]` so clients
  never special-case null.
- The warning text promises only what is known: it says the job *will be
  stopped if* it reaches the ceiling, not that it will fail — the estimate
  overstates duration, so many warned runs will in fact complete.
- Users with genuinely too-large datasets now get the bad news before paying,
  which is the entire point; the residual risk is a user who ignores a correct
  warning, which is their judgement to make.
- The throughput constant is the softest number in this codebase. It is one
  run, one GPU (L4), one model (Qwen3-4B). When the catalog grows, this
  estimate says nothing about other hardware — Phase B's quoting work replaces
  it.

## Rollback

Delete the warning computation in `create_job` and the `feasibility` import;
the column can stay (it is nullable and unread). No configuration toggles
this — a warning that a flag can silently switch off is the unread-constant
failure shape ADR-0002 exists to describe.
