# ADR-0054 — Divergence is detected on the streamed loss and offers a single retry as a choice

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/010-recovery-and-fault-injection.md`
- **Issue:** [#36](https://github.com/thp728/temper/issues/36)

## Context

A run whose loss becomes meaningless burns its full duration and hands back a
worthless result. The fault surface (#24) already gives us a way to make that
happen on demand: the `divergence` fault multiplies the resolved learning rate
by 10⁶ so the loss genuinely becomes NaN, and the fake provider narrates the
meaningless loss as `{'loss': nan}` and lets the run complete with that
worthless result. The recovery's job is the abort; the fault's job is only to
be the meaningless loss, on both tiers.

Two prior issues shape the room to work in:

- **#53 streams held-out loss** and **#49 classifies trainer output into typed
  events with a progress kind.** The criterion says detection "uses the
  measurements the platform already streams" — so the detector reads the
  existing `metric` stream rather than adding a second measurement path. Adding
  one would be a second source of truth about loss that could drift from the
  chart the user already sees.

- **#24's spec constraint:** trainer-side faults are an environment switch the
  trainer reads; provider-side faults are behaviour of the provider seam. The
  divergence fault is trainer-side, so provider-side faults are out of scope.
  #34 is the closest sibling in spirit — it also consumes #24's faults, but on
  the provider and reconciler side. This work stays on the trainer's loss.

Three facts from the research set the thresholds, and the issue says to use
them rather than invent ones. `docs/research-reports/report-b.md` §5.7 says
to abort when **"loss becomes NaN/Inf (immediate abort — unrecoverable without
rollback); loss increases >2× its trailing 50-step average for >20 consecutive
steps (divergence); grad_norm exceeds ~100 repeatedly"** despite clipping, with
§6 restating the warn threshold as **"loss > 2× trailing-average for 20 steps,
or any NaN"**. Those are the numbers a reviewer can look up, not numbers this
code chose. `grad_norm` is not streamed today — `events.py` deliberately does
not promote it, because it is diagnostic rather than progress — so the
grad_norm half of the same paragraph cannot be read without a second
measurement path, and the criterion forbids one.

The remaining criteria are easy to get subtly wrong, and the issue names
them:

- **"A single retry at a reduced learning rate is offered as a choice rather
  than performed automatically"** — because a diverging run usually means the
  data or the rate is wrong, and repeating it is rarely the answer. Making it
  automatic would be easier to test and wrong in product terms.

- **"Thresholds are configuration with their derivation recorded beside them"**
  — the way ADR-0036's dataset ceiling records 21.6 MB/s × 60 s. A threshold
  with no derivation is a magic number with a comment.

- **"A real run is made to diverge on hardware, and aborts with the right
  reason"** needs GPUs that are not available in this wave. #66 did the honest
  thing last wave and wrote that a fake run "proves the path's shape, not the
  live-hardware observation". This issue does the same: the hardware observation
  is stated as outstanding in the PR body rather than quietly reinterpreted as
  met.

## Decision

**Divergence is detected on the training-loss stream the platform already
emits, aborts with a plain cause and a stable code, surfaces instability short
of divergence as a warning, and offers a single half-learning-rate retry as a
choice. Thresholds are configuration with their derivation beside them.**

- **Detection reads the existing `metric` stream and the non-finite log line.**
  `orchestrator._consume` holds a `DivergenceDetector` per job, fed each
  `loss` from `metric` events and also the textual `{'loss': nan}` that
  `events.classify` keeps as `log` (non-finite numbers are deliberately not
  promoted to metrics because a NaN is real information but not a point on a
  chart, and it survives in the log line either way). No second measurement
  path is added. The detector is `temper_core.divergence` — pure, no I/O, no
  framework imports — so the thresholds are tested without hardware as a table
  of losses and an expected verdict.

- **Thresholds are the published figures.** In both `temper_core.divergence`
  and `apps/control-plane/src/temper_control_plane/config.py`:

  * `DIVERGENCE_MULTIPLIER = 2.0` — report-b: ">2×"
  * `DIVERGENCE_WINDOW = 50` — report-b: "trailing 50-step average"
  * `DIVERGENCE_CONSECUTIVE = 20` — report-b: "for >20 consecutive steps"
  * `WARNING_CONSECUTIVE = 5` — instability is the *same* exceedance seen for
    fewer steps than a divergence. 5 is one quarter of the 20-step divergence:
    early enough to warn while the user can still act, late enough that a
    single spike is not a warning. The research names the divergence (20) and
    the second instability signal (grad_norm >100); grad_norm is not streamed,
    so the warning reads the same loss exceedance for 5 steps.
  * **NaN/Inf is immediate divergence** — report-b's immediate abort. The
    sophisticated recovery (roll back ~100 steps, PaLM/OPT) is named as v2;
    v1 aborts.

  All four are configuration via `TEMPER_DIVERGENCE_*` and `TEMPER_WARNING_*`,
  defaulting to the figures above. Changing a number changes the derivation
  comment beside it, not a silent literal.

- **The abort is a plain cause with a stable code, and it stops the job
  before the teardown.** A diverged loss raises `OrchestratorError` with code
  `training_diverged` and a sentence that tells the user the loss became
  meaningless, the run was stopped early so they are not billed for hours that
  cannot produce anything, to try a lower rate or check the data, and that a
  single half-rate retry is available as a choice. The raise happens inside
  `_consume`, so the `finally` that tears the machine down runs, the
  confirmation that the machine is gone precedes the terminal state, and the
  job ends `failed` with that code — never `training_failed` and never silent.

- **Instability is a warning, not an abort.** The same exceedance for
  `WARNING_CONSECUTIVE` (5) consecutive steps emits a `log` event with
  `code: training_instability` and `warning: true` and the plain sentence
  "Training instability detected — loss is spiking ...". The run continues;
  the warning is rendered as a banner on the running view and on the finished
  record, beside the events. Divergence is the abort; instability is the
  banner that precedes it. The detector warns once per unstable run rather
  than on every step, so the history is not flooded.

- **A single retry at half the learning rate is offered as a choice, never
  performed automatically.** `POST /v1/jobs/:id/retry` creates a new job whose
  `learning_rate` is half the failed job's (`RETRY_LR_FACTOR = 0.5`, the
  report's "optionally auto-retrying once at half LR", now offered rather than
  automatic because a diverging run usually means the data or the rate is
  wrong). The new job carries `retry_from` so the single-retry offer can be
  enforced (a job that already has a retry child is refused with
  `already_retried`) and the history can name what came from what. The
  finished-job page shows the retry as a button only when the job is
  `training_diverged`; clicking it calls the endpoint and links to the new job.

- **The fake provider's divergence fault trips the detector.** The fault's
  `{'loss': nan}` line is the non-finite loss the detector treats as immediate
  divergence, even though it never becomes a metric. A fake run therefore
  proves the path's shape end to end; the hardware observation — a real run
  whose sabotaged learning rate genuinely diverges and is aborted with
  `training_diverged` — is stated as outstanding in the PR body rather than
  reinterpreted as met, exactly as #66 did.

## Alternatives considered

**Add a second measurement path (e.g. a trainer callback that reports
divergence explicitly).** Rejected: the criterion says detection "uses the
measurements the platform already streams", and a second path would be a second
source of truth about loss that could drift from the chart. The detector reads
what the chart already reads.

**Detect divergence on `held_out_loss` as well.** Rejected: that series
belongs to the overfitting detector (#53, #77) and is evaluated per epoch, not
per step; mixing it into the per-step divergence rule would conflate two
signals with different cadences. Training loss is the high-frequency signal
this recovery watches.

**Auto-retry once at half LR inside the orchestrator.** Rejected: the issue's
reasoning is attached to the criterion — repeating a diverging run is rarely
the answer — and making it automatic is easier to test but wrong in product
terms. The retry is a button, not a transition.

**Use grad_norm >100 as the instability warning (report-b's second signal).**
Rejected: `events.py` deliberately does not promote `grad_norm`; adding it
would be the second measurement path the criterion forbids. The same loss
exceedance for fewer steps is the honest warning within the existing stream;
if grad_norm is streamed later, this decision is the place to revisit it and
the ADR records that.

**Emit a new `warning` event kind.** Rejected as the minimal change: the
events table's `kind` is `state|metric|log|error` in the contract, and a new
kind would be a contract change for a banner the interface can already render
from a `log` event carrying `code: training_instability` and `warning: true`.
If a first-class warning kind is wanted later, it can supersede this without
changing the detector.

**Make thresholds literals in the detector.** Rejected: "thresholds are
configuration with their derivation recorded beside them" means the number
carries where it came from, the way ADR-0036's ceiling does. Literals in the
detector are the violation that criterion exists to catch; the numbers live in
`config.py` with the report-b citations beside them and are injected into the
pure detector, so derivation is co-located with configuration.

## Consequences

- `temper_core.divergence` ships a pure `DivergenceDetector` and
  `is_non_finite_loss_line`, tested without hardware as a table of losses and
  expected verdicts (including NaN, sustained exceedance, warning, and reset).
- The orchestrator aborts on `training_diverged` with the stable code and the
  plain sentence, tearing the machine down before marking the job failed; the
  fake provider's `divergence` fault trips it, so a deliberately broken fake
  run is observed to abort with the right reason.
- Instability is surfaced as a `training_instability` warning event and a
  banner on both the running and finished views; it does not abort the job.
- `POST /v1/jobs/:id/retry` offers the single half-rate retry as a choice;
  the job record's `retry_from` links the retry to its parent, and a second
  retry is refused with a stable code.
- The config's `TEMPER_DIVERGENCE_*` knobs carry the report-b derivations
  beside them, and the detector's defaults are the same published figures —
  one definition, two readers, never retyped.
- The hardware-only acceptance criterion — a real run made to diverge on a
  GPU — is stated as outstanding in the PR body, not claimed via the fake.

## Rollback

Revert `temper_core.divergence`, the detector wiring in `orchestrator._consume`,
the `POST /v1/jobs/:id/retry` endpoint and `jobs.retry_diverged_job`, the
`retry_from` column and the config's divergence knobs, and the web's retry
button plus instability banner; jobs return to completing with a worthless
result when loss diverges, exactly the shape the fault surface was left in
by #24 so that the recovery could be demonstrated rather than reasoned about.
