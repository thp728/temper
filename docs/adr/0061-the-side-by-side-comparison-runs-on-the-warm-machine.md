# ADR-0061 — The side-by-side comparison runs on the warm machine, compares the chosen checkpoint, and never fails the run

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/011-evaluation-and-delivery.md`
- **Issue:** [#69](https://github.com/thp728/temper/issues/69)

## Context

A job finishes and hands back a file. Nothing answers the question the user
actually has — *did it work, and is it better than what I started with* — and
the comparison a non-specialist can read is the same prompt before and after
tuning, side by side. Spec 011 asks for exactly that: the same held-out
prompts run through the base model and through the tuned model, produced on a
machine that is already warm with both models loaded.

Four prior decisions constrain the room, and none is ours to change:

- **#53 (ADR-0048)** produced the held-out split, and its ordering is
  load-bearing: deduplication happens *before* the split, so a duplicated row
  can never land on both sides. The comparison's prompts must come from that
  split, not from a second sampling of the dataset — otherwise the criterion
  "the user is not shown answers the model memorised" becomes false in exactly
  the case that matters.
- **#62 (ADR-0049)** records the chosen result checkpoint by held-out loss.
  The comparison must compare the model selection actually chose, not the last
  one written, and the selection rule is a pure function in
  `temper_core.checkpoint`.
- **ADR-0010** says a value two components must agree on is defined once and
  never retyped. Decoding settings in particular are read by the generator and
  displayed by the interface.
- **The paid run.** Someone has paid for the machine by the time the comparison
  runs. Spec 011 is explicit: evaluation failure never fails the run — the
  reason is recorded and the artifact is still delivered.

The physical reality that shapes the whole design: the machine is destroyed
immediately after the control plane collects result.json. There is no window
in which the control plane could select the checkpoint, tell the trainer, and
have the trainer still be warm. The comparison either runs *inside* the
trainer, before result.json is final, or it does not run on the warm machine.

## Decision

**The side-by-side comparison runs in the trainer, on the warm machine, right
after training — before result.json is written — and it can never fail the
run.**

- **The prompts are the platform's held-out rows, never a second sample.** The
  trainer reads its own `eval.jsonl` — the same rows the held-out loss was
  measured on — and selects a fixed, deterministic slice of it. The slice size
  and seed are named constants in `comparison.py` (`COMPARISON_PROMPT_COUNT`,
  `COMPARISON_PROMPT_SEED`), so the same split and seed select the same
  prompts, reproducibly from the run's record. The held-out answer is trimmed
  from the prompt before either model generates, so neither model is handed
  its own target.

- **The tuned side is the checkpoint selection actually chose, decided by the
  same pure rule the control plane records.** `temper_core.checkpoint`'s
  `select_best_checkpoint` ships flat into the image beside the entrypoint
  (like `split.py`), and the trainer builds the same step+loss records the
  control plane selects over and applies the same function. The comparison
  records the chosen step and basis on the result, and an agreement test pins
  that the trainer's choice equals the domain's — a drift cannot silently
  compare a different model than the run answers with. The control plane still
  records `best_checkpoint` at terminal time over the *verified, retained*
  set; the trainer chooses over the machine's own checkpoints at comparison
  time. The two agree in the normal path, and the comparison's record names
  the step it actually compared either way.

- **Decoding settings are fixed, defined once, and recorded with the result.**
  `COMPARISON_DECODING` in `comparison.py` is the single definition: the
  generator reads it, and the result records it, so the interface displays the
  settings from the record without retyping them (ADR-0010). Recorded means
  stored on the result and shown, which is what lets a reader tell whether two
  outputs are comparable — that is a test, not a docstring.

- **Generation renders through the same template the run trained with.**
  `tokenizer_default` plus the detected thinking mode, resolved by the same
  `resolve_template` the export-time probe uses, so what the models answer is
  the same question the training rows asked. Both sides use the same decoding
  settings, so the difference shown is the models', not the settings'.

- **Evaluation failure never fails the run.** `run_comparison` in
  `comparison.py` never raises: a generator that throws becomes `ok: false`
  with the reason recorded and any partial rows dropped. The entrypoint wraps
  even model loading in the same rule, because loading is the most likely
  failure on a warm machine. The negative test makes the comparison throw and
  asserts the job still reaches its terminal state with the artifact intact and
  the reason recorded — at the unit level (injected raising generator) and at
  the job level (a completed job whose recorded comparison failed).

- **The comparison is published typed, exactly as recorded.** The control
  plane presents `comparison` from the result document as a typed field on the
  published job record (the client is generated from the contract, per
  ADR-0023). The finished-record page renders both sides per prompt, the
  decoding settings, the chosen checkpoint, and — when the comparison failed —
  the recorded reason, without dressing a comparison failure as a failed job.

## Alternatives considered

**Run the comparison in the control plane after the best checkpoint is
recorded.**
Rejected on the physical reality: the machine is destroyed the moment
result.json is collected, so there is no window, and provisioning a second
machine to compare would cost another cold start for work that takes moments
where the weights already are. Spec 011 is explicit that the comparison is
produced on the machine that is already warm.

**Have the trainer wait for the control plane's recorded `best_checkpoint`
before choosing which checkpoint to compare.**
Rejected: the trainer is the only warm place, and by the time the control
plane has verified and recorded the choice the machine is gone. The trainer
applies the same pure rule over the same shape of records, and the agreement
is pinned by tests; the comparison's record names the step it used, so the
record is honest even in the divergence case (a checkpoint that failed to
upload, or was evicted by retention, is on the machine but not in the control
plane's candidate set).

**Sample the dataset a second time for prompts.**
Rejected: this is the whole point of #53's ordering. A second sampling lets a
duplicated row appear on both sides and shows the user answers the model
memorised, in exactly the case the held-out split exists to prevent.

**Let the interface define or display decoding settings of its own.**
Rejected: a second definition of a value two components must agree on is
exactly what ADR-0010 forbids. The generator reads the constant; the interface
reads the record.

**Use fixture generation output in the trainer and claim the comparison works.**
Rejected: a fixture of plausible-looking output satisfies the renderer and
proves nothing about the generator. The real generation path is exercised by a
hardware-marked trainer test that runs a real model and asserts on the real
text it produces; the browser journeys' simulated machine carries a canned
comparison only so the *renderer* is exercised end to end, and the PR body
says plainly that this does not substitute for real generations from a real
run.

## Consequences

- The trainer records `comparison` in result.json after training, narrates the
  phase on the machine's output (so the stall detector sees activity), and the
  comparison ships flat as `comparison.py` with `checkpoint.py` added to the
  image's `COPY` and `TRAINER_SOURCES`.
- The published job record gains a typed `comparison` field; the finished-job
  page renders it side by side with the decoding settings and the chosen
  checkpoint. A failed comparison shows its reason; pre-existing rows (no
  recorded comparison) render nothing.
- `decoding` is always recorded, even on a failed comparison, so a reader
  knows what settings a comparison (or an attempted one) was made under.
- The one acceptance criterion that cannot be satisfied without hardware — the
  comparison rendered with *real* generations from a *real* run — is stated as
  outstanding in the PR body rather than quietly reinterpreted as met by the
  fake's canned comparison; what was used instead (the hardware-marked
  generator test, the journey against the canned record) is named, and why it
  does not substitute.

## Rollback

Revert the trainer's comparison step (the `comparison.py` import and the
`result["comparison"]` call), drop the `comparison` field from the published
record and its page section, and the finished record returns to not comparing
the two models. Nothing else depends on the comparison's presence — a row
without one reads exactly as it did before.
