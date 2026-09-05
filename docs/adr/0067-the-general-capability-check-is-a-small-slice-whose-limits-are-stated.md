# ADR-0067 — The general-capability check is a small slice, labelled a smoke test, whose limits are stated

- **Status:** accepted
- **Date:** 2026-08-30
- **Spec:** `docs/specs/011-evaluation-and-delivery.md`
- **Issue:** [#73](https://github.com/thp728/temper/issues/73)

## Context

Fine-tuning degrades general capability. The effect is real, and no other
signal in the product catches it: the held-out loss measures how well the
model learned the *task*, and nothing says whether it can still answer
general questions it was not fine-tuned on. Spec 011 asks for a small
general-capability check that runs before and after, is reported as a delta,
states its sample size and uncertainty, and is labelled a smoke test rather
than a benchmark — a large regression must be surfaced prominently.

The comparison from issue #69 already established the shape the platform
follows for warm-machine evaluation: a pure module (`comparison.py`) whose
decisions are fixed constants, run by the entrypoint on the machine that is
already warm, with the result recorded and published typed. This check is the
same shape, for a different question.

## Decision

**The general-capability check is a fixed, versioned slice of general
multiple-choice questions, run through the base model and through the chosen
checkpoint on the already-warm machine, and reported as a delta with its
sample size, per-question weight and paired standard error stated. It is
labelled a smoke test in the interface, and a tuned model that loses at least
`LARGE_REGRESSION_QUESTIONS` correct answers to the base is flagged on the
record so the interface surfaces it prominently.**

- **The slice is small, fixed, general and versioned.** `CAPABILITY_QUESTIONS`
  in `apps/trainer/capability.py` is a module constant — the same eight
  general-knowledge questions for every job, deliberately unrelated to
  whatever the user fine-tuned on (a model that learned the task and forgot
  arithmetic or geography is exactly what the check catches). `CAPABILITY_SLICE_VERSION`
  travels with every result, so a number from slice v1 is never compared to a
  number from slice v2 as if they were the same measurement. Eight questions
  because each costs warm-machine time and the check exists to catch a large
  regression, not to rank models.

- **Scoring is deterministic and stated.** Each question is multiple-choice;
  the model's reply is parsed for the option letter (`parse_answer`), and a
  reply that names no letter is *wrong* rather than dropped — a model that
  cannot produce an answer letter is itself evidence of degradation. Each row
  of the record carries the question, the domain, the correct letter, both
  models' replies, what was parsed, and whether each side was right, so a
  reader can spot-check the score instead of trusting it.

- **The delta is reported with its uncertainty.** The delta is the tuned
  score minus the base score over the *same* questions — a paired
  before/after — and its standard error is computed from the per-question
  differences (`delta_standard_error`), the honest statistic for paired data.
  The sample size (`total`) and the per-question weight (1/n) travel with the
  result, and the interface states all three: the number beside its sample
  size, the per-question weight, and the standard error of the change.

- **A large regression is a flag on the record, not a judgement call in the
  client.** The threshold — the tuned model answered at least
  `LARGE_REGRESSION_QUESTIONS` (2) fewer questions correctly than the base —
  is defined once in `capability.py`, recorded on the result as
  `regression_threshold` and `large_regression`, and surfaced prominently by
  the interface from the record (ADR-0010: a value one side decides on is
  defined once). The interface never re-decides what "large" means.

- **It runs on the already-warm machine, and the two eval steps share one
  model load.** The entrypoint selects the checkpoint and loads the base and
  tuned generators once, then hands the same pair to the comparison and the
  capability slice. This is the physical reality the comparison already
  accepted (ADR-0061): the machine is destroyed the moment the control plane
  collects result.json, and provisioning a second machine would cost another
  cold start for work that takes moments where the weights already are. Not
  loading twice is the same reasoning applied to a second warm load.

- **Evaluation failure never fails the run.** `run_capability_slice` never
  raises: a generator that throws becomes `ok: false` with the reason
  recorded, and the entrypoint wraps the whole phase — selection, loading,
  answering — under the same rule. A failed slice is recorded under
  `capability` and the artifact is still delivered (Spec 011's "evaluation
  failure never fails the run").

- **It is not a benchmark, and the interface says so.** The section heading
  and copy label the check a smoke test for catastrophic forgetting, not a
  benchmark. Spec 011 is explicit that a small slice honestly labelled is
  useful, and the same slice dressed as a benchmark is misleading; the copy
  and the ADR say the same thing.

## The limits of the slice, stated

These are the reasons the check is a smoke test and not a benchmark, recorded
so the claim cannot be mistaken for something stronger:

- **Eight questions is a sample, not a signal.** Each question is worth
  12.5% of the score, and the standard error of the change is correspondingly
  wide. The check exists to catch a *large* regression — losing two or more
  questions to the base — and is not sensitive to small ones. A change within
  the stated standard error proves nothing.
- **The questions are a fixed sample, not a task suite.** They cover a
  handful of domains (astronomy, geography, history, arithmetic, biology,
  language) and sample each thinly. They do not measure "general capability"
  as a construct; they are a canary for catastrophic forgetting.
- **The slice is general, so it can miss task-specific degradation.** It is
  deliberately unrelated to the user's task, which is the point for catching
  forgetting and the reason it says nothing about whether the model is *good
  at the task* — the held-out loss and the side-by-side comparison answer
  that.
- **The score is one decoder's answer.** The slice is scored by parsing the
  model's own text for an option letter, under the fixed decoding settings
  the comparison uses. A model that refuses the format scores as wrong even
  when it "knows" the answer; that is a deliberate, stated choice (an
  unparseable answer is evidence), not a measurement of knowledge.
- **A general evaluation harness is out of scope.** Task suites, leaderboards
  and custom metrics are Spec 011's explicit out-of-scope; the slice is
  labelled as a smoke test and must never be relabelled as a benchmark in
  copy or interface.

## Alternatives considered

**A benchmark-style score (many questions, a leaderboard-style aggregate).**
Rejected: it is Spec 011's out-of-scope, costs warm-machine time the product
does not budget for, and the honest label for a handful of questions is a
smoke test, not a benchmark. The bar here is "a small slice with a
stated margin of error is honest; the same slice presented as a benchmark
score is not."

**Score by free-text rubric matching instead of multiple choice.**
Rejected: automatic rubric scoring of arbitrary text is a smaller project
than the check itself, and its own errors would need a benchmark to
characterise. Multiple choice with a parsed letter is deterministic, stated,
and legible in the record.

**Let the interface decide what a "large regression" is.** Rejected: a
threshold rendered in two places is a value two components must agree on, and
ADR-0010 says that value is defined once. The machine records the flag and
the threshold; the interface reads both.

**Give the slice its own decoding settings.** Rejected: the spec requires
fixed decoding settings recorded with the result, not per-eval settings. One
definition of the fixed settings — the comparison's — covers both eval steps,
so there cannot be two sets to drift (ADR-0010).

**Run the capability slice on a second machine, or load the models twice.**
Rejected: the machine is warm, and a second machine is another cold start for
work that takes moments where the weights already are (the same reasoning as
ADR-0061). The two eval steps share one load between them.

## Consequences

- The trainer records `capability` in result.json after the comparison,
  narrates the phase on the machine's output (so the stall detector sees
  activity), and `capability.py` ships flat in the image's `COPY` and
  `TRAINER_SOURCES`.
- The published job record gains a typed `capability` field; the finished-job
  page renders the smoke-test section — the two scores with the sample size
  beside them, the change, the per-question weight and the standard error —
  and surfaces a large regression prominently from the recorded flag. A failed
  slice shows its reason; pre-existing rows (no recorded capability) render
  nothing.
- `entrypoint.run_machine_comparison` gains a `loaded` parameter and a shared
  `_load_warm_models` helper, so the comparison and the capability slice load
  the two models exactly once between them. The comparison's own behaviour
  and tests are unchanged when called without a pre-loaded pair.
- The one acceptance criterion that cannot be satisfied without hardware —
  the slice answered with *real* generations from a *real* run — is stated as
  outstanding in the PR body rather than quietly reinterpreted as met by the
  fake's canned record; what was used instead is named, and why it does not
  substitute.

## Rollback

Revert the trainer's capability step (the `capability.py` import, the shared
load, and the `result["capability"]` call), drop the `capability` field from
the published record and its page section, and the finished record returns to
not checking general capability. The comparison is untouched by the rollback —
it keeps the shared load, which changes nothing about its behaviour.
