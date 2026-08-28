# ADR-0049 — A model outside the catalog is usable once a probe reports on it

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/009-models-methods-and-advanced-mode.md`
- **Issue:** [#58](https://github.com/thp728/temper/issues/58)

## Context

The catalog is a curated, tested, pinned set of models with a good argument
behind it: detection of an arbitrary model is easy, support is not, and every
family brings its own template quirks and tokenizer edge cases. But the
catalog's own documentation calls curation *a default rather than a
limitation* while the code behaved as a limitation — `catalog.get` refusing
anything not on the list meant "any model, including large ones" was untrue in
practice. Spec 009's answer: the catalog stays the recommended, tested path,
and *any other* model repository may be used at a pinned revision once a
compatibility probe reports on it. The probe does the testing in front of the
user, and its result is shown rather than merely enforced.

Three things landed in earlier waves that this issue builds on rather than
beside:

- **#48** put model facts behind a seam (`temper_core.models.ModelFacts`,
  resolved by `apps/control-plane`'s `models.py`). Spec 009's criterion is
  explicit that the probe reads facts through the same seam the predictor
  uses, so the two cannot drift — a probe with its own resolver is a second
  source of truth about a model, and ADR-0010 exists to stop exactly that.
- **#54** made memory block at job creation while time and cost warn
  (ADR-0043), through `selection.NoFittingHardwareError`. This issue's
  criterion "the predicted memory fits something available" is that same
  check, and it must be reused, not reimplemented.
- **#59** landed the export-time template probe and taught the vocabulary a
  probe needs: a probe that has only ever passed is not evidence, and each
  failure path deserves its own fixture.

## Decision

**Any model repository may be admitted at a pinned revision once a
compatibility probe reports on it, and the probe's result is persisted and
shown before a job can be created against that model.**

**The probe reads facts through the predictor's own seam.** `admission._resolver`
is `quote.models_for_quote()` — the exact seam the quote, the hardware search
and the creation-time refusal read through. In production both are
`new_models()`; under test both are whatever the quote seam has been pointed
at. There is deliberately no second resolver: two implementations of *how many
parameters does this model have* would drift, and the drift would surface as a
model the probe accepted and the predictor mispriced. The memory line is the
same `memory.predict_peak` arithmetic `selection.select_hardware` and the #54
refusal are built on — one function, refined once, shared by all three.

**A pinned revision is required.** `POST /v1/models/probe` refuses anything
but a 40-character commit SHA with the stable code `unpinned_revision` — the
catalog's own pinning rule (a model can change under a completed run, which
breaks the reproducibility claim the pinning exists to support), applied to
every path, not just the curated one.

**Blocking and warning are product decisions, with the reasons kept beside
them** — the probe's verdict is a set of findings, each carrying its severity:

- **The revision resolves.** A pinned reference that cannot be resolved cannot
  be trained against at all. This blocks (`revision_unresolvable`), and it is
  the seam's own refusal surfaced as a probe verdict, so a failed admission is
  persisted and shown rather than silently absent.
- **A chat template is present.** Without one, chat data cannot be formatted,
  and the alternative is hand-writing role delimiters — the modal silent
  failure. Blocks (`missing_chat_template`).
- **Padding differs from end-of-sequence.** A tokenizer whose pad token equals
  its EOS token, unmasked, teaches the model never to stop. Blocks
  (`padding_collides_with_eos`).
- **The licence resolves.** An unresolvable licence is shown as unknown rather
  than guessed — a visibly missing licence beats a wrong one, the posture the
  models seam already takes. Warns (`license_unresolvable`).
- **The architecture is tested here.** Mixture-of-experts and unknown
  architectures change target-module selection, memory scaling and packing
  behaviour, so they are labelled untested. They warn rather than block,
  because curation is a default and not a boundary — blocking on the third
  would make "any model, including large ones" untrue in practice, which is
  the exact claim this issue exists to retire. Warns (`untested_architecture`).
- **The predicted memory fits something available.** Computed at the product
  defaults against the cards the platform can provision, taking the smallest
  card that holds the peak. Per ADR-0043, memory *blocks at creation*; at
  admission it warns with the same arithmetic (`memory_does_not_fit`), so the
  user sees the numbers the launch would refuse with. "Available" here means
  *provisionable*: current availability is a fact about the moment, and a
  persisted probe must not record a moment as if it were a property of the
  model — the live availability search still runs at job creation (#54),
  against the real configuration, where it belongs.

**The result is persisted and gates creation.** A probe — passing, warning or
blocked — is stored in `admitted_models` at admission, and `jobs.create` (the
one creation path) refuses a launch against a model whose stored probe has
blocking findings, with those findings shown (`model_probe_blocked`). A model
with no stored probe is `unknown_model`. A blocked model is persisted too, so
the reason is shown and the user does not retry the same thing (spec 009's
user story 3). The catalog needs no probe: it is the tested default, and the
probe is the testing done in front of the user *for anything else*.

## Alternatives considered

**Give the probe its own model-facts resolver.**
Rejected outright: that is a second implementation of "how many parameters
does this model have", and ADR-0010's drift — the probe accepts a model the
predictor misprices — is the specific failure this spec's criterion names. The
probe reads through `quote.models_for_quote`, the predictor's own seam, so
there is exactly one implementation and the two cannot drift by construction.

**Block admission on memory at probe time (a model that fits no card at
defaults is refused up front).**
Rejected: ADR-0043 already decided where the memory refusal lives — at job
creation, against the *real* configuration, because sequence length, batch
size and rank are not known at admission. A model that does not fit at the
defaults may fit a lighter configuration, and refusing it at admission would
strand work the creation path would have accepted. The probe shows the
arithmetic and warns; the creation path refuses with the same numbers.

**Warn rather than block on a missing chat template or on
padding==end-of-sequence.**
Rejected: these are not matters of taste or moment. A model without a chat
template cannot format chat data without hand-written role delimiters (the
silent failure this probe exists to prevent), and a model whose padding
collides with EOS cannot be trained to stop. Both are deterministic properties
of the model at the pinned revision, and both make the model unusable here, so
both block. This is the half that inherits ADR-0043's "memory blocks" posture:
a wrong *block* here has no false positive, because the check is deterministic.

**Make an unresolvable licence block.**
Rejected: a model without a resolvable licence is still usable; what the user
needs is to *know* the licence is unknown and make their own call about
obligations. Blocking would take a legal-risk disclosure and turn it into a
hard no, which is exactly the over-eager boundary this issue removes. The
models seam's own posture — a visibly missing licence beats a guessed one — is
inherited unchanged.

## Consequences

- `temper_core.probe` is a pure module: verdict + findings over `ModelFacts`
  and the defaults, testable without a network, with a fixture for each
  failure path (spec 009's testing decision verbatim). The control-plane half
  (`admission.py`) is thin: resolve facts through the predictor's seam, run
  the probe, persist, and gate creation.
- The launch screen shows imported models under their own group, each with its
  persisted verdict, findings and memory line; a blocked model is shown with
  its reasons and cannot be selected. The probe form is deliberately *not* a
  nested `<form>` inside the launch form — a nested form submits natively with
  its own fields as query parameters and silently drops the `dataset_id`; the
  browser journey caught that in development.
- Re-probing the same pinned reference is idempotent (the `(repo, revision)`
  pair is unique), which matters because a pinned revision is immutable: the
  facts cannot change, so re-probing would only ever return the same verdict.
- #54's ADR-0043 consequence is realised: an imported model whose facts
  predict no fit anywhere is refused at creation with the same arithmetic the
  search used, and the probe surfaced that arithmetic at admission.

## Rollback

Remove the `admitted_models` table, `admission.py`, `temper_core.probe` and
the probe endpoints, and revert `jobs.create`/`quote_for_launch` to resolve
`catalog.get` only; a model outside the catalog is refused as `unknown_model`
again, and the probe's claims about blocking and warning stop being true, so
this record would need to be superseded rather than edited.
