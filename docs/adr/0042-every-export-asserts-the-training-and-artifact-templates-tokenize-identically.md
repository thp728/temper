# ADR-0042 — The export-time template probe asserts the artifact reproduces the training template

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/009-models-methods-and-advanced-mode.md`
- **Issue:** [#59](https://github.com/thp728/temper/issues/59)

## Context

The template used in training resolves by default and nothing asserts it
resolved to the right one. That was already mandatory because thinking mode is
detected from the dataset rather than fixed — the probe is what catches a wrong
detection. Exposing the template controls (issue #80) makes it the only thing
standing between an advanced user and a model that trains cleanly and answers
wrongly: a wrong template passes every obvious health check and only surfaces
as garbage generations.

Spec 009 fixes the shape of the probe — a fixed conversation tokenised through
the template used in training and through the template serialised into the
artifact, identical ids required, running on every export regardless of
overrides — but leaves three questions to the implementation: where it runs,
what the artifact's serialised template is, and how a failure is reported so
that it is actionable rather than decorative. Issue #59 also fixes the testing
bar explicitly: *a probe that has only ever passed has no evidence of working*,
so a deliberately mismatched template must be proven caught.

## Decision

**The probe runs in the trainer at export time, and its result travels with the
artifact record.** It lives in `apps/trainer/template_probe.py`, shipped into
the image beside the entrypoint, because only the trainer holds the tokenizer
and the resolved config at the moment of export: the control plane stores the
probe's record (inside `result.json`, which is the artifact record per
ADR-0035), it does not perform the probe. The tokenizer is the probe's only
external dependency and is passed in rather than imported, so the host suite
drives it with a double and no image dependency leaks into tests.

**The two sides are the training template and the artifact's serialised
template, both as concrete strings, and they come from different sources.**
The training side is what Axolotl applied — the tokenizer's own template for
the `tokenizer_default` directive, an explicit template as given. The
artifact's serialised side is the concrete template the trainer records with
the artifact, never the directive: a directive is for the trainer to
interpret, a string is what a server can apply. The serialised side reads the
template controls the **job spec records as chosen** (where #80's override
surface writes its result), falling back to the config when nothing was
recorded. Two sources, deliberately: a recorded override that training never
applied — or the reverse — is a real divergence the probe can catch, not a
value compared with itself. Today the spec records nothing, so both sides
resolve to the same concrete template and a pass is the round-trip: the
concrete template the artifact records tokenises identically to what training
applied.

**A mismatch fails the export with the stable code `template_probe_mismatch`,
and a probe that cannot run at all fails closed with
`template_probe_unavailable`.** A guard that silently disappears when it cannot
run is no guard, and the templates are load-bearing for every advanced override.
The failure travels the ordinary path — `error_code` in `result.json`, and the
control plane's `_attempt` already fails a result whose `ok` is false with the
named code — so a user gets a coded, explainable refusal rather than a trained
but wrong adapter.

**The failure message names what differs, not merely that something did.** It
names both template strings when they differ, the id counts on both sides, the
first differing id, and the rendered text around the first difference. "The
templates differ" tells no one what to fix; this message points at the exact
place and text.

**The testing decision is the point of the suite.** A deliberately mismatched
template — a serialised template that is not the training template — must fail
the probe, and the tests assert the message names both sides. The happy path is
the contract but not the evidence; Spec 009's rule is that a probe with no
failing test has no evidence of working.

## Alternatives considered

**Run the probe in the control plane at job finalisation.**
Rejected: the control plane does not hold the tokenizer, the model or the
resolved config — it would have to re-download the model and re-implement what
training did, a second implementation that could disagree with the first. The
probe must compare against what actually trained, which only the machine knows
at export time. The control plane's half is recording the result, which it
already does for every `result.json` field.

**Compare the serialised template against a stored snapshot of the training
template rather than re-tokenising the conversation.**
Rejected: comparing template *strings* only catches a changed string, not a
string that renders differently once kwargs (thinking mode) are applied. The
probe's load-bearing case is precisely a template that resolves differently
under the detected `enable_thinking`, so the comparison has to be over what the
template *does* to a real conversation — token ids — not over the template text.

**Silently record a "not run" result when the tokenizer cannot be loaded at
export.**
Rejected: that is the guard disappearing exactly when it is needed, and it
reads as a pass to every downstream consumer. The tokenizer was just used for
training, so an export that cannot load it is genuinely broken; failing closed
with `template_probe_unavailable` is the honest report.

**Record the directive (`tokenizer_default`) as the artifact's serialised
template.**
Rejected: a directive requires its reader to re-resolve it against a model,
which is the same silent-failure surface the probe exists to close. The record
carries the concrete Jinja string, so a downloader reproduces formatting
without interpreting a directive.

## Consequences

- `apps/trainer/template_probe.py` ships in the trainer image; the Dockerfile
  `COPY` and `TRAINER_SOURCES` both carry it, pinned to agree by
  `test_trainer_context.py`.
- Every export runs the probe after `collect_artifacts` and before the artifact
  upload; a failing probe prevents the upload and the job reports no success.
- `result.json` carries a `template_probe` record with the verdict and, on
  failure, the stable code, the message and the serialised template; the
  simulated machine's canned success carries the same shape so every surface
  that reads a finished job reads one that includes the probe.
- Two new stable codes: `template_probe_mismatch` and
  `template_probe_unavailable`.
- The probe's tokeniser seam is the place #80's template overrides meet the
  export: whatever the override surface records, the probe proves at export
  that it resolved consistently with training.

## Rollback

Remove the probe call from `main()`, delete `template_probe.py`, its test file,
and the build-context entries, and drop the `template_probe` block from the
simulated machine's canned result. The stable codes disappear with the code
that raised them. The cost of the revert is the guard itself: nothing then
asserts the artifact's template matches the training template, which is the
exact hole this record closes.
