# ADR-0034 — The advanced surface is generated from the trainer's own schema

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/009-models-methods-and-advanced-mode.md`
- **Issue:** [#33](https://github.com/thp728/temper/issues/33)

## Context

The product exposes five adjustable values and refuses everything else. The
refusal is right and the list is arbitrary: it is what the first version
needed, not what the trainer supports. Meanwhile the settings that matter most
-- template resolution, end-of-sequence handling, loss masking, quantisation,
target modules, precision, seed -- are locked with no way to reach them, which
is a limitation wearing a safety costume: every serious tool in this space
exposes them.

The tension is that *expose every dial* and *refuse unknown keys loudly* must
both survive. Hand-curating the exposed list keeps the refusal but keeps the
arbitrariness, and the list can drift from the trainer: a field the trainer
gains is invisible until someone notices, and a field it loses stays on the
page offering a dial connected to nothing. Spike 7 measured the shape of the
problem: the pinned image's config schema (`AxolotlInputConfig`) carries 388
fields, of which 44 configure the platform's own plumbing, and of the
constraints Axolotl enforces, 113 `model_validator` plus 27 `field_validator`
run as arbitrary Python that the schema itself does not express.

## Decision

**The exposed surface is generated from the configuration schema of the pinned
trainer image, at build time.** The schema is captured once per pinned image as
a checked-in snapshot (`packages/contracts/axolotl-schema.json`) -- the
introspection of `AxolotlInputConfig` from inside the image, deterministic for
a given digest -- and the surface is derived from it. That makes the surface
exactly as wide as the trainer and no wider: it cannot claim support for
something the trainer lacks, cannot silently drop something the trainer has,
and cannot drift when the pinned image moves, because a digest change without a
regenerated schema fails a test that pins the snapshot to the Dockerfile's
`FROM`.

**Every field lands in exactly one of three tiers, and the tier is a checked-in
data file, not code.** `packages/contracts/axolotl-field-tiers.json`
classifies all 388 fields as `calculated` (the platform sets it; the common
path never sees it), `exposed_with_named_failure_mode` (reachable, with the
specific thing that goes wrong written beside it), or
`known_but_unsupported` (present in the trainer, not offered, with the reason
stated). Data rather than code so a classification is reviewable in a diff and
defensible in review: moving a field from one tier to another is a one-line
change to a JSON file, not an edit to a form component.

**A field in no tier fails the completeness check.** `temper_core.surface`
refuses to import with a schema field that has no tier (a hole through which an
unclassified control would reach a user), a tier entry that is not a field, a
duplicate, or an unknown tier name. The test suite asserts the same on the
real files: the completeness assertion rather than a sample.

**The refusal vocabulary is inherited from the trainer, not invented.** The
pre-launch gate (`surface.validate_overrides`, used by the job-creation path)
now refuses three shapes with three stable codes:

- `unknown_hyperparameter` -- a key the trainer does not know (not in the
  schema), echoed back, per the unchanged unknown-key rule. *Unknown now means
  unknown to the trainer.*
- `unsupported_hyperparameter` -- a key the trainer knows but this product does
  not expose, refused with its tier's reason. This is the difference between a
  refusal that says *why* and one that says "unknown key".
- `invalid_hyperparameter` -- an exposed value that violates a constraint the
  schema *does* express (enum membership, bounds), refused before anything is
  priced or provisioned.

**The combinations that cannot be known ahead of time are named as such.** The
schema expresses almost nothing (19 enum-typed fields, one bounded field); the
113 model validators and 27 field validators are the runtime-only surface a
generated form cannot see, so they are named in the schema snapshot and the
generated surface rather than pretended away -- precisely the risk Spec 009
calls out: a combination only the trainer rejects at runtime would otherwise be
accepted by a generated form and fail minutes into a paid machine.

**The trainer's known-key set is the schema.** The entrypoint's guard -- which
keys it accepts rather than echoing back as `rejected_overrides` -- reads the
schema snapshot shipped into the image, so the trainer refuses exactly what it
does not know, and the domain test pins the entrypoint's set to
`surface.known_keys()` so the two cannot drift.

**The reachable set today is unchanged.** The exposed tier holds the ten
hyperparameters that were already overridable (plus the platform's own
`simulated_failure_code`), so the create path behaves identically; what changes
is that every other key now gets a reason-specific refusal, and widening the
exposed tier (issue #80) is a diff in a data file, not a rewrite.

## Alternatives considered

**Hand-curate the exposed list, as today, and keep it growing by hand.**
Rejected: this is the status quo that produced five arbitrary knobs, and it
cannot satisfy the spec's reviewer-facing criterion -- a hand-listed surface
gives no way to *see* that it cannot drift. It also makes "unknown" mean
"absent from a hand-written list" rather than "unknown to the trainer", which
is a lie the moment the trainer and the list disagree. The list survives, but
as data (the tier file), generated into the surface rather than written as a
second copy.

**Generate the field list from the schema but classify in code.**
Rejected: a classification encoded in `if name.startswith(...)` branches is
reviewable only by reading code, and a change to a judgement becomes an edit to
a function. The whole point of Spec 009 is that a classification is a long list
of judgements that deserves a diff. The tier file is the judgement; the
generator is how it was produced, and the checked-in file is authoritative.

**Expose everything, with no tiers.**
Rejected: it destroys the refusal. An advanced surface that lets a user set
`save_only_model` or `special_tokens` with no guard, no failure mode and no
pre-launch validation would put the highest-value silent failures one click
away -- the opposite of the safety the refusal exists to provide. The tiers
exist so exposure is honest about what can go wrong.

**Ship only the reachable list to the interface, keeping the schema private.**
Rejected: then the interface could not render "known but unsupported" fields
with their reasons, and the spec's user story -- *I want a setting the trainer
supports but this product does not to be visible with its reason, so that I do
not wonder whether it was overlooked* -- would be unanswerable.

## Consequences

- `packages/contracts/` gains `axolotl-schema.json` (the snapshot) and a
  complete `axolotl-field-tiers.json`; `just contracts` generates
  `advanced-surface.json` from the two, and `just contracts-check` fails the
  gate on drift.
- `temper_core.surface` is new: the completeness check, the tier views, the
  known-key set, the runtime-only validator names, and the pre-launch gate.
  `hyperparams.ALLOWED_OVERRIDES` now derives from the surface, and
  `trainer-defaults.json` no longer declares the reachable list, so the page,
  the gate and the frozen spec read one definition of what is reachable.
- The trainer entrypoint reads `axolotl-schema.json` (shipped into the image)
  for its known-key set; the Dockerfile `COPY` and the build context carry the
  file.
- The create path refuses three shapes with three codes instead of one, and
  the refusals carry the reason text from the tier file.
- The 113 model validators and 27 field validators are named in the schema
  snapshot as the combinations that cannot be known ahead of launch.
- The exposed tier is deliberately the current reachable set; issue #80 widens
  it behind an explicit disclosure surface, and #54 and #58 extend the same
  pre-launch gate.

## Rollback

Delete `axolotl-schema.json`, the tier file and `advanced-surface.json`,
remove `temper_core.surface`, restore `allowed_overrides` to
`trainer-defaults.json`, and return the entrypoint's known-key set to the
hand-written list. The create path reverts to the single
`unknown_hyperparameter` refusal. The cost of the revert is the drift the
generated surface exists to remove.
