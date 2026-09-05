# ADR-0075 - Hyperparameters edit in place, and explanations open on demand

- **Status:** accepted
- **Date:** 2026-09-05
- **Supersedes:** nothing
- **Spec:** Spec 005 (launch surface) as amended by wireframe review
- **Issue:** none - surfaced during a step-2 UI review, not filed as a ticket.

## Context

The launch wizard's hyperparameters step showed frozen values in spec cards,
with editing split across two other surfaces: predictor decisions beside
their full reasons in `QuoteView`, and trainer settings behind the
"Advanced settings" disclosure in `AdvancedSurface`. Review of the step
against the wireframe asked for what the step plainly is: an input form.
Every hyperparameter editable in place with its default filled in,
explanations (decision reasons, failure modes) one "?" click away rather
than printed on the page.

Two hard limits shape what "editable" and "appropriate input" can mean:

1. The launch gate accepts only `surface.overrideable_keys()` (the exposed
   tier plus platform-internal keys); every other trainer key is refused
   before launch (`temper_core.surface`). Ten keys are exposed today, all
   typed int or float.
2. `SurfaceField` publishes `type`, `reason`, `failure_mode` and `tier`
   only. No bounds, no enums, no option vocabularies.

## Decision

- The step-2 cards host the only editors. Each exposed key renders a numeric
  or text input (generated from the schema's own `type`, the seam
  `temper_core.surface.value_kind` names) populated with its effective value;
  committing an empty or default-equal value clears the override. Calculated
  keys render read-only with the tier's reason one hover away, never a
  control whose value the launch would refuse.
- `QuoteView` gains `explain="dialog"` for the wizard (label, value,
  control, "?" dialog with reason and alternatives); the finished-job
  record keeps the inline default, where the explanation is the product.
- Failure modes sit behind per-row "?" tooltips (shadcn Tooltip,
  already in-repo). The "Advanced settings" disclosure now lists only the
  unsupported tier, retitled as Axolotl settings Temper doesn't support
  yet; the adjustable-versus-refused essay is removed with it.

## Alternatives considered

- **Sliders, selects and toggles for hyperparameters.** Rejected: without
  published bounds or vocabularies their ends and options would be invented
  constraints the server never promised. When the contract grows bounds,
  `EditableSpecRow` is the seam.
- **Epochs/Max Steps mode tabs.** Rejected: both keys are exposed and the
  launch merges overrides over defaults, so "unset" is inexpressible -- a
  toggle that cannot unset the other side would lie about what freezes.
- **Duplicate editors in cards and disclosure.** Rejected: two controls per
  accessible name, and two places to keep the commit rule identical.
- **Dialog explanations on the finished record.** Rejected: there the
  explanation is the product being read, not a form being filled.

## Consequences

- `AdvancedSurface` takes only `{ surface }`; its editor, subtitle and
  essay tests moved to the wizard (`LaunchForm.test.tsx`).
- The launch journeys assert the new placements (dialogs, tooltips,
  in-place values) and must run under `just e2e` -- no browser runner
  exists in this environment, so they are updated unread.
- Follow-up refinements land in the same change: a "Reset defaults" button
  clears every override in one act; the step-2 decisions render bare inside
  the cards they tune (no "Why this configuration" section on a form step);
  tooltips render dark single-column through the primitive's `className`
  seam rather than a fork.
- A second review pass, same change: decision selects show the predictor's
  value as selected (reselecting it unpins; there is no placeholder option
  to confuse with a value), method/precision render above the rank rows they
  govern, the card-01 method pill goes away as duplicative, and the
  sequence-length hyperparameter row steps aside while its decision control
  is present -- one control per value, since both write the same number.
