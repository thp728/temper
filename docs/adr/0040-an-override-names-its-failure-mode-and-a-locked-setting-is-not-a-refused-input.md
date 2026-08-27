# ADR-0040 — An override names its failure mode, and a locked setting is not a refused input

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/009-models-methods-and-advanced-mode.md`
- **Issue:** [#80](https://github.com/thp728/temper/issues/80)

## Context

The correctness settings are the highest-frequency silent-failure surface in
the system: they pass every obvious health check and surface only as bad
output. That argues for safe defaults. It does not argue for making them
unreachable — every serious tool in this space exposes them, and hiding them
was defended as safety and is better described as a limitation wearing a
safety costume.

Issue #33 already reconciled *expose every dial* with *refuse unknown keys
loudly*: the exposed set is generated from the pinned trainer's own schema,
each field lands in exactly one of three tiers, and the exposed tier carries
the specific thing that goes wrong. Issue #79 made the plan's decisions
overridable with the rest recomputing. What remained was to put the first two
together: let the exposed tier be overridden, record the overrides in the job
spec, and make the boundary between what is adjustable and what is refused a
product surface rather than an accident.

That boundary is the intellectual core of this issue, and Spec 009 draws it in
a way this record exists to name precisely: **a locked setting and a refused
input are different things.** A locked setting has a correct value the product
chose, and an informed user may choose otherwise. A refused input has no
correct interpretation at all — no value of any control resolves it, because
the ambiguity is in the input, not in the configuration.

## Decision

**Every exposed setting is overridable, behind an explicit disclosure.**
The advanced surface is rendered from the published `GET /v1/surface`
document (the same generated tiers `just contracts` materialises), never
hand-listed, so the panel cannot drift from what the pre-launch gate refuses.
It sits behind a disclosure because opening it is a deliberate act, and each
exposed field carries its reason and, inline, the specific thing that goes
wrong if it is set badly — not a general caution.

**A setting the trainer supports but this product does not is visible with
its reason rather than absent.** The `known_but_unsupported` tier is rendered
(searchable), so a field is never wondered about as overlooked, and passing
one is refused with that reason.

**Overrides are recorded in the job specification and shown on the finished
run.** The user's advanced overrides ride the same recompute as plan
decisions, are frozen into the job's hyperparameters (coerced to the schema's
type), and the finished run shows them under "Settings you changed" — the run
says what it actually used, after it is over.

**An override that makes the job infeasible is refused before launch with the
same arithmetic.** A hyperparameter override is a demand, not an absent
estimate: `build_quote` refuses an infeasible configuration with the same
peak-vs-capacity numbers the predictor used (`configuration_does_not_fit`),
on the recompute and on the launch, so taking control of the correctness
surface never costs a provisioned machine that OOMs minutes in.

**A locked setting and a refused input are different things, and the
interface says so.** The distinction is rendered on the advanced surface and
the dataset-level refusals are structurally not overridable:

- A **locked setting** has a correct value the product chose — e.g. `seed`
  is pinned so runs reproduce, `train_on_inputs` is off so loss masks to the
  assistant turn. The product's choice is a starting point; for the settings
  Temper exposes, an informed user may choose otherwise, and each exposed
  setting says what goes wrong if it is set badly. A locked setting Temper
  does not expose is refused with its reason, never silently dropped.
- A **refused input** has no correct interpretation at all. The two named
  examples, both from dataset validation:
  - **A mixed thinking-mode dataset** — assistant turns mixing reasoning
    traces with plain answers are ambiguous by construction; there is no
    value of any control that decides how to interpret the file, so it is
    refused with its lines named (`mixed_thinking`) and no override can
    launch it.
  - **A dataset below the minimum usable row count** — with fewer than the
    floor (10) usable rows, no adapter a run could produce is meaningful;
    there is no correct setting to override to, so it is refused
    (`too_few_rows`) and cannot be overridden.
  Both reach the launch as `dataset_invalid` before any hyperparameter is
  considered — the refusal happens where validation lives, and the advanced
  surface cannot route around it. That is an absent control, not a hidden one.

## Alternatives considered

**Expose the settings without a disclosure.**
Rejected: the correctness surface is where the silent failures live, so
opening it should be an explicit act with the consequences named; a wall of
ten dials beside a first-time launch reads as a hazard rather than a choice.

**Render only the overridable tier and keep the rest absent.**
Rejected: Spec 009's user story 20 is explicit that a setting the trainer
supports but this product does not must be *visible with its reason*, so a
user never wonders whether it was overlooked. The `known_but_unsupported`
tier is shown, searchable, rather than hidden.

**Make the refusals overridable — let an advanced setting reinterpret a mixed
dataset.**
Rejected, and this is the distinction the record is about. A mixed
thinking-mode dataset is ambiguous by construction: forcing an interpretation
would silently resolve what the product explicitly refuses to guess about.
There is no correct value of any trainer setting that decides what a file
"means", so it stays refused. The same holds for a dataset below the row
floor, where the missing ingredient is data, not a dial.

**Refuse only at launch, letting the plan show the overrides as accepted.**
Rejected: the pre-launch refusal is the point — it runs the same arithmetic
the predictor used *before* anything is provisioned, so a mistake costs
nothing. Showing a config as acceptable and refusing it only at the moment of
commitment would repeat the silent-failure pattern this surface exists to
remove.

**Coerce nothing and pass the browser's strings straight to the trainer.**
Rejected: a string that reaches a numeric field is a silent type drift, and
the trainer's pydantic model must not be relied on to be lax. The resolver is
the single place a value is normalised to the schema's type.

## Consequences

- `GET /v1/surface` publishes the generated document (typed at the HTTP
  boundary) and the interface renders its advanced panel from it; the exposed
  tier also publishes a render/coerce type derived from the schema snapshot.
- `POST /v1/quotes` accepts `hyperparameters` and runs them through the same
  `surface.validate_overrides` gate as a launch; `build_quote` treats a
  hyperparameter override as a demand and refuses infeasibility with the
  arithmetic.
- `hyperparams.effective` coerces overrides to the schema's type; launched
  overrides are frozen into the job record coerced, and the finished run
  shows them.
- The advanced panel explains the adjustable-versus-refused distinction with
  both named examples; the dataset-level refusals remain non-overridable by
  construction (they occur in validation, before hyperparameters exist).
- The runtime-only validators the schema cannot express remain named rather
  than pretended away, exactly as ADR-0034 recorded.

## Rollback

Drop the `GET /v1/surface` endpoint and the advanced panel, revert
`build_quote`'s demand handling, remove the coercion, and the correctness
settings return to being default-locked with no reachable path — the state
this issue exists to retire.
