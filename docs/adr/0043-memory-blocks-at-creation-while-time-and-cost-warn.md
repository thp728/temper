# ADR-0043 — Memory blocks at job creation while time and cost warn, and why the two halves differ

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/005-predictor-and-quote.md`
- **Issue:** [#54](https://github.com/thp728/temper/issues/54)

## Context

The predictor has two halves, and this record exists to name where the
warn-rather-than-block posture applies and where it deliberately does not.

ADR-0007 established the posture for the whole feasibility surface: the
duration estimate is a crude number from one measured run, a wrong block is
worse than a wrong warning, and the job launches with the warning attached.
ADR-0031 inherited it exactly for the quote: time and cost warn, and a launch
is never refused because a quote is missing. Issue #72's own PR made the
inheritance explicit for an unpricable configuration — provider unreachable,
nothing free, a model that will not resolve — which produces **no quote**,
never an error, on the plan screen *and* on the launch path.

But spec 005 draws a line inside that posture that the quote work inherited
whole and the override work (#79, #80) only touched when the user *demanded* a
specific configuration:

> **Memory is arithmetic, and it blocks.** Whether a configuration fits a card
> is deterministic — parameters, gradients, optimizer state, activations. Being
> wrong means an out-of-memory failure minutes into a paid machine, so a
> configuration predicted not to fit is refused with the arithmetic shown.
>
> **Time and cost are estimates, and they warn.** ... quoted as a range,
> labelled as an estimate, calibrated against every run ... and they never
> block.

Before this issue, an unpinned launch (no overrides, no advanced
hyperparameters) that predicted no fit anywhere was *not* refused: `build_quote`
returned None — no quote — and the job was created, to be refused at
provisioning by the orchestrator's `provider_capacity_unavailable` backstop.
The refusal existed, but it fired after a job record was created and carried
none of the arithmetic that would tell the user what to change. An override
made the same refusal fire early, with the arithmetic (#80), because an
override is a demand. This issue asks why a *plain* job that cannot fit should
get less protection than one whose user demanded a configuration.

## Decision

**A configuration predicted not to fit any available card is refused at job
creation — with the arithmetic shown — whether or not the user demanded it.**
The memory half blocks. The refusal:

- **reuses one computation.** The refusal and the hardware search share
  `memory.predict_peak`/`headroom_gb` through `selection.NoFittingHardwareError`,
  which already carries the peak and the capacity that did the refusing. The
  creation path does not recompute anything; it renames the search's own error
  for the HTTP boundary. There is deliberately no second arithmetic that agrees
  today and drifts tomorrow — that drift is the specific failure ADR-0010
  exists to prevent.
- **shows what was needed, what was available, and where the shortfall is.**
  The payload carries `peak_gb`, `capacity_gb` and `shortfall_gb` (= peak over
  capacity), plus the method, card and device count the refusal was computed
  against — the same numbers `memory.headroom_gb` used to refuse.
- **names what the user could change.** The message says it: a smaller model, a
  shorter sequence, or a lighter training method such as qlora. Each lever is
  real arithmetic — fewer parameters shrink every pool, a shorter sequence
  shrinks activations, a lighter method shrinks weights and the trainable set —
  and the method lever is dropped when the refused method is already qlora,
  because recommending a heavier one would mislead.
- **carries a stable code**: `configuration_does_not_fit`, the code #80 already
  established for the memory refusal. One code, one meaning, whether the
  configuration was demanded or not.

**Time and cost warn, and they inherit ADR-0007 exactly.** A configuration that
cannot be priced — provider unreachable, nothing free, a model that will not
resolve — still produces no quote, not a refusal, and the launch still happens.
This half is the one ADR-0007's posture was built for, and #72's PR inherited
it explicitly for the unpricable configuration; this record does not disturb
it.

**The line between the two halves is the cost of being wrong, not the presence
of an override.** Naming that line is the point of this record:

- Being wrong about **memory** means an out-of-memory failure *minutes into a
  machine the user is paying for*. The machine is billed from provisioning, so
  an OOM is not a wrong prediction that costs an estimate — it is real money
  spent on a run that could not have worked. Memory is also deterministic
  arithmetic over the model's own facts, not extrapolation: whether a card
  holds a predicted peak is not a matter of measurement error, it is a
  computation.
- Being wrong about **time or cost** costs an estimate. The throughput figure
  is the softest number in the model — one measured run, a published spread
  spanning nearly an order of magnitude — and a wrong duration or price is a
  wrong number the user sees *before* committing, correctable by calibration
  (#77). Nothing is spent on an estimate being wrong.

So the predictor's time/cost half inherits ADR-0007's warn-rather-than-block
posture, and the memory half deliberately does not. The two halves differ
because the cost of a wrong refusal differs: a wrong memory block has no false
positive (the arithmetic is deterministic), while a wrong time block would
strand work that would have run.

**Availability alone is not a refusal at creation.** A configuration that
*would* fit but is not currently free — `provider_capacity_unavailable` — is a
fact about the moment, not about the configuration. Refusing it at creation
would block work that could run once hardware frees, which is exactly the
over-eager refusal this issue warns against ("a refusal that fires too eagerly
is worse than none, because it blocks work that would have run"). It keeps the
warn posture: no quote, the job launches, and the orchestrator's provisioning
backstop names it if nothing frees up. The one exception is an override that
cannot be honoured (#79): a demand that cannot be met is refused on
availability grounds too, because taking control must not silently fall back to
the predictor's pick.

**A configuration that fits is never blocked on memory grounds.** The negative
is tested as hard as the positive: the same job that an L4 refused launches the
moment a card that holds its predicted peak is available. The refusal fires on
the arithmetic, not on the model's existence.

## Alternatives considered

**Refuse availability at creation too (fits but nothing free).**
Rejected: that is a fact about the moment, not about the configuration, and a
refusal there blocks work that would have run as soon as hardware frees. The
provisioning backstop already names it without a machine having been created.
This is the over-eager refusal the issue's criteria single out.

**Keep the warn posture for unpinned memory non-fits (only overrides refuse).**
Rejected: that would give a plain job *less* protection than a demanded one,
for the worst failure mode there is. The reason the refusal exists is not that
the user asked for something specific — it is that the job cannot fit, and
"cannot fit" is not an estimate. This is the status quo this issue exists to
retire, in which the job record was created and the refusal surfaced at
provisioning without arithmetic.

**Refuse at provisioning only, where `select_hardware` already runs.**
Rejected: a job record then exists that was never going to run, and the user
sees the failure as a failed job rather than a refusal before anything is
spent — which is the issue's title. The creation path already computes the
quote through the same search, so the refusal there costs nothing extra and
reuses the same computation.

**Invent a second memory arithmetic at the creation boundary.**
Rejected outright: two computations that agree today and drift tomorrow is the
specific failure ADR-0010 exists to prevent. The creation refusal is the
search's own `NoFittingHardwareError`, renamed; if the memory model is refined,
the search and the refusal refine together because they are one thing.

## Consequences

- `quote_for_launch` — the creation path shared by the JSON API and the launch
  form — refuses a memory non-fit with `configuration_does_not_fit` even when
  nothing was overridden; `POST /v1/jobs` returns the coded 400 with the
  arithmetic before any job row exists.
- The estimate surfaces (`GET`/`POST /v1/quotes`) keep the warn posture: a
  non-fit is still `None`, not an error, because those are the surfaces the
  estimate appears on and an estimate does not block the surface it appears on
  (spec 005). The refusal the user acts on is the one at the moment of
  commitment.
- `selection.NoFittingHardwareError` now also carries `shortfall_gb` (peak over
  capacity), defined once on the error so the search and the refusal surface
  cannot disagree about where the shortfall is.
- The override path (#79, #80) is unchanged: a demanded configuration refuses
  on both memory and availability grounds, through the same `_no_fit_refusal`.
- #58 will admit models from outside the catalog through this same refusal
  path: an imported model whose facts predict no fit anywhere is refused at
  creation with the same arithmetic, because the refusal consumes model facts
  through the same `models` seam the search reads. The seam is left clean; the
  import itself is not this issue's work.
- ADR-0007, ADR-0031 and this record now read as one story: the warn posture
  covers the estimate halves (duration, cost, availability), and this record
  draws the boundary where the cost of being wrong stops being an estimate.

## Rollback

Revert `quote_for_launch` to pass the estimate posture (drop
`refuse_unfittable`) and remove the `shortfall_gb` field; the memory refusal
returns to firing only for overrides, and an unpinned non-fit is refused at
provisioning again. The ADR's claim that memory blocks at creation stops being
true, so it must be superseded rather than edited if that change is ever
accepted.
