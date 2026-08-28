# ADR-0051 — A failure path is not done until it has been caused deliberately

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/010-recovery-and-fault-injection.md`
- **Issue:** [#24](https://github.com/thp728/temper/issues/24)

## Context

Every recovery path this product needs is verified by something going wrong:
an out-of-memory run, a diverging loss, an interrupted worker, a silent
machine, an orphaned machine, a destroy the provider refuses. Nothing could
make any of those happen on demand, so every one of those recoveries was a
claim about code that had never run against the thing it recovers from. A
recovery path that has only ever been reasoned about is the same class of
artifact as a large green suite over a product that could not train — the
project has already been burned by exactly that shape of confidence.

The fix is structural, not behavioural: build the ability to cause each
failure first, then build the recoveries, then cause the failures against
them. This record is the first half — the fault surface, shipped before any of
the recoveries it will prove. It also states the rule the recoveries are
measured against, which is the durable decision this record exists to hold.

## Decision

**A failure path is not done until it has been caused deliberately, and the
cause is made by the fault surface.**

- **The fault surface is part of the product, off by default.** It is
  configuration per job — the dict form of the existing
  `simulated_failure_code` hyperparameter, naming one of the six faults —
  and off is enforced at the single creation path: a fault-spec job is
  refused unless `TEMPER_FAKE_PROVIDER` (the zero-cost tier) or
  `TEMPER_FAULT_SURFACE` (the deliberate real-hardware tier) is set. Off is a
  safety property, not a convenience: a fault surface that can be switched on
  by accident in front of a user is worse than none.

- **Provider-side faults live in the existing provider seam; trainer-side
  faults are read from the environment.** No new module and no new boundary.
  The fake provider honours all six faults by reading the job spec exactly as
  it already read the reserved `simulated_failure_code` (ADR-0026) — that
  mechanism is grown, not replaced. The real trainer reads its fault from an
  environment variable the control plane writes, and makes the fault
  genuinely happen on hardware. Provider-side faults can therefore never fire
  against a real machine (only the fake reads them); trainer-side faults need
  the surface switched on twice over — at creation and at the point the
  control plane writes the trainer's environment.

- **Every injected fault is named in the run's history.** The orchestrator
  writes a launch event naming the fault; the machine's or trainer's stream
  narrates it at the moment it fires; a run that produces its own result
  document carries a `simulated_` code (oom, divergence); and the fault spec
  is frozen into the job record. A deliberately broken run can never be
  mistaken for a real one, which is what makes a reviewer able to trust what
  they are watching. The faults that fail through the platform's ordinary
  machinery (a killed worker, a silent machine, a refused destroy) carry the
  platform's ordinary codes, and the history is what names them; the
  contract deliberately declares a code only where the run carries one.

- **The vocabulary is data, defined once.** The six faults, their codes and
  which side makes each real live in
  `packages/contracts/fault-surface.json`, read by the control plane and
  shipped into the trainer image. The trainer cannot import `temper_core`
  (ADR-0010), so the two sides read the same file rather than the same
  module; a trainer test pins the wire agreement.

## Alternatives considered

**A fault-injection framework as a new seam.** Rejected: it would be a seam
introduced to test seams, and the spec names that exact shape as the thing to
avoid. The provider protocol already is the seam provider-side faults belong
to, and an environment switch is the cheapest honest seam for the trainer.

**A second, parallel hyperparameter for the surface, leaving
`simulated_failure_code` untouched.** Rejected: the reserved key and the
surface are the same mechanism at two ages — a platform-internal key that
reaches the machine in the job spec. Growing the one key (a string keeps the
old early-exit affordance; a dict names a fault) keeps one definition, one
validation path, and no second rail for a reviewer to learn.

**A fault spec that is honoured unconditionally once written into the job
spec.** Rejected: that is a fault surface that can fire against real
hardware. The launch guard and the trainer-env gate are the price of the
surface being safe to ship before its recoveries.

**Simulating the recoveries' behaviour in the fake rather than the faults.**
Rejected: the fake's job is to make the failure happen and let the run's own
machinery act on it, so a recovery written later tests against the same fake
and the same codes. Where a fault would pre-empt a recovery, the fake does not
— `divergence` narrates the meaningless loss and lets the run complete with a
worthless result, exactly as the sabotaged real trainer would, so the abort
stays the divergence recovery's (#36) job on both tiers.

**Allowing provider-side faults on the real tier, letting the run complete
normally while its history claims a deliberate break.** Rejected: no real
provider honours a provider-side fault, so the "Simulated fault injected"
event would be a false claim — the naming guarantee firing the wrong way.
Provider-side faults are refused on the real tier (`fault_not_causable`);
only the fake tier can cause them.

## Consequences

- The six failures can be caused on demand at zero cost and, for the
  trainer-side faults, genuinely on hardware. Every recovery ticket that
  follows (#35, #36, #60, #61, #34) is verifiable by turning a fault on and
  watching the recovery happen, which is the whole point.
- `simulated_failure_code` now means two things, with the string form kept
  for the pre-existing failed-job journey; the surface supersedes it as the
  thing a user can be offered honestly, and the docs say so.
- A fault-spec job is refused loudly when the surface is off, which is a
  behaviour a caller could previously have had silently ignored; the refusal
  is the safety property working, not a regression.
- The fake provider gained fault-reading behaviour that only fires on a dict
  spec; the canned success and the reserved string path are unchanged, so
  existing journeys and tests are unaffected.

## Rollback

Remove the dict branch from the fake provider and the trainer's env reading,
drop the launch guards in `jobs.create` and `orchestrator.run_job`, and the
string form of `simulated_failure_code` reverts to being the only form; the
contract file and `docs/fault-surface.md` are documentation of a surface that
no longer exists and can be deleted with it.
