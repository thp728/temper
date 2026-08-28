# ADR-0057 — Teardown is confirmed across consecutive observations

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/010-recovery-and-fault-injection.md`
- **Issue:** [#34](https://github.com/thp728/temper/issues/34)

## Context

After a destroy the provider's listing is eventually consistent. Spike 5
measured it and `spike/teardown.py` recorded the correction as C17: the
machine reads absent, then reappears as `destroying`, then goes absent for
good. The control plane's confirmation read at the first absence and called
it done. That is a check that succeeds during the gap in which the machine
is still billing, and a reconciler that treated a `destroying` machine as a
stray would try to destroy it again — or worse, count it as leaked — while
the provider was still acting on the first request.

A `destroy` that the provider refuses was retried three times and then asked
once whether the machine was still listed. An orphaned GPU bills until
someone notices, so the path that cannot destroy a machine must be the
loudest in the system rather than a silent return.

None of this had been exercised against a destroy that actually failed.
`#24` built the fault surface deliberately before the recoveries and gave
this ticket the two faults it needed: `destroy_refused` (the provider
refuses a destroy) and `orphan` (a machine left with no job that owns it).
The criterion "the path is exercised against a fault that makes the provider
refuse" is that fault, and it exists now.

## Decision

**Teardown is confirmed by absence across consecutive observations, and a
machine reported as `destroying` is treated as not yet confirmed rather than
as a stray.**

- **Confirmation requires `TEARDOWN_CONFIRM_SAMPLES` consecutive absent
  listings.** The constant is three, the same number `spike/teardown.py`
  uses, and the interval and timeout are `TEARDOWN_CONFIRM_INTERVAL_S` and
  `TEARDOWN_CONFIRM_TIMEOUT_S`. An absent that is followed by a present
  resets the counter; only a run of absences proves the machine is gone.
  This closes the gap in which a destroyed machine transiently reads absent
  before reappearing as `destroying`.

- **`destroying` is present, not a stray.** The provider seam now exposes
  `list_machines()` returning `Machine` with a `status` field
  (`running`|`destroying`), and `list_machine_ids()` is derived from it.
  `JarvisLabsProvider.list_machines` maps the SDK's `status`/`state` field
  through that vocabulary; `FakeProvider` can report `destroying` for a
  configurable window after a successful destroy or be scripted via
  `list_sequence` to emit the exact absent→destroying→absent shape. The
  orchestrator's confirmation loop treats `destroying` as present — it logs
  "still destroying — not yet confirmed" and resets the consecutive counter
  — rather than emitting a `STRAY` or counting it as confirmed. The
  reconciler will read the same status for the same reason.

- **A destroy the provider refuses is retried and escalated.** The retry
  loop is three attempts with `DESTROY_RETRY_DELAY_S` between them, each
  failure recorded as "Destroy attempt failed". When the attempts are
  exhausted the job records "destroy refused after 3 attempts: …; still
  billing — manual removal required". Confirmation still runs, and if the
  machine remains listed the loop times out and records the loud terminal
  signal: `STRAY MACHINE {id} still listed — destroy it manually, it is
  billing`. A transient failure (one refused, then accepted) retries and
  does not escalate.

- **An undestroyable machine is recorded loudly.** There is no silent
  forgetting. The two loud signals are the per-attempt errors and the final
  `STRAY … it is billing` error, both with the machine id and the billing
  warning. They precede the terminal state transition so a client that stops
  polling at terminal still sees them.

- **Exercised against `destroy_refused`.** The zero-cost provider's
  `destroy_refused` fault (times=N, stays_listed) makes the fake refuse
  `destroy` N times and stay listed. The new `test_teardown_confirmation`
  drives the orchestrator against it and asserts the retry count, the
  escalated destroy-refused message, and the `STRAY` billing warning; the
  existing `test_a_refused_destroy_leaves_the_machine_reported_loudly` now
  passes through the consecutive-confirmation path. The eventual-consistency
  shape is exercised by scripting `list_sequence` to return absent, then
  present, then consecutive absent, and by scripting an explicit
  `destroying` status.

## Alternatives considered

**A single absent observation as confirmation.** Rejected: it is the known
weakness this ticket closes. Spike 5 observed the absent→destroying→absent
window; a single sample reads the gap as success and reports teardown
complete while the machine is still billing.

**Treat `destroying` as absent (or as a stray to be destroyed again).**
Rejected: `destroying` is the provider acknowledging the destroy and still
acting on it. Treating it as absent declares success early; treating it as a
stray makes the teardown loop or a reconciler issue a second destroy against
a machine already being destroyed, which is noisy at best and double-counting
at worst. "Not yet confirmed" is the honest state.

**Retry once or not at all.** Rejected: a single transient provider error
must not be what leaves a machine billing. Three attempts with a back-off is
what the spikes used and what the existing tests assert; keeping it preserves
that guarantee while the new loud escalation replaces the previous silent
single-check.

**Silent forgetting of an undestroyable machine.** Rejected: an orphaned GPU
bills until someone notices. The cost of this criterion being fudged is real
money, measured per minute in INR (L4 at ₹41.31/hr, `account.currency()`).
The `STRAY … it is billing` error exists so the failure is unmistakable.

**A fault-injection framework as a new seam.** Rejected for the same reason
as ADR-0051: provider-side faults belong to the provider seam. `destroying`
and the absent→destroying→absent sequence are simulated by extending the
existing `FakeProvider` (`list_sequence`, `destroying_for`) rather than
adding a new module.

## Consequences

- `Provider` now has `list_machines() -> Sequence[Machine]` with `Machine.status`;
  `list_machine_ids()` remains as a derived view so existing callers keep
  compiling. `JarvisLabsProvider` maps the SDK's instance state to that
  vocabulary; the fake provider honours `list_sequence` and `destroying_for`
  so the confirmation rule can be tested without hardware.
- `orchestrator._teardown` retries `destroy`, logs each failure, escalates
  when exhausted, then polls `list_machines` until three consecutive absences
  or a timeout, treating `destroying` as present. The `STRAY` escalation
  remains the loud billing warning and still precedes the terminal transition.
- `test_teardown_confirmation.py` exercises the four behaviours the issue
  names — consecutive confirmation, destroying-as-not-yet-confirmed, retry
  and loud escalation, and the `destroy_refused` fault — on the zero-cost
  provider; the real-hardware proof remains outstanding by design and is
  stated as such.
- A teardown now takes up to `TEARDOWN_CONFIRM_TIMEOUT_S` (30s in production,
  patched to 2s in tests) rather than a single listing. The cost is a few
  seconds per job for a guarantee that the previous single sample could not
  give.

## Rollback

Revert `provider.Machine.status`, `Provider.list_machines`,
`FakeProvider.list_sequence`/`destroying_for`, and `orchestrator._teardown`
to the single-check form; the three constants and
`test_teardown_confirmation.py` can be deleted. Jobs will again confirm at
the first absent and will not distinguish `destroying` from absent.
