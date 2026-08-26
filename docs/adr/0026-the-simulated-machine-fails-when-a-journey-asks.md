# ADR-0026 — The simulated machine fails when a journey asks it to

- **Status:** accepted
- **Date:** 2026-08-26
- **Spec:** `docs/specs/007-the-application-shell.md`
- **Issue:** [#40](https://github.com/thp728/temper/issues/40)
- **Amends:** [ADR-0024](0024-the-browser-journeys-launch-against-an-app-side-fake-provider.md) (which it does not supersede)

## Context

Spec 007's testing decisions name "land on a failed job and read the reason"
as one of the journeys that must run on every push, and issue #40's acceptance
criteria require browser coverage of reading a failure. ADR-0024's switch made
launches drivable, but `completed_run()` only ever succeeds: nothing in main
could produce a failed job without hardware. Issue #24 ("failures can be caused
on demand") owns product-level fault injection and is still open, so the gap
had to be filled at the same tier as the canned success -- the app's own double
-- rather than waited for.

## Decision

**The `FakeProvider` instance behind `TEMPER_FAKE_PROVIDER` reads the jobspec
out of every script it receives, and when the spec's hyperparameters carry
`simulated_failure_code`, the machine ends early through the ordinary result
document with that code.** The failure travels the real path -- coded result,
`OrchestratorError`, coded terminal state, plain-language message frozen into
the record -- so the finished-job view cannot tell a requested failure from a
real one by shape. The key is documented in `fake_provider.py` and honoured
nowhere else; a real trainer refuses unknown keys loudly, so launching with it
against real compute produces an honestly-failed job too.

## Alternatives considered

**Two backends under different env selections (one always succeeding, one
always failing).** Rejected: Playwright boots one control plane per run, the
Next rewrite targets exactly one origin, and per-spec server selection would
rebuild Playwright's webServer model around a testing convenience.

**Failing naturally -- deleting the stored dataset file mid-run.** Rejected:
the resulting record carries `internal_error` and a Python exception string
containing an absolute filesystem path, which is neither the plain language the
acceptance criterion asks for nor something a page should be driven to render.

**Waiting for #24.** Rejected: the failed-job journey is this spec's gate, and
a gate nobody can pass gets quietly abandoned.

## Consequences

- The failed-job journey runs on every push for zero rupees, asserting on what
  a user actually reads: stable code, plain-language reason, retained history.
- One reserved hyperparameter exists that is test affordance rather than
  product surface; #24 replaces it with something a user can be offered
  honestly, and until then it stays out of every UI.
- The events endpoint now publishes typed models (`JobEvent`, `EventPage`) --
  not a new decision, simply ADR-0023's rule applied to the last untyped job
  endpoint the finished-job view consumes.

## Rollback

Remove the reserved-key branch from the fake and the journey that uses it. The
canned success of ADR-0024 is untouched.
