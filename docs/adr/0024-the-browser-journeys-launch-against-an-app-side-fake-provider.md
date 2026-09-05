# ADR-0024 — The browser journeys launch against an app-side fake provider

- **Status:** accepted
- **Date:** 2026-08-26
- **Spec:** `docs/specs/007-the-application-shell.md`
- **Issue:** [#38](https://github.com/thp728/temper/issues/38)

## Context

Spec 007's testing rule puts the journeys in `just check`: "these run without
hardware and without money, which is what makes them runnable on every push."
Up to #28 that was cheap to honour, because the ported journey (upload and
validate) never touches the provider. Issue #38 ports model choice and
**launch** -- the first browser surface whose acceptance criteria cannot be met
without driving the money path. A launched job goes through
`orchestrator.launch`, which builds a real provider by default; on a machine
holding credentials (`spike/.env` is read at import), a Playwright run could
provision real machines and bill against the account.

The suite-wide rule has always been "no test constructs a real provider"
(`conftest.py` enforces it by refusing `_connect`). But conftest guards pytest;
a live uvicorn serving Playwright is outside its reach, and `reuseExistingServer`
makes a developer's plain dev server indistinguishable from the one Playwright
boots.

## Decision

**The control plane grows one explicit, off-by-default switch:
`TEMPER_FAKE_PROVIDER=1` substitutes the in-package `FakeProvider` for any
launched job, and `/health` advertises which implementation is in force.**
Playwright boots the backend with the switch set, and every launch journey
refuses to begin unless `/health` answers `provider: "fake"`.

Three properties make this safe rather than merely convenient:

- **Off by default.** `just dev`, production and every other entry point are
  unchanged; nothing about the real path moves.
- **The stub is the app's own double**, not a network mock: the same API, the
  same database, the same orchestrator transitions. The journeys prove the
  product's behaviour, not a recording of it.
- **The guard is enforced where the mistake would happen.** A reused server
  without the switch fails the journey before any launch, with a message that
  says what to stop -- not after a VM exists.

## Alternatives considered

**Let the launch fail naturally on missing credentials.** Rejected twice over:
on credential-less machines it makes "launch" mean "watch it fail", which tests
the failure path while claiming to test the feature; and on machines *with*
credentials -- the maintainers' -- it provisions real machines.

**Mock at the HTTP layer with Playwright's `page.route`.** Rejected: faking
`/v1/jobs` responses stubs the API out of its own test. The shell's contract
with the control plane is exactly what these journeys exist to exercise, and a
route mock proves only that the mock works.

**A separate test-only ASGI app wrapping `main.app` with patched
orchestration.** Rejected as ceremony: two entry points to keep in agreement,
where one environment variable reads identically everywhere and is visible in
`/health`.

## Consequences

- The launch journeys complete a job end to end -- queued through `complete`,
  adapter downloadable -- on every push, for zero rupees.
- `config.FAKE_PROVIDER` is a lie about compute only: everything beside the
  provider is real, so the journeys' verdicts stay meaningful.
- `/health` now publishes which provider is in force; anything that drives
  launches programmatically can ask before spending.

## Rollback

Remove the flag read in `provider.new_provider()`, the `/health` field and the
journey-side guard together. The FakeProvider itself stays: it predates this
decision and the pytest tier depends on it independently.
