# ADR-0071 — The zero-cost path is a labelled demonstration, structurally blind to the transport

- **Status:** accepted
- **Date:** 2026-08-30
- **Spec:** `docs/specs/012-clone-and-run.md`
- **Issue:** [#63](https://github.com/thp728/temper/issues/63)

## Context

A reviewer whose key is not registered, or who does not want to spend, has
nothing to see: every fact about the current state points the wrong way, and
the evaluation ends at the setup step. The existing double already implements
the whole compute interface — scripted output, staged failures, controlled
timing, held pauses ([ADR-0024](0024-the-browser-journeys-launch-against-an-app-side-fake-provider.md))
— and is a configuration switch away from being a front door.

But a demonstration mode that looks like a real run is the single most
dishonest thing this repository could ship. And it would be *specifically*
dishonest here, not merely in general: the defect that cost this project a
real run was a transport defect — a line-ending translation applied on the way
to a remote shell ([ADR-0027](0027-the-transport-is-proven-against-a-real-endpoint.md)) —
and the double is structurally blind to exactly that class, because it never
crosses a connection at all. A reviewer who only ever ran the zero-cost tier
and was not told this would have evaluated a puppet show.

## Decision

**The zero-cost tier is the existing double promoted, labelled in the
interface on every page it serves, with its limitation stated specifically
rather than as a general caution.**

- **A single setting selects it, and nothing else differs.** `TEMPER_FAKE_PROVIDER`
  is the one switch; the same API, the same database, the same orchestrator
  transitions, the same interface. `models` resolution, the remote-dataset
  seam and the tokenizer all follow it because they already did (ADR-0024,
  ADR-0028, ADR-0047). Every process that reads it — control plane, worker,
  web shell — parses it with the same boolean rule (`config._flag` and its
  mirror in `apps/web/src/lib/demo-mode.ts`), so the banner and the provider
  cannot disagree about which mode is in force, and the documented `=0`
  actually means the real tier. In the reviewer-facing one-command stack the
  value is written once, in a single compose anchor that all three services
  read (`compose.yaml`), so a flip to the real tier changes one line and
  cannot desynchronise the marking from the provider. An unrecognised value
  is refused by the backend at boot — a running stack cannot carry one — and
  read as the real tier by the shell, which is why the two halves are one
  parse rule with the refusal policy living where a misconfiguration stops
  the process rather than mislabels a page.

- **Every page in that tier is marked as a demonstration, and the marking
  cannot be turned off in that mode.** The marking is a server-rendered banner
  in the web shell's root layout, driven by that same single setting, so it
  rides every route and a screenshot of any page includes it. There is no
  dismiss control: a banner a visitor can dismiss is a banner that can be
  missing from the very screenshot meant to prove the mode was labelled
  (Spec 012's own rule).

- **The documentation names the specific limitation, and this record is part
  of that documentation.** The double never crosses a connection, so the
  zero-cost tier is structurally blind to transport defects — and a transport
  defect is exactly what broke a real run here. What the tier proves is the
  whole journey's shape: upload, validate, choose, launch, watch, collect.
  What it cannot prove is any claim about the transport, and nothing in the
  interface or the docs claims otherwise.

- **A sample dataset and one completed run are seeded so the tier has
  something to show immediately.** The seed is part of the mode, not a second
  setting: on the zero-cost tier, against an empty database, startup plants
  the checked-in sample (`samples/sample-chat.jsonl`) and drives one job to
  completion through the real `orchestrator.run_job` against the fake
  provider — the only way to guarantee the seeded record is shaped exactly
  like a run a reviewer would drive themselves, because it is one. A database
  that already holds a dataset or a job is never seeded, and the real tier
  never seeds.

- **Browser tests run against this tier on every push**, which the journeys
  already did (ADR-0024), and now assert the marking as well
  (`e2e/demo-marking.spec.ts`), so the demonstration path — and its labelling —
  cannot rot.

### The one interaction with ADR-0066

The seeded run is driven by `orchestrator.run_job` inside the control plane's
startup, which is the one place this repository calls the orchestrator from
that process. This is a deliberate, recorded exception, scoped so it cannot
reintroduce what ADR-0066 removed. ADR-0066's rule is that the *request path*
does not start threads and `jobs.create` never calls the orchestrator, because
a thread in that process is how a machine ends up billing unowned. A
synchronous, one-shot, startup-only call in the zero-cost tier has none of
those properties: the double crosses no connection and bills nothing, so there
is no machine to orphan and no spend to watch — which is precisely what the
worker exists to protect against. It is not the request path, it starts no
thread, it runs only in the zero-cost tier, and it runs only against an empty
database. Recorded here rather than hidden. **This record supersedes the
control-plane `AGENTS.md` sentence "This process does not run jobs." for this
one boot-time seed**; the module's own file is amended to say so, so a reader
who starts there meets the exception where the rule lives.

## Alternatives considered

**A hosted demonstration.** Rejected — Spec 012 names it out of scope: the
zero-cost mode runs on the reviewer's machine. Hosting it would also require
deploying the stack, which Spec 012 also names out of scope.

**A second implementation of the compute interface for the demo.** Rejected —
the double already implements the interface, and promotion means selecting it
by configuration, not writing a second thing to maintain and to drift.

**Marking only in the documentation.** Rejected on Spec 012's own words: the
interface says what it is, not only a file nobody reads. And a banner a
screenshot would not include is not a marking, which is why the banner is
server-rendered and non-dismissible.

**A dismissible or configuration-turned-off marking.** Rejected — the marking
cannot be turned off in that mode, by construction and by review.

**Seeding a hand-written terminal record at the database layer.** Rejected —
a hand-written row is a second source of truth that would drift from what a
real run produces, which is the failure mode ADR-0010 exists to prevent. The
seed runs the real path instead.

**A separate "seed the demo" setting.** Rejected — the mode and its
demonstration are one thing; a second knob is how the marking and the provider
come to disagree.

## Consequences

- The reviewer-facing one-command stack (`docker compose up`, issue #29) is
  now visibly a demonstration: every page carries the banner, and the stack
  boots straight to a sample dataset and a completed run to inspect.
- The zero-cost tier is a *journey* proof, never a transport proof. Claims
  about the transport rest on ADR-0027's real-endpoint tier and on the
  hardware tier; this record is the boundary that keeps those apart.
- The control plane's startup may take a moment longer on a fresh zero-cost
  boot (it drives one canned run to completion); a non-empty database skips
  it entirely.
- ADR-0066's separation is preserved everywhere except the single boot-time
  seed, and that exception is scoped to the zero-cost tier.

## Rollback

Revert this issue's commits: delete `seed_demo.py`, `apps/web/src/lib/demo-mode.ts`,
`apps/web/src/components/DemoBanner.tsx` and their tests, remove the banner
from the root layout and `TEMPER_FAKE_PROVIDER` from the web service in
`compose.yaml`, and restore `config.py`'s `FAKE_PROVIDER` to its earlier
`bool(os.environ.get(...))` line. The zero-cost tier remains selectable by
the single setting; it just stops showing something immediately and stops
saying what it is on every page.
