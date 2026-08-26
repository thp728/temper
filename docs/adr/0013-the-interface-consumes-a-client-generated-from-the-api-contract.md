# ADR-0013 — The interface consumes a client generated from the API contract

- **Status:** accepted
- **Date:** 2026-08-26
- **Spec:** `docs/specs/007-the-application-shell.md`
- **Issue:** [#28](https://github.com/thp728/temper/issues/28)

## Context

Spec 007's rule for keeping the interface honest with the server: a field that
changes shape must break a **build**, not a page. That requires the browser
side to have no hand-written knowledge of response shapes — which in turn
requires the API to *publish* shapes.

It did not. `POST /v1/datasets` returned `{"id": ..., "filename": ..., **
report_dict}`, and both `GET /v1/datasets/{id}` and `GET /v1/datasets`
returned the raw database rows, filesystem `path` included. An OpenAPI
document generated from those handlers
describes no schema at all: any client generated from it types every dataset
response `unknown`, and the "generated" client degenerates into hand-written
types with a generator attached — the drift it exists to prevent, laundered
through tooling.

## Decision

**Three things, shipped together, because any one alone is decorative:**

1. **The dataset endpoints publish Pydantic models**
   (`contracts_models.py`: `ValidationIssue`, `PreviewTurn`, `PreviewRow`,
   `DatasetReport`, `DatasetUploaded`, `DatasetRecord`, `DatasetList`). The
   contract now describes the shape of every dataset endpoint; FastAPI
   validates responses against it, so the document cannot silently
   diverge from what is sent. Publishing them removed `path` from every
   dataset response -- an absolute filesystem path reaching a browser was a
   leak of machine layout that no consumer used.
2. **`apps/web` generates its client from the checked-in contract with Orval**
   (ADR-0011's choice). The generated output is gitignored; the transport is a
   small hand-written mutator (`src/lib/api/mutator.ts`) that owns fetch, the
   same-origin base URL, multipart boundaries, and typed extraction of the
   API's stable error codes. Types, paths and call signatures are never hand-
   written. The gate compiles against freshly generated output, so a breaking
   contract change fails `just check`, not a user.
3. **Unported screens are proxied through the shell's origin** (Next.js
   rewrites): `/v1/*` always, `/jobs/*` until issue #38 ports them. A journey
   therefore never crosses origins mid-flow, and the ported upload screen can
   hand off to the existing create-job page today without either surface being
   rebuilt early.

On the spec's third property — *the two interfaces never coexist*: they coexist
only inside this transition, and only because #28 ships one screen of five. The
rule stands as written: each later screen replaces its proxy entry in the same
change that ports it, and when the last screen lands, the old pages and their
templates are deleted in that same change. The proxy entries make the interim
state explicit in config rather than ambient in the codebase.

## Alternatives considered

**Hand-write a typed client now, generate later.** Rejected outright: it is
the drift, hired immediately.

**Generate from a live server.** Rejected in ADR-0011 already; restated here
because this change makes the checked-in file load-bearing: generation needs no
running control plane in CI, and contract changes appear as reviewable diffs.

**Have Next.js route handlers proxy the API instead of rewrites.** Rejected for
now: a route handler per path is code that must be kept in sync with nothing;
rewrites are declarative and fail loudly (500) when the control plane is down,
which during development is information, not noise.

## Consequences

- The API's response models are now part of its public behaviour; changing a
  field name fails the web build. This is the point.
- Two sources of truth exist for one shape (`validation.Report.to_dict()` and
  `DatasetReport`) bound by a test
  (`test_upload_response_is_exactly_the_published_model`). When Phase B moves
  reports into `packages/contracts/` (#82), the duplicate collapses there.
- The web toolchain (pnpm via corepack, Node 22) becomes a prerequisite for
  `just check`. Cold-clone cost is accepted by Spec 007 and mitigated by
  Spec 012's one-command stack gate.

## Rollback

Delete `apps/web`, remove the `response_model` arguments, regenerate
`packages/contracts/openapi.json`. Nothing else references the models; the old
pages never stopped working.
