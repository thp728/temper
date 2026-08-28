# ADR-0044 — The server-rendered pages are deleted, and the two interfaces never coexist

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/007-the-application-shell.md`
- **Issue:** [#47](https://github.com/thp728/temper/issues/47)
- **Completes the promise made in:** [ADR-0023](0023-the-interface-consumes-a-client-generated-from-the-api-contract.md)

## Context

Spec 007 names a property the repository must not have: two interfaces for one
product. Its rule is that the old pages are deleted *in the change that ports
the last screen*, so there is never a window where both exist. ADR-0023
recorded the forward promise — "when the last screen lands, the old pages and
their templates are deleted in that same change" — and the proxy entries that
made the interim state explicit in `next.config.ts`.

The port is complete: upload and report (#28), model choice and launch (#38),
the job list and record (#40), and live watching (#39) are all ported, and
since #39 the only remaining rewrite is the API's own `/v1/:path*`. Two
earlier issues left the deletion half-done and said so: #39 deleted the old
watch-page proxies, and #59 noted that `apps/control-plane/.../web.py` was not
physically deleted. What remained was the full server-rendered surface — the
`web.py` router, its seven Jinja templates, and its stylesheet — still wired
into the FastAPI app, still rendering HTML, still covered by a dedicated test
file that drove the old routes.

## Decision

**The server-rendered pages are deleted, and the control plane becomes an API
only.** Concretely:

- `web.py` and its router are removed, and `app.include_router(web_router)` is
  deleted from `main.py`. The API no longer renders HTML: the routes `/`,
  `/upload`, `/datasets/{id}`, `/jobs`, `/jobs/new`, `/jobs/{id}`,
  `/jobs/{id}/cancel` and `/static/styles.css` no longer exist. The next.js
  rewrite list was already just `/v1/:path*`, and stays that — no route serves
  the old interface, inside or outside the shell.
- The templates directory and `static/styles.css` are deleted with it.
- The tests that covered the old pages (`tests/test_web.py`, plus the one
  API-suite test that hit the old `/jobs/new` page) are removed, and every
  behaviour they asserted is asserted somewhere against the new application —
  mapped below, because deleting a test is not retargeting it.
- The README and the control plane's `AGENTS.md` are updated to say the
  interface is the shell, so a reader is not asked which surface is real.

This keeps the repo's position on ADR-0023's second property: the interface
consumes a client generated from the API contract, and the two interfaces
**never** coexist — no longer a transition promise but the state of the
repository.

## Alternatives considered

**Keep the old pages.** They worked, cost nothing, were walked through a full
journey on real hardware, and a reader can argue a server-rendered page is
easier to review than a build step. Rejected on Spec 007's own terms: two
interfaces in one repository is a question with no good answer, a fix made for
one surface has to be made for the other, and a maintained-but-unreached
router (`web.py` had kept accumulating Phase B fixes — streaming validation,
the token count) is a second product being maintained in secret. The old
surface also rendered from the same code paths, so keeping it gives reviewers
a second thing to keep honest for no user.

**Delete the pages but keep the tests, "just in case".** Rejected: a test
suite whose fixtures construct the pages it drives is a second interface under
test. The port's tests already assert the user-visible behaviour against the
shell, and the mapping below shows nothing was silently dropped.

**Keep `web.py` around as a reference for the port.** Rejected: the port is
complete and behaviour-preserving (that is what the test mapping argues); a
deleted interface kept as reference is dead code a later reader has to
distinguish from live code, which is exactly the "which is real" confusion the
never-coexist rule exists to prevent. Git history is the reference.

## Consequences

- The control plane is a `/v1` API. Its OpenAPI document is the complete
  description of what a client can do; there is no second surface to keep it
  honest with.
- `tests/test_web.py` is gone. Its behaviours now live as follows:

  | Old assertion (removed) | Where it is asserted now |
  | --- | --- |
  | Upload page offers a labelled file form | `UploadForm.test.tsx` (labelled control + named submit) |
  | Form upload redirects to the report | `UploadForm.test.tsx` (pushes to `/datasets/{id}`); e2e upload journey |
  | Report shows counts, thinking mode, preview | `ReportView.test.tsx`; e2e upload journey |
  | Rejected report names lines and codes | `ReportView.test.tsx`; e2e upload journey |
  | Mixed thinking-mode block explained | `ReportView.test.tsx` (new: `mixed_thinking` code + plain-language message) |
  | Warning present, proceed still offered | `ReportView.test.tsx`; e2e upload journey |
  | Unknown dataset report 404s | `test_api.py::test_missing_dataset_is_404` (new); the shell renders "Not found" for it |
  | Validating dataset shows progress | `ValidationProgressView.test.tsx` |
  | Oversized upload shows its code | `UploadForm.test.tsx` (`dataset_too_large`) |
  | No-filename upload refused by name | `test_api.py::test_an_upload_without_a_filename_is_refused_by_name` (same handler, `main.upload_dataset`) |
  | Model choice lists licence and pinned revision | e2e launch journey (licence + 40-hex revision on the card); `test_api.py` catalog tests |
  | Frozen hyperparameters shown | e2e launch journey (spec section) |
  | Spec says it freezes at launch | e2e launch journey |
  | One launch action | e2e launch journey (`toHaveCount(1)`) |
  | Feasibility warning before launch | e2e launch journey; `test_api.py` launch-preview tests |
  | Launch creates a job | e2e launch journey (URL `/jobs/job_`); `test_api.py` job tests |
  | Invalid/unknown dataset refused at launch | e2e launch journey; `test_api.py` launch-preview tests |
  | Watch shows state, elapsed, machine, price | `JobRecordView.test.tsx`, `RunningJobView.test.tsx` |
  | Latest loss with its step | `JobRecordView.test.tsx`, `RunningJobView.test.tsx`; e2e running journey |
  | State change announced to screen readers | `RunningJobView.test.tsx` (new: the State region is `aria-live="polite"`) |
  | Unknown job's record 404s | `test_api.py::test_missing_job_is_404`; the shell renders "Not found" for it |
  | Cancel offered with its consequence stated | `RunningJobView.test.tsx`; e2e running journey |
  | Cancel offered in every working state, spend visible while accruing | `RunningJobView.test.tsx` (machine line appears as provisioning reports it) |
  | Terminal jobs hide the cancel control | `JobRecordView.test.tsx` (new) |
  | Completed job offers the adapter with its config | `JobRecordView.test.tsx`; e2e finished journey (zip download) |
  | Failed job shows its code, keeps its log | `JobRecordView.test.tsx`; e2e finished journey |
  | Cancelled reads as a decision, not a defect | `JobRecordView.test.tsx`; e2e running journey |
  | Stalled job says it was stalled | `JobRecordView.test.tsx` (`gpu_stalled` + safety-limit wording) |
  | Over-long job says it hit the ceiling | `JobRecordView.test.tsx` (new: `gpu_max_duration_exceeded` + reason) |
  | Machine-destroyed confirmation visible | `JobRecordView.test.tsx` (failure message carries "machine destroyed") |
  | Job list shows each job with its outcome | `JobsView.test.tsx` (complete / failed / cancelled, new) |
  | List links to each record | `JobsView.test.tsx`; e2e finished journey |
  | Jobs list reachable from the upload page | e2e journeys navigate through the shell header |
  | Empty list says so and offers the way in | `JobsView.test.tsx` |
  | Finished record keeps full event history | `JobRecordView.test.tsx` |
  | Live updates without refresh | `RunningJobView.test.tsx`; e2e running journey (stream) |

  The old "no JavaScript on the pages" assertions are the one family not
  carried forward as written, because they tested the old architecture, not a
  user-visible behaviour of the shell: a SPA uses scripting. What they were
  guarding — that a terminal record carries no polling machinery — survives as
  `JobRecordView.test.tsx` (no `script`/meta-refresh on a terminal record) and
  as the stream hand-back (`RunningJobView.test.tsx`).
- The `just check` gate no longer boots a server that renders HTML; the e2e
  journeys drive the shell, which is the surface Spec 007 requires every flow
  to be verified through.
- **Verifying criterion four surfaced a real defect in the shell, fixed here.**
  The report page's token count (issue #42) is produced by a background
  phase, and the landing update used `<meta http-equiv="refresh" content="2">`.
  A meta-refresh is scheduled when its tag is parsed and is *not* cancelled by
  a later client-side navigation, so a user who launched and moved on to the
  job page was yanked back to the report ~2s later — the browser navigated the
  whole tab to `/datasets/{id}` mid-launch, abandoning the running job page.
  That made the launch journeys flaky (intermittent, timing-dependent, present
  on the base commit) and is a direct breach of "the whole journey remains
  walkable". The counting update now rides on a client component
  (`TokenCountPoll`) that lives in the page's React tree: while the user is on
  the report it re-renders the route on the same cadence the meta-refresh
  used, and when the user navigates away the component unmounts and the poll
  stops, so nothing can navigate the tab out from under the next page.
- `jinja2` remains a declared dependency of the control plane though nothing
  imports it now; removing it from `pyproject.toml` and the lockfile is
  deferred to avoid lockfile churn colliding with the parallel wave.

## Rollback

Restore `web.py`, the `templates/` and `static/` trees and
`app.include_router(web_router)` from git history, and the old pages serve
again. The tests removed here are in the same history. Nothing else needs to
change: the shell and the API are unaffected by their absence.
