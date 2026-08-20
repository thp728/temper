# Spec 003 — Phase A user interface

**Status:** ready for tickets
**Phase:** A (target: Fri 2026-08-21)
**Depends on:** Spec 001 (the watch page needs the streaming loop and the cancel action)
**Produces:** ADR-0003 (fp32 adapter artifacts)
**Assumes:** ADR-0001 (event channel over SSH stdout), produced by Spec 001

## Problem Statement

The product's entire surface is an HTTP API. The checkpoint asks whether **a user** can go from a dataset to an adapter through the product, and a curl transcript does not answer that question — it demonstrates that the API works, which is a different claim.

The gap is sharpest at validation. A rejected dataset produces line-numbered errors naming exactly which rows to fix, and that report is currently reachable only as a JSON response body. It is the most useful thing the product produces before a run starts, and it is the point where users are most likely to give up, and it is invisible to anyone not reading raw JSON.

## Solution

Four server-rendered pages covering the whole journey: upload, choose, watch, collect. No JavaScript framework, no build step, no component library.

This interface is **deliberately disposable**. Phase B replaces it with a typed single-page application, and building that on the last contiguous day before the checkpoint would spend the day on tooling rather than on the journey. The cost of throwing this away is a few hours; the cost of not having it is that the checkpoint's question goes unanswered.

From the user's perspective:

- Upload a dataset and see the validation report as a readable page, with errors against their line numbers.
- Choose a base model, see the frozen job specification, and launch.
- Watch the job: state, elapsed time, live log output, the latest loss, and a cancel control.
- Collect the adapter, or read why the job failed.

## User Stories

1. As a user, I want to upload a dataset from a web page, so that I can use the product without an HTTP client.
2. As a user, I want to see immediately whether my dataset was accepted, so that I know if I can proceed.
3. As a user whose dataset was rejected, I want each error shown against the line number it came from, so that I can fix the file without guessing.
4. As a user whose dataset was rejected, I want the errors to be readable text rather than raw JSON, so that I do not need to parse a response body.
5. As a user whose dataset produced warnings but is usable, I want to see the warnings and still be able to proceed, so that advice does not become an obstacle.
6. As a user, I want to see how many rows were found and how many are usable, so that I can tell whether the file parsed as I expected.
7. As a user, I want to see a preview of the first few rows as the product understood them, so that I can confirm the schema was read correctly.
8. As a user, I want to be told whether thinking mode was detected in my dataset, so that I understand how the model will be trained.
9. As a user whose dataset mixes thinking and non-thinking rows, I want the blocking error explained on the page, so that I understand why an otherwise valid file was refused.
10. As a user, I want to choose from the available base models on a page, so that I do not need to consult the catalog endpoint separately.
11. As a user choosing a model, I want to see its licence and pinned revision, so that I know what I am training and under what terms.
12. As a user, I want to see the hyperparameters my job will use before I launch, so that there are no surprises after I have committed to spending.
13. As a user, I want to understand that the job specification is frozen at launch, so that I know a later change cannot retroactively alter what my job did.
14. As a user creating a job from a very large dataset, I want the feasibility warning shown before I launch, so that the warning reaches me while I can still act on it.
15. As a user, I want a single obvious action to start the job, so that launching is unambiguous.
16. As a user watching a job, I want the current state shown prominently, so that I can tell at a glance where it is.
17. As a user watching a job, I want elapsed time shown, so that I can judge whether it is taking longer than expected.
18. As a user watching a job, I want log output to appear as it is produced, so that the page is evidence of progress rather than a spinner.
19. As a user watching a job, I want the newest output visible without scrolling, so that I see the current state first.
20. As a user watching a job, I want the latest loss value shown as a number, so that I can see the model is learning.
21. As a user watching a job, I want the machine type and hourly price shown, so that I know what I am spending.
22. As a user watching a job, I want a cancel control, so that I can stop a run I no longer want.
23. As a user who clicks cancel, I want clear confirmation that cancelling produces no adapter, so that the consequence is understood before it happens.
24. As a user watching a job that ends, I want the page to show the terminal outcome without me refreshing manually, so that I am not left staring at a stale page.
25. As a user whose job succeeded, I want a download link for the adapter, so that I can collect the thing I paid for.
26. As a user who downloads an adapter, I want the accompanying configuration file too, so that the adapter is loadable rather than just present.
27. As a user whose job failed, I want the reason shown in plain language with its stable code, so that I can act on it or report it.
28. As a user whose job failed, I want the log output still available after failure, so that I can see what happened before the failure.
29. As a user whose job was cancelled, I want the page to say so distinctly from failure, so that my own action is not presented as a defect.
30. As a user whose job was killed for stalling or exceeding the duration limit, I want that stated specifically, so that I understand it was a safety limit.
31. As a user, I want to see confirmation that the machine was destroyed, so that I trust I am no longer being billed.
32. As a user, I want a list of my previous jobs, so that I can return to a completed run and download it again.
33. As a user returning to an old job, I want its full event history, so that a finished run is as inspectable as a running one.
34. As a user on a keyboard, I want to complete the whole journey without a mouse, so that the product is usable without pointing.
35. As a user with a screen reader, I want the state of a running job announced when it changes, so that progress is perceivable without sight.
36. As a user, I want pages to work without JavaScript enabled for everything except live updates, so that the core journey does not depend on scripting.

## Implementation Decisions

### Scope and shape

Four pages, server-rendered from the existing control plane, sharing one stylesheet:

1. **Upload and report** — the upload form and, after submission, the validation report: counts, detected thinking mode, row preview, and errors and warnings against their line numbers.
2. **Create job** — model selection with licence and pinned revision, the hyperparameters to be frozen, any feasibility warning, and the launch action.
3. **Watch** — state, elapsed time, machine type and price, live log tail, latest loss, and the cancel control.
4. **Jobs list** — previous jobs with their outcomes, linking back to the watch page, which doubles as the record for a finished job.

The watch page and the finished-job page are **the same page**. A job's record is the same thing during and after the run; splitting them would duplicate rendering and create two places for the outcome to be described differently.

### Templates and styling

Server-rendered templates from the existing control plane, with one hand-written stylesheet. No component library, no build step, no bundler. The interface is disposable by design and should not acquire tooling that outlives it.

### Live updates

The watch page polls the existing events endpoint, which already supports fetching only events after a given identifier. Polling is a deliberate choice for Phase A: server-sent events with replay are Phase B's work and depend on infrastructure that does not exist yet.

Polling stops when the job reaches a terminal state. **The teardown confirmation is visible before that point** — Spec 001 reorders it precisely so that a client which stops on a terminal state does not miss it.

Everything except live log updates works without JavaScript. The page renders the current state server-side on each load, so a user with scripting disabled sees a correct, if static, view.

### Metrics display

The latest loss is shown as a number, with its step. No chart. Metric events are already structured, so adding a chart later reads the same data — but drawing one on the last contiguous day before the checkpoint is the wrong use of the day.

### Accessibility

Semantic markup, labelled form controls, a visible focus order that matches reading order, and a live region for the job state so that changes are announced rather than only rendered. This is a baseline, not the full audit; the audit is Phase B hardening.

### Error presentation

Every error shown to a user carries its stable code alongside the human-readable message. Validation errors carry their line number. This is the same contract the API already honours; the interface must not weaken it by summarising errors into prose that loses the reference.

### Decision records

This spec owns one decision record, because this is where the decision becomes visible to a user: the adapter download is the first place anyone sees how large the artifact is.

- **ADR-0003 — Adapters ship as fp32.** That the artifact is exactly the weights that were trained rather than a post-hoc cast; that the size consequence is roughly double the alternative and is accepted deliberately; that the behaviour is standard for the quantised training path rather than a defect; and that a reduced-precision export becomes an explicit user choice in Phase B, recorded per job, rather than a silent default.

This record documents behaviour that already exists and is not changing. It is written because the reasoning is currently nowhere, and an undocumented artifact size reads as an oversight rather than a decision. The alternative rejected — casting on save to halve the download — is the one a reader will ask about, so it belongs in the record explicitly.

## Testing Decisions

**What makes a good test here.** Tests assert that a user-visible fact reaches the page: that a rejected line's number appears in the rendered report, that a failed job's error code appears on its page, that a cancel control is present for a running job and absent for a finished one. They do not assert markup structure, class names, or layout.

**Prior art.** The existing control-plane suite uses a test client against the application; these tests use the same client to request pages and assert on their content. The provider fake from Spec 001 drives job states without provisioning anything.

**Modules under test.** The page handlers, through the HTTP seam. No separate rendering layer is tested in isolation — the template and its handler are one unit from the user's perspective, and testing them separately would test the templating engine.

**Cases that must exist.**

- An invalid dataset's report page contains the offending line numbers and the error codes.
- A mixed thinking-mode dataset's report page explains the block.
- A valid dataset with warnings renders the warnings and still offers the launch action.
- The create-job page shows each catalog model with its licence and revision.
- A large dataset's feasibility warning appears before launch, not after.
- The watch page renders each non-terminal state, and shows the cancel control for them.
- The watch page for a completed job offers the adapter download and hides the cancel control.
- The watch page for a failed job shows the error code and the retained log output.
- The watch page for a cancelled job describes it distinctly from failure.
- Every page renders with scripting unavailable.

## Out of Scope

- **The Phase B interface.** Typed single-page application, component library, data-fetching layer and charts are all Phase B, and this interface is expected to be deleted when they arrive.
- **Server-sent events.** Phase B, with the pub/sub infrastructure it requires.
- **Loss curve charts.** The data is structured for it; drawing is Phase B.
- **Authentication and any per-user view.** A sanctioned gap for the whole project.
- **Dataset management** beyond upload and report — no renaming, deleting, or re-validating.
- **Re-running a job with modified hyperparameters.** Phase B.
- **A full accessibility audit.** Baseline semantics now; audit in Phase B hardening.
- **Visual design.** Legible and unstyled beats decorated and disposable.

## Further Notes

The validation report is the highest-value page in this set and the reason the interface is worth building at all. It is the product's best existing work — line-numbered, coded, actionable — and it is currently invisible to anyone who is not reading a JSON response by hand.

The disposability is a deliberate, recorded trade rather than an oversight, and should be described that way if the interface is discussed: it exists to answer the checkpoint's question on the day the checkpoint is asked, and its replacement is already specified.
