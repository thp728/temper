# ADR-0058 — The finished job's event history is paginated and disclosed

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/006-streaming-transport-progress-and-artifacts.md`
- **Issue:** [#56](https://github.com/thp728/temper/issues/56)

## Context

After #49, layer-pull and model-download lines are promoted into a superseding
`progress` kind rather than accumulating as events, with the raw lines retained
as job output. That removes the cause of the defect that #56 reports: a real
run wrote several hundred events, the overwhelming majority container-layer pull
progress, and the event store returns oldest-first with a cap of 500, so the
served page for a completed job stopped partway through the image pull and
never showed that the artifact was verified, the machine destroyed, or the job
finished.

Measurement on the current main after #49:

* **Typical completed run (journey fake `completed_run`):** 21 events — 5 state
  transitions, the 5 log lines that surround the pull/download (building image,
  held-out split, running training), 2 metric events, verification/checkpoint
  events, destroy and completion. Well below the 500 cap, so truncation no
  longer occurs. Full record fits on one page and the finished page shows its
  own outcome without paging.
* **Same run before #49 would have been ~31 events** (the 10 promoted pull/
  download lines as log events). A large real pull with hundreds of layers
  produced several hundred before; after promotion it would be two superseding
  progress rows plus the same ~21 events.

So the cause is gone. What remains is the guardrail the spec flags: *"where a
limit still applies, the interface says what it is showing and of how many"*
and *"nothing is hidden — the full record remains reachable"*. A silent
truncation is the defect; a stated one is a feature. The cap itself stays — a
job that genuinely produces very many events (e.g. verbose trainer output) must
still be paginated.

The boundary for #56 is the finished job page's event list, its pagination or
limit disclosure, and the reachability of the full record. Live watching is
#39's stream and must be unaffected.

## Decision

**The event page is paginated by `after`/`limit` with a disclosed total; the
finished record renders the disclosure and offers paging to the tail.**

- **API:** `GET /v1/jobs/{id}/events` gains `?after=` (cursor) and `?limit=`
  (1..500, default 500). The response gains `total: int` — the job's event
  count regardless of the window — so a client can state `"Showing 500 of 734
  events"` without a second request. `limit` >500 is refused with
  `invalid_limit`. Progress (`progress`) and retained output (`output`) travel
  on the same page unchanged — they supersede per phase, so they remain small
  by construction.

- **Persistence:** `db.count_events(job_id)` and `db.get_events(job_id,
  after_id, limit)` with `COUNT(*)`/`LIMIT`. The existing oldest-first order is
  kept; paging with `after=last_id` walks forward. `total` is computed per
  request; no new table.

- **Interface:** The finished record's `Output` region becomes `EventLog`
  (client component). It receives the first page (`events`, `total`) from the
  server and renders:
  * disclosure — `Showing N of M events` plus, when `N < M`,
    `— paginated (limit 500). The full record remains reachable.`
  * the log (role="log") of the events currently held,
  * when truncated, a `Load more events (M-N remaining)` control that fetches
    `?after=lastId` and appends the next page in chronological order,
    updating the disclosure.

  A typical 21-event run shows `Showing 21 of 21 events` with no control.
  A 734-event run shows `Showing 500 of 734 events — paginated …` and one
  click loads the remaining 234, including `Artifact verified`, `Machine …
  destroyed`, and `Training complete`. The full record is reachable without
  leaving the page, and the same `after` paging the live stream uses is reused
  — no second mechanism.

- **Live watching:** Unchanged. `RunningJobView` continues to consume the
  server-pushed stream over the durable log; the stream's polling loop is not
  paginated differently.

## Alternatives considered

**Return the newest 500 when truncated so the tail is visible by default.**
Rejected: it would hide the earliest events (queued, hardware selection) that
explain how the run started, and inconsistency with the `after` cursor the
stream and polls already use would create two ordering stories. The chosen
path keeps oldest-first everywhere; the tail is one page away and disclosed.

**Remove the cap for finished jobs (return all events).**
Rejected: the spec explicitly keeps the cap as a guardrail — a job that
genuinely floods must be paginated. Returning all would also regress memory
for a pathologically verbose trainer.

**Server-side fetch-all for the finished record (render everything despite the
cap).**
Rejected: same as above — it would hide the limit rather than disclose it.

**Separate total endpoint.**
Rejected: one request should be enough to render the disclosure; `total` rides
on the page.

## Consequences

- A typical promoted run (~21 events) renders without truncation; measurement
  is recorded in the PR body (before/after) and the `event count falls far
  enough that truncation no longer occurs` criterion is satisfied by evidence
  rather than by building a fix for a gone cause.
- Where the limit still applies, the page is not silent — disclosure plus
  paging satisfies `"where a limit still applies, the interface says what it is
  showing and of how many"` and `"nothing is hidden — the full record remains
  reachable"`.
- The contract gains `total`; the client is regenerated. The interface change
  is isolated to the finished record's event region; provenance manifest (#70),
  divergence detection (#36), teardown (#34), MoE label (#65) and live stream
  (#39) are untouched.

## Rollback

Revert `EventPage.total`, `limit`/`after` handling, and `EventLog` to the
plain log div; the finished page returns to silent oldest-first `LIMIT 500`.
The `total` column is derived, so no migration.
