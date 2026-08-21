# ADR-0006 — Validation runs off the event loop

- **Status:** accepted
- **Date:** 2026-08-21
- **Spec:** `docs/specs/002-dataset-transport-and-limits.md`
- **Issue:** [#9](https://github.com/thp728/temper/issues/9)

## Context

The upload handler was declared `async def` and then performed synchronous,
CPU-bound validation inside itself. An `async def` handler runs on the event
loop, so every millisecond of validation froze **every other request** — not
the uploading client's request, everyone's.

Validation cost was measured at roughly 80 ms per megabyte. A 200 MB upload —
half the new size limit — froze the entire server for about 16 seconds; a 1 GB
upload would freeze it for over a minute. At the sizes actually uploaded so
far (7.5 KB) this was invisible: sub-millisecond blocking reads as
responsiveness.

**This is recorded as a correction, not quietly fixed.** The handler was
written async because the endpoint *serves* an async framework, not because
anything in it awaits.

## Decision

**The upload handler is a sync (`def`) handler.** FastAPI dispatches sync
handlers to its worker pool, so a long validation blocks only its own request;
the event loop keeps serving everything else. This is a one-keyword change
with a measured 16-second blast radius at 200 MB.

A regression test pins it:
`test_upload_limits.test_large_upload_does_not_block_concurrent_requests`
stubs validation to block until released, verifies the server cannot answer
`/health` for longer than a second after the block begins, and fails if the
handler ever moves back onto the loop. (Writing it surfaced a measurement
trap worth recording: when the loop is frozen, the test coroutine is frozen
with it, so latency measured *from the test coroutine* starts late enough to
hide the defect entirely. The measure is wall-clock from the moment
validation begins blocking to the moment `/health` answers.)

## Alternatives considered

**Make validation itself asynchronous (streaming, chunked yields).** Rejected
for now. It is the real fix for memory as well as responsiveness and belongs
with Phase B's storage work; going async-only for the event-loop problem
duplicates the fix at a much higher price.

**Run validation in a separate process / task queue.** Rejected for Phase A.
Correct at scale, but it adds a worker topology to a one-process deployment
whose actual CPU work is bounded by the new size limit at roughly five
seconds. The threadpool is the proportionate answer; Phase B's Temporal
workflows supersede it.

## Consequences

- Upload latency no longer predicts anyone else's latency.
- Sync handlers use threadpool workers; concurrent large uploads each hold a
  worker for their validation. Bounded by the size limit, acceptable at this
  scale, revisited with Phase B's queue.
- The correction is visible in the handler's docstring and pinned by a test,
  so "it used to block the whole server" stays checkable rather than anecdotal.

## Rollback

Change `def upload_dataset` back to `async def` (and the body back to
`await file.read()`). The concurrency regression test fails within seconds of
the change — that failing is its job.
