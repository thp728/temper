# Control plane

FastAPI, SQLite, a thread per job, server-rendered HTML. Phase B replaces every one of those and
keeps the domain logic. Moving to `apps/control-plane/` and `packages/core/` under issue #15.

## Seams that Phase B depends on

These were written so the migration would be bounded. Preserve them.

- **Validation is pure.** Functions over parsed rows. No I/O, no framework imports. This is what
  becomes `packages/core/`.
- **No SQL in request handlers.** All persistence behind `db.py`-style functions. Their names and
  return shapes survive Phase B; their bodies change.
- **The orchestrator is a function over a job record.** It does not know whether the record came
  from SQLite or Postgres, or whether a thread or a Temporal activity called it. Keep the steps
  separable and individually idempotent, because **creating a VM twice is the failure that costs
  money.**

## Rules

**Validation errors name the line.** A rejection that does not say *which line* leaves the user
guessing at a file they cannot see. This is where users churn first.

**Every API error carries a stable machine-readable code**, a user-safe message, and a line or field
reference where one applies.

**`result.json` is always written, including on failure.** The orchestrator never parses logs to
learn what happened.

**Unknown job keys are refused loudly**, at the top level *and* inside `hyperparameters`, and echoed
back as `rejected_overrides`. An override the caller believes is in effect but is not is worse than
a refusal.

**Mixed thinking-mode datasets block** with a line-numbered error. They are ambiguous by
construction. Detection lives here and mirrors `trainer/thinking.py`.

**CPU-bound work stays off the event loop.** Validation costs roughly 80 ms per megabyte, and an
`async def` handler doing it synchronously freezes every request, not just its own. See
[ADR-0006](../docs/adr/0006-validation-runs-off-the-event-loop.md). The same reasoning is why the S3
client runs in a threadpool rather than blocking a handler
([ADR-0011](../docs/adr/0011-one-command-runs-every-task-and-one-defines-green.md)).

**Hyperparameter defaults are mirrored from the trainer by hand today, and that is a known defect.**
`hyperparams.py`, `feasibility.py:38` and `test_feasibility.py:108` each hold a copy. Issue #82
collapses them into one definition in `packages/contracts/`. Do not add a fourth copy.

## Testing

The provider is stubbed suite-wide. `conftest.py` refuses any attempt to construct a real client,
because a suite that can reach the billing account by accident eventually does. Pass `FakeProvider`.

The one exception is the transport tier ([ADR-0013](../docs/adr/0013-the-transport-is-proven-against-a-real-endpoint.md)):
`test_transport_endpoint.py` drives `JarvisLabsProvider.push/fetch/stream` against a local in-process
SSH endpoint, built with `object.__new__` so no client — and no credential path — is ever constructed.
Everything else keeps the fake.

Phase B integration tests use real Postgres, Redis and MinIO through testcontainers. Faking the
database in the spec whose content is *which* database defeats the purpose.
