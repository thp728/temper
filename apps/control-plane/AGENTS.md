# Control plane

The API: FastAPI over PostgreSQL (ADR-0064). The server-rendered pages this
once also served were deleted when the last screen was ported into
`apps/web` (Spec 007) — this app now answers the `/v1` contract only, and
the interface that consumes it lives in the web app. The domain logic sits in
`packages/core` (moved here from this app's own source under issue #15).

**This process does not run jobs.** `jobs.create` inserts a `queued` row and
returns; `apps/worker` claims it and calls `orchestrator.run_job` from a
separate process (ADR-0066). The request path starts no threads. The one
exception is the zero-cost tier's boot-time demonstration seed
(`seed_demo.maybe_seed`, ADR-0067): at startup, on an empty database and only
in that tier, it drives one canned run through the real `run_job` so the
reviewer-facing stack has a completed run to inspect immediately. It is
synchronous, startup-only, spends nothing and crosses no connection, so it
cannot leave the unowned billing machine ADR-0066's worker exists to prevent.

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
a refusal. Unknown *hyperparameter* keys are refused at creation with the stable code
`unknown_hyperparameter` (#83): resolution happens before launch now, so a key that fell through
would be dropped by the resolver without ever reaching the trainer's guard.

**Mixed thinking-mode datasets block** with a line-numbered error. They are ambiguous by
construction. Detection lives here and mirrors `trainer/thinking.py`.

**CPU-bound work stays off the event loop.** Validation costs roughly 80 ms per megabyte, and an
`async def` handler doing it synchronously freezes every request, not just its own. See
[ADR-0006](../docs/adr/0006-validation-runs-off-the-event-loop.md). The same reasoning is why the S3
client runs in a threadpool rather than blocking a handler
([ADR-0011](../docs/adr/0011-one-command-runs-every-task-and-one-defines-green.md)).

**The control plane resolves; the trainer applies (#83).** The job spec written at launch carries
`hyperparams.effective(overrides)` whole; the trainer holds no defaults and resolves nothing -- one
resolver, and it is the visible one. The table that resolver reads is
`packages/contracts/trainer-defaults.json` (#82), read through `temper_core.hyperparams` and shipped
into the trainer image at build time. Never declare these values as literals anywhere: a scan in
`apps/trainer/tests/test_agreement_with_the_domain.py` fails on any module-level literal bound to
those names. It watches the names the copies historically travelled under, so a paste is caught; a
fresh name for the same numbers is not, and stays a review concern. The trainer's required-key set
is pinned to the resolver's output by an app-side test, because a default added to the resolver
without the trainer learning to read it would fail every launch on the machine.

## Testing

The provider is stubbed suite-wide. `conftest.py` refuses any attempt to construct a real client,
because a suite that can reach the billing account by accident eventually does. Pass `FakeProvider`.

The one exception is the transport tier ([ADR-0027](../docs/adr/0027-the-transport-is-proven-against-a-real-endpoint.md)):
`test_transport_endpoint.py` drives `JarvisLabsProvider.push_stream/fetch_stream/stream`
against a local in-process
SSH endpoint, built with `object.__new__` so no client — and no credential path — is ever constructed.
Everything else keeps the fake.

Phase B integration tests use real Postgres, Redis and MinIO through testcontainers. Faking the
database in the spec whose content is *which* database defeats the purpose.
