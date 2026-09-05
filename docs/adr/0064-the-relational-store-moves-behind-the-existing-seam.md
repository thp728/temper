# ADR-0064 — The relational store moves behind the existing persistence seam

- **Status:** accepted
- **Date:** 2026-08-29
- **Spec:** `docs/specs/008-the-stack-foundation.md`
- **Issue:** [#43](https://github.com/thp728/temper/issues/43)

## Context

Job state lived in a local SQLite file, one process holding it, a thread per
job. That file cannot express a claim on a row surviving contention, so there
was no path to running orchestration anywhere but inside the request-serving
process, and a restart lost every running job with nothing to show for it.
"a thread in the web process, a local file, and a directory" is not a
production answer to any question asked about durability.

Spec 008 named the bet this issue tests: **persistence sits behind functions
rather than scattered through handlers, so the migration is bounded to what
those functions write to.** `db.py`'s function names and return shapes were
supposed to survive; only their bodies would change. This record states
whether that held, because the spec asked for the answer to be recorded
whichever way it came out, not only if it came out clean.

## Decision

PostgreSQL replaces SQLite behind `db.py`. Every public function in that
module (`create_job`, `set_state`, `get_job`, `list_jobs`, `add_event`, the
dataset and admitted-model functions, the row-shaping helpers) keeps its name,
its arguments and its return shape. Callers outside `db.py` — the orchestrator,
the API handlers, `main.py`'s health check — needed no changes to keep
working, and none were made to them.

**Migrations are versioned and exercised both directions**
(`migrations.py`). `0001_baseline` recreates the schema `db.py` held in
SQLite, translated column-for-column into PostgreSQL types. `0002_phase_b_
tables` adds the tables Spec 008 names as what Phase B needs — quote, attempt,
artifact and manifest, metric series, checkpoint — as typed tables alongside
the JSON columns `db.py`'s functions still read and write. `test_migrations.py`
migrates a fresh database to head, rolls every migration back (and rolls back
just the newest one, leaving the baseline in place), and proves a rollback
loses the *data* a table held, not only the table — a `down` function nobody
has run is not a rollback, so every one of these paths is actually executed by
a test rather than merely defined.

**A state change and its event are still one transaction.** `db.set_state`
already documented the intent under SQLite ("a status that moved with no event
recorded is a run you cannot account for"); it is the same guarantee under
`psycopg`'s connection context, and `test_transactions.py` proves it by
forcing the event write to fail after the status `UPDATE` has already been
sent and asserting the status change is absent too, not by reading the
`with connect()` block and trusting it.

**The job spec's immutability is unchanged, because it was never a database
constraint to begin with.** `create_job` is still the only function that
writes `hyperparams_json`, `base_model` or `base_revision`; no function added
or touched here updates them. This was true under SQLite by the same
mechanism (no UPDATE statement existed) and is worth naming as a limit of the
guarantee: it holds by *convention* — no code path writes those columns after
insert — not by a database-enforced immutability constraint. A future change
that added an UPDATE touching those columns would violate the property
silently. Making it structurally impossible (a trigger, a separate
insert-only table, a `REVOKE UPDATE` on those columns) was considered and
deferred: it is real hardening but changes what `jobs` looks like to write
against, which is exactly the redesign this issue's boundary excludes.
Recorded here as the honest gap rather than closed quietly.

**Money already followed the "integer minor unit plus currency" rule where it
mattered, and this migration extends the pattern rather than inventing it.**
`SPEND_CEILING_MINOR` (ADR-0063) and `temper_core.quote`'s `cost_low_minor` /
`cost_high_minor` / `minor_unit_for(currency)` predate this issue. `0002`'s
`quotes` table carries that same convention into typed columns. **Two columns
were deliberately left unconverted**: `jobs.price_per_hour` and
`jobs.storage_cost_usd_per_hour` stay `DOUBLE PRECISION`. `price_per_hour` is
a rate `temper_core.actuals` already rounds into a minor-unit total itself,
downstream, at the one place that total is needed — converting it here would
move that rounding earlier without being asked to. `storage_cost_usd_per_hour`
is seeded at `0.0137` USD/hour in `test_calibration.py`, sub-cent precision by
design (ADR-0030 prices storage as a separate low-volume USD line); rounding
it to whole minor units at rest would silently corrupt a rate a downstream
computation still needs exact. This is a conversion considered and rejected,
not an oversight — the acceptance criterion is a rule about values that are
*money at rest*, and a rate is not that even when its unit is a currency.

## Whether the seam held

**Partially, and the exceptions are worth naming precisely rather than folded
into "mostly."**

The functional seam held completely. `packages/core`, the orchestrator, the
validation path and the state machine are untouched — every domain module the
spec named as needing to survive unchanged does. That is the boundary the
issue drew, and it is the one that mattered for the domain logic's own
correctness.

**What did not hold was the *test infrastructure's* seam, because it was never
actually a seam — it was a shared assumption that the database was a file.**
Three tests reached past `db.py`'s functions directly at `db.DB_PATH`, a
module attribute rather than anything `db.py`'s functions expose, and each
failure names a specific place SQLite's shape leaked past where the spec
claimed persistence was hidden:

- **Every test's isolation** (`conftest.py`'s `isolated` fixture, and by
  extension every test using `client`) depended on `db.DB_PATH` being a
  mutable attribute pointing at a private file. This was never a documented
  part of `db.py`'s public contract, but it was load-bearing for the entire
  suite's isolation strategy. A real server has no equivalent knob — there is
  no "point this at your own private database" for free the way "point this
  at your own private file" was. `conftest.py` now starts one throwaway
  PostgreSQL container per session and clones a fresh database per test from
  an already-migrated template, which is a real, if larger, mechanism where a
  four-line monkeypatch used to suffice.
- **`test_startup_and_health.py`'s "starts with no configuration" test**
  could rely on SQLite needing no running process; a client-server database
  has no serverless default, so "no configuration set" can no longer mean "no
  external process required," only "no `TEMPER_*`/`JL_*` override required,
  given a database reachable at the documented default address." The test now
  stands one up at exactly that address before spawning its
  scrubbed-environment subprocess. This is a real, permanent change in what
  the criterion can mean for this product, not a test artifact: **a
  reviewer's first run genuinely now depends on a database being reachable**
  where it previously depended on nothing but a writable directory.
- **The same file's way of *causing* a broken database** (pointing `DB_PATH`
  at a location that could not be created) was intrinsically filesystem-shaped
  and has no PostgreSQL analogue; a connection string breaks by naming an
  address nothing answers on instead. The observable failure kind changed too
  — `OperationalError`, not `NotADirectoryError` — a real, visible consequence
  of the engine change, not a rewording.
- **`test_storage_paths.py`** asserted every writable location resolves to a
  path under this repository's `/data/`. That was true of the database when
  it was a file and is false of a networked one by construction — not a bug
  in the assertion, a fact about what changed.
- **`test_db.py`'s three legacy-column-rename tests were deleted, not
  edited.** They pinned a SQLite-specific compatibility shim
  (`_rename_location_columns` / `_rewrite_legacy_locations`, issue #22) that
  rewrote a database written by the pre-#22 build. No PostgreSQL deployment of
  this product ever ran the pre-#22 schema, so there is nothing for a rewrite
  to find; carrying the shim (and its tests) forward would have meant testing
  dead code against an engine it was never written for.

**A second, unplanned finding: the seam's *performance* contract leaked, and
the fix belonged inside the boundary, not outside it.** The full suite first
measured at 2306 seconds (460 tests) against a warm CI gate that used to run
end-to-end in about ninety seconds. Measured with `pytest --durations` rather
than guessed: a single orchestrator test opened 63 separate `psycopg`
connections — `db.connect()` opened and closed a fresh one on every call, by
design, on the stated theory that pooling could wait until it was a measured
problem. At roughly 17ms to open and close a connection on the machine this
was measured on, that is over a second of pure connection overhead inside one
test's `call` phase, recurring on every `db.py` call the orchestrator makes —
dozens per job, since progress and output are promoted one database write at a
time. Per-test database cloning (`CREATE DATABASE ... TEMPLATE`, `DROP
DATABASE`) was measured too and found real but secondary: roughly 70ms and
50ms respectively, against the seconds a connection-heavy test was paying.
`db.connect()` now draws from a `psycopg_pool.ConnectionPool` keyed by
`DATABASE_URL` (evicting the pool for any other URL on lookup, since tests
monkeypatch that value per test and a stale pool would serve connections to a
database that either is not the current test's or no longer exists) rather
than dialing fresh every call — this is a body-only change, so it stays
inside the "function names and return shapes stay" boundary, but it is exactly
the kind of thing "the body changes" was supposed to absorb without a second
finding. It very nearly wasn't absorbed silently: nothing about `db.py`'s
external contract signaled that the engine swap had an order-of-magnitude
performance consequence until the suite's own wall-clock time said so.
Full before/after numbers are in the PR body, measured the same way both
times.

**The honest summary:** the spec's bet on the seam was correct for the code
the seam was drawn around — the orchestrator, the validation path, the state
machine never needed to change, and that is the load-bearing claim Spec 008
made and this migration is meant to prove. The bet did not extend to the test
harness's isolation strategy or to the engine's operational shape (a server
that must be running, a connection cost that compounds), because those were
never inside the seam's stated boundary to begin with — they were assumptions
riding along with SQLite that nobody had occasion to name until this issue
made them visible by breaking.

## Alternatives considered

**Keep SQLite for tests, PostgreSQL for the deployed process.** Rejected per
AGENTS.md's own testing rule: "these tests run against real services started
as throwaway containers by the suite itself" — faking the database in a spec
whose entire content is *which* database defeats the purpose, and a suite
that passes against SQLite while the deployed process runs PostgreSQL proves
nothing about the actual engine's constraint behaviour, transaction semantics
or the migration's own correctness.

**Alembic for migrations**, matching the reference architecture the original
SQLite comment cited and rejected for a four-evening build. Rejected again
here: this repository's convention is hand-rolled verbs (`justfile`) over
framework ceremony, `migrations.py`'s forward/back pair per migration is
plainly readable without a second tool's model to learn, and the roundtrip
guarantee this issue asks for (`test_migrations.py`) does not need a
migration framework to prove — it needs `up` and `down` functions that are
actually called.

**Truncate tables between tests instead of cloning a database per test.**
Considered when the timing problem surfaced, and rejected once connection
pooling turned out to be the dominant cost: cloning added roughly 120ms/test
(70ms create + 50ms drop), pooling saved on the order of a second per
connection-heavy test. Truncation would have removed the smaller cost while
leaving the real one in place, and it trades a stronger isolation guarantee
(a fresh schema per test, immune to a stray sequence or constraint left behind
by a prior test) for a weaker one, for a saving that measurement showed was
not where the time was going.

**A single, session-wide transaction per test, rolled back at teardown**
(the classic Postgres test-speed pattern). Considered and rejected as a
larger change than this issue's boundary allows: `db.py`'s functions each open
their own connection and commit their own transaction — that per-call
transaction is exactly the mechanism `test_transactions.py` proves the
one-transaction guarantee through. Wrapping a whole test in one outer,
never-committed transaction would require every `db.py` call to use
savepoints against a shared, injected connection instead of committing for
real, which changes `connect()`'s contract in a way visible to every caller,
not merely its body.

## Consequences

- `db.py`'s public functions are unchanged in name and return shape; every
  caller outside this file needed no edits.
- A developer or CI runner needs Docker (for the test suite's throwaway
  container) and, for anything running the app directly rather than through
  `docker compose up` (`just dev`, `just e2e`), a reachable PostgreSQL at the
  documented default address (`just db-up`).
- `just e2e` and `just dev` now share one PostgreSQL database where SQLite
  gave each a private file for free; a journey's reset wipes a developer's
  own local data if run concurrently. Recorded in `playwright.config.ts`'s own
  comment as an accepted, real regression in isolation, not fixed in this
  issue — it needs either an init script or `db.py` gaining the ability to
  create a database, both outside persistence's function bodies.
- Two more `psycopg`-specific dependencies (`psycopg[binary,pool]`,
  `testcontainers[postgres]`) join the workspace.

## Rollback

Revert this issue's commits. `db.py` returns to SQLite, `migrations.py` and
its tests are removed, `compose.yaml` drops the `postgres` service, and
`config.DATABASE_URL` reverts to `config.DB_PATH`. Nothing outside `db.py`
needs to change either direction, which is the seam holding exactly where it
was supposed to.
