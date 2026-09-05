# ADR-0011 — One command runs every task, and one command defines green

- **Status:** accepted
- **Date:** 2026-08-25
- **Spec:** `docs/specs/008-the-stack-foundation.md`
- **Issue:** [#15](https://github.com/thp728/temper/issues/15)

## Context

There is no task runner, no continuous integration, and no definition of done
that a machine can check. Quality tooling has been run by hand: a `.ruff_cache`
sits in the repository root from an ad-hoc invocation, and the test suite is
whatever `python -m pytest -q` finds.

Issue #27 asks for lint, types and tests on every push, with one line in it
that decides the shape of everything here: *"the pipeline runs the same
invocation a developer runs locally, not a parallel definition."* Without a
single entry point, the gate list lives in a workflow file and again in a
README, and they diverge the first time a step is added. Then the pipeline and
the developer disagree about what green means, and the pipeline wins by being
the one that blocks.

Two things make this harder than it looks. The repository is about to hold two
languages with entirely separate toolchains. And development happens on Windows
11 while the pipeline runs on Linux, against a product that has never executed
on anything but Windows, whose environment documentation is a list of
Windows-specific traps, and which has already lost a real run to a line-ending
bug on exactly that boundary.

## Decision

### One entry point

**`just`.** Every task in the repository is a `just` recipe, and the pipeline
invokes the same recipes a developer does.

```
setup      uv sync && pnpm install
dev        docker compose up
check      the gate, below
fmt        formatters, write mode
lint       linters
types      type checkers, both halves
test       tests, hardware markers deselected
test-gpu   the excluded set. Requires JL_API_KEY. Never runs in check
contracts  regenerate openapi.json, then the TypeScript client
migrate    alembic upgrade head
image      build the trainer image locally
```

One rule for the file: **no logic in a recipe.** Every line is a single
readable invocation of a real command, so anyone without `just` installed reads
the file and runs the line. That is what makes the extra prerequisite a
convenience rather than a dependency, which matters for issue #71, where a
stranger clones this onto an untested machine. `just --list` prints the table
above, which doubles as the README's command section.

### One definition of green

`just check` runs the gates cheapest-first and stops at the first failure,
reporting which gate failed rather than that something did.

**Blocking:** format check, lint, type check, unit tests, integration tests,
and the OpenAPI drift check. Once #43 lands, the migration round-trip check
joins them: a fresh database migrates to head and every migration rolls back.

**Reporting only:** coverage, printed as a number with no threshold. A coverage
gate on a repository this young buys tests written to satisfy the gate.

**Excluded by default, visibly:** anything requiring `JL_API_KEY` or a GPU,
behind a pytest marker that `test` deselects and `test-gpu` selects. #27 asks
for the exclusion to be explicit, so it is a line you can read rather than an
absence you have to notice.

### Four points on one line

The gate contract is not one mechanism, it is four, and they are described
together because that is what makes them legible:

1. **Pre-commit hooks** run format, lint and a gitleaks scan on staged files.
   `repo: local` hooks calling the same recipes, so there is still one
   definition and the hook is a trigger rather than a second declaration. This
   is a fast filter, not the gate. Types and tests are deliberately excluded: a
   hook that takes a minute is a hook that gets `--no-verify`'d by the second
   day, and a bypassed hook is worse than no hook because you stop trusting it.
2. **`just check`** locally, the full gate, run when you want certainty.
3. **The pipeline** runs `just check` on `ubuntu-latest` on every push and pull
   request.
4. **Branch protection** on `main` requires the pipeline check to pass. It does
   *not* require a reviewer, which on a solo repository would block merging
   your own work.

Work reaches `main` through a branch and a pull request, one per issue,
squash-merged, with `Closes #N` in the body. Minimum ceremony: no templates, no
required reviewers, no labels beyond the triage set that already exists.

### The pipeline runs on Linux only

`ubuntu-latest`, one job. A Windows job was considered and rejected for a
reason worth stating because the first instinct is backwards: **the development
machine is the Windows coverage.** Every local `just check` is a Windows test
run. The platform this code has never met is Linux, and that is precisely what
the pipeline provides. `.gitattributes` (ADR-0010) handles the specific
cross-platform failure at its source rather than testing for it afterwards.

Two workflows, no more. `check.yml` runs the gate. `trainer-image.yml` builds
the trainer image and pushes it to GHCR by digest (#44), which has to be a
GitHub workflow because it needs a `packages: write` token only the runner
holds. Nothing deploys.

### Integration tests use real services, twice

Testcontainers starts Postgres, Redis and MinIO as session-scoped fixtures.
Real dependencies, clean state per run, and Ryuk cleans up even when the suite
crashes.

Spec 008 says these tests run "against the real services started by the same
command a developer uses", and testcontainers satisfies the first half of that
but not the second: it starts Postgres from its own image definition, not from
`compose.yaml`. A broken environment variable or healthcheck in the compose
file would leave every test green and still fail for the reviewer who types
`docker compose up`.

So both. Testcontainers for the suite, plus a pipeline job that runs
`docker compose up` and polls `/health` until every dependency reports ready.
The first tests the code against real services; the second tests the
composition, which is what that sentence was protecting. The spec paragraph is
clarified to say so.

One constraint that falls out and needs recording before #51 reaches for
Temporal's official compose file: **the stack has to start on an ordinary
laptop.** Temporal's published composition includes its own Postgres plus
Elasticsearch, and Elasticsearch is the hungriest thing in it. Temporal runs
auto-setup against the Postgres already in the stack, in its own database, with
the Web UI and without Elasticsearch.

### Every tool, named

Issue #15 originally said no language-specific tool choice would be made here,
on the reasoning that those belong to the tickets introducing each half. That
reasoning assumes a team, where a shared decision becomes a bottleneck. There
is one author and eight days, and deferring means sixty tickets each
re-deciding. The criterion is amended rather than quietly violated.

**Python.** uv for environments and a single cross-platform `uv.lock`, over
pip and venv (platform-specific pins, manual interpreter install) and Poetry
(slower, losing ground to uv). Ruff for format and lint, replacing Black,
flake8, isort, pyupgrade and bandit with one binary and one config block. mypy
strict over pyright, on the strength of Pydantic's first-class plugin support,
which matters when most of the codebase is Pydantic models. pytest with
testcontainers. SQLAlchemy 2.0 async with psycopg3 and Alembic, over SQLModel:
SQLModel merges the API model with the table, and `job_spec` is immutable and
composed with attempt and quote data on the way out, so those shapes diverge by
design. SQLModel would also make the domain model an ORM model by
construction, breaking ADR-0010's rule that `packages/core` imports no
framework. pydantic-settings, which does not replace environment variables but
reads them into one typed object that fails at boot rather than four hours into
a job. structlog for JSON logs with a correlation identifier threaded through
contextvars. Redis with redis-py. Scalar for API documentation over Swagger UI
and Redoc, on full OpenAPI 3.1 support and a built-in client. uvicorn, one
process per container, so worker count is an operational knob rather than a
code constant.

**boto3, behind `run_in_threadpool`, inside spec 006's storage seam.** aioboto3
was the obvious choice and was rejected for a specific mechanism: aiobotocore
works by patching botocore's internals, so it has to pin botocore to a narrow
range and lags releases structurally rather than through neglect. This project
lost checkpoint-resume to four packages moving independently (spike 3), and the
entire Axolotl decision exists because of it. Taking the same bet in a smaller
room for an S3 surface of four operations is not worth it. The threadpool
wrapper is about five lines, it honours **ADR-0006**, which exists because
blocking the event loop already bit this codebase once, and the seam makes it a
one-file reversal if it ever appears in a profile.

**JavaScript.** pnpm over npm: the `workspace:*` protocol stops a stale
registry copy silently resolving in place of the local package, which is the
same failure uv's workspace prevents on the Python side. Next.js 16 App Router
with React 19 and TypeScript strict. A Vite SPA would be smaller and faster to
build, and there is no authentication, no SEO and no public content to justify
server rendering today — but authentication as middleware, multi-tenancy as
nested layouts and public content as server-rendered pages are all cuts this
project documents in its scope records, and Next.js answers all three
while a Vite SPA answers none without a rewrite. **The framework is chosen for
the flows that were cut, not the flows that ship.** Tailwind v4 and shadcn/ui,
where you own the component code outright. TanStack Query v5, with `staleTime`
set explicitly, because it defaults to 0 and every mount refetches — wrong for
a job list that changes on a state transition rather than a timer. ESLint with
`eslint-config-next` plus Prettier, over Biome: Biome is one fast Rust binary
and would be symmetric with Ruff, but the frontend grows a compatibility
report, a generated advanced form, a plan editor and an evaluation comparison
before this is done, and an unfamiliar linter is not a tool you can defend
under questioning. Vitest with Testing Library over Jest. Playwright for
end-to-end.

**Orval** generates the TypeScript client, over Hey API. Hey API has momentum
and is the successor to `openapi-typescript-codegen`, but it is pre-1.0 and its
own documentation tells you to pin an exact version, which is not a sentence
worth saying about a core dependency. Orval is stable at v7 and generates
TanStack Query hooks, Zod schemas and MSW mock handlers from one config. The
mocks matter more than they look: they let component tests run with no backend,
which is the difference between a fast test and one that needs Postgres and
MinIO to boot.

`openapi.json` is checked in; the generated client is gitignored. Generating
the client from a live server would make drift impossible but would require a
running control plane for every web build, including in the pipeline. Reading
from a checked-in file removes that, and makes an API contract change visible
in a diff, which is worth having in a public repository. The drift check exists
for one specific failure: someone edits a Pydantic response model and does not
re-run `just contracts`. The gate is `just contracts && git diff --exit-code
packages/contracts/openapi.json`.

**Supply chain.** gitleaks in the pipeline and in the pre-commit hook, scanning
history rather than only the diff. This is not optional: `spike/.env` holds a
live provider key. Catching a secret before it enters history is a different outcome from catching
it after, where the remedy is rewriting history and rotating a key. Dependabot
scoped to GitHub Actions only; action version drift is a genuine supply-chain
risk and it is one config file, while opening it to pip and npm with days left
produces noise and no value.

### Deliberately absent, with thresholds

**OpenTelemetry traces and a Prometheus/Grafana pair.** The textbook answer,
roughly a day of work, and a reviewer looks at it for thirty seconds. What
exists instead: structured logs with a correlation identifier, a `/health`
endpoint reporting each dependency separately, Sentry wired behind a DSN that
is unset locally, per-job metric series in Postgres, and Temporal's Web UI —
every workflow, activity, retry and failure, browsable, free with the stack.
For a project whose core is the orchestration layer, that is a better
answer than a dashboard built in an afternoon. **Threshold:** tracing earns its
place when there is more than one service worth correlating across and the
correlation identifier stops being enough.

**Turborepo and Nx** — see ADR-0010. **Zustand** — no cross-screen client state
that is not server state; not added until something demands it. **React Hook
Form and Zod** — deferred with a trigger rather than absent: they arrive with
#33, whose advanced form is generated from the trainer's schema with dozens of
fields and cross-field validation, and Orval generates the Zod schemas from the
same document.

## What implementation found, 2026-08-25

Appended after carrying this out. The decision is not edited; these are the
places the description above did not survive contact.

**The pre-commit hooks call the tools, not the recipes.** This record says
"`repo: local` hooks calling the same recipes, so there is still one
definition". They call `uv run ruff` directly, because pre-commit passes the
staged filenames to the hook and a `just` recipe takes no filenames — calling
the recipe would lint the whole tree on every commit, which is the slow hook
this record warned would get bypassed. One definition survives in the sense
that matters: both read the same `[tool.ruff]` block. The claim about recipes
does not.

**gitleaks is a remote hook, not a local one**, pinned by `rev`. And the
history scan this record attributes to it is the pipeline's job (#27); the hook
scans staged changes, which is the half that catches a secret before it exists
in history at all.

**Nothing carries the `hardware` marker yet**, so `just test-gpu` selects
nothing. The marker is registered and the exclusion is real; the set it excludes
is currently empty, and fills as the tickets that need a GPU land.

**`just dev` starts the control plane, not the stack.** The stack needs a
compose file, which is #29.

**`coverage` was an orphan.** This record has coverage reporting as part of the
gate; the first implementation put it in a recipe nothing called. It now runs
inside `test`, so the number is printed on every gate run and still thresholds
nothing.

**The mypy suppression is wider than described.** It names four error codes, and
one of them, `misc`, is mypy's catch-all rather than a member of the nullable
family. Three findings arrive under it. Recorded in #85 rather than narrowed,
because there is no narrower code to name.

## Alternatives considered

**make.** The trodden path, and rejected on one measured constraint. Make is
not installed on Windows, and installing it is the easy part: recipes run under
`SHELL`, which defaults to `cmd`, so every recipe needs `SHELL := bash` plus
Git Bash on PATH. Every comparison written in 2026 flags Windows-without-WSL as
make's failure case, and that is exactly this situation, on a project that has
already lost a run to a cross-platform boundary bug. `just` is Make's syntax
with Make's Windows problem removed, which is not straying far.

**uv as the runner.** Preferred if it worked, because it is already a
dependency. It has no task runner; the request is an open issue.

**poethepoet.** The Python-ecosystem answer, works well with uv. Rejected
because half the tasks here are `pnpm build`, `docker compose up` and
`docker build`, and a Python-only runner would mean a second mechanism for
them.

**npm scripts at the root.** Zero extra installation for anyone who already has
Node. Rejected because chaining shell commands in a `"scripts"` field across
Windows and Linux is precisely where cross-platform breaks, and it would make
Node a prerequisite for backend-only work.

**Run every gate and summarise, rather than fail fast.** Argued for on the
grounds that lint, type and test failures often come from one edit, and fixing
them one round trip at a time is worse. Rejected: fail-fast is the convention,
it is simpler to explain, and #27's requirement to report *which* gate failed
is satisfied either way.

**Faking the database and object store in tests.** Rejected outright. Faking
the dependency in the spec whose entire content is *which* dependency defeats
the purpose.

**A `windows-latest` pipeline job.** See above. Rejected once the reasoning was
corrected.

## Consequences

- `just` becomes a prerequisite for contributing. Mitigated by the no-logic
  rule, and documented in the README alongside uv, pnpm and Docker.
- Naming every tool here contradicts issue #15's original acceptance criterion,
  which is amended on the issue with this reasoning rather than left to be
  discovered as a violation.
- The pipeline needs `packages: write` for GHCR and Docker available on the
  runner for testcontainers. Both are standard on `ubuntu-latest`.
- Branch protection means `main` is always green, and it means a failing gate
  blocks your own merge. That is the point.
- Testcontainers plus a compose smoke check is two mechanisms for one property.
  Accepted, because they test different things and only one of them is what
  #71's reviewer will actually type.

## Rollback

Each piece unwinds independently, which is why the runner and the layout are
two records rather than one. Deleting the `justfile` and inlining its lines
into the workflow reverts the runner without touching the tree. Removing
branch protection reverts the merge policy. Replacing a named tool is a
dependency change plus a config file, except mypy and Ruff, which are load
bearing in the sense that the code has been written to satisfy them.
