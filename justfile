# Windows only: under the default sh, pnpm hands its children a POSIX-style
# PATH, and the processes the e2e journeys spawn cannot resolve native tools
# like `uv`.
set windows-shell := ["pwsh", "-NoProfile", "-Command"]

# One entry point. The pipeline runs these recipes, not a parallel definition of
# them, which is the whole reason this file exists.
#
# One rule: no logic in a recipe. Every line is a single readable invocation of a
# real command, so anyone without `just` installed reads the line and runs it.
# That makes the prerequisite a convenience rather than a dependency.
#
# See docs/adr/0011-one-command-runs-every-task-and-one-defines-green.md.

# Show every task. This is the index; the README points here rather than listing.
default:
    @just --list

# Everything a fresh clone needs before anything else works.
setup:
    uv sync

# The definition of green. Cheapest gate first, stops at the first failure.
# The web half runs after contracts-check so client generation reads a
# contract that has just been proven current. `e2e` is inside the gate
# because Spec 007's rule is that the journeys run on every push; they cost
# no hardware, which is what makes that affordable.
check: fmt-check lint types contracts-check test web-install web-browsers web-client web-lint web-types web-test db-up e2e

# Formatting, as a gate rather than as a fix.
fmt-check:
    uv run ruff format --check .

# Formatting, as a fix.
fmt:
    uv run ruff format .

lint:
    uv run ruff check .

types:
    uv run mypy packages/core/src apps/control-plane/src apps/worker/src

# Hardware and credential tests are deselected here, and only here. Coverage
# prints a number and gates nothing: a threshold would buy tests written for it.
#
# Needs Docker running: the persistence suite (issue #43) starts its own
# throwaway PostgreSQL container per session rather than faking the database
# in a spec whose entire content is *which* database. No `db-up` required --
# the suite manages its own container's whole lifecycle.
test:
    uv run pytest -m "not hardware" --cov=packages/core/src --cov=apps/control-plane/src

# The excluded set, run deliberately. Costs money: provisions real machines.
# Nothing carries the marker yet, so this selects nothing until the tickets that
# add hardware tests land. `--exitfirst` keeps an empty selection from reading as
# a failure.
test-gpu:
    uv run pytest -m hardware --exitfirst

# Regenerate the API contract and the advanced surface. The web client
# generates from the API file; the advanced surface (issue #33) is generated
# from the pinned trainer's schema and its tier data, so the interface and the
# refusal gate read what build time produced. The gate regenerates and
# typechecks against the result, so drift fails a build.
contracts:
    uv run python -m temper_control_plane.contracts
    uv run python -m temper_core.surface

# Fails when a checked-in contract no longer matches what the application emits.
contracts-check: contracts
    git diff --exit-code packages/contracts/openapi.json
    git diff --exit-code packages/contracts/advanced-surface.json

# The control plane alone, against local defaults, with reload. Needs a
# database reachable at the default address -- `just db-up` starts one. For
# the whole stack -- database, control plane and web shell -- in one command,
# use `just up`.
#
# The control plane serves the API but drives no jobs: orchestration lives in
# the worker (issue #51). `just dev` alone leaves every launch sitting in
# `queued` forever, so run `just worker` beside it.
dev:
    uv run uvicorn temper_control_plane.main:app --reload

# The worker, which claims queued jobs and drives them. The other half of
# `just dev`: without this nothing advances a job past `queued`, and a
# control plane restarted mid-job leaves that job non-terminal until a worker
# picks it up again. `just up` runs both, so this is only for the host path.
#
# The package, not the submodule -- `python -m temper_worker.worker` imports
# the module, defines its functions and exits 0 having done nothing.
worker:
    uv run python -m temper_worker

# --- the database (issue #43) ------------------------------------------------
# A named, long-lived container rather than compose: `just dev`/`just e2e` run
# the control plane directly on the host (not through `docker compose up`), so
# they need a database already listening at the default address independent of
# the compose stack's own postgres service.

# Start (or reuse) a local PostgreSQL at the zero-configuration default
# address, for `just dev` and `just e2e`.
db-up:
    docker run -d --name temper-db -p 5432:5432 -e POSTGRES_USER=temper -e POSTGRES_PASSWORD=temper -e POSTGRES_DB=temper postgres:16

db-down:
    docker rm -f temper-db

# --- the whole stack (issue #29) --------------------------------------------
# One command starts every service the product needs. Defaults to the
# zero-cost tier (the in-package fake provider), so a reviewer with no account
# and no secrets can walk the whole journey; real compute is one line in
# compose.yaml away. Images build on first run; add --build to rebuild after
# source changes.
up:
    docker compose up

# Stop the stack. The named volume keeps data, so a later `just up` resumes it
# (stopping and restarting preserves data).
down:
    docker compose down

logs:
    docker compose logs -f

# --- the web application (apps/web) -----------------------------------------
# Every recipe is a single invocation; read the line and run it if you lack
# `just` or `corepack`.

# Build the trainer image from the same sources a real job builds from.
image:
    uv run python -m temper_control_plane.trainer_image

# Publish the trainer image to the registry, verify it is pullable by digest,
# and rewrite the digest contract the orchestrator reads (#44). The pipeline
# (`.github/workflows/image.yml`) runs this same command, then opens the pull
# request that lands the digest. Needs docker and registry credentials.
publish-image:
    uv run python -m temper_control_plane.publish_trainer_image

# --- the web application (apps/web) -----------------------------------------
# Every recipe is a single invocation; read the line and run it if you lack
# `just` or `corepack`.

# Web dependencies. Frozen: the lockfile is the supply-chain boundary.
web-install:
    corepack pnpm --dir apps/web install --frozen-lockfile

# The browser binaries the journeys need. Idempotent and near-instant when
# already present; this is what makes `just check` pass from a cold clone.
web-browsers:
    corepack pnpm --dir apps/web exec playwright install chromium

# Regenerate the API client from the checked-in contract. The output is
# gitignored; what keeps the halves honest is that web-types compiles against
# whatever this produces, so a contract change that breaks the interface
# fails here rather than in front of a user.
web-client:
    corepack pnpm --dir apps/web generate:client

web-lint:
    corepack pnpm --dir apps/web lint

web-types:
    corepack pnpm --dir apps/web typecheck

# Component tests. No backend, no network: the generated client is mocked.
web-test:
    corepack pnpm --dir apps/web test

# Browser journeys against both halves running for real. Costs no hardware:
# upload and validation never touch the GPU provider, which is why these can
# run everywhere. Boots both servers itself.
e2e:
    corepack pnpm --dir apps/web exec playwright test

# The shell half of the journey, with reload. Run it beside `just dev` in a
# second terminal -- `just` runs each recipe line in its own shell, so
# backgrounding the control plane here would orphan it the moment this line's
# shell exits. The port each side uses is defined once, in
# apps/web/src/lib/backend.ts.
dev-web:
    corepack pnpm --dir apps/web dev

# Install the fast pre-commit filter. Format, lint and secrets on staged files.
hooks:
    uv run pre-commit install
