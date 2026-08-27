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
check: fmt-check lint types contracts-check test web-install web-browsers web-client web-lint web-types web-test e2e

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

# The control plane alone, against local defaults, with reload. ADR-0011 has this
# starting the whole stack; that needs a compose file, which is #29.
dev:
    uv run uvicorn temper_control_plane.main:app --reload

# --- the web application (apps/web) -----------------------------------------
# Every recipe is a single invocation; read the line and run it if you lack
# `just` or `corepack`.

# Build the trainer image from the same sources a real job builds from.
image:
    uv run python -m temper_control_plane.trainer_image

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
