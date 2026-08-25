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
check: fmt-check lint types test contracts-check

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

# Hardware and credential tests are deselected here, and only here.
test:
    uv run pytest -m "not hardware"

# Reports a number and gates nothing: a threshold here buys tests written for it.
coverage:
    uv run pytest -m "not hardware" --cov=packages/core/src --cov=apps/control-plane/src

# The excluded set, run deliberately. Costs money: provisions real machines.
test-gpu:
    uv run pytest -m hardware

# Regenerate the API contract and the client the interface consumes.
contracts:
    uv run python -m temper_control_plane.contracts

# Fails when the checked-in contract no longer matches what the application emits.
contracts-check: contracts
    git diff --exit-code packages/contracts/openapi.json

# The control plane, against local defaults, with reload.
dev:
    uv run uvicorn temper_control_plane.main:app --reload

# Build the trainer image from the same sources a real job builds from.
image:
    uv run python -m temper_control_plane.trainer_image

# Install the fast pre-commit filter. Format, lint and secrets on staged files.
hooks:
    uv run pre-commit install
