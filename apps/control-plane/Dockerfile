# The control plane, as a container in the one-command stack (issue #29).
#
# Build context is the repository root (see compose.yaml): uv needs the
# workspace root and the single lockfile to resolve this package, and the
# lockfile is the supply-chain boundary just as it is for `just setup`.
#
# Only the runtime dependency set is installed (`--no-dev`): pytest, ruff,
# mypy, httpx and moto have no place in an image. `--package` installs exactly
# this app and its workspace dependency temper-core, not the other members.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# uv pinned to a release, copied from its own image, so the install step is
# reproducible without installing a package manager through pip.
COPY --from=ghcr.io/astral-sh/uv:0.7.0 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY apps/control-plane apps/control-plane
COPY apps/worker apps/worker
COPY apps/trainer apps/trainer
COPY packages/core packages/core
# The data files temper_core reads at import (fault surface, trainer defaults,
# the pinned schema and its tiers) -- the contract travels with the code.
COPY packages/contracts packages/contracts
COPY spike spike

RUN uv sync --frozen --no-dev --package temper-control-plane --package temper-worker

ENV PATH="/app/.venv/bin:$PATH"

# The object store's persistent data lives under /data, mounted as a named
# volume by compose.yaml so stopping and restarting preserves it. Overridden
# here rather than baked into defaults so the image itself still starts with
# nothing configured and the volume is the only thing pointing at /data.
# The database (issue #43) is a server, not a path under here -- compose.yaml
# points TEMPER_DATABASE_URL at its own postgres service explicitly, which is
# the compose network's own concern, not this image's.
ENV TEMPER_STORAGE_ROOT=/data/objects

EXPOSE 8000

# One process per container, per ADR-0011: worker count is an operational knob,
# not a code constant.
CMD ["uvicorn", "temper_control_plane.main:app", "--host", "0.0.0.0", "--port", "8000"]
