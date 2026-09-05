# The application shell, as a container in the one-command stack (issue #29).
#
# Build context is the repository root (see compose.yaml): the API client
# generates from packages/contracts/openapi.json, which lives outside
# apps/web, and the whole app must be installed through its own lockfile
# (the supply-chain boundary, same as `just web-install`).
#
# Two stages: the builder installs, generates the client from the checked-in
# contract and runs a production build; the runtime image ships only what
# `next start` needs. `next build` bakes the /v1 rewrite target into the
# routes manifest, so the build-time value must be the address the running
# server will use -- the control plane's name on the compose network, which is
# the one place this stack ever runs (the dev path uses `next dev`, which
# reads TEMPER_BACKEND_URL from its environment at start).

# --- builder ---------------------------------------------------------------
FROM node:22-slim AS builder

RUN corepack enable

WORKDIR /workspace

COPY apps/web/package.json apps/web/pnpm-lock.yaml apps/web/pnpm-workspace.yaml apps/web/
COPY packages/contracts/openapi.json packages/contracts/openapi.json
COPY apps/web apps/web

# The rewrite target, fixed by the compose network. An ARG so a deployment
# that renames the service can override it without editing the Dockerfile.
ARG TEMPER_BACKEND_URL=http://control-plane:8000
ENV TEMPER_BACKEND_URL=$TEMPER_BACKEND_URL

# Run pnpm from the directory that pins it, not from /workspace. `--dir`
# tells pnpm where to work but does not move corepack's resolution root:
# corepack reads `packageManager` from the working directory's package.json,
# finds none at /workspace, and silently falls back to whichever pnpm it
# ships with. That version is not pinned by anything here -- `node:22-slim`
# is a moving tag -- so a base-image bump shipped pnpm 11.25.0 against a
# project pinned to 11.24.0 and the build began failing with no change to
# this repo. Working from apps/web means the pin in its package.json is the
# one definition both corepack and pnpm read.
WORKDIR /workspace/apps/web

RUN corepack pnpm install --frozen-lockfile
RUN corepack pnpm generate:client
RUN corepack pnpm build

WORKDIR /workspace

# --- runtime ----------------------------------------------------------------
FROM node:22-slim

# next start resolves .next, next.config and the project root from its working
# directory, so the runtime stage's WORKDIR is the app directory itself.
WORKDIR /workspace/apps/web

COPY --from=builder /workspace/apps/web/package.json ./
COPY --from=builder /workspace/apps/web/next.config.ts ./
COPY --from=builder /workspace/apps/web/.next ./.next
COPY --from=builder /workspace/apps/web/node_modules ./node_modules
COPY --from=builder /workspace/apps/web/src ./src

# The control plane's address inside the compose network is supplied by
# compose.yaml (build arg + runtime env, one YAML anchor). The default below
# only covers building the image directly, outside compose.
ENV TEMPER_BACKEND_URL=http://control-plane:8000
ENV PORT=3000

EXPOSE 3000

# `next start` invoked through the installed binary rather than pnpm: the
# runtime image ships node_modules already installed, and pnpm's
# verify-deps-before-run auto-install would refuse the intentionally
# pre-approved build scripts (esbuild, unrs-resolver) it never ran.
CMD ["node", "node_modules/next/dist/bin/next", "start", "-p", "3000"]
