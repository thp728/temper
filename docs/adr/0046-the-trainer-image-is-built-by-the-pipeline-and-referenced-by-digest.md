# ADR-0046 — The trainer image is built by the pipeline and referenced by digest

- **Status:** accepted
- **Date:** 2026-08-27

## Context

The trainer image was built on the machine, by the machine: the control plane
streamed the sources (`trainer_build.TRAINER_SOURCES`) to the VM and the VM ran
`docker build`. The base was pinned by digest
([ADR-0015](0015-trainer-image-pins-axolotl-by-digest.md)), which makes *the
base* reproducible, but nothing captured *our layer* — the digest of the image
that actually ran was whatever the machine happened to produce that day. "The
image is pinned by digest" was only half true, and the half that was true was
the half we did not build.

Issue #44's requirement is the other half: the image a job runs must be the
image the pipeline built, and a change to that image must be deliberate and
visible rather than silent.

## Decision

A dedicated pipeline workflow (`.github/workflows/image.yml`) builds the trainer
image from the same named `TRAINER_SOURCES` a job used to build from, publishes
it to the container registry, verifies it is pullable by digest, and records the
digest in `packages/contracts/trainer-image.json`. The orchestrator reads that
contract and the machine **pulls `<image>@<digest>`** — it never builds.

The digest is the contract and the tag is a comment, exactly like the
Dockerfile's FROM line. A digest change lands as its own reviewed pull request
opened by the workflow, which is the mechanism that makes it a deliberate,
visible change rather than a silent one.

Nothing installs into the image at run time. The image is built from the same
list `just image` builds from, so the pipeline can only ever publish what a
local build would have produced — the two cannot diverge.

## Alternatives considered

*Keep building on the machine* (rejected — the pipeline-published image would
be dead weight: nothing would reference it, and a machine build could silently
diverge from anything the pipeline produced. The ticket's whole point is that
what runs is exactly what was built).

*Reference the published image by tag* (rejected — a tag is movable, so
"what runs is exactly what was built" would depend on nobody ever moving a
tag. The product already knows the discipline: the digest is the contract and
the tag is a comment).

*Build on the machine as a fallback when nothing is published* (rejected — a
fallback reintroduces the divergence the change removes, and "sometimes the
machine builds" is the worst shape for reproducibility: it runs on a billing
VM and its output is exactly what cannot be reproduced or audited).

## Consequences

A real job refuses loudly (`image_not_published`) until the pipeline has
published an image, because a job that cannot name the image it would run must
not spend money finding that out. The checked-in contract starts unpublished
and the first publish lands as the first digest-change pull request.

The base pin and the layer pin are now two different forcing functions, both
tested: the schema snapshot stays pinned to the Dockerfile's `FROM` digest
(`test_the_schema_snapshot_matches_the_pinned_image_digest`), and the published
layer's digest is recorded in `trainer-image.json`, which changes only through
the workflow's own pull request.

The first publish costs a cold ~8.5 GB base pull and push on the runner; the
workflow is scoped to trainer-source changes plus manual dispatch so that cost
is paid deliberately, not on every push.

## Rollback

Keep `orchestrator._remote_script`'s machine-build path (the sources tar and
the `docker build`) in history; restoring it makes the published image
informational again, and nothing else changes — the machine-build path never
depended on the contract.
