# ADR-0010 — The repository is laid out as `apps/` and `packages/`

- **Status:** accepted
- **Date:** 2026-08-25
- **Spec:** `docs/specs/008-the-stack-foundation.md`
- **Issue:** [#15](https://github.com/thp728/temper/issues/15)

## Context

Phase A ran as one process out of one directory. `api/` held the FastAPI app,
the domain logic, the provider adapter and the tests, all as one flat Python
package. That was the right shape for proving the loop, and it is the wrong
shape for what comes next.

Phase B adds a second language and splits the single process into three. A
Next.js application, a worker that claims jobs and drives machines, and a
control plane that no longer runs training on a thread inside itself. Sixty
open tickets are about to assume a directory shape. Deciding it after twenty of
them have landed means renaming across a tree that has grown, and two halves
that each invented their own answer to the same question.

Two smaller problems come along for the ride, and both are already causing
damage:

**Shared values are copied by hand.** `DEFAULTS` and `ALLOWED_OVERRIDES` are
Python literals in `trainer/entrypoint.py` at lines 50 and 63. They are
mirrored by hand in `api/hyperparams.py`, `api/feasibility.py:38` and
`api/test_feasibility.py:108`. Two of those copies carry comments saying
"duplicated rather than imported" out loud. Four copies of one definition is
three opportunities for a silent divergence in the numbers that decide what a
model trains with.

**Line endings are handled in application code.** There is no `.gitattributes`.
`api/orchestrator.py:112` strips `\r\n` out of every trainer file as it packs
the tarball, with a comment recording that a CRLF Dockerfile "fails inside the
container in ways that read as anything but a line-ending bug". That is a
workaround for something git should be doing, written after the bug cost a real
run.

The prior target layout lived in the private architecture document and had four
defects: it put GitHub workflow definitions in `infra/ci/`, which GitHub does
not read; it gave shared Python domain code no home while asserting that the
control plane and worker import the same domain packages; it described
`packages/contracts/` as JSON schemas without accounting for three consumers
with three different mechanisms; and it split code across `apps/`, `services/`
and `gpu/` with no rule saying which was which.

## Decision

**Two code roots, one rule: if it ships, it is an app; if it is imported, it is
a package.**

```
apps/
  web/                    Next.js application. Generated API client, gitignored
  control-plane/          FastAPI: routes, settings, ASGI app
  worker/                 claims jobs, runs activities, hosts the reconciler
  trainer/                Dockerfile plus two files. No runtime dependencies
packages/
  core/                   pure domain, importable by both Python apps
  contracts/              artifacts crossing a boundary where import is impossible
compose.yaml
compose.override.yaml
.github/workflows/
spike/  docs/  justfile  pyproject.toml  uv.lock  .gitattributes
```

Every Python member is `src/<package_name>/` with a sibling `tests/`. The src
layout is what makes tests exercise the installed package rather than the
working directory, which is the difference between catching a packaging mistake
here and catching it on a GPU.

Five rules follow from the tree, and they are the part that matters.

**`packages/core` is pure and framework-free.** Validation, hyperparameter
resolution, feasibility, the state machine, event classification, the
error-code registry and the money type. No FastAPI import, no SQLAlchemy
import, no I/O. Both Python apps import it; it imports neither. This is the
seam spec 008's migration is bounded by, and it exists in substance already —
Phase A's rules about pure validation and an orchestrator that is a function
over a job record are ports and adapters arrived at by a different road.

**`packages/contracts` holds only generated artifacts that cross a boundary
where importing is impossible.** `openapi.json`, emitted from the FastAPI
application and checked in. The trainer field-tier data, which moves out of
`docs/data/`. Nothing hand-written lives there, and Python types shared between
two Python processes are code and belong in `packages/core`.

**A value two components must agree on is defined once and read, never
retyped.** The four copies of the trainer defaults collapse to one definition
in `packages/contracts`, read by the control plane and copied into the trainer
image at build time. A test asserts no copy has crept back.

**`spike/` is a one-time-use directory, documented for reference.** Its
findings files are the measured evidence that lets numbers in this repository
be called measured, and they stay. Its scripts are the investigation that
produced them, not product code. When code graduates out of it into the
product, its tests move with it: streaming logic and its tests go to
`packages/core`, GPU availability and price filtering and its tests go to
`apps/worker`. Whatever remains has no tests because none of it is product
code.

**Line endings are git's job.** `.gitattributes` sets `* text=auto eol=lf`,
with `*.sh text eol=lf` and `*.ps1 text eol=crlf` stated explicitly. The
orchestrator's normalisation stays as a second line of defence, but it is no
longer the only one.

Two supporting choices. Python is pinned to 3.12 in `.python-version`, matching
the trainer image, so the control plane and the container cannot drift on
syntax or standard-library behaviour. The Node version is pinned by `.nvmrc`
and the `packageManager` field.

## Alternatives considered

**Stay flat.** Keep `api/` where it is and add `web/`, `worker/` and
`contracts/` beside it. Nothing moves, no imports change, the existing suite
passes byte-identically, and this is finishable in twenty minutes rather than
an afternoon.

Rejected because the flat tree has no answer to the question that forced this
decision: where does code shared between the control plane and the worker live?
The only flat answer is that the worker imports the control plane, which makes
the process split cosmetic. A worker that imports the web application's package
is not a separate process in any sense that matters; it is the same program
started twice. The second problem is that flat gives no place for the JavaScript
half that is not a peer of `api/`, and a reader opening the root sees eight
directories with no rule telling them which are deployables and which are
support.

**The three-root split from the architecture document** (`apps/` for web,
`services/` for Python, `gpu/` for images). Rejected because the rule
distinguishing an app from a service was never written down, and under
questioning "why is the web application an app and the control plane a service"
has no answer that survives a follow-up. A reader learns the trainer runs on a
GPU from the README and the architecture diagram, which is where that belongs,
not from a directory name that costs a third root.

**Turborepo or Nx.** Rejected. They exist to cache builds and compute affected
graphs across many JavaScript packages. There is one. Adopting a monorepo tool
for a single Next.js application is ceremony that a reviewer would reasonably
ask about. The `apps/` and `packages/` names are borrowed as a *convention*,
deliberately without the tooling that popularised them, and this record says so
in those words so nobody goes looking for a `turbo.json` that does not exist.

**`infra/compose/`.** Rejected. Compose reads `compose.yaml` from the root, and
one file does not need a directory. If it grows a MinIO bucket-init script and
a Postgres seed, it can become one then. `infra/ci/` is rejected outright
because GitHub Actions reads `.github/workflows/` and nowhere else, so that
directory could only ever have held files nothing executes.

**Domain-driven design layering** (`domain/`, `application/`,
`infrastructure/`). Rejected. The property those folders exist to enforce — a
domain that depends on nothing — is already enforced here by `packages/core`
being a separate package with its own dependency list, which is a stronger
constraint than a naming convention because a wrong import fails resolution
rather than a code review. Three more folders and no new guarantee.

**Colocated tests, as they are today.** Rejected because colocation defeats the
src layout: tests inside the package are tests that ship in the distribution
and run against the working directory rather than the installed code.

## Consequences

- **The existing suite does not pass byte-identically, and claiming it would
  have been dishonest.** Every test import changes, and `REPO_ROOT =
  Path(__file__).parent.parent` in `api/orchestrator.py:56` and
  `api/config.py:18` changes depth. What does not change is a single assertion.
  That is the measure spec 008 actually cares about: if an assertion has to
  change, the seam was leakier than claimed, and that finding gets recorded
  rather than patched over.
- `apps/worker/` and `apps/trainer/` are created with a `pyproject.toml` and a
  README before anything fills them, because their existence is what stops #51
  and #44 choosing different names.
- `apps/trainer/` declares no runtime dependencies and never will. Its base
  image already contains the resolved torch, transformers, peft and trl matrix,
  and installing anything reintroduces the version problem that pinned base
  exists to avoid. Its `pyproject.toml` exists so pytest can find
  `test_output.py` and `test_thinking.py`. That is stated in its README because
  "why does this package declare no dependencies" is otherwise a puzzle.
- Adding `.gitattributes` does not renormalise files already in the index. The
  `git add --renormalize .` sweep lands with the tree move as one reviewable
  commit rather than mixed into unrelated work.
- `docs/data/axolotl-field-tiers.json` moves to `packages/contracts/`, which
  changes a path referenced by spec 009 and issue #33.

## Rollback

The move is a sequence of `git mv` calls plus an import rewrite. Reverting is
the same operation in reverse, and the test suite passing is the check that it
worked in either direction. The rules are harder to unwind than the tree: a
codebase that has spent a week assuming `packages/core` cannot import FastAPI
will have grown around that constraint, which is the point of writing it down
now rather than discovering it later.
