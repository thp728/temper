# Final pass: findings

A pre-submission walk of the whole UI against the real provider, 2026-09-05.
The database was wiped first so nothing here is contaminated by e2e leftovers.

Two of these are fixed in the same branch as this file; the rest are recorded
and not yet touched.

## Blockers

### 1. The trainer image has never been published

`packages/contracts/trainer-image.json` carries `"published": false` and a null
digest, so **no real job can run at all**. A launch is refused at provisioning:

```
image_not_published — The trainer image has not been published yet, so a real
job cannot launch. Run the pipeline's publish workflow and merge the pull
request it opens, which lands the digest this launch would pull.
```

The refusal itself is well built. It fires before anything is provisioned, so
the failed launch cost nothing, and it names its own remedy. But until
`.github/workflows/image.yml` runs and its PR merges, the GPU half of the
product cannot be demonstrated.

**Status: open. This is the one thing standing between here and a real run.**

### 2. Nothing started the worker

Orchestration moved out of the control plane into `apps/worker` (issue #51).
`just dev` starts only the API, and no recipe started the worker, so a launch
sat in `queued` forever. This is also what left a job stranded in `packaging`
with its own log already reading "Machine 4242 destroyed": the process that
would have advanced it did not exist.

The compose file runs all three services, so this only bit the host path, which
is the path the README sends a developer down.

**Status: fixed. Added a `just worker` recipe and pointed `just dev`'s comment
at it.**

## The gate is red on main

Every recent `check` run fails, including the one titled "Fix pre-existing
format, lint and type gate failures". `AGENTS.md` says `just check` is the
definition of green and that main is protected by it, so this matters more than
a normal red build.

The failure is in the compose build:

```
[ERROR] This project is configured to use 11.24.0 of pnpm.
Your current pnpm is v11.25.0
```

`apps/web/Dockerfile` runs `corepack pnpm --dir apps/web install` from `/app`.
`--dir` tells pnpm where to work but does not move corepack's resolution root,
and there is no `package.json` at `/app`, so corepack never reads the pinned
`packageManager` field and falls back to whatever pnpm it ships with. The base
image is `node:22-slim`, a moving tag, so a node patch bumped the bundled pnpm
to 11.25.0 and the build started failing with no change to this repo.

The fix is to make corepack resolve from the directory that pins the version,
either by running the install with `apps/web` as the working directory or by
activating the pinned version explicitly before installing. Both survive the
next base-image bump; neither is applied yet.

**Status: open, diagnosed, one-line fix.**

## Correctness

### Pre-flight verified four things and missed the one that mattered

The review step reported "4 of 4 verified" and left Launch enabled, then the
job failed a second later on `image_not_published`. Its four assertions covered
the model, the dataset report, VRAM headroom and the cost estimate, all of which
describe a job that *would* run. None asked whether a launch could start.

**Status: fixed.** `GET /v1/jobs/spec` now publishes `trainer_image_published`,
the review step asserts it as a fifth row, and the Launch button is disabled
with the stable code shown when it is false. An absent field passes open, so an
older control plane never blocks a launch. Two component tests cover both
directions.

### `/health` says the database is fine when it is unusable

With every table dropped, `/health` still answered:

```json
{"ok":true,"provider":"real","dependencies":{"database":{"ok":true}}}
```

while `GET /v1/jobs` and `GET /v1/datasets` both returned `internal_error`. The
check proves a connection, not a schema. That is the difference between "the
database is reachable" and "the application can serve", and the endpoint claims
the second. Compose wires container health to this endpoint, so a control plane
in this state reports healthy and keeps taking traffic.

**Status: open.**

### Compose cannot run real compute

The header comment says real compute is "one switch away: flip the single
`x-zero-cost-mode` anchor to 0". It is not. The provider shells out to `ssh` and
depends on the host's ssh-agent, and says so in its own error text ("check
`ssh-add -l` lists the JarvisLabs key"). The worker container mounts no key and
receives no agent socket, so flipping the anchor produces authentication
failures at readiness rather than a working run.

Compose is a genuinely good zero-cost tier. The claim above it just promises
more than it delivers, which is the kind of thing that fails under questioning.

**Status: open. Either mount a key and forward the agent, or narrow the claim.**

## Navigation

- **Models** is a top-level sidebar entry leading to a page that says the model
  explorer is coming in a later slice. Honest text, but it is a dead end in the
  primary navigation.
- **`/calibration`** aggregates prediction-versus-actual across every finished
  job, states its basis carefully, and is one of the better pages here. It has
  no navigation entry at all and is reachable only by typing the URL.

The two problems cancel out: calibration deserves the slot Models currently
occupies.

## What holds up well

Worth recording, because most of the surface is in good shape.

**Refusals are consistent.** Every one carries a stable code, a reason and a
correlation id, at every layer. `dataset_invalid` comes back identically from
the spec preview and from the launch, so a refusal cannot appear late.
`unsupported_hyperparameter` quotes the specific reason that parameter is not
exposed rather than a generic rejection.

**Validation explains itself.** A deliberately broken file came back with
per-line codes and, for each, why it matters: "No assistant turn. The assistant
turn is the training target, so this row teaches nothing." Not just what failed.

**The Hugging Face import is real.** It pulled all 52,002 rows of
`tatsu-lab/alpaca` and refused it as instruction-format while naming the shape
it expects, then imported 21,555 rows of `HuggingFaceTB/smoltalk` cleanly with
chat schema auto-detected and normalisation warnings raised.

**The explanations are the strongest part of the product.** "Why this
configuration" gives each decision its reasoning and the alternatives that lost:
"predicted peak is 5.4 GB; the L4 (24.0 GB) is the cheapest card currently
available that holds it with 18.6 GB to spare". Prediction-versus-actual then
scores those predictions against measurement and labels which figures were
measured and which derived. The evaluation tab runs a general-capability smoke
test, flags a regression, and states its own standard error and sample size
rather than presenting eight questions as a benchmark.

## Not yet exercised

Everything below needs the trainer image published first:

- a real run on an L4, start to teardown
- the delivery formats, merged and quantised
- the serving endpoint and an inference round trip
- cancel mid-run, and retry on a failed job
