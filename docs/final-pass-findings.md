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

## The reconciler destroys machines it should own

The worst finding here, and it costs real money each time it fires.

A run died with `orphaned_machine` while its job sat in `preparing` with the
machine id already on the row. The pass logged `listed: 1, owned: 0`. Every
half of that checks out in isolation: the ownership query returns the id when
asked directly, and `set_state` persists it. The disagreement only exists at
runtime, so a read-only probe (`scripts/reconciler_probe.py`) printed both
sides on an interval until it caught the moment:

```
11:05:59 machine=498686 type=int status='running' owned_set=[] matches=False
11:06:06 machine=498686 type=int status='running' owned_set=[] matches=False
11:06:11 machine=498686 type=int status='running' owned_set=[498686] matches=True
```

For about twelve seconds the machine is created, listed and billing while no
job row claims it. The reconciler runs every thirty seconds and destroys
whatever it cannot account for, so a pass landing in that window kills a live
machine mid-provision. One run in six hit it.

**And the record then hides it.** That run's final `error_code` is
`ssh_unreachable`, not `orphaned_machine`, because two writers raced for the
job's terminal state and the less accurate one won:

```
05:29:03  reconciler destroys 498681 as unowned, marks the job orphaned_machine
05:29-32  the orchestrator, unaware, keeps polling SSH on a machine that is gone
05:32:42  destroy attempt 1/3 -- Instance 498681 not found
05:32:53  destroy refused after 3 attempts
05:32:58  gives up after the full 300s SSH timeout, overwrites with ssh_unreachable
```

Only the reconciliation log still names the real cause. Anyone reading the job
record sees a machine that would not accept SSH, and would go looking at the
network or the key. Three hundred seconds of a paid machine's timeout were
also spent waiting for a host that had already been destroyed.

The cause is structural. `provider.create` blocks until the SDK returns, but
the instance exists at the provider the moment the call lands, and the
orchestrator only learns its id from the return value. So it cannot record
ownership any earlier than it does. `await_ready` was deliberately split out
of `create` so the id is recorded before readiness can fail, and that
reasoning is right; the window *inside* `create` is the part nothing covers.

The machine is already named `temper-{job_id[:12]}` when it is created, so
the name carries ownership from the first instant the instance exists. If
`Machine` carried that name and the reconciler matched on it as well as the
id, the window closes. That is a change to the seam, the reconciler and its
tests in a money-critical path, so it is proposed here rather than applied.

A grace period would also work and is weaker: a genuine orphan then bills for
the length of the grace.

**Status: fixed.** `Machine` carries the provider-side name, `machine_name`
defines it once for the orchestrator and the reconciler to read, and a machine
whose name belongs to a live job is matched even before its id lands. An empty
name is explicitly not ownership, so a provider that reports no name leaves id
matching as the only test rather than protecting every unnamed machine at
once. Three tests cover the window, a terminal job's named machine still being
destroyed, and an unnamed machine staying unprotected.

## Smaller things from the hardware pass

- **SSH readiness is one successful probe.** `await_ready` returns on the
  first `ssh true` that exits zero. A freshly booted machine's sshd commonly
  accepts one connection and then restarts, and the very next connection was
  reset: a run died with `source_upload_failed` and
  `kex_exchange_identification: Connection reset` seconds after readiness
  passed. Transient, since a later run got past it, but the check asserts
  reachability where it means stability.

- **A diagnosable failure got a laundered code.** A missing object came back
  as `internal_error` with the message `ObjectNotFound:
  'datasets/ds_....jsonl'`. The condition is specific and the message says so;
  the code does not, against this repo's own rule that every error keeps a
  stable one.

- **Switching storage backends silently orphans existing datasets.** Bytes
  written under the filesystem backend are not in the S3 bucket, and the only
  signal is a run that fails after provisioning. Nothing warns at launch that
  a dataset's object is not in the store the job will read from.

- **A test reads developer environment.**
  `test_the_module_singleton_follows_the_configured_backend` asserts the
  default backend, so it fails for anyone whose `.env` selects another one.
  It should pin the configuration it asserts.

## "Try your model" bills for a GPU and serves a template

The most serious finding here, because it is not a bug so much as a feature
that is not there.

Starting an endpoint provisions a real L4 at 41.31 INR/hr. Inference then
returns:

```
[qwen3-4b] tuned response to: How do I connect my weather station to Wi-Fi?
[qwen3-4b] tuned response to: What is the capital of France?
```

The prompt echoed inside a template. The model is never loaded. The machine
exists, bills, and does nothing but hold the string together.

`serving.py` says so itself:

> The generation: a canned completion in the fake tier, the real inference
> path on real hardware. [...] A real implementation would `provider.stream`
> or `provider.fetch_stream` a generation from the machine's inference server
> here.

There is no branch. The canned line runs unconditionally, so the comment's
"the real inference path on real hardware" describes something that does not
exist. Anyone reading the source to check would be told the opposite of the
truth, which is worse than the missing feature.

Either implement the generation, or stop provisioning a machine for it and
say plainly on the screen that serving is not built yet. Spending a user's
money to return their own prompt is the one outcome that cannot be defended,
and the comment claiming otherwise is what turns a gap into a
misrepresentation.

**Status: fixed, and unproven on hardware.** The generation is
implemented (ADR-0076): `start_endpoint` pushes `apps/trainer/serve.py` to
the machine, runs it under the published trainer image with the job's
adapter applied to its base, and waits for the server to report the weights
loaded *before* any key is minted -- a machine that never gets there is
destroyed through the confirmed teardown path and the start refuses with
`endpoint_model_not_ready`. `infer` branches on `provider.is_remote` and
asks that server for the completion; the canned line now belongs to the
simulated tier alone. The endpoint also stores its machine's SSH handle
(migration 0008), without which an inference request had a machine id and
no way to reach the machine.

**Proven on a real L4, 2026-09-06.** Machine 498869, nine minutes of
warm-up, one prompt, teardown confirmed by listing. About 7 rupees.

```
[15:25:10] starting endpoint (provisions an L4 and loads the model)
[15:34:17] start -> 201 after 547s
[15:34:26] {"completion": "<think>\nOkay, the user is asking for the
           capital of France. I need to make sure I provide the correct an...
[15:35:13] listing 1/10: []  teardown CONFIRMED
```

That is the model, not the template, and it is the first time this
platform's endpoint has answered with something it generated.

The run bought three findings that no amount of reading had produced.

### The answer arrived and was thrown away

The 547-second start succeeded and the inference returned **502
`endpoint_generation_failed`** against a completion sitting right there in
the message. `provider.stream` folds stderr into stdout by design, one
ordered channel being what the transport delivers, so `ssh` saying

```
Warning: Permanently added '217.18.55.26' (ED25519) to the list of known hosts.
```

lands in the same text as the reply, and parsing the whole stream as JSON
refused a genuine answer.

The readiness probe went through the same banner unharmed, because it tests
for the substring `"ready": true` rather than parsing. Two calls on one
channel, one strict and one not, and only the strict one broke.

**Status: fixed.** The generation is preceded by a marker line and only what
follows it is parsed, which is how the trainer's own result crosses this
channel (`orchestrator.RESULT_MARKER`) for the same reason. Three tests, one
quoting the banner from this run verbatim; with the old parse restored it
fails with the exact message the run produced.

### Thinking mode was requested off and the model thought anyway

The endpoint passed `TEMPER_ENABLE_THINKING=0` for a job whose dataset
detected `enable_thinking: false`, and the answer opened with a thinking
block regardless.

Qwen3's template, read from the pinned revision, decides on a top-level
variable:

```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
```

`is defined` is the whole difficulty. `serve.py` passed the value as
`chat_template_kwargs={"enable_thinking": False}`. Newer transformers merges
that into the render context; older transformers forwards it as a template
variable *of that name*, leaving `enable_thinking` undefined and the test
failing open. Thinking stays on and nothing says so.

**Status: fixed in `serve.py`**, which now spreads the value as a keyword so
it reaches the render context on any version.

**Open, and worth more than the fix.** The trainer's own comparison uses the
same `chat_template_kwargs=` shape (`entrypoint._ModelGenerator.generate`,
`template_probe._tokenise`). If the image's transformers does not recognise
that parameter, then every side-by-side comparison this platform has shown
was rendered under a template the run did not train with, and
`build_config`'s own comment says the same value must be applied at serving.
One observation from the serving path is not proof about the training path.
What settles it is rendering one conversation both ways inside the pinned
image and diffing the two strings. That is nearly free on the next machine
that exists, and it has not been run.

### The interface shows nothing while it warms

Nine minutes of `GET .../endpoint` answering 404, because a `starting` row
and "no endpoint yet" are the same answer to that route. The job's event log
does carry a line saying an endpoint is starting, so the state is recorded,
just not where someone waiting would look.

**Status: open.**

## The endpoint could never have worked, and the UI hid the reason

Two bugs stacked, both fixed.

Provisioning refused every time:

```
Instance creation failed: Disk size must be at least 100 GB for V2 VM
instances. Requested: 20 GB (code=400)
```

`serving.py` hardcoded 20 GB, reasoning that "the value is small by
construction, not a constant two components must agree on" and that 20 GB sat
"well below the 50 GB floor some providers enforce" -- a guessed floor, and
the wrong one. It is exactly such a constant: `temper_core.disk` already
holds `PLATFORM_MIN_DISK_GB = 100`, measured in spike 5, and the training
path enforces it. Serving now reads the same definition.

And the failure was invisible. The section polls `GET .../endpoint` every
fifteen seconds and called `setError(null)` on the 404 that means "no
endpoint yet" -- the ordinary state of every unserved job. So a failed start
displayed `endpoint_provision_failed` and had it wiped within fifteen
seconds, leaving a button that appeared to do nothing at all. The poll no
longer clears an error it did not cause; only a user action does.

That combination is why this was never caught: the feature failed on every
real launch, and the interface erased the evidence each time.

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

## What the hardware pass actually proved

Real runs on a real L4, once the trainer image was published:

- provisioning, image pull by published digest, weights download, Axolotl
  loading 398 shards and training, teardown confirmed by listing, and cost
  accounting against the frozen rate
- the live job stream, after the compression fix: the page moved itself from
  `PREPARING` to `TRAINING` with progress, where before it froze on "waiting
  for SSH" for an entire eleven-minute run
- every refusal path, at no cost: invalid datasets, unsupported
  hyperparameters, unknown models, an unpublished image, an undeliverable
  artifact
- Hugging Face import against two real repositories

Total spend across the pass: about twelve rupees. Every machine was destroyed
and every teardown confirmed by listing; the account ended with none.

## Artifact delivery is still unproven, and the tunnel was the wrong tool

The last run trained to completion on a real L4 and then failed on delivery:

```
05:49:14  [trainer] comparison: 1 held-out prompt(s) through base and chosen checkpoint (step 5)
05:49:27  [trainer] capability: 8 general question(s) through base and chosen checkpoint (step 5)
05:52:08  [trainer] training complete in 116.9s
05:54:32  client_loop: send disconnect: Connection reset
```

Training, the held-out comparison and the capability check all ran. Then two
and a half minutes of silence -- the upload window -- and the connection
reset. Nothing but the dataset ever reached the bucket.

The cause was the tunnel, not the platform. A cloudflared quick tunnel cannot
carry an artifact: 70 MB returned 503, and on retest even 5 MB timed out at
524. The probe that "proved" the tunnel worked had pushed 27 bytes, which is
the kind of test that proves nothing -- an upload path has to be probed at
the size it will actually carry. The adapter here is roughly 66 MB (33M
trainable parameters in bf16).

Real delivery needs storage the machine can reach that will accept a
hundred-megabyte PUT: an S3 bucket, R2, or MinIO behind something sturdier
than a quick tunnel.

## Still not exercised

- the delivery formats, merged and quantised
- cancel mid-run

The serving endpoint came off this list on 2026-09-06: a real L4 loaded the
model and answered a real prompt, and what that run found is recorded above.

Retry needs no hardware and turns out not to exist as a general control: the
API offers a retry only for `training_diverged`, at half the learning rate,
and the interface offers it in exactly that case. That is a deliberate
design, and it means an infrastructure failure leaves the user re-entering
the wizard rather than re-running the job.
