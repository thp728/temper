# Final pass: findings

A pre-submission walk of the whole UI against the real provider, 2026-09-05.
The database was wiped first so nothing here is contaminated by e2e leftovers.

Most of these are fixed in the same branch as this file, several of them
proven on real hardware afterward; the ones still marked open are recorded
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

**Settled, at no hardware cost, and fixed.** The trainer's own comparison
used the same `chat_template_kwargs=` shape
(`entrypoint._ModelGenerator.generate`, `template_probe._tokenise`). This
did not need a GPU to answer: `docker run --rm <pinned digest> python -c
...` against the exact image the trainer runs, on CPU, rendered one
conversation both ways. Wrapped (`chat_template_kwargs={"enable_thinking":
False}`) produced no thinking directive at all; spread
(`enable_thinking=False`) produced the correct
`<think>\n\n</think>\n\n`. Confirmed on transformers 5.14.1, the version
this image actually ships.

Training itself turned out fine: Axolotl's own `chat_template.py` builds
its `chat_template_kwargs` dict from config and spreads it with `**` before
calling the tokenizer -- the correct form, already. The bug was isolated to
this platform's two post-training uses of the same tokenizer call: the
held-out base-vs-tuned comparison shown on every job's results page, and
the export-time template-divergence probe that exists specifically to catch
a training/serving template mismatch. Both silently dropped
`enable_thinking`, so the probe could not have caught the exact class of
bug it was built for, and every comparison on every finished job to date
was rendered with thinking mode not applied, whatever the dataset's
detected setting said.

**Status: fixed.** Both call sites now spread the kwargs, matching
`serve.py`'s fix and Axolotl's own pattern. The test double in
`test_template_probe.py` modelled the wrapped form as if it worked, which is
why nothing here had ever failed; it now only recognises the spread form,
and a new test (`test_a_thinking_mode_mismatch_on_an_identical_template_fails`)
asserts the probe catches a thinking-mode divergence on an otherwise
identical template. Reverting the fix and rerunning that one test
reproduces exactly this failure: `ok=True` where two renders that should
differ don't.

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

## The delivery formats: the estimate was wrong and the image cannot make one

Run on a real L4, 2026-09-06: `job_6c459c2cfbd1`, a 12-row dataset with
`merged` and `quantised` ticked at launch. It failed after 25 minutes and
INR 17.36, and it was worth every paisa.

### The quote ignores packaging, confirmed

Predicted before the run, from reading: `quote_for_launch` receives
`dataset_id`, `base_model`, `hyperparameters` and `overrides` — the
launch's `delivery_request` goes to `jobs.create` and never reaches the
quote — and `quote.PHASES` ends at `teardown`. Measured after it:

```
phases the quote priced: ['provisioning', 'readiness', 'image_pull',
                          'model_download', 'training', 'teardown']
packaging priced: False
```

The job's own actuals carry a `packaging` stage the quote has no
counterpart for. So this is not a coefficient that is off; there is no
phase to be off.

| | predicted | actual |
| --- | --- | --- |
| duration | 168–438s | 1513s |
| cost | INR 1.94–5.03 | INR 17.36 |

**3.5x the high end on duration, 3.45x on cost**, against a cold run's
2.7x already recorded above. Two multiplicative misses, and this one grows
with the model: merging is proportional to the base, so a bigger model
makes the estimate worse rather than better.

The arithmetic underneath is more embarrassing than the ratio. Axolotl
reported `train_runtime: 17.37` — seventeen seconds of actual training
inside a 1455-second "training" stage. Everything else was the held-out
comparison, the capability check, the merge and the delivery attempt, none
of which the estimate models at all.

### The image cannot produce the quantised format

```
[trainer] ERROR: DeliveryFailure: the GGUF converter is not on the image's
PATH; this export cannot produce the quantised local format. Refusing
rather than shipping a file that is not the format its name claims.
```

`quantised` is offered in the launch UI, described in `/v1/jobs/spec` as
"the 'run it on your own machine' format", accepted by the launch, priced
at nothing, and then cannot be produced by the published image. The merge
succeeded first — "Writing model shards: 100%" — so the run got all the way
to the last step before discovering it.

The refusal itself is exactly right, and says so: it will not ship a file
that is not the format its name claims. What is wrong is where it happens.
Nothing between the checkbox and the GPU asks whether the image can honour
the request, so the user pays for a full run to find out. `/v1/jobs/spec`
already publishes `trainer_image_published` and `artifact_deliverable`
before a launch; the producible formats belong in the same answer.

**Status: fixed.** `trainer-image.json` now records `producible_formats` for
the published image (`adapter`, `merged`; not `quantised`, since the GGUF
converter isn't on its PATH). `/v1/jobs/spec` publishes a `producible` flag
per delivery format, `jobs.create` refuses a launch that requests a format
the image can't produce with `delivery_format_not_producible` before
anything is provisioned, and the wizard disables the checkbox with the
reason in place of letting the click reach the API. Verified in the browser
against the real control plane: the spec's `producible` map came back
`{adapter: true, merged: true, quantised: false}`, the quantised checkbox
rendered disabled with "the published trainer image cannot produce this
format, so asking for it would train, bill, and fail at the last step",
merged stayed tickable, and Launch stayed enabled with merged ticked alone.
No GPU needed for this one — the whole point is that the refusal now
happens before a machine exists.

### The failure reached the user as `training_failed`

The job record says:

```
training_failed — Trainer's result document did not parse:
Expecting ',' delimiter: line 1 column 4 (char 3)
```

Training succeeded. It took seventeen seconds and converged (loss 5.46 to
2.54, held-out 6.76 to 3.99). What failed was a delivery format, and the
trainer named it precisely. That diagnosis then died in transit: the result
document did not parse, so the orchestrator raised its own generic code and
the real reason survives only in the event log.

This is the third instance of one shape in this document — a specific,
already-diagnosed failure arriving under a generic name — and the second
found on hardware today.

**Worse, the parse error discarded its own evidence.** A byte offset into
text nobody kept, on a machine destroyed before anyone read the record.
Two paid runs would be needed to learn what the first one already knew.

**Status: partly fixed.** The parse failure now quotes what it could not
parse, so the next occurrence is diagnosable from the record it leaves. The
laundering of the delivery failure into `training_failed` is not fixed: a
result document that fails to parse is not a training failure, and a
delivery refusal deserves its own stable code.

### Merging is silent for twelve minutes, against a fifteen-minute limit

```
Output resumed after 738s of silence (stall limit 900s)
```

The merge writes nothing while it runs. It came within 162 seconds of
tripping the stall detector, which would have destroyed a machine that was
working correctly. The margin is not a design; it is where a 4B model
happens to land. A 7B would trip it -- and did.

**Confirmed on hardware, 2026-09-06.** A launch against a real Llama-2-7B
(`unsloth/llama-2-7b-chat`, admitted through the compatibility probe with no
findings) and a 300-row customer-service dataset, `merged` delivery only,
ran QLoRA training to completion -- the job's own events show the merge's
`Loading weights: 100%` and `Writing model shards: 100%` lines -- and was
then killed by the reconciler-adjacent guard: `failed (gpu_stalled)`,
"The job produced no output for 900s and was treated as stalled." The
model shards finished writing at 22:41:34; the next event was the failure
itself. The merge and the artifact upload for a real 7B model produced zero
output for longer than the stall budget, and the guard did exactly what it
is built to do: destroy a machine it cannot tell from a wedged one. Cost:
INR 27.41, four minutes short of the training-only run's total.

**Status: fixed, and confirmed on hardware.** `apps/trainer/delivery.py`
gains a `heartbeat` context manager: while the merge (or quantise)
conversion runs and while each format's upload runs, a background thread
logs progress every 120 seconds -- comfortably inside the 900-second
budget -- so a slow-but-healthy export keeps talking instead of going
silent. `run_delivery` and the canonical artifact's own upload in
`entrypoint.py` both take the same `log`. A test proves it: a merge and
upload each stubbed to run past one fast heartbeat interval must emit
their message; reverting the wrapping (verified) makes that test fail
with `assert False` in exactly the way the hardware run did.

Re-run on 2026-09-07 against the republished image, the identical job that
had died with `gpu_stalled` (`job_ac4aedcb89ea`, same Llama-2-7B, same
customer-service dataset): the event log shows
`[trainer] producing merged (120s elapsed)`, `(240s elapsed)`, `(360s
elapsed)`, `(480s elapsed)`, `(600s elapsed)` -- ticking through and past
the 900-second point the previous run never reached the far side of. The
merge and upload together ran roughly 24 minutes this time (a bigger
merged model than the earlier run: 10.7 GB against the delivery-formats
run's 6.4 GB, since a 7B base merges into more bytes than a 4B one), and
the job reached `complete` with `merged` delivered, no error, teardown
confirmed by listing. Cost: INR 32.88.

### The chat template is dumped into the job log

Roughly 190 events of raw Jinja, twice, one line per template line. It is
the resolved template being logged, and it drowns the run it belongs to —
the events API caps at 500, so on this run the template alone consumed most
of the readable history.

**Status: open. Cosmetic, and it makes every other finding harder to find.**

## Cancel mid-run is a clean pass

The third hardware run, and the first one today that went exactly as
designed. `job_a8d08080fdd1` launched, was cancelled once it reached
`training` (a machine already running, not a queued job — a cancel that
lands before a machine exists proves nothing about teardown), and settled at
`cancelled` with no `error_code` and no artifact:

```
duration_s: 106.0
cost_minor: 122  (INR 1.22)
stages: provisioning 13.8s, preparing 47.7s, training 42.2s, packaging: never reached
```

Teardown confirmed by listing, independently, twice: the account showed no
machines both immediately after and on a second check later. ADR-0003's
claim — cancelling destroys the machine, produces no adapter, and is not a
failure — holds on real hardware.

## A fourth run confirms the delivery gate and the template fix together

`job_613caee67086`, `merged` delivery only, launched after the
producibility gate (ADR-0077) and the `chat_template_kwargs` fix both
landed. Every part of the launch that had just been built or fixed held on
real hardware:

- The gate let `merged` through without a second look — only `quantised`
  would have been refused, and that path is already covered without
  hardware.
- The comparison rendered under the fixed, spread form of
  `chat_template_kwargs`. Neither the base nor the tuned generation opened
  a thinking block, matching this dataset's detected setting -- the
  behaviour the wrapped form left to chance and the spread form enforces.
- Reconciler passes read `listed: 1, owned: 1` throughout, as they should.
- Teardown confirmed by listing: the account showed no machines
  immediately after completion.

```
duration_s: 1774.4, cost_minor: 2036 (INR 20.36)
stages: provisioning 14.4s, preparing 47.6s, training 1710.3s, packaging: not measured separately
predicted: duration [168.5, 437.6]s, cost [INR 1.94, INR 5.03]
```

A fourth data point on the quote's blind spot: **4.05x the predicted high on
both duration and cost**, in the same direction as the 2.7x and 3.45x
already on record, and for the same reason -- `packaging` still has no
phase in the quote. The gap between "training" (1710s measured) and
Axolotl's actual training time is the same story as the delivery-formats
run: the comparison, the capability check and the merge all run inside a
stage the quote calls "training" and prices as if it were only that.

## A realistic job, end to end through the UI: real dataset, real 7B model, real inference

Requested explicitly as a realistic-scale test, not a smoke test: a real
customer-service dataset (300 conversations parsed from
`NebulaByte/E-Commerce_Customer_Support_Conversations`'s raw transcripts
into the platform's chat schema, 995/1000 parsed cleanly) and a real
Llama-2-7B chat model (`unsloth/llama-2-7b-chat`, admitted through the
compatibility probe with no findings -- most Llama-2-7B mirrors ship no
chat template at all; this one does).

Two full training runs launched this way (one direct-API, one through the
actual wizard -- `job_90fb4748b2d7`, provisioning through review with the
producibility gate, the pre-flight's 6/6 checks, and the disabled
`quantised` box all behaving exactly as designed) both completed cleanly
with `merged` delivered, confirming the heartbeat fix (above) on hardware
a second time. The UI run also surfaced one thing a direct API call would
never have caught: **port 8000 was down from an earlier restart**, and the
wizard's spec-preview call failed with a raw `http_500` instead of a coded
refusal -- not a product bug, an operator error, but it is exactly the
class of failure `AGENTS.md`'s error-code rule exists to prevent showing
to a real user, and it was indistinguishable from one until the control
plane's own log was checked.

### Starting an endpoint is one long blocking call, and a client that gives up loses the key for good

Found testing "try your model" against the trained adapter, as asked.
`POST /v1/jobs/{id}/endpoint` does not return until the machine is
provisioned **and** the model is fully loaded (ADR-0076's design: mint the
key only once `/health` reports ready) -- for this 7B model, about nine
minutes. The first UI attempt worked (eventually) because the browser tab
was left on the page. The second showed the user this:

```
Error
http_500: Internal Server Error
```

The control plane's own log tells the real story: the POST kept running
for the full nine minutes and returned **201** -- the endpoint started
successfully, machine 499448, a real charge. The request just outlived
whatever sits between the browser and the backend (most likely Node's
default `requestTimeout`, 300000ms = 5 minutes, since neither
`next.config.ts` nor the workflow overrides it, and 5 minutes is short
of what a 7B model's warm-up needs but long enough that nothing smaller
had ever tripped it). The frontend's `EndpointSection` holds the one-time
API key only in React component state (`created.api_key`); the request
that would have set it never resolved client-side, so the key was shown
to nobody and cannot be recovered -- the server stores only its hash. The
endpoint was live, billing, and permanently unusable. Confirmed by
listing and stopped by hand: about nine minutes, roughly INR 6.

This is the same shape of bug ADR-0066 already closed once, for job
launches: a request whose real duration is minutes, made through an API
call that blocks until finished, is a client-timeout and a lost-work bug
waiting for whichever request happens to run the longest. `jobs.create`
was fixed to insert a row and return immediately, with the worker
finishing the job out of band. `POST .../endpoint` still has the old
shape.

**Status: open, and the fix is known.** Start the endpoint the same way a
job starts: insert a `starting` row and return immediately (the
`starting`-row and machine-name protections ADR-0076 already built for the
reconciler race make this a small extension, not new machinery), let the
existing 15-second poll (`fetchEndpoint`) discover `running` the way it
already discovers every other state, and mint the key into a field the
poll can read once -- not only the response of the call that happens to
still be listening when it finishes. Confirmed live inference through the
resulting endpoint separately, by direct API call once the key was in
hand: the tuned model answered a customer-service prompt in character
("I'm sorry to hear that you lost the receipt... may I have your order
number, please?"), so the serving path itself is not in question -- only
the shape of the call that starts it.

## Still not exercised

Four ran on hardware this pass: the served endpoint answered a real
prompt, a launch with both delivery formats ticked found the quote's blind
spot and an unproducible format, cancel mid-run tore down cleanly, and a
`merged`-only launch confirmed the producibility gate and the template fix
together. What each one found is recorded above.

Retry needs no hardware and turns out not to exist as a general control: the
API offers a retry only for `training_diverged`, at half the learning rate,
and the interface offers it in exactly that case. That is a deliberate
design, and it means an infrastructure failure leaves the user re-entering
the wizard rather than re-running the job.
