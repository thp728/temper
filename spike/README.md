# Phase 0 vertical spike

Purpose: settle the provider assumptions that everything else rests on, before writing product code. The architecture document names JarvisLabs VM automation as the highest technical risk in the build, and says *"do not begin broad product implementation until this path works."*

This is a probe, not a prototype. It deliberately does no training. A spike that also trains cannot tell you which half broke.

> **A note on the `§` references below.** They point at a private working
> architecture document that is not part of this repository, and they are kept
> because they are what the findings were written against at the time. Nothing
> here depends on reading it: each reference is summarised where it is used.

## What it answers

| # | Question | Why it blocks the build |
| --- | --- | --- |
| 1 | Does auth work, is there balance? | Trivial, but a wrong env var wastes an evening |
| 2 | Is an SSH key registered? | A hard prerequisite for `--vm`, and the failure message doesn't say so |
| 3 | Which GPUs are free *for VMs*? | The CLI docs warn *"a GPU may have free devices for containers but not VMs"*, meaning separate capacity pools and possibly separate pricing |
| 4 | Can the SDK create a VM at all? | `--vm` is documented for the CLI, not the SDK. If it is CLI-only, orchestration must shell out to `jl`, which is a materially worse story |
| 5 | Do startup scripts upload and attach? | The only mechanism for getting anything onto a bare VM |
| 6 | Is Docker present and usable? | Correction C1. §14 assumes it. The SDK can't supply a custom image, so if Docker isn't there the whole immutable-image approach needs replacing |
| 7 | Does GHCR pull work? Does `--gpus all` work? | Docker existing is not Docker useful. Separates "no Docker" from "no egress" from "no nvidia-container-toolkit" |
| 8 | Is outbound HTTPS open? | The callback protocol and every R2 signed URL depend on it, and it fails *late* |
| 9 | Does teardown always happen? | An orphaned GPU bills until someone notices. This is the rehearsal for the orchestrator's `finally` path |

## Setup (once)

```powershell
pip install jarvislabs
```

Credentials go in `.env` next to this file, which is git-ignored. Copy `.env.example`:

```
JL_API_KEY=<from jarvislabs.ai/settings/api-keys>
```

An SSH key is mandatory before any VM can be created, because VMs have no password login and the public key is baked in at boot:

```powershell
ssh-keygen -t ed25519 -C "jarvislabs" -f "$HOME\.ssh\id_ed25519"
# if you set a passphrase, enable the agent so scripts don't hang on a prompt:
#   (admin, once)  Set-Service ssh-agent -StartupType Automatic; Start-Service ssh-agent
#   (normal)       ssh-add "$HOME\.ssh\id_ed25519"
```

Then register the public half (`.pub`) through the JarvisLabs dashboard or `jl ssh-key add`.

## Running it

```powershell
python -u spike.py --dry-run   # preflight only, provisions nothing, costs nothing
python -u spike.py             # full run, destroys the instance afterwards
python -u spike.py --keep      # leaves it up for manual poking (COSTS MONEY)
```

Use `-u`. Python buffers stdout when redirected, so without it a backgrounded run shows nothing until it exits.

Run `--dry-run` first. It costs nothing and catches the likely blockers, meaning a missing SSH key or no VM-capable GPU free.

Output is a console narrative plus `findings.json`, which is git-ignored because it carries machine IDs and your balance.

## Cost

Picks the cheapest VM-capable GPU with a free device. Measured 2026-08-17: L4 at ₹41.31/hr. Billing is per-minute and a full run is roughly 3–4 minutes, so around ₹2–3.

Three things to know before running it:

- Prices are INR on this account. `account.currency()` returns `INR`, not USD. The USD figures on the pricing page are a different denomination of the same rate.
- VMs force a 100 GB minimum disk, billed at $0.10/GB-month while the instance exists.
- `--keep` leaves the instance billing. Destroy it with `jl destroy <machine_id>`.

## Results, 2026-08-17

Run against the real account. Total spend across all attempts: ₹9.64.

Resolved favourably. A `--vm` instance is Ubuntu 22.04.5 with Docker 29.3.1 preinstalled and running, `nvidia-container-toolkit` present, GHCR pull working, `docker run --gpus all` working, and egress open to ghcr.io, huggingface.co and pypi.org. The L4 shows 23,034 MiB on driver 580.126.20. Python 3.10.12 and git are present, with no `uv`. The immutable-image approach holds.

Measured, replacing guesses: a VM reaches `Running` in 15–17s, but SSH refuses connections for a further 42s, so it is usable at roughly T+60s rather than T+16s. Teardown succeeded first attempt, 4 times out of 4, with no strays.

**The blocking finding: startup scripts do not run on `--vm` instances**, and `create()` accepts `script_id` without error. Confirmed by reading cloud-init on a live VM: the user-data contains only JarvisLabs' own config, and `runcmd` has no trace of the supplied script. Nothing fails, the job just never starts. `probe.sh` therefore cannot self-execute, and the diagnostics above were gathered over SSH by hand. See correction C11.

What that means for the build: the bootstrap needs rewriting around the orchestrator SSHing in and driving the pull itself, which puts SSH key handling in the control plane. The alternative, container instances, forfeits custom images, which is the thing Docker-on-VM was buying.

## Spike 2 results, 2026-08-17, all checks passed

`spike2.py` plus `bootstrap.sh`. The bootstrap is removed: the near-duplicate scripts collapsed to [bootstrap6.sh](bootstrap6.sh), which represents the final approach, and earlier versions are in git history. The SSH-driven bootstrap works. Exit 0, instance destroyed, no strays.

| Test | Result |
| --- | --- |
| Bootstrap over SSH, unattended | pass, 94s wall clock |
| Image pull (`pytorch:2.5.1-cuda12.4-cudnn9-runtime`) | pass, 3.3 GB in 72s, 46 MB/s |
| Digest pinning (resolve, then re-pull by digest) | pass, immutable runs are achievable |
| `torch.cuda` inside container | pass, NVIDIA L4, bf16 supported, torch 2.5.1+cu124 |
| Artifact out via bind mount | pass, 263 KB tensor survived container exit |
| C7, public port | pass, `http://…:8000/` returned HTTP 200. C7 retracted, VM serving works |

Cold start, measured end to end, is about 131s: 13s create, 46s SSH-ready, 72s image pull. That replaces a guess in the quote's ETA, and the image pull is the biggest single component.

### The finding that matters most

VMs come up with a public IP and `ufw` inactive, reporting `Status: inactive` on a fresh box. The port test succeeded because nothing is filtering. A container bound to `0.0.0.0:8000` was reachable from the open internet, unauthenticated, within seconds.

The architecture states *"training VMs expose no public application ports."* The platform provides the opposite. Anything the trainer or vLLM binds is exposed by default. Firewalling belongs in bootstrap rather than in a later hardening pass, and no tenant data should touch a VM before it exists. Logged as correction C12.

## Spike 3 results, 2026-08-17

`spike3.py` plus `bootstrap3.sh` (removed, as above). Two attempts, and the first failed usefully. QLoRA runs. Resume does not.

| Test | Result |
| --- | --- |
| QLoRA on Qwen3-4B | pass, 4 steps, loss 3.99, peak VRAM 5.31 GB of 24 GB |
| Trainable parameters | pass, 33,030,144, exactly the count derived from `config.json` beforehand |
| Model load / train | 32s / 10.9s, adapter 132.2 MB, checkpoints at steps 2 and 4 |
| Stack install | pass, 15s |
| Checkpoint resume | fail, blocked by a version chain, see below |
| Firewall, `ufw` default-deny | fail, reports `active`, protects nothing |
| Firewall, `DOCKER-USER` on published port | fail, rule installs, never matches |
| Firewall, bind `127.0.0.1` | pass |

### Resume is impossible on an unpinned stack, and it is a three-link chain

Installing *latest* gave `transformers 5.15.0`, `trl 1.10.0`, `peft 0.20.0` and `bnb 0.50.1` on `torch 2.5.1`:

1. TRL 1.x's `SFTConfig` no longer inherits `TrainingArguments`, so `warmup_ratio` raises `TypeError`. One of the defaults the research settled on cannot be expressed.
2. `save_safetensors` was also rejected, so checkpoints were written as torch `.bin`.
3. transformers 5.x then refuses to load them: *"we now require users to upgrade torch to at least v2.6… does not apply when loading files with safetensors."*

The result is that *"a run survives a forced interruption"* fails, not from a design flaw but from four packages moving independently. The architecture names dependency instability as a risk and prescribes pinned images. This is that risk firing on the first real attempt. Pin the stack before building anything else.

### Two firewall mitigations that look correct and do nothing

- `ufw` default-deny reported `Status: active` while the port stayed open. `ufw` filters `INPUT`, and Docker publishes via NAT and `FORWARD`, so container traffic never traverses `INPUT`.
- A `DOCKER-USER` rule on the published port installed cleanly and never fired. By `DOCKER-USER` the packet is already DNAT'd, so the destination port is the container's (80), not the published one (8000).
- Binding to `127.0.0.1` worked: it answered on localhost and timed out from outside.

The rule for the build is that training containers publish nothing. The trainer needs no inbound port. Serving needs one, and it must match the *container* port in `DOCKER-USER`, and be verified from outside rather than assumed.

### Known spike bug, not a platform issue

Host-side `sha256sum` on the adapter failed with `Permission denied`, because the container writes as root and the host reads as `ubuntu`. Same trap as C10. The adapter was written correctly at 132.2 MB, and the verification step needs `sudo`.

## Spike 4 results, 2026-08-18: the trainer image works

`spike4.py` plus `bootstrap4.sh` (removed, as above) and [`../apps/trainer/`](../apps/trainer/README.md). Three attempts, and the first two failed on my own tooling rather than the platform.

| Test | Result |
| --- | --- |
| Image builds from pinned digest | pass, 183s, 8.5 GB |
| Job runs through the `/job` to `/out` contract | pass, 120.9s, 64 rows, exit 0 |
| `axolotl train` accepts `warmup_ratio` | pass, the exact argument TRL 1.x removed |
| Checkpoint resume | pass, resumed from `checkpoint-2` to step 6 in 54s |
| Adapter format | pass, safetensors rather than torch `.bin`, 132.2 MB, sha `d311285e…` |
| Adapter config | pass, `r=16 alpha=32 rslora=False`, targets are all 7 linear modules |
| Unknown job keys refused | fail, found a real bug in my entrypoint, since fixed |

C14 is closed. Resume is the thing spike 3 could not do at all. Pinning to Axolotl's tested stack by digest, rather than resolving six packages by hand, removed the `transformers`, `trl` and `torch` conflict that made checkpoints unloadable. The *"a run survives a forced interruption"* criterion is now achievable.

The `adapter_config` is worth reading closely, because it confirms the design end to end. `target_modules` came back as `gate_proj, down_proj, v_proj, o_proj, q_proj, k_proj, up_proj`, all seven, exactly the list derived from `config.json` in the research notes. `alpha=32` is `2r`. `rslora=False` is correct at `r=16`.

### The bug this caught in my own code

The job spec deliberately carried `not_a_real_key`, and it was not refused. `build_config` validated keys inside `hyperparameters` but never the top level, so a caller misspelling `max_steps` as `maxSteps` would have got a full-length training run with no warning. Both levels are validated now, with `_comment*` keys ignored by design.

### Two failed runs, and neither was the platform

Attempts 1 and 2 reported *"ssh ready, no answer within 240s"*. I logged that as a provisioning-reliability finding (C16) and it was wrong. I had launched those runs through Git Bash, which ships its own `ssh` and cannot see the Windows `ssh-agent` service, so the passphrase-protected key was unusable. Verbose SSH showed `Server accepts key` immediately followed by `Permission denied (publickey)`, meaning the key was provisioned correctly the whole time. Relaunched through PowerShell, SSH connected in 37s.

C16 is withdrawn. Two lessons are kept. A readiness check must distinguish *unreachable* from *authentication failed*, because those have opposite remedies and mine collapsed both into "no answer". And changing tooling mid-investigation is a confound worth testing before blaming the provider.

### Still open

- **fp32 adapters.** 132.2 MB is 33M × 4 bytes, and bf16 would halve it. Undecided.
- **Qwen3 thinking mode**, meaning `<think>` blocks versus a dataset without them (C2). Unhandled.
- **`chat_template: tokenizer_default`** ran without error, but that it resolved to the *right* template has not been asserted. The export-time probe from the research notes is the check.
- **Not pushed to GHCR.** The image is built per-VM. Pushing needs a token and ideally CI.

## Spike 4: what to test next

Everything below is blocked on pinning the stack. C14 means the current "install latest" approach cannot resume a run, so this is the next thing built rather than the next thing tested.

1. **Build the trainer image with an exact lockfile** and push to GHCR by digest. Resolve the version set that makes resume work, almost certainly `torch >= 2.6` plus whichever TRL expresses the intended defaults, or an explicit decision to drop `warmup_ratio`. Then re-run spike 3 unchanged against that image, where it becomes the regression test.
2. **Resume, properly.** Kill mid-run rather than re-instantiating in-process, and confirm the step counter, optimizer state, RNG state and dataloader position all restore.
3. **Adapter integrity** with `sudo` on the host-side hash, and a decision on fp32 versus bf16 adapters. 132.2 MB against roughly 66 MB is a real artifact-size difference.
4. **tokens/sec on a real dataset.** The 4-step run was too short to calibrate MFU, and the estimator's 35–50% band is still the softest number in the cost model.
5. **Serving**, once training is stable: vLLM on a VM, port matched correctly in `DOCKER-USER`, verified unreachable except through the gateway.

There is still no reason to spike container instances. The docs are explicit that *"templates are container-based, so Docker cannot run inside them"*, so they cannot run a custom image, which is the entire reason VMs were chosen. C7's retraction removed the last argument for containers in serving too.

## Reading the result

`spike.py` prints a "WHAT THIS MEANS" section that maps the probe output onto the decision it forces. The one that matters:

- Docker present, daemon up, GHCR reachable and GPU passthrough working means C1 resolves favourably, the bootstrap design holds as written, and C1 is marked resolved in the architecture.
- No Docker means C1 resolves unfavourably, and a decision is required between installing Docker in the startup script, provisioning a `uv` environment from a lockfile, or abandoning `--vm` for a container template. Each forfeits something different, and several sections of the architecture need rewriting rather than patching. Log it in the decision record the same evening.

## Afterwards

1. Update correction C1 in the architecture document from open to resolved, with what was found.
2. Log any forced decision the same evening, per the brief's standard that reasoning gets recorded rather than reconstructed.
3. Record the measured boot time. It feeds the duration estimate in the quote, which is the product's differentiator, and a guessed boot time makes the quote wrong from day one.

---

# Phase B spikes, 2026-08-23

Spec: [`docs/specs/004-phase-b-spikes.md`](../docs/specs/004-phase-b-spikes.md).
Issues [#16](https://github.com/thp728/temper/issues/16) to [#20](https://github.com/thp728/temper/issues/20).

Phase B's scope changed shape on 2026-08-23. The catalog stops being two dense
models and becomes anything the provider can hold, multi-GPU and full
fine-tuning move from cut to in, and the hyperparameter surface becomes whatever
the pinned Axolotl accepts. Every one of those rested on a number nobody had
measured. These five spikes take the measurements before the tickets assume
them.

Unlike spikes 1 to 4, their findings files are committed. See the negations in
[`.gitignore`](../.gitignore). The spike directory is the evidence base, and a
finding that exists only in a terminal scrollback is not evidence. None of the
five records a credential or a balance, and spike 8 deliberately records that
the balance was *readable* while withholding the figure.

| # | Question | GPU | Answer |
| --- | --- | --- | --- |
| 5 | How much disk, and how fast do weights arrive? | yes | 7200 GB ceiling, 364 MB/s. 70B is not disk-bound |
| 6 | Does the pinned image see 2 GPUs, and does FSDP run? | yes | It shards, checkpoints and resumes, and the loss is `nan` |
| 7 | How wide is Axolotl's config schema? | no | 388 fields, and only 12% of its constraints are in the schema |
| 8 | What does the SDK actually expose? | no | Pause exists. ADR-0003 flagged for reopening |
| 9 | How fast does validation stream? | no | File ×20 gives memory ×1.14, and token counting must split off |

What they produced beyond the findings:
[ADR-0009](../docs/adr/0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md)
(supersedes ADR-0004's property 2, on spike 5 and 6's measured sizes),
a correction to [ADR-0005](../docs/adr/0005-the-dataset-size-limit-is-derived-from-measured-memory.md)
(4.8× was measured again as 5.93×), and a reopening flag on
[ADR-0003](../docs/adr/0003-cancellation-is-destructive.md), because the provider can pause.

## Spike 5: disk and download throughput

`spike5.py` plus `bootstrap5.sh` (removed, as above), producing [`findings-spike5.json`](findings-spike5.json).
Two runs, a few rupees. The kill criterion did not fire.

| Question | Answer |
| --- | --- |
| Storage ceiling | 7200 GB, and the platform *names it*: requesting 8000 returns `hdd: ensure this value is less than or equal to 7200` |
| Ceiling honoured? | Yes. 4000 GB requested, 3877 GiB usable, writable at 149 MB/s |
| Is storage inside the GPU-hour? | No. The L4 rate stayed at ₹41.31/hr with a 40× larger disk |
| Download rate | Qwen3-8B, 16.4 GB in 45s, so 364 MB/s, 2.7s per GB |
| Steady or bursty? | Bursty. Window rates 0–730 MB/s, CoV 0.59 |
| Teardown | Destroyed first attempt, absent in 3 consecutive listings |

The ceiling is measured, not bisected. The spec said to *"bisect to find what
is"* accepted. No bisect was needed, because the platform names its own bound in
the refusal, which is a better answer than the largest value a ladder happened
to try. The first version of this finding recorded *"between 4000 GB and 8000
GB"*, bracketing from the ladder while the exact figure sat in the rejection
text the same run had already stored. Corrected in `findings-spike5.json`, with
the correction kept.

The ceiling is not the constraint anybody expected. Planning assumed the 100 GB
VM floor was close to the ceiling and that 70B might not fit. It is 72× the
floor. 70B full fine-tuning is not disk-bound on this provider, and the catalog
claim stands without a caveat about disk.

The download rate is 7 to 30 times better than the range the spec was reasoning
about. The spec's own framing was *"at 200 MB/s that is 12 minutes; at 50 MB/s
it is 47"*, and the whole `preparing` ETA hung on which. Measured at 364 MB/s, a
141 GB 70B model arrives in about 6 minutes, so the preparing phase is not the
dominant term in the quote, and the trainer image pull (87–183s, measured in
spike 4) is now comparable to it.

Two things that number is not. It is a floor: `hf_transfer` was deliberately
left off, because the trainer image does not use it and quoting a rate the
product does not take would be dishonest. And it is one region, one repo, one
time of day, so it is a range rather than a promise.

Bursty matters more than the mean. Windows ranged from 0 to 730 MB/s over a
45-second download. A live ETA computed from a 30-second window will oscillate
badly. The quote must smooth it or show a range, and it must not show a number
that jumps.

### What the first run got wrong

Two of my own bugs, both worth keeping:

- **`pip3: command not found`.** The download step never ran. A `--vm` instance
  has `python3` and no `pip3`, and spike 1's note that *"Python 3.10.12 and git
  are present, no `uv`"* never checked for pip. `bootstrap5.sh` now resolves it
  in three steps, trying the binary, the module, then `apt-get`, and puts
  `~/.local/bin` on PATH, because that is where the console script lands and it
  is not on a non-login PATH.
- **An inverted verdict.** The first version recorded
  `storage_billed_in_the_gpu_rate: true` *because the rate did not move*, which
  is exactly backwards. A rate that ignores disk size means storage is billed
  separately, and a quote derived only from GPU-hours understates a large-disk
  job. The field is now named
  `storage_billed_separately_from_the_gpu_hour`.

And one unit conflation: the platform's "4000" is decimal GB, `df`'s "3877G" is GiB. Both are recorded now, because reporting one alone reads as a discrepancy.

## Spike 6: two devices and sharded training

`spike6.py` plus [bootstrap6.sh](bootstrap6.sh), producing [`findings-spike6.json`](findings-spike6.json).
Three runs on 2× L4 at about ₹82/hr, roughly ₹20 in total. Two of the three
failed on my own tooling, and both failures are kept below because each one very
nearly became a wrong finding about the platform.

The verdict is split, and the split is the point.

| Claim | Result |
| --- | --- |
| 1. `num_gpus=2` attaches two devices | pass, 2× NVIDIA L4 on the host |
| 2. The pinned image sees both | pass, `torch.cuda.device_count() == 2` in the container |
| 3a. FSDP FULL_SHARD shards and steps | pass, 3 of 3 steps, `rc=0`, on Qwen3-0.6B |
| 3b. Checkpoint format | pass, `torch.distributed.checkpoint`, `.distcp` shards |
| 3c. Sharded resume | pass, resumed to step 4, `rc=0` |
| 3d. Did it actually train? | fail, loss `12.16 → 0 → 0`, `grad_norm: nan` on every step |

Claim 2 was the one with no evidence at all. The image was built and
digest-locked before multi-GPU was in scope, so nothing asserted the toolkit
passed more than one device through. It does.

Claim 3c was the one expected to fail. Sharded checkpoints are a format this
product has never written, and resume is proven at single-GPU (spike 4) with no
reason to carry over. It carried over.

### Taking steps is not training

In both model cases the loss starts plausible and collapses to zero on step 2, with `grad_norm: nan` reported on every step including the first.

I nearly shipped this as "capstone is a go." The first version of the
interpretation asked only whether the step counter moved and whether a
checkpoint appeared, and a run that shards correctly, steps, checkpoints,
resumes, and learns nothing passes that test completely. The check now asks
about the loss, and the verdict reads: *the mechanism works and the numerics do
not.*

The 8B full-FT capstone does not go on the calendar on this evidence. A sharded
run that produces a plausible-looking artifact from garbage weights is a *more*
expensive failure than one that will not launch, because it produces an
artifact, and nothing downstream would catch it. The `nan` gets explained first,
and is tracked as [#81](https://github.com/thp728/temper/issues/81).

Three candidates, none of them tested here, recorded as candidates rather than
as a cause: bf16 with FSDP2 and gradient checkpointing; the synthetic dataset
carrying only about 28 trainable tokens per step under assistant-only loss
masking; or `flash_attention: false` forcing an eager attention path.

### The numbers the predictor did not have

Peak VRAM per device, measured from `nvidia-smi` during training. The predictor's sharding arithmetic had zero real anchors and now has two:

| Model | Peak per device | Of 23,034 MiB |
| --- | --- | --- |
| Qwen3-0.6B full FT | 4,678 MiB | 20% |
| Qwen3-4B full FT | 21,162 MiB | 92% |

The 4B case did not complete, and that is a capacity result rather than a
mechanism one, because the small case had proved the mechanism on the same
machine minutes earlier. 48 GB across two cards is not enough for a 4B full
fine-tune with `adamw_torch`. Running two models exists precisely so those two
failures cannot be confused, and the earlier single-model runs could not tell
them apart.

Sharded checkpoints are large: 3.9 GB for 0.6B and 13.6 GB for 4B, with the
optimiser state two thirds of it. Extrapolated, and marked as extrapolated, a
70B sharded checkpoint is in the hundreds of GB, which is a question for
ADR-0004's transport rules and not only for the disk ceiling.

### Two failures that were mine, not the platform's

**Run 1, `permission denied ... /var/run/docker.sock`.** Recorded as *"FSDP does
not run in the pinned image"*. The probe had never reached the image. The
`ubuntu` user on a `--vm` instance is not in the `docker` group, and
`bootstrap4.sh` (removed, see git history) already used `sudo docker`. The
knowledge simply did not carry over. This is correction C16 repeating: a tooling
failure filed as a platform one. The fix is not only `sudo`. The probe now
refuses to report anything about FSDP when the container reports no devices,
because *a probe that could not run is not evidence about what it would have
found.*

**Run 2, "0 steps completed" beside "a checkpoint was written".** A
self-contradicting finding. The step counter grepped the training log for a
pattern the log did not use, while 25 GB of `.distcp` shards sat on disk. Steps
now come from `trainer_state.json`, and a written sharded checkpoint is treated
as proof the mechanism ran regardless of the exit code. The same run reported
`ChildFailedError, exitcode 1` with no cause, because `torchrun` swallows child
stderr. `--tee 3` now captures it, and it is how the loss collapse was found at
all.

Run 3 is the one whose numbers are above.

### One more footgun, closed

A `--dry-run` wrote its preflight over `findings-spike6.json`, destroying a
completed run's measurements. A preflight that costs nothing must not be able to
delete evidence that cost money. Dry runs now write to
`findings-spike6-dryrun.json` and say that the real file was left alone.

The verdict in the findings file was recomputed from the *same* captured probe
report after the interpretation gained the numerics check. No machine was
provisioned a second time. The measurements are unchanged and only the reading
of them moved.

## Spike 7: Axolotl's config schema

`spike7.py` plus `introspect_axolotl.py`, producing [`findings-spike7.json`](findings-spike7.json)
and [`packages/contracts/axolotl-field-tiers.json`](../packages/contracts/axolotl-field-tiers.json).
No GPU, no VM, no money. It runs the pinned digest locally, because reading the
schema from a pip-installed Axolotl would measure a different trainer than the
one that runs jobs.

Axolotl `0.19.0.dev0`, `axolotl.utils.schemas.config.AxolotlInputConfig`.

| | |
| --- | --- |
| Total fields | 388 |
| After excluding infrastructure | 344 |
| Carrying a default | 386 |
| Required | 2 |
| Free-form (`Any` or bare dict) | 19 |
| Fields with schema bounds (`ge`/`le`/pattern) | 1 |
| Fields typed as an enum or `Literal` | 19 |
| `model_validator` hooks | 113 |
| `field_validator` hooks | 27 |

The spec asked whether *"forty is a UI, four hundred is a different product"*. It is 344. Tiering is mandatory rather than a nicety.

### The risk the spike existed to surface, and it fired

Roughly 12% of Axolotl's constraints are visible to a schema reader. One field
carries a numeric bound. Nineteen are enums. 113 `model_validator` hooks hold
the rest, and a `model_validator` is arbitrary Python, so a form generated from
the schema cannot know what it will refuse until the job is already running.
That is precisely the failure the spec named: *a generated form will happily
accept combinations that fail four minutes into a paid job.*

That 12% is a ceiling rather than an estimate. It counts each validator as one rule, and one validator commonly encodes several.

The verdict is to derive the form and hand-write the rules. That is neither the
clean derivation the Advanced-mode design assumed nor the hand-enumerated
fallback. The field list, types and defaults are generated from the pinned
digest, so they cannot drift from the trainer. The cross-field rules are
hand-enumerated and reviewable in a diff, because the schema does not contain
them.

All twelve correctness settings from `AGENTS.md` are expressible in the config,
under names now recorded. One is thinner than expected: loss masking resolves to
`train_on_inputs` alone, because `roles_to_train` and `train_on_eos` are not
top-level fields in this version.

### The tier classification, and the number that actually sizes the ticket

Twenty fields, drawn as a seeded random sample of the 344. Random rather than
hand-picked on purpose, because a sample of fields the product already uses
would be quick to classify and would underestimate the rest. It took 27 minutes
of real work, 1.35 min/field, which projects to 7.7 hours for a full pass.

But the field count is the wrong number to estimate from. 13 of 20 came out
`known_but_unsupported`: DPO, Q-GaLore, LISA, EAFT, whole methods this product
does not do, each needing a one-line reason rather than a design. If 65% holds
across the 344, the surface needing real design work is about 120 fields rather
than 344, and that is the number the ticket should carry.

The highest-value entry in the sample is `special_tokens`. Set `pad_token` equal
to `eos_token` and the loss mask hides every end-of-sequence token, so the model
is trained never to stop. Training completes, the loss curve looks fine, and the
failure is invisible until inference.

## Spike 8: the SDK surface

`spike8.py`, producing [`findings-spike8.json`](findings-spike8.json). No GPU,
no money, and every live call is a read. 37 capabilities across 7 namespaces and
43 public methods, each marked separately for documented, in-SDK, and
tested-here, because *"documented"* and *"works on VMs"* have already been shown
to be different claims here (C11: startup scripts accepted and silently
ignored).

The `documented` column is transcribed from the vendor's own `SKILL.md`, which
ships inside the wheel, so unlike the website it cannot drift away from the code
under test between one reading and the next. Six of its claims are asserted as
substrings, so a reworded doc fails loudly instead of leaving a stale
transcription looking current.

### Pause exists

`instances.pause()` and `instances.resume()` are both in the SDK, and `SKILL.md`
states: *"Paused (compute billing stopped, storage billing continues, data
persists)"*. A paused machine was observed in the live account, so the backend
holds the state and it is not merely an attribute.

[ADR-0003](../docs/adr/0003-cancellation-is-destructive.md) is flagged for
reopening, and the flag says why it is not a simple win. A paused machine still
bills for storage and has no run that owns it, which is exactly the shape the
reconciler exists to destroy. "Pause instead of destroy" is only cheaper if
somebody eventually destroys it.

Two caveats the docs are explicit about, and that any implementation must
handle: resume is region-locked, and resume may return a new `machine_id`, so a
stored id must be re-read.

### The rest, briefly

- **Nothing pushes.** No webhooks, no event stream, and no server-side log API.
  The CLI's own `logs` command SSHes in. Every state change is discovered by
  polling. [ADR-0001](../docs/adr/0001-event-channel-over-ssh-stdout.md) chose
  stdout-over-SSH, and it turns out there was no provider alternative to reject.
- **Spot is containers-only.** `is_spot` with `template="vm"` raises. A product
  that needs VMs for Docker cannot have spot pricing. That closes a cost avenue
  rather than opening one, and it is better closed on paper.
- **Persistent filesystems exist** (`create`, `list`, `edit`, `remove`,
  attachable at create or resume). They are the obvious home for a model cache
  that would otherwise be re-downloaded per job.
- **Serverless deployments exist** (`jl deploy`), and they are the one place the
  platform reports cost per resource. Out of scope for this submission, recorded
  so the cut is informed rather than accidental.

### What I got wrong, found by running it

The first version of the table recorded *"per-instance cost to date"* as absent,
reasoning that no call is named `cost`. The live probe disproved it: every
`Instance` row carries a `cost` float and a `runtime` string. The product
computes spend from uptime × hourly rate and never reads the provider's own
figure, so there is a second, independent number available to reconcile against,
which is worth having when the first is an estimate. The row is corrected in
place and labelled as a correction.

## Spike 9: streaming validation throughput

`spike9.py` plus `streaming.py` and `test_streaming.py`, producing
[`findings-spike9.json`](findings-spike9.json). No GPU, no money.

`streaming.py` is a streaming rewrite of `api/validation.py`, written so the
measurement is of the real work rather than of a benchmark. 26 tests pin it
against the shipped validator: for any dataset small enough to validate both
ways, both must reach the same verdict, the same counts and the same line
numbers. Without that, a streaming pass that is fast because it checks less
would report a throughput that means nothing.

| Dataset | in-memory (shipped) peak RSS | streaming peak RSS | streaming MB/s |
| --- | --- | --- | --- |
| 1 GB | +5,930 MB, 5.93× the file | +4.1 MB | 23 |
| 5 GB | not run, would need about 30 GB | +3.8 MB | 18 |
| 20 GB | not run, would need about 96 GB | +3.6 MB | 24 |

File size ×20 gives peak memory ×1.14. That is the whole claim, and it holds.
The retained memory does not merely stay flat, it drifts *down*, because what is
retained is a fixed set of caps and not a function of the file at all.

Streaming is also faster than the in-memory path, at 23 MB/s against 15 MB/s at
1 GB. The ceiling is not a memory-versus-speed trade. The in-memory path spends
its time allocating. It is simply a property of code that predates the need.

### Token counting has to become its own phase

The quote is priced per training token, so counting is a hard prerequisite of
the quote. Tokenisation is 79 to 88% of a tokenising pass, and it drags the
throughput from about 22 MB/s to 3.4 MB/s.

| | |
| --- | --- |
| Validate alone | about 21.6 MB/s, so 1.3 GB inside a 60s wait |
| Validate and tokenise | 3.4 MB/s, so 5 GB takes 1,485s |

The kill criterion in the spec fires. Validation and token counting split into two phases: validate on upload, count tokens asynchronously before the quote, and the quote gains a `counting` state.

The 60-second budget is a judgement rather than a measurement, and is labelled as one in the findings. It is the point at which a synchronous upload page stops reading as a wait and starts reading as a hang.

### This run shared the machine, and why the verdict survives it

These passes ran on a laptop that was concurrently pulling a multi-GB Docker
image for spike 7 and orchestrating spikes 5 and 6. The MB/s figures are a floor
rather than a rate, which is visible in the data, where the same pass reports 18
MB/s at 5 GB and 24 MB/s at 20 GB, and no larger file is genuinely faster.

Rather than assert the number is good enough, the findings ask how much faster the machine would have to be to change the answer:

| If tokenisation were | 5 GB takes |
| --- | --- |
| as measured | 1,485s |
| 2× faster | 743s |
| 4× faster | 371s |
| 10× faster | 149s |

A 24.8× speed-up would be needed to fit the 60s budget. The split
recommendation does not depend on the contended measurement. Re-running on an
idle machine would sharpen the number and would not move the decision.

The memory numbers are unaffected. Peak RSS does not care what else is on the CPU, and that is the half of this spike that carries the claim.

### Correction to ADR-0005

[ADR-0005](../docs/adr/0005-the-dataset-size-limit-is-derived-from-measured-memory.md)
derives the 1 GB limit from a measured 4.8× memory multiplier. Measured again at
a real 1 GB file, it is 5.93×, so a 1 GB dataset peaks near 6 GB rather than 4.8
GB. The multiplier was extrapolated from small files and the extrapolation was
optimistic. The limit is still defensible on the development machine, with less
headroom than the record claims. Corrected in place rather than quietly fixed.

### Where the streaming path would have leaked

The line-number guarantee is free while streaming, because a line number is a
counter. The samples are not. `trainer/thinking.py`'s `detect()` accumulates one
line number per assistant turn before capping them at the end, which is bounded
by the file rather than by the cap. The streaming path caps as it goes, and a
test drives 2,000 rows through it to prove the retained list stays at five.

Everything retained is capped: the first N errors, the first N warnings, a
three-row preview, twenty key sets, and ten sample line numbers. `error_count`
is exact while the list is capped, because *"every line is broken"* is a
different problem from *"line 4,102 is broken"*.

### Two deliberate divergences, recorded rather than hidden

- **Encoding errors name a line.** The in-memory path decodes the whole file at
  once, so it can only report a byte offset with `line: null`. Decoding a line
  at a time can say which line, which is strictly better for the user.
- **Rows split on newline only.** `str.splitlines()` also splits on U+000B,
  U+000C, U+001C–U+001E, U+0085, U+2028 and U+2029, so a JSON string containing
  a literal U+2028 is two rows to the shipped validator and one row here. JSONL
  is newline-delimited by definition, so this is the shipped code's accident
  rather than its intent.

### A bug found by reading, not by a failing test

`_iter_lines` tested the *chunk* for a byte-order mark rather than the
accumulated buffer. At `chunk_bytes=1` a BOM is split across three reads, so two
of its bytes stayed at the head of line 1 and every row failed to parse. The
parametrised chunk-size tests now go down to 1 byte for exactly this reason.

## Phase B spikes: what is still open

Each of these is a known gap rather than an oversight, and each is written down because the alternative is remembering it.

1. **The `nan` in spike 6**, tracked as
   [#81](https://github.com/thp728/temper/issues/81). The single highest-value
   open question here: FSDP shards, checkpoints and resumes, and the loss
   collapses to zero with a `nan` grad_norm on the first step. Candidates,
   untested: bf16 with FSDP2 plus gradient checkpointing; a synthetic dataset
   with about 28 trainable tokens per step under assistant-only masking;
   `flash_attention: false` forcing an eager path. The 8B capstone is blocked
   on this.
2. **8 devices is not 2 devices.** Spike 6 proved a 2-device NCCL topology on
   one card type. The architecture's claim about 8 cards on one host stays
   *measured but unexercised*, so do not upgrade the wording on this evidence.
3. **ADR-0009 is written but unexercised.** Spikes 5 and 6 supplied the sizes
   that force it, a 7200 GB ceiling and a 13.6 GB sharded checkpoint at 4B, and
   [ADR-0009](../docs/adr/0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md)
   supersedes ADR-0004's property 2 on that evidence. Nothing has yet written
   an artifact to a pre-signed URL from a JarvisLabs VM.
4. **The storage line item is unmeasured.** Spike 5 established that the
   GPU-hour rate does not move with disk size, so storage bills separately. The
   documented figure is $0.10/GB-month, and a spike lasting minutes cannot
   observe a GB-month. The quote needs the real number.
5. **Pause is documented, not measured.** Spike 8 read the vendor's claim that
   pausing stops compute billing and keeps storage billing. Nobody has paused a
   machine here and watched the meter. Reopening ADR-0003 should start by doing
   that.
6. **324 of 344 Axolotl fields are unclassified.** Spike 7 classified a
   20-field sample to time the pass. `docs/data/axolotl-field-tiers.json` says
   so in its `_status`.
7. **Spike 9's throughput was measured on a busy machine.** It is a floor rather
   than a rate. The findings show the verdict survives a 10× speed-up, so this
   is worth sharpening rather than urgent.
8. **The on-VM probes are silent while they run.** `subprocess.run` captures
   stderr and prints it only on return, so a spike that provisions a billing
   machine shows nothing for twenty minutes. Streaming it would make cost
   exposure visible as it accrues. Deliberately not changed after the fact,
   because the committed scripts are the ones that produced the committed
   findings.

## Files

The bootstrap scripts collapsed to one. Spikes 2 to 5 each carried a
near-duplicate bootstrap script (`bootstrap.sh`, `bootstrap3.sh` through
`bootstrap5.sh`). They were the right artifact while the investigation ran and
read as clutter once it concluded. They are removed, and
[bootstrap6.sh](bootstrap6.sh), the one representing the final approach, is what
remains. Re-running an early spike needs its script retrieved from git history.
The findings files all stay, because they are the measured evidence.

**Phase 0 (spikes 1 to 4)**

- [spike.py](spike.py), orchestration, run from your machine
- [probe.sh](probe.sh), runs *on* the VM as a startup script and writes `/root/probe-report.json`
- `findings.json`, generated output, git-ignored because it records the account balance

**Phase B (spikes 5 to 9)**

- [spike5.py](spike5.py), disk ceiling and download throughput (its bootstrap is removed, see above)
- [spike6.py](spike6.py) plus [bootstrap6.sh](bootstrap6.sh), two devices, FSDP, sharded resume
- [spike7.py](spike7.py) plus [introspect_axolotl.py](introspect_axolotl.py), Axolotl's config schema
- [spike8.py](spike8.py), the provider SDK surface
- [spike9.py](spike9.py) plus [streaming.py](streaming.py), streaming validation throughput
- [teardown.py](teardown.py), shared by 5 and 6. Destroy, then *prove it* across
  consecutive listings (C17). The one thing that must not vary between two
  spikes that both provision GPUs is the code that turns them off
- [test_streaming.py](test_streaming.py) and [test_spike5.py](test_spike5.py),
  the parts that can be tested without spending money. `python -m pytest spike/ -q`
- `findings-spike5.json` through `findings-spike9.json`, committed deliberately

This directory is a diagnostic artifact, not the product. It is kept in the repository because the findings are the evidence base the architecture decisions rest on, and a claim whose measurement lives somewhere else is a claim you cannot check.
