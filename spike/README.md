# Phase 0 Vertical Spike

**Purpose:** settle the provider assumptions that everything else rests on, before writing product code. [reference-technical-architecture.md](../reference-technical-architecture.md) §32 names JarvisLabs VM automation as the highest technical risk in the build, and §30 says *"do not begin broad product implementation until this path works."*

**This is a probe, not a prototype.** It deliberately does no training. A spike that also trains cannot tell you which half broke.

## What it answers

| # | Question | Why it blocks the build |
| --- | --- | --- |
| 1 | Does auth work, is there balance? | Trivial, but a wrong env var wastes an evening |
| 2 | Is an SSH key registered? | **Hard prerequisite for `--vm`**, and the failure message doesn't say so |
| 3 | Which GPUs are free **for VMs**? | The CLI docs warn *"a GPU may have free devices for containers but not VMs"* — separate capacity pools, possibly separate pricing |
| 4 | Can the SDK create a VM at all? | `--vm` is documented for the CLI, **not** the SDK. If it's CLI-only, orchestration must shell out to `jl` — a materially worse story |
| 5 | Do startup scripts upload and attach? | The only mechanism for getting anything onto a bare VM |
| 6 | **Is Docker present and usable?** | ⚠️ **Correction C1.** §14 assumes it. The SDK can't supply a custom image, so if Docker isn't there the whole immutable-image approach needs replacing |
| 7 | Does GHCR pull work? Does `--gpus all` work? | Docker existing ≠ Docker useful. Separates "no Docker" from "no egress" from "no nvidia-container-toolkit" |
| 8 | Is outbound HTTPS open? | §15's callback protocol and every R2 signed URL depend on it, and it fails *late* |
| 9 | Does teardown always happen? | An orphaned GPU bills until someone notices. This is the rehearsal for the `finally` path in §15 |

## Setup (once)

```powershell
pip install jarvislabs
```

**Credentials** — put the key in `.env` next to this file (git-ignored; copy `.env.example`):

```
JL_API_KEY=<from jarvislabs.ai/settings/api-keys>
```

**SSH key** — mandatory before any VM can be created, because VMs have no password login and the public key is baked in at boot:

```powershell
ssh-keygen -t ed25519 -C "jarvislabs" -f "$HOME\.ssh\id_ed25519"
# if you set a passphrase, enable the agent so scripts don't hang on a prompt:
#   (admin, once)  Set-Service ssh-agent -StartupType Automatic; Start-Service ssh-agent
#   (normal)       ssh-add "$HOME\.ssh\id_ed25519"
```

Then register the **public** half (`.pub`) via the JarvisLabs dashboard or `jl ssh-key add`.

## Running it

```powershell
python -u spike.py --dry-run   # preflight only — provisions nothing, costs nothing
python -u spike.py             # full run, destroys the instance afterwards
python -u spike.py --keep      # leaves it up for manual poking (COSTS MONEY)
```

**Use `-u`.** Python buffers stdout when redirected, so without it a backgrounded run shows nothing until it exits.

**Run `--dry-run` first.** It costs nothing and catches the likely blockers — missing SSH key, no VM-capable GPU free.

Output: console narrative plus `findings.json` (git-ignored — it carries machine IDs and your balance).

## Cost

Picks the cheapest **VM-capable** GPU with a free device. Measured 2026-08-17: **L4 at ₹41.31/hr**. Billing is per-minute, a full run is ~3–4 minutes, so **around ₹2–3**.

⚠️ Prices are **INR** on this account — `account.currency()` returns `INR`, not USD. The USD figures on the pricing page are a different denomination of the same rate.

⚠️ VMs force a **100 GB minimum disk**, billed at $0.10/GB-month while the instance exists.

⚠️ **`--keep` leaves the instance billing.** Destroy it with `jl destroy <machine_id>`.

## Results — 2026-08-17

Run against the real account. **Total spend across all attempts: ₹9.64.**

**Resolved favourably.** A `--vm` instance is Ubuntu 22.04.5 with **Docker 29.3.1 preinstalled and running**, `nvidia-container-toolkit` present, GHCR pull working, `docker run --gpus all` working, egress open to ghcr.io / huggingface.co / pypi.org. L4 shows 23,034 MiB, driver 580.126.20. Python 3.10.12 and git present, no `uv`. **The immutable-image approach holds.**

**Measured, replacing guesses:** VM reaches `Running` in **15–17s**, but SSH refuses connections for a further **42s** — usable at roughly **T+60s**, not T+16s. Teardown succeeded first attempt, 4/4, with no strays.

**⚠️ The blocking finding:** **startup scripts do not run on `--vm` instances**, and `create()` accepts `script_id` without error. Confirmed by reading cloud-init on a live VM — the user-data contains only JarvisLabs' own config and `runcmd` has no trace of the supplied script. **Nothing fails; the job just never starts.** `probe.sh` therefore cannot self-execute, and the diagnostics above were gathered over SSH by hand. See correction C11.

**What that means for the build:** §14's bootstrap needs rewriting around the orchestrator SSHing in and driving the pull itself, which puts SSH key handling in the control plane. The alternative — container instances — forfeits custom images, which is the thing Docker-on-VM was buying.

## Spike 2 results — 2026-08-17, all checks passed

`spike2.py` + `bootstrap.sh`. **The SSH-driven bootstrap works.** Exit 0, instance destroyed, no strays.

| Test | Result |
| --- | --- |
| Bootstrap over SSH, unattended | ✅ 94s wall clock |
| Image pull (`pytorch:2.5.1-cuda12.4-cudnn9-runtime`) | ✅ **3.3 GB in 72s — 46 MB/s** |
| Digest pinning (resolve → re-pull by digest) | ✅ **immutable runs are achievable** |
| `torch.cuda` inside container | ✅ NVIDIA L4, **bf16 supported**, torch 2.5.1+cu124 |
| Artifact out via bind mount | ✅ 263 KB tensor survived container exit |
| **C7 — public port** | ✅ `http://…:8000/ → HTTP 200` — **C7 retracted, VM serving works** |

**Cold start, measured end to end: ~131s** — 13s create + 46s SSH-ready + 72s image pull. That replaces a guess in the quote's ETA, and the image pull is the biggest single component.

### 🚨 The finding that matters most

**VMs come up with a public IP and `ufw` inactive** (`Status: inactive` on a fresh box). The port test succeeded *because nothing is filtering* — a container bound to `0.0.0.0:8000` was reachable from the open internet, unauthenticated, within seconds.

Architecture §23 states *"training VMs expose no public application ports."* **The platform provides the opposite.** Anything the trainer or vLLM binds is exposed by default. **Firewalling belongs in bootstrap, not in §7 hardening**, and no tenant data should touch a VM before it exists. Logged as correction C12.

## Spike 3 results — 2026-08-17

`spike3.py` + `bootstrap3.sh`. Two attempts; the first failed usefully. **QLoRA runs. Resume does not.**

| Test | Result |
| --- | --- |
| **QLoRA on Qwen3-4B** | ✅ 4 steps, loss 3.99, **peak VRAM 5.31 GB of 24 GB** |
| Trainable parameters | ✅ **33,030,144 — exactly the count derived from `config.json` beforehand** |
| Model load / train | 32s / 10.9s · adapter 132.2 MB · checkpoints at steps 2 and 4 |
| Stack install | ✅ 15s |
| **Checkpoint resume** | ❌ **blocked by a version chain — see below** |
| Firewall — `ufw` default-deny | ❌ reports `active`, protects nothing |
| Firewall — `DOCKER-USER` on published port | ❌ rule installs, never matches |
| **Firewall — bind `127.0.0.1`** | ✅ **works** |

### 🔴 Resume is impossible on an unpinned stack — a three-link chain

Installing *latest* gave `transformers 5.15.0` + `trl 1.10.0` + `peft 0.20.0` + `bnb 0.50.1` on `torch 2.5.1`:

1. **TRL 1.x's `SFTConfig` no longer inherits `TrainingArguments`** — `warmup_ratio` raises `TypeError`. One of the defaults the research settled on cannot be expressed.
2. `save_safetensors` was also rejected → checkpoints written as torch `.bin`.
3. transformers 5.x then **refuses to load them**: *"we now require users to upgrade torch to at least v2.6… does not apply when loading files with safetensors."*

**Net: §31's "a run survives a forced interruption" fails** — not from a design flaw, but from four packages moving independently. The architecture names dependency instability as a risk and prescribes pinned images; **this is that risk firing on the first real attempt.** Pin the stack before building anything else.

### 🚨 Two firewall mitigations that look correct and do nothing

- **`ufw` default-deny** reported `Status: active` while the port stayed open. `ufw` filters `INPUT`; Docker publishes via NAT/`FORWARD`, so container traffic never traverses `INPUT`.
- **`DOCKER-USER` rule on the published port** installed cleanly and never fired — **by `DOCKER-USER` the packet is already DNAT'd**, so the destination port is the container's (80), not the published one (8000).
- **Binding to `127.0.0.1` worked**: answered on localhost, timed out from outside.

**Rule for the build: training containers publish nothing.** The trainer needs no inbound port. Serving needs one, and must match the **container** port in `DOCKER-USER` — and be verified from outside, never assumed.

### Known spike bug (not a platform issue)

Host-side `sha256sum` on the adapter failed with `Permission denied` — the container writes as root, the host reads as `ubuntu`. Same trap as C10. The adapter was written correctly (132.2 MB); the verification step needs `sudo`.

## Spike 4 results — 2026-08-18 — THE TRAINER IMAGE WORKS

`spike4.py` + `bootstrap4.sh` + [`../trainer/`](../trainer/README.md). Three attempts; the first two failed on my own tooling, not the platform.

| Test | Result |
| --- | --- |
| Image builds from pinned digest | ✅ **183s**, 8.5 GB |
| Job runs through `/job` → `/out` contract | ✅ **120.9s**, 64 rows, exit 0 |
| `axolotl train` accepts `warmup_ratio` | ✅ — the exact argument TRL 1.x removed |
| **Checkpoint resume** | ✅ **resumed from `checkpoint-2` → step 6 in 54s** |
| Adapter format | ✅ **safetensors** (not torch `.bin`) — 132.2 MB, sha `d311285e…` |
| Adapter config | ✅ `r=16 alpha=32 rslora=False`, targets = all 7 linear modules |
| Unknown job keys refused | ❌ **found a real bug in my entrypoint** — fixed |

**C14 is closed.** Resume is the thing spike 3 could not do at all. Pinning to Axolotl's tested stack by digest — rather than resolving six packages by hand — removed the `transformers`/`trl`/`torch` conflict that made checkpoints unloadable. §31's *"a run survives a forced interruption"* criterion is now achievable.

The `adapter_config` is worth reading closely, because it confirms the design end to end: `target_modules` came back as `gate_proj, down_proj, v_proj, o_proj, q_proj, k_proj, up_proj` — **all seven**, exactly the list derived from `config.json` in [wiki/foundations.md](../wiki/foundations.md). `alpha=32` is `2r`. `rslora=False` is correct at `r=16`.

### The bug this caught in my own code

The job spec deliberately carried `not_a_real_key`, and it was **not** refused. `build_config` validated keys inside `hyperparameters` but never the top level — so a caller misspelling `max_steps` as `maxSteps` would have got a full-length training run with no warning. Now both levels are validated, with `_comment*` keys ignored by design.

### Two failed runs, and neither was the platform

Attempts 1 and 2 reported *"ssh ready — no answer within 240s"*. I logged that as a provisioning-reliability finding (C16) and **it was wrong**. I had launched those runs through Git Bash, which ships its own `ssh` and cannot see the Windows `ssh-agent` service, so the passphrase-protected key was unusable. Verbose SSH showed `Server accepts key` immediately followed by `Permission denied (publickey)` — the key was provisioned correctly the whole time. Relaunched through PowerShell, SSH connected in **37s**.

**C16 is withdrawn.** Two lessons kept: a readiness check must distinguish *unreachable* from *authentication failed*, because those have opposite remedies and mine collapsed both into "no answer"; and changing tooling mid-investigation is a confound worth testing before blaming the provider.

### Still open

- **fp32 adapters.** 132.2 MB is 33M × 4 bytes. bf16 would halve it. Undecided.
- **Qwen3 thinking mode** — `<think>` blocks versus a dataset without them (C2). Unhandled.
- **`chat_template: tokenizer_default`** ran without error, but that it resolved to the *right* template has not been asserted. The export-time probe from [report-a.md](../2026-finetuning-research/report-a.md) §4.2 is the check.
- **Not pushed to GHCR.** The image is built per-VM. Pushing needs a token and ideally CI.

## Spike 4 — what to test next

**Everything below is blocked on pinning the stack.** C14 means the current "install latest" approach cannot resume a run, so this is the next thing built, not the next thing tested:

1. **Build the trainer image with an exact lockfile** and push to GHCR by digest. Resolve the version set that makes resume work — almost certainly `torch >= 2.6` plus whichever TRL expresses the intended defaults, or an explicit decision to drop `warmup_ratio`. **Then re-run spike 3 unchanged against that image**; it becomes the regression test.
2. **Resume, properly.** Kill mid-run rather than re-instantiating in-process, and confirm the step counter, optimizer state, RNG state and dataloader position all restore.
3. **Adapter integrity** with `sudo` on the host-side hash, and a decision on fp32 vs bf16 adapters — 132.2 MB versus ~66 MB is a real artifact-size difference.
4. **tokens/sec on a real dataset.** The 4-step run was too short to calibrate MFU; the estimator's 35–50% band is still the softest number in the cost model.
5. **Serving**, once training is stable: vLLM on a VM, port matched correctly in `DOCKER-USER`, verified unreachable except through the gateway.

**Still no reason to spike container instances.** The docs are explicit — *"templates are container-based, so Docker cannot run inside them"* — so they cannot run a custom image, which is the entire reason VMs were chosen. C7's retraction removed the last argument for containers in serving too.

## Reading the result

`spike.py` prints a "WHAT THIS MEANS" section that maps the probe output onto the decision it forces. The one that matters:

- **Docker present, daemon up, GHCR reachable, GPU passthrough works** → C1 resolves favourably; §14 holds as written; mark C1 resolved in the architecture's §0.
- **No Docker** → C1 resolves unfavourably, and a decision is required between (a) installing Docker in the startup script, (b) a `uv`-provisioned environment from a lockfile, or (c) abandoning `--vm` for a container template. **Each forfeits something different**, and §5, §14 and principle 10 all need rewriting rather than patching. Log it in [decisions.md](../decisions.md) the same evening.

## Afterwards

1. Update correction **C1** in [reference-technical-architecture.md](../reference-technical-architecture.md) §0 from 🔴 to resolved, with what was found.
2. Log any forced decision in [decisions.md](../decisions.md) — **the same evening**, per the brief's standard that reasoning gets recorded, not reconstructed.
3. Record the measured boot time. It feeds the duration estimate in the quote, which is the product's differentiator, and a guessed boot time makes the quote wrong from day one.

## Files

- [spike.py](spike.py) — orchestration, run from your machine
- [probe.sh](probe.sh) — runs **on** the VM as a startup script; writes `/root/probe-report.json`
- `findings.json` — generated output, git-ignored in the real repo

⚠️ **This directory is a diagnostic artifact, not the product.** When the real repo exists, the spike moves to it (or to `infra/jarvislabs/` per §6) — it should not stay in the vault long-term.
