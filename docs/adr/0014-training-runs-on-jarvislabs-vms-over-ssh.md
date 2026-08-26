# ADR-0014 — Training runs on JarvisLabs VMs with an SSH-driven bootstrap

- **Status:** accepted
- **Date:** 2026-08-17

> **A note on the number and the date.** This decision was recorded on
> 2026-08-17 in the private working vault where the first thirteen decisions
> were logged, and copied into this directory on 2026-08-26
> ([#26](https://github.com/thp728/temper/issues/26)). Its file number reflects
> when the record landed here, not when the decision was made; the wording is
> the original, with private-vault links replaced by descriptions of what they
> pointed at.

## Context

JarvisLabs offers two execution models and neither is the "run my container image" primitive that RunPod and Modal treat as default. Container instances are fixed templates — the docs state plainly that *"templates are container-based, so Docker cannot run inside them"* — so they cannot run a custom image at all. That rules them out, because a pinned image is the only defensible answer to the dependency-instability risk the reference architecture named: transformers, TRL, PEFT, bitsandbytes, FlashAttention, PyTorch and CUDA move together, and an unpinned environment reinstalled per run is not reproducible.

VMs can run Docker, and two spikes proved the path end to end. Docker 29.3.1 ships preinstalled with its daemon running, `nvidia-container-toolkit` is present, a 3.3 GB PyTorch image pulled in 72s, **digest pinning round-tripped** (resolve, then re-pull by digest), `torch.cuda` saw the L4 with bf16 support inside the container, and a tensor written through a bind mount survived container exit and returned to the orchestrator.

The startup-script hook does not work on VMs — `create()` accepts `script_id` and silently ignores it — so the bootstrap moves to SSH. That was proven in spike 2: a bootstrap script piped over SSH, unattended, 94s wall clock (the early per-spike bootstrap scripts have since been collapsed; [bootstrap6.sh](../../spike/bootstrap6.sh) is the one that remains).

## Decision

Execute training on JarvisLabs **VM** instances (`template="vm"`), with the orchestrator connecting over SSH to pull a digest-pinned Docker image and run the container. **Not** container instances, and **not** the startup-script bootstrap the reference architecture described.

## Alternatives considered

*Container instances with a startup script installing dependencies* — rejected: no custom image, so no immutability, and the dependency-instability risk stands unmitigated. *Keep the startup-script bootstrap on VMs* — rejected as impossible, not undesirable; it does not execute. *Shell out to the `jl` CLI* — unnecessary, since `template="vm"` reaches VM mode from the SDK directly.

## Consequences

**Tradeoffs, and they are real:**

- **Spot pricing is forfeited entirely.** VMs reject `--spot`, and spot is up to 56% cheaper (H100 ₹112.59 against ₹255.15). This is not a deferred feature — it is unreachable on this path.
- **The GPU catalog narrows.** A30 and A100-40GB are container-only. Any policy naming A100-40GB as the 8B default is unimplementable.
- **The control plane now holds SSH keys**, which is new attack surface and new key-rotation work that the startup-script design avoided.
- **Cold start is ~131s** before training begins: 13–17s to `Running`, ~46s to SSH-reachable, 72s to pull the image.

## Rollback

Container instances remain viable for **serving**, where a custom image matters much less and the platform port-proxy is a genuine convenience. If SSH bootstrap proves unreliable under load, serving can move to containers independently without touching the training path.

**⚠️ Blocking prerequisite this creates:** VMs come up with a public IP and `ufw` **inactive**. Spike 2 reached a container on port 8000 from the open internet, unauthenticated, within seconds. *"Training VMs expose no public application ports"* asserts the opposite of what the platform provides. **Firewalling is part of bootstrap, not hardening**, and no tenant data goes near a VM until it is in place.
