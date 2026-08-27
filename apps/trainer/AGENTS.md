# Trainer

The pinned container that runs one job on one machine. `README.md` holds the image contract, the
`/job` to `/out` table, and what each spike proved. This file holds the rules that constrain
changes. Moving to `apps/trainer/` under issue #15.

## The pin

**The digest is the contract. The tag is a comment.** Changing either is a deliberate act that
re-runs the GPU smoke test.

**Never `pip install` inside this image.** The base carries a torch, transformers, peft, trl and
bitsandbytes matrix that Axolotl's maintainers resolve and test together. Installing anything
reintroduces the exact resolution problem the pinned base exists to avoid: spike 3 lost
checkpoint-resume to four packages moving independently. That is also why this package declares no
runtime dependencies and never will. Its `pyproject.toml` exists so pytest can find the two test
files.

**Anything the entrypoint imports must be listed in the `COPY`.** `thinking.py` was missing from it
until 2026-08-19. The image built fine and died at import on the GPU, before the `try/finally` that
writes `result.json`, surfacing as `training_failed: "Trainer produced no result.json"` — an error
pointing at training and saying nothing about the image.

**Axolotl owns the training loop; we own the contract.** Do not call TRL or PEFT directly. Those
APIs move underneath you.

## Correctness settings

**Default-locked, not hidden.** Chat-template resolution, `train_on_inputs: false`, EOS handling,
NF4 double-quant, all-linear LoRA targets, bf16, seed. These are the highest-frequency
silent-failure surface: they pass every obvious health check and only surface as garbage
generations, so the common path never touches them.

**Advanced mode exposes every one of them**, with the failure mode named inline and the override
recorded in the job spec. Every serious platform in this space exposes these, and hiding them
permanently is a limitation dressed as a safety feature. The export-time template probe is what
makes exposure safe.

**The trainer resolves nothing (#83).** The job spec arrives carrying the full resolved set; this
entrypoint applies it as given and holds no defaults of its own. A spec missing a required key
fails loudly (`spec_incomplete`, keys named) rather than falling back; unknown keys are echoed
back as `rejected_overrides`. Since #33 the known-key set is generated from the pinned image's own
schema (`axolotl-schema.json`, shipped into the image): *unknown* means unknown to the trainer,
not absent from a hand-written list, and `test_agreement_with_the_domain.py` pins the trainer's
sets to exactly what `temper_core.hyperparams.effective` and `temper_core.surface.known_keys`
produce.

**α tracks r at resolution time.** Move `lora_r` without `lora_alpha` and α recomputes as `2r`
before launch, in `temper_core.hyperparams` -- never pair a new rank with a stale scale, and
never recompute it here.

**rsLoRA is inferred at `r >= 32`**, in the resolver before launch, never exposed.

**Thinking mode is detected from the dataset**, applied identically at training and serving.
`thinking.py` here and the control plane's validation must agree.

## Multi-GPU

Provisioning, not architecture. JarvisLabs VMs take up to 8 GPUs, `num_gpus` is a create parameter,
and 8 devices are free on every VM-capable type. 8× H100 is 640 GB, which covers 70B full
fine-tuning on one host via `accelerate` FSDP FULL_SHARD, which Axolotl configures.

⚠️ **Partly exercised as of spike 6 (2026-08-23), and the split matters.**

- **Proven on 2× L4:** the pinned image sees both devices, FSDP FULL_SHARD shards and steps, a
  `.distcp` checkpoint is written, and sharded resume works.
- **Not proven:** the loss collapses to zero with a `nan` grad_norm. **The mechanism runs and the
  numerics do not.** Taking steps is not training. Tracked as issue #81.
- **Untouched:** 8 devices is a different NCCL topology from 2.

**Never describe multi-GPU training as working without naming the `nan`.**

## Security

**The trainer publishes no ports.** `ufw` does not filter Docker-published ports, and a
`DOCKER-USER` rule matched on the published port never fires, because the packet is already DNAT'd
and carries the *container* port. Not publishing is the only mitigation that holds.

The machine is a pure compute node reached only by the control plane, with one exception: it may
write its own artifact to a pre-signed URL scoped to a single key
([ADR-0009](../docs/adr/0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md)).
