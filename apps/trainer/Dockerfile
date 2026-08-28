# Trainer image — pinned by digest, never by tag.
#
# WHY AXOLOTL AS THE BASE, rather than building from a PyTorch image:
#
# Spike 3 demonstrated the cost of resolving the version matrix ourselves.
# Installing "latest" gave transformers 5.15.0 + trl 1.10.0 on torch 2.5.1, and
# three things chained: TRL 1.x's SFTConfig no longer inherits TrainingArguments
# so `warmup_ratio` raised TypeError; `save_safetensors` was rejected so
# checkpoints wrote as torch .bin; and transformers 5.x then refused to load
# them because torch < 2.6. Net effect: checkpoint/resume was impossible.
#
# That matrix — transformers, trl, peft, bitsandbytes, torch, CUDA — is exactly
# what the architecture's §32 names as the key dependency risk. Axolotl's
# maintainers resolve and test it on every build, and Axolotl's own config
# surface still exposes `warmup_ratio`, so our intended defaults remain
# expressible regardless of what TRL does to its API. Delegating the matrix to
# people who test it is a better answer than pinning six packages by hand and
# re-resolving them every time one moves.
#
# Base carries torch 2.12 (>> the 2.6 floor that blocked resume), CUDA 13.0,
# Python 3.12. ~8.5 GB, so budget ~3 min for a cold pull at the ~46 MB/s
# measured on JarvisLabs.
#
# PIN DISCIPLINE: the digest is the contract. The tag is a comment. Changing
# either is a deliberate act that re-runs the GPU smoke test.
FROM axolotlai/axolotl:main-20260817-py3.12-cu130-2.12.0@sha256:29327e75d7ae0348e001809df5c11151887fdeec63fe4bc5c4330917777c701e

LABEL org.opencontainers.image.title="jarvislabs-finetune-trainer" \
      org.opencontainers.image.description="QLoRA trainer for the fine-tuning platform" \
      org.opencontainers.image.source="https://github.com/thp728/tp-life-os" \
      com.jarvislabs-finetune.base-digest="sha256:29327e75d7ae0348e001809df5c11151887fdeec63fe4bc5c4330917777c701e"

# Our layer only. No pip installs: adding packages here would reintroduce the
# resolution problem this base exists to avoid. If something is genuinely
# missing, it belongs in a pinned requirements file with its own review, not an
# unpinned RUN pip install.
#
# Everything the entrypoint imports, plus the one data file that defines
# what it will validate. The contract ships in the same layer as the code:
# defaults are data in `packages/contracts/trainer-defaults.json` (#82),
# read by `temper_core.hyperparams` to resolve the spec before launch (#83),
# and the image bakes the same file so data and code share one digest once
# the pipeline builds it. `thinking.py` arrived after this Dockerfile was
# written and after spike 4 last built the image, so the COPY was never
# updated -- and because `from thinking import ...` is a top-level import,
# the container would have died before the `finally` that writes
# result.json, surfacing to the orchestrator as the useless
# "trainer produced no result.json". Caught by reading, not by paying for
# it.
#
# The build context is assembled, not a directory you can point at. `thinking.py`
# lives in packages/core because the control plane validates thinking mode with
# the same module this image runs, so the orchestrator flattens every source into
# one context before building. `just image` does the same thing locally, and
# TRAINER_SOURCES in the orchestrator is the list both use. A source added here
# and not there fails test_trainer_context.py.
#
# `axolotl-schema.json` (issue #33) is the pinned image's own configuration
# schema: the entrypoint reads it so that a key unknown to the trainer is
# refused loudly and echoed back, with 'unknown' meaning unknown to the trainer
# rather than absent from a hand-written list.
#
# `checkpoints.py` (issue #37) is the checkpoint uploader the entrypoint
# imports; it must ship in the same layer or the container dies at import,
# before the `finally` that writes result.json.
#
# `split.py` (issue #53) is the held-out split the entrypoint runs at the
# start of every job: it lives in packages/core (the control plane shares the
# same dedup-and-split behaviour) and is flattened into this context like
# `thinking.py`, so it must be in the COPY too -- a top-level import, so a
# missing COPY dies at import, before the `finally` that writes result.json.
#
# `fault-surface.json` (issue #24) is the fault surface's vocabulary: the six
# fault names, the `simulated_` code each carries and which side makes it
# real. The trainer reads it as data to validate the fault spec its
# environment carries, so it ships beside the entrypoint exactly like
# `trainer-defaults.json` -- one definition, read by the control plane and the
# trainer alike.
COPY entrypoint.py checkpoints.py thinking.py split.py template_probe.py trainer-defaults.json axolotl-schema.json fault-surface.json /opt/trainer/

# Written by the orchestrator, read here. Declared so the contract is visible
# in `docker inspect` rather than only in the run command.
#   /job  — job.json + dataset.jsonl   (read-only)
#   /out  — checkpoints, adapter, result.json, logs
VOLUME ["/job", "/out"]

WORKDIR /out
ENTRYPOINT ["python", "/opt/trainer/entrypoint.py"]
