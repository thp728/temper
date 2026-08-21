# ADR-0008 — Adapters ship as fp32

- **Status:** accepted
- **Date:** 2026-08-21
- **Spec:** `docs/specs/003-phase-a-ui.md`
- **Issue:** [#13](https://github.com/thp728/temper/issues/13)

> **A note on the number.** Spec 003 calls this "ADR-0003", which is its
> position in the private working log where the first thirteen decisions were
> recorded. The in-repo numbering was already at 0007 when this directory was
> created, and 0003 is taken, so the record lives here as 0008. When the vault
> entries are copied into this directory before the repository goes public,
> this file keeps its number and the cross-reference above keeps the two
> straight.

## Context

The adapter download is the first place a user sees how large the artifact is —
and it is larger than most people expect. The first real run's safetensors
header was read to find out why: **all 504 adapter tensors are F32**, which is
why a Qwen3-4B LoRA at r=16 is a **132 MB** download rather than the ~66 MB a
half-precision cast would produce.

That is not a bug in the export path. It is what the quantised training path
*is*: under QLoRA (NF4 + double quant, bf16 compute), the frozen base model is
stored in 4-bit while **the trainable parameters are kept in fp32 master
weights**. This is standard and correct — bf16 gradients accumulated directly
into bf16 weights would lose the update noise that fp32 absorbs, and every
mainline quantised-training stack does the same. The adapter saved at the end
is exactly the tensor set that was trained, in the dtype it was trained in.

The claim in `trainer/entrypoint.py` that said "adapters in bf16" *was* a bug —
a documentation bug. The behaviour was right; the comment described a system
that does not exist. The comment now records the measurement instead.

## Decision

**The adapter ships as the weights that were trained: fp32, no post-hoc cast.**
What a user downloads is byte-for-byte the artifact whose SHA-256 the container
computed and the control plane verified.

The size consequence — roughly double the half-precision alternative — is
accepted deliberately for Phase A. A silent cast on save would halve the
download and quietly introduce a second representation of the artifact that no
code path verified: the hash check covers the transfer, not the cast, and a
bad cast would ship undetected. An unverified transformation between "what was
trained" and "what was delivered" is exactly the class of silent failure this
project treats as the enemy.

## Alternatives considered

**Cast to bf16/fp16 on save.** Halves the download, and bf16 loses nothing an
inference load would miss. Rejected for Phase A because it adds an unverified
step between training and delivery; accepted as a *future option*, below.

**Cast only at download time, keeping fp32 on disk.** Same integrity problem,
plus a per-request transform on the serving path. Strictly worse than casting
at save.

**Ship both precisions.** Doubles storage to spare the user a decision they
have not been given the information to make. Rejected.

## Consequences

- The download is ~132 MB for a 4B r=16 adapter, not ~66 MB. Users on slow
  links pay for the fidelity the training actually used.
- The artifact loads everywhere PEFT does; no conversion step sits between
  download and use.
- **Reduced-precision export becomes an explicit user choice in Phase B**,
  recorded per job in the run spec like every other correctness setting — with
  the hash computed after any cast, so the verification chain stays intact.
  Until then there is deliberately no flag: a silent default that halves the
  artifact is the failure mode, not a convenience.
- The size is visible to users on the watch page's download link from issue
  #13 onward, which is why this record is written now rather than at Phase B.

## Rollback

Casting on save is a three-line change in the packaging path plus moving the
hash computation after it. Nothing about the training contract changes, which
is precisely why it must stay a recorded decision rather than a quiet one.
