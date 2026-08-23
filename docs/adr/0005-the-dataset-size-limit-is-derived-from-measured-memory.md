# ADR-0005 — The dataset size limit is derived from measured memory

- **Status:** accepted — **the multiplier it derives from was measured too low; see below**
- **Date:** 2026-08-21
- **Spec:** `docs/specs/002-dataset-transport-and-limits.md`
- **Issue:** [#9](https://github.com/thp728/temper/issues/9)

> **Correction — spike 9, 2026-08-23.** The 4.8× multiplier below was measured
> across five dataset sizes, none of them near the limit it was used to derive.
> Measured again at a real 1 GB file, the shipped in-memory validator peaks at
> **5.93× the file size** — so a 1 GB dataset peaks near 6 GB of RSS, not the
> 4.8 GB this record predicted. **The number was extrapolated from small files
> and the extrapolation was optimistic.** The 1 GB limit is still defensible
> on the development machine, with less headroom than claimed.
>
> The same spike measured the streaming alternative: peak RSS **+4 MB and flat
> from 1 GB to 20 GB**, and *faster* than the in-memory path (the in-memory
> path spends time allocating). So the ceiling is not a trade between memory
> and speed — streaming wins on both, and the ceiling is simply a property of
> code that predates the need. See `spike/findings-spike9.json`.
>
> This record is not superseded here, because streaming validation is not
> built yet. It is corrected: the derivation stands, the input was wrong.

## Context

There was no upper bound on an uploaded dataset. Validation enforces a floor
(ten usable rows) and nothing else, because nothing larger than 7.5 KB had
ever been uploaded. The failure mode for a large upload was therefore not an
error but an out-of-memory crash of the control plane — untyped, unnamed, and
taking the whole process down with it.

## Decision

**Datasets larger than 1 GB are refused at upload**, before validation runs,
with HTTP 413 and the stable code `dataset_too_large`. The refusal names both
the limit and the actual size of the file, so the user knows exactly how much
to trim.

The number is **derived rather than chosen**. Validation holds the whole
dataset in memory, and its peak resident memory was measured across five
dataset sizes, one fresh process each (method and table in
[ADR-0004](0004-the-machine-is-a-pure-compute-node.md)): it converges to
**4.8× the file size**. At 4.8×, a 1 GB dataset peaks around 4.8 GB — roughly
15% of the development machine's memory, with headroom for concurrent work.
The derivation is recorded beside the value in `api/config.py`, so changing
the number without revisiting the reasoning requires deliberately deleting
the reasoning.

It is **configurable** (`TEMPER_MAX_DATASET_MB`) because the right limit
depends on the deployment's memory, not on this code. An invalid value stops
the process at import rather than falling back silently, same contract as the
runtime limits.

The refusal message and the documentation describe this as **a limit of the
current in-memory validation path, not a product rule**, and name streaming
validation as what removes it. Streaming belongs with Phase B's storage work;
until then the limit is honest about being temporary.

This is **deliberately below the 25 GB named baseline**. That figure is a
property of a multi-node fleet; on a single-GPU job with a 24-hour ceiling a
dataset that size cannot finish anyway. Advertising a limit the system cannot
honour is worse than being visibly below it.

## Alternatives considered

**Match the 25 GB baseline.** Rejected. It would be a number the current
path cannot honour on this hardware — the crash it claims to prevent would
return, just later and less legibly.

**Build streaming validation now.** Rejected for this ticket. Line-by-line
parsing removes the limit entirely and is the correct end state, but it is a
rewrite of the validator landing mid-checkpoint-week; the bound plus an honest
message is the proportionate fix, and the message points at the real one.

**Hard-code the limit.** Rejected. A deployment with more memory cannot raise
it without a code change, and a limit nobody can adjust gets treated as
arbitrary — which, unexplained, it would be.

## Consequences

- A too-large upload fails in milliseconds with a typed error instead of
  seconds-to-minutes later as an OOM kill of the shared process.
- Datasets between ~1 GB and what training could theoretically use are
  unreachable until streaming lands. This is stated in the error, not hidden.
- The limit travels with the machine's memory only if an operator sets it;
  the default assumes development-machine scale.

## Rollback

Delete the size check in `upload_dataset` and the constant in `config.py`.
The refusal tests fail loudly, which is the intended behaviour of a rollback
guard.
