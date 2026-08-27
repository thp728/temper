# ADR-0034 — The dataset size limit is a product limit derived from measured throughput

- **Status:** accepted — supersedes [0005](0005-the-dataset-size-limit-is-derived-from-measured-memory.md)
- **Date:** 2026-08-27
- **Spec:** `docs/specs/006-streaming-transport-progress-and-artifacts.md`
- **Issue:** [#31](https://github.com/thp728/temper/issues/31)

## Context

ADR-0005 set the upload ceiling at 1 GB because validation held the whole
dataset in memory: its peak resident memory was measured at 4.8x the file
size, later re-measured at 5.93x on a real 1 GB file (spike 9), and a dataset
that large was the largest the process could honour without an out-of-memory
crash. The ceiling was therefore a property of *how validation was written*,
not a product rule -- and the record said so, naming streaming validation as
what would remove it.

Spike 9 also measured the streaming alternative before this ticket was
written: peak RSS flat at **+4 MB from 1 GB to 20 GB**, and *faster* than the
in-memory path. The memory multiplier was the only thing the ceiling was
derived from, and the rewrite removes it. That makes the ceiling a number in
search of a reason: **1 GB is no longer what the machine can honour; it is
now whatever the product decides to let a user wait for.**

The second number that matters is throughput. Spike 9 measured the streaming
pass at a **mean 21.6 MB/s** (validate alone) across 1 GB, 5 GB and 20 GB
files -- a floor, because the measurement machine was contended. At that rate
a 1 GB dataset validates in roughly 46 seconds, and a user who has uploaded
it sits on a page with nothing to show for the wait.

## Decision

**Validation streams, and the upload ceiling becomes a configured product
limit derived from the measured throughput.**

Validation now reads the dataset one row at a time from a bounded chunk
stream: peak memory stays flat as the file grows, every error still names its
line, and the report -- counts, thinking mode, preview, line-numbered
problems -- is unchanged. Nothing retained scales with the file; the error
and warning lists are capped, and the report carries the true totals beside
the capped lists so a file broken on every line is told it is broken on every
line.

The ceiling is now **1.3 GB by default**, derived as: the measured mean
streaming rate (21.6 MB/s, a floor) times a 60-second tolerable synchronous
wait. The rate is **measured**; the 60-second tolerance is **a judgment** --
the point at which an upload stops reading as a wait and starts reading as a
hang. Both halves are recorded here and beside the value in `config.py`, so
changing the number means revisiting the reasoning, not just retyping it. It
is configuration (`TEMPER_MAX_DATASET_MB`), because the right number depends
on the deployment's tolerance for a synchronous wait.

An oversized upload is still refused **before it is read**, naming both the
limit and the actual size with the stable code `dataset_too_large`. A lying
or absent Content-Length cannot defeat the ceiling: the ingest path also
refuses mid-stream once the true size is known.

Validation now runs in the **background**: the upload answers with the
dataset's id, and the record exposes `progress` while validation works. The
progress is the measured bytes-through-the-pass, published so a large upload
does not look like a frozen page. This is the other half of the "tolerable
wait" question -- the wait is no longer silent.

## Why ADR-0005 is superseded rather than corrected

0005 was already corrected once, in place, when spike 9 re-measured the
multiplier (5.93x, not 4.8x) -- and that correction was the right shape while
streaming was still a plan. It is superseded now because **the thing the
record was about no longer exists.** The record's whole argument runs from a
memory multiplier to a ceiling that protects the process from itself; the
streaming rewrite deletes the multiplier, so the record's reasoning cannot be
patched into agreement with the code. A decision that states a rule in words
broader than the property it defends (0005's opening claim was "the limit is
derived from measured memory") is exactly the case the decision directory
exists for: a new entry saying where the earlier reasoning ran out is
stronger evidence of judgment than an entry that was quietly widened.

0005 itself is left unedited, per the convention. Its correction note -- that
streaming validation would remove the limit -- is now simply what happened.

## Alternatives considered

**Keep the 1 GB number.** Rejected. It was derived from a memory multiplier
that no longer exists, so it would be an unanchored legacy figure presented
as if it meant something. A number with a dead derivation is worse than a
number with a debatable one.

**Derive the ceiling from the same 60-second wait but validate synchronously,
no progress.** Rejected. It keeps the frozen page the spec's user story
names. The background validation with observable progress is what makes the
60-second ceiling something a user can actually experience.

**Remove the ceiling entirely.** Rejected. Streaming makes memory flat, but a
ceiling still protects something real: a synchronous validation that runs for
longer than any user should wait, tying up the control plane's worker threads
on one request. And the platform's own 25 GB baseline is out of reach on a
single-GPU job anyway; advertising a limit the system cannot honour is worse
than a visible product rule.

**Report the capped error list without the totals.** Rejected. A user whose
file is broken on every line must be told it is every line, not silently
shown a hundred errors and left to re-upload blind. The suppression counts
keep a capped report honest, which is why they are part of the report's
published shape.

**Split token counting into the upload pass.** Rejected, as spike 9
recommended: tokenising dominates beyond any tolerable wait (3.4 MB/s vs
21.6 MB/s for validate alone), so counting stays a separate phase before the
quote, not part of the synchronous upload.

## Consequences

- Peak memory stays flat as dataset size grows, asserted by a test that fails
  if a later change re-materialises the dataset (an uncapped list, rows
  collected before thinking detection, a whole-file read).
- The ceiling goes from 1 GB to 1.3 GB and is now a product decision; the
  record beside the config value says where it came from and what to revisit.
- An upload is observable while it validates: the record carries progress,
  and the pages render a proportion-complete bar instead of a frozen form.
- The report gains suppression totals (`error_count`, `*_suppressed`); every
  existing field keeps its meaning and value.
- A validation that raises in the background records a coded failure rather
  than stranding the dataset at "validating" forever. (A whole-process death
  mid-validation leaves the row "validating", the same recovery story a job
  that dies with the process has; the page tells the user to upload again.)

## Rollback

Restore the previous validator and ceiling: revert
`temper_core/validation.py`, `config.py`, `datasets.py` and the contract
models. The streaming property tests fail loudly if the validator is replaced
with a materialising one, which is the intended behaviour of a rollback
guard.
