# ADR-0041 — Checkpoints are written off the machine as they are produced, and verified before they are presented

- **Status:** accepted
- **Date:** 2026-08-27
- **Supersedes:** nothing (extends [ADR-0009](0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md))
- **Spec:** `docs/specs/010-recovery-and-fault-injection.md`
- **Issue:** [#37](https://github.com/thp728/temper/issues/37)

## Context

A checkpoint that exists only on a machine that is about to be destroyed is not
a recovery mechanism. Spec 010's resumption depends on checkpoints surviving
their machine, which means they have to be in object storage **as they are
written**, not collected when the run ends — and issue #37 makes each
acceptance criterion explicit: written as produced, each recording its step and
its held-out loss, bounded and configurable retention, upload that does not
stall training, and *never presenting a partially written checkpoint as
complete*.

That last criterion is the one with no cheap answer. A checkpoint directory is
several files written over time, some of them tens of gigabytes; "it is in the
store" is not the same claim as "it is complete", and the whole point of the
criterion is that nothing downstream may mistake the first for the second.

The machinery for a machine writing to the store already exists. [ADR-0009]
(0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md) lets the
machine write its artifact to a scoped, key-bound, expiring grant, and
[ADR-0035](0035-the-control-plane-verifies-a-machine-written-artifact-by-streaming-it-back.md)
decided how the control plane verifies what landed: stream the stored object
back through the SHA-256 the machine reported, never holding it whole. The
question this record answers is how that machinery shapes checkpoints, and
specifically how "complete" is made into a property of the record rather than a
hope.

## Decision

**Each checkpoint is written to object storage as it is produced, as one
tarred object under a bounded set of slot keys, through the ADR-0009 scoped
grant.** Concretely:

1. **The control plane mints `CHECKPOINT_RETENTION` scoped grants per job**
   (`config.py`), one per key `checkpoints/{job}/slot-{0..N-1}`, each with the
   same remaining-duration-ceiling lifetime as the artifact's grant. The
   machine holds no credential that outlives the job.
2. **The trainer uploads in a background thread while Axolotl trains.** It
   watches `/out/run/` for a `checkpoint-N` directory and ships it only when
   it is *complete*: its own `trainer_state.json` exists and its `global_step`
   equals `N`. `trainer_state.json` is the last file written in a checkpoint
   save, so its presence with the agreeing step is the machine's local
   signal that the save finished. A directory without it is a save cut off
   mid-way and is never uploaded.
3. **Each checkpoint is one tarred object, streamed** — the directory is tarred
   in blocks to a temporary file, the file is stream-hashed, and the PUT
   streams the body with a declared Content-Length, exactly the ADR-0009 shape.
   Nothing on the machine holds a checkpoint whole.
4. **The i-th successful upload overwrites slot `i mod N`.** Storage never
   holds more than `CHECKPOINT_RETENTION` checkpoint objects per job, so
   retention is bounded by construction rather than by a deletion pass, and
   the bound is configuration.
5. **The trainer reports each upload in `result.json`** — step, slot, loss and
   held-out loss (read from the checkpoint's own log history), the SHA-256,
   and whether the upload succeeded.
6. **The control plane verifies every reported checkpoint by streaming the
   stored object back through the reported SHA-256** (the ADR-0035 shape, one
   chunk at a time), and **records a checkpoint as complete only when that
   matches.** Anything short of a match — a failed upload, a slot with nothing
   landed, a checksum mismatch — is recorded as *not* complete, with the
   reason, and is never presented as a recoverable checkpoint.

### How "a partially written checkpoint is never presented as complete" holds

It is three mechanisms that do not rely on hope:

- **The storage layer publishes whole objects.** A single PUT to a scoped URL
  either lands in full or publishes nothing; the filesystem backend's
  `put_stream` stages to a `.part` name and renames only when every chunk
  arrived. A slot key never holds a half-written object.
- **The report is two-phase.** The trainer assigns a step to a slot and only
  records the assignment *after* its PUT succeeded, so a machine that dies
  mid-upload never claims the slot for a step it did not finish writing.
- **The control plane gates on verification.** Nothing is presented as a
  checkpoint until the stored bytes have been streamed back and matched the
  machine's own checksum. A report without a matching object is a *not
  complete* record, not a checkpoint.

**A marker object was considered and rejected.** The storage seam has no list
operation, so a marker object could not be discovered by the control plane
except through the same `result.json` report that already carries the
step-and-checksum record — and adding one would need a second grant per
checkpoint and double the objects, for nothing the report does not already
say. The two-phase report *is* the marker; the control plane's verified record
is the gate.

**Per-step keys were considered and rejected.** One object per step would make
storage hold a checkpoint per step written, which is unbounded on an
epoch-driven run (no `max_steps`), and the grants would have to be minted for
a set the control plane cannot know at launch. The bounded slot ring makes
retention a property of the design, not of a cleanup pass.

## Why this is not a regression of ADR-0009/0035

ADR-0009's properties survive intact. The machine still holds no long-lived
credential, cannot enumerate or read anything, and its write authority is a set
of one-key, expiring URLs — the same grant the artifact uses, reused rather
than re-invented. The control plane still owns the objects' identity: it
decides the keys, mints the grants, and verifies what landed. The bytes never
cross the control plane, which is the flat-memory guarantee the checkpoint
sizes — larger than adapters — make non-negotiable. Verification streams from
the store to the control plane, one chunk at a time, exactly as ADR-0035
decided for the artifact.

## Alternatives considered

**Route checkpoint bytes through the control plane and have it write them.**
Rejected on the same ground ADR-0009 rejected it for artifacts, only stronger:
checkpoints are larger than adapters, and the control plane's flat-memory
guarantee (ADR-0035, `test_verifying_a_large_artifact_stays_flat_in_memory`)
exists precisely because the control plane never holds a whole stored object.

**A marker object written after the payload, discovered by the control plane.**
Rejected: the seam has no list operation, so discovery would still require the
`result.json` report; a marker doubles grants and objects; and the report-plus-
verification already provides the completeness gate.

**Give the machine a broader credential to write checkpoints.**
Rejected outright. The scoped grant is the mechanism, and widening it to a
credential that can write many keys would undo ADR-0009's scoping.

**Trust the machine's report without verification.**
Rejected on the ADR-0035 ground: identity was never delegated. A checkpoint
that hashes back to what the machine reported is what a resumption can use;
a report with no matching bytes is not a checkpoint.

**Per-step keys with the control plane deleting beyond retention.**
Rejected: the grant set is unbounded when `max_steps` is absent, and retention
by deletion adds a reconciler-shaped obligation the slot ring avoids by
construction.

## Consequences

- **Storage holds at most `CHECKPOINT_RETENTION` checkpoint objects per job**
  (default 3, matching `save_total_limit`; independently configurable via
  `TEMPER_CHECKPOINT_RETENTION`). The machine overwrites the oldest slot.
  A checkpoint whose slot a newer checkpoint overwrote is recorded as
  **superseded** — eviction by a design decision, never misreported as a
  checksum failure.
- **Upload never stalls training.** The uploader is a background thread;
  training never joins it except in the entrypoint's `finally`, where one
  final sweep ships the last checkpoint — the one a resumption would most
  want — even if the poll never saw it. A failed upload is retried a bounded
  number of times and then recorded as a failure. Checkpoints are shipped in
  the order they complete (oldest first); retention keeps the newest, which
  are the ones a resumption needs.
- **Verified checkpoints are recorded on the job row** (`checkpoints_json`),
  each with its step, its loss where one exists, and the key its bytes were
  verified at. Failed or missing ones are recorded as *not complete* with the
  reason, superseded ones as *not complete* with `superseded`. A checkpoint
  problem never fails the job: the adapter is the deliverable, and a bad
  checkpoint does not make a trained adapter untrained.
- **A failed run's checkpoints are still recorded.** The machine reports them
  in `result.json` even when the run ends in failure, and the control plane
  verifies and records what survived before the job reports failure — a
  crashed run's checkpoints are recovery material, not orphaned bytes.
- **Cancellation discards checkpoint objects**, mirroring the artifact: a job
  the user stopped keeps no recovery material.
- **The record shape is deliberately open.** Checkpoints live under their own
  `checkpoints/` key namespace, distinct from `artifacts/`, and the job row
  carries them in their own field, so issue #32 can give stored objects a
  declared kind and classify checkpoints without rewriting this.
- **A checkpoint directory that Axolotl deletes (via `save_total_limit`)
  before the uploader ships it is simply not in storage.** Bounded retention
  includes the machine's own disk; the newest checkpoints are the ones a
  resumption needs, and they are shipped as soon as they complete.

## Rollback

Stop minting checkpoint grants (`_remote_script` drops the `checkpoint_grants`
block) and the trainer leaves checkpoints on `/out` exactly as it leaves the
artifact without a grant. The storage seam, the grant machinery, the key
helpers and the verification step all survive; the trainer's uploader is inert
without grants. Nothing else in the job contract changes.

**This is unexercised on real hardware.** The write path is proven against the
filesystem-backed grant and a loopback store; no real VM has yet PUT a
checkpoint to a pre-signed URL. Spike 2 established the prerequisite (outbound
HTTPS from a VM), and ADR-0009's "unexercised" caveat applies here unchanged.
