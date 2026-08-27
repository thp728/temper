# ADR-0035 — The control plane verifies a machine-written artifact by streaming it back

- **Status:** accepted
- **Date:** 2026-08-27
- **Supersedes:** nothing (implements [ADR-0009](0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md))
- **Spec:** `docs/specs/006-streaming-transport-progress-and-artifacts.md`
- **Issue:** [#50](https://github.com/thp728/temper/issues/50)

## Context

[ADR-0009](0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md)
lets the machine write its artifact directly to a scoped grant, which removes
the control plane from the write path entirely: the machine PUTs the artifact
to a pre-signed URL and the control plane never sees the bytes cross itself.
But the same record states, in point 5, that the control plane still owns the
artifact's identity — *a job is not complete because the machine says so* —
and issue #50's acceptance criterion makes verification explicit: *the
machine reports the checksum it computed, and the control plane verifies it
against the stored object before the job may report success.*

That raised a question ADR-0009 did not answer: **how does the control plane
verify an object it never handled, without becoming the proxy ADR-0009
removed it from?**

## Decision

**The control plane verifies the machine-written artifact by streaming the
stored object back through the storage seam and hashing it, one bounded chunk
at a time, comparing the result to the checksum the machine reported.** The
job may only report success once that hash matches.

Two properties make this the right shape rather than a regression of
ADR-0009:

- **Memory stays flat.** Verification streams `get_stream` through a SHA-256,
  holding one chunk however large the artifact is. The machine-write removed
  the *control plane's* memory from the write; this keeps it out of the
  verify too. The flat-memory guarantee of issue #30/#92 survives across the
  whole artifact path — write, verify, download — and each leg is guarded by
  a test that fails if a whole-object read returns (this repo already watched
  one such guarantee die at PR #95 and be restored at PR #97; a guarantee
  with no guard does not survive the next large merge).
- **The object is what is verified.** The machine's checksum is the ground
  truth for the bytes it trained; the object behind the seam is what a user
  will download. Streaming the object through the same hash proves those two
  are the same bytes, which is the whole point of the criterion.

The grant's lifetime is **what remains of the job's own duration ceiling**,
counted from the job's creation timestamp (the same origin the ceiling counts
from), per ADR-0009 point 3: the machine cannot run longer than the ceiling, so
a URL minted for the remaining ceiling expires no later than the job can
legitimately end — an abandoned URL is not a standing grant.

Two adjacent choices, decided here so the record is complete:

- **The machine writes only the weights; the config still travels in
  `result.json`.** The adapter is not loadable without `adapter_config.json`,
  but that object is small by construction, so it rides the event channel the
  way it always did and is stored by the control plane. The machine's write
  is reserved for the payload that grows without bound.
- **A result that names an artifact but whose object never lands fails the
  job.** With the machine writing directly, absence after a claimed upload is
  a failed upload, not a completed job without an artifact — the old "an
  empty read is no adapter" behaviour belonged to the pull path, where the
  control plane itself was the reader. The job may not report success
  (ADR-0009 point 5).

## Why this is not a regression of ADR-0009

ADR-0009 removed the control plane from the *write* path: the machine's bytes
no longer cross its memory or its network link on the way to storage. Nothing
here reinserts them — the re-read for verification happens between the control
plane and its own store, over the seam, and it is a read, not a proxy for a
transfer between two other parties. The machine still holds no credential,
still cannot enumerate anything, and still writes exactly one object to one
key. What is added is the control plane's half of the machine's own report: a
claim without a check is not an identity, and identity was never delegated.

The cost is honest: verification is a second full read of the artifact from
the store. On the object-store backend that is real egress. It is cheaper
than the alternative it replaced — streaming the artifact *through* the
control plane from machine to store, which billed the control plane's memory
and link for the whole transfer *and* still had to be hashed — and it is the
price of not trusting the machine, which the acceptance criterion demands.

## Alternatives considered

**Trust the machine's reported checksum without re-reading.** Rejected: it
empties the criterion. The point of "the control plane verifies it against
the stored object" is that the machine's word and the stored bytes are
checked against each other; believing the machine makes the check a
formality.

**Trust the store's own checksum instead of re-reading.** Rejected. S3's ETag
is not the object's SHA-256 — it is an MD5 for single-part PUTs and an
opaque multi-part value otherwise — and MinIO's behaviour is its own. The
machine's SHA-256 is the contract the trainer reports; nothing the store
returns says whether *those* bytes match *that* hash, which is the question
being asked.

**Have the machine sign the upload so the store returns a content hash.**
Rejected as speculative: it requires the store to echo a payload hash the
presigned PUT never asked it to compute, and it moves the ground truth away
from the machine's own report. The streaming re-read is standard, portable,
and needs nothing from the store but a working `get`.

**Verify by metadata (size and existence) only.** Rejected: a truncated
upload can have the right size only by coincidence, and existence says nothing
about integrity. The machine's checksum exists precisely to catch truncation,
so refusing to use it would leave the criterion's failure mode undetected.

## Consequences

- **The control plane reads each artifact twice from storage: once to verify,
  once when a user downloads it.** Both legs stream, so memory is unaffected;
  the object-store backend pays egress for the verify read.
- **`artifact_corrupt` and `artifact_unverified` are terminal job failures.**
  A machine that wrote nothing, wrote something else, or reported no checksum
  ends the job `failed` with a stable code, and the stored object is deleted
  so nothing reachable by key reads as the deliverable.
- **Cancellation discards machine-written objects.** A job cancelled after
  the machine's write landed deletes the artifact's objects before the job
  reports `cancelled`; an upload that lands after the delete is an orphan in
  the sense ADR-0009 already records (nothing reconciles orphans yet).
- **The trainer's fingerprint is computed in bounded blocks.** Reporting the
  checksum no longer requires reading the artifact whole on the machine
  either.

## Rollback

Stop minting the grant and revert to pulling the artifact over SSH
(`fetch_stream`), restoring `_fetch_adapter`. The storage seam, the grant
machinery and the checksum contract all survive unchanged; the verification
step is where the two transport shapes meet, and that is the single function
to swap. The machine side reverts by ignoring the `artifact_upload` block,
which is optional in the job spec by construction.
