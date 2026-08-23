# ADR-0004 — The machine is a pure compute node

- **Status:** accepted — **property 2 superseded by [ADR-0009](0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md) (2026-08-23)**
- **Date:** 2026-08-21
- **Spec:** `docs/specs/002-dataset-transport-and-limits.md`
- **Issue:** [#8](https://github.com/thp728/temper/issues/8)

## Context

Every byte a training job needs has to reach a machine that exists for the
length of one job and is destroyed afterwards. How those bytes travel was
decided twice, differently, inside one module: the trainer's sources were
shipped as a binary archive pushed over the SSH connection, and the dataset —
the larger payload by orders of magnitude — was hex-encoded into the remote
shell script, doubling in size as text, held in memory several times over, and
embedded in a pipe feed sized for commands. The correct mechanism existed forty
lines away and had been written first. Nothing noticed, because nothing larger
than 7.5 KB had ever been uploaded.

Fixing the transport forced the prior question: what is the machine *for*, and
therefore who is allowed to reach what?

## Decision

**The machine is a pure compute node.** It runs one job, produces one artifact,
and is destroyed. It has three properties:

1. **It is reachable only over the control plane's own connection.** Today that
   is SSH; the control plane holds the credentials, opens every connection, and
   pushes and pulls every byte. The machine accepts no inbound traffic beyond
   SSH from the control plane, and the trainer publishes no ports at all.
2. **It never contacts object storage.** Not in Phase A, where there is no
   object storage; not in Phase B, where S3-compatible storage exists and the
   machine still does not touch it. All storage is control-plane-side, in both
   phases.
3. **Everything it receives travels over the same channel, by one mechanism:**
   a binary archive written to the machine's standard input by the control
   plane. Trainer sources and user datasets alike. Datasets go into their
   archive as raw bytes — line-ending normalisation is correct for shell
   scripts and Dockerfiles and silently corrupting for user data, so it is
   applied to the former and never to the latter.

This makes transport a **control-plane concern rather than a networking one**.
There is no network topology to secure around the machine — no credentials to
distribute to it, no bucket policy to scope, no egress path to audit — because
the machine initiates nothing and receives everything from the one component
that already owns storage and secrets. The transport question reduces to "how
does the control plane write bytes down its own connection", which is testable
without a GPU and was, in fact, untested for the dataset until now.

### The measured memory multiplier

Recorded here because the dataset size limit introduced alongside this change
is derived from it, and a future change to that limit needs to know how the
number was obtained.

Peak resident memory for validating an uploaded dataset was measured across five dataset sizes, one fresh process each, peak working set read from the
operating system:

| Dataset | Peak RSS | Ratio |
| --- | --- | --- |
| 10 MB | 67.6 MB | 6.8× |
| 25 MB | 139.2 MB | 5.6× |
| 50 MB | 256.9 MB | 5.1× |
| 100 MB | 492.0 MB | 4.9× |
| 200 MB | 959.4 MB | 4.8× |

The ratio converges to **4.8× the file size**; the higher figures at small
sizes are fixed interpreter overhead amortising against a smaller file. The
curve is still declining slightly at 200 MB, so the true asymptote is a little
below 4.8× — it is used as the conservative figure. The earlier working
assumption in discussion was 5–8×, which was high at the top end.

These are measured numbers, not estimates, but they were measured on one
development machine with one Python build; a different runtime shifts the
constant, not the shape.

## Alternatives considered

**Let the machine fetch its own dataset from object storage.** Rejected, and
this decision exists to record why. It is the obvious design at scale: presign
a URL, let the trainer pull, and the control plane never carries the payload.
It requires giving the machine network egress, storage credentials scoped to
one job, and a trust boundary around code running on rented GPUs — three new
things to get right, on the component with the shortest life and least
scrutiny. It also inverts the failure model: today a failed transfer raises in
the control plane, where it is coded, logged and torn down; a fetch that fails
on the machine surfaces as a training failure and sends the user to read the
wrong logs. Phase B introduces object storage anyway, and the machine still
will not reach it — the control plane remains the only component that does.

**Keep hex-in-script and raise the pipe buffer / stream the script in chunks.**
Rejected. It treats the symptom (the transport chokes on large payloads) while
keeping the disease (user data passing through a text encoding designed for
embedding in commands). Doubling the dataset as hex text is paid in memory,
transfer time and pipe pressure at every size, and no buffer size fixes a
mechanism whose cost is proportional to the square of what it should be.

**scp / rsync for the dataset, archive for the sources.** Rejected. Two
transport mechanisms means two things to test, two failure modes, and two sets
of host-key and authentication semantics — for no gain, since the push channel
already moves arbitrary bytes and had just proven itself on the sources. One
mechanism for everything sent to a machine is the property that made this
ticket's regression tests writable at all.

**Normalise line endings in datasets too, for consistency with the sources
archive.** Rejected on correctness. A CRLF Dockerfile breaks visibly at build
time; a CRLF training dataset changes every row invisibly and trains a model on
data the user did not upload. Where the two payloads differ in requirement, the
code now says so in the two functions rather than sharing one that fits neither.

## Consequences

- The dataset reaches the machine byte-identical or not at all, and the
  regression tests assert exactly that — including carriage returns and
  non-ASCII text, the two cases the old path got wrong or would have.
- The remote script is a fixed few kilobytes regardless of dataset size. It no
  longer carries the dataset, so the streaming stdin feed no longer handles a
  payload that grows with the user's file.
- Large uploads are bounded by what the control plane can hold, not by what a
  shell script can carry — the size limit derived from the multiplier above
  (1 GB, below the 25 GB named baseline, deliberately and openly) lands with
  the bounded-upload change and is enforced entirely control-plane-side.
- Phase B's MinIO deployment stores datasets and artifacts for the *control
  plane*; the machine's behaviour does not change when it arrives. The
  transport becomes "control plane downloads from storage, pushes to machine" —
  a new producer feeding the same channel, not a new topology.
- What this forecloses: any future design where the trainer resumes a download,
  reads a manifest from a bucket, or pulls a base model directly from a hub
  cache shared across jobs. If a job someday needs assets too large to push,
  the decision gets revisited with those numbers — not drifted around.

## Rollback

Reintroduce the hex embedding in `_remote_script` and delete
`_dataset_tarball` and its push. Nothing else depends on the archive: the
provider seam (`push`) is unchanged, the script is self-contained again, and
the regression tests would fail loudly, which is the intended behaviour of a
rollback guard.
