# ADR-0009 — The machine may write its own artifact to a pre-signed URL scoped to one key

- **Status:** accepted
- **Date:** 2026-08-23
- **Supersedes:** [ADR-0004](0004-the-machine-is-a-pure-compute-node.md), in part — property 2 only
- **Spec:** `docs/specs/004-phase-b-spikes.md`
- **Issue:** [#16](https://github.com/thp728/temper/issues/16)
- **Evidence:** [`spike/findings-spike5.json`](../../spike/findings-spike5.json),
  [`spike/findings-spike6.json`](../../spike/findings-spike6.json)

## Context

[ADR-0004](0004-the-machine-is-a-pure-compute-node.md) settled how bytes reach a
training machine, and property 2 was categorical: **the machine never contacts
object storage.** Everything travels over the control plane's own SSH
connection, in both directions, in both phases. That was the right call for the
sizes it was written against — the largest artifact this product had ever
produced was a **132 MB adapter**, and pulling 132 MB down an SSH channel the
control plane already owns costs nothing worth designing around.

Phase B changed the sizes, and spikes 5 and 6 measured how far.

**Disk stopped being the constraint.** Spike 5 found the storage ceiling is
**7200 GB** — the platform names it in its own refusal — and provisioned 4000 GB
to confirm the parameter is honoured. So the catalog is not disk-bound, and 70B
configurations are on the table rather than off it.

**Artifacts stopped being small.** Spike 6 wrote a real sharded checkpoint
through the pinned image and measured it: **3.9 GB for a 0.6B full fine-tune,
13.6 GB for 4B**, in `torch.distributed.checkpoint` `.distcp` shards, of which
roughly two thirds is optimiser state. That is the first time this product has
produced an artifact in that class, and it is a *measured* number rather than an
estimate.

Extrapolating the same shape to the configurations the catalog now admits puts a
70B sharded checkpoint in the hundreds of gigabytes. **That extrapolation is
marked as an extrapolation** — nothing here has trained a 70B model — but the
direction is not in doubt, and the transport decision has to be made before the
run, not during it.

Pulling hundreds of gigabytes through the control plane means that every byte
crosses the control plane's memory and its network link, on the way from a
machine that is billing to storage the control plane owns. The control plane
becomes a proxy for data it does not need to see, and the machine stays alive —
and billing — for the whole transfer.

## Decision

**The machine may write its own artifact to a pre-signed URL, scoped to one
key, for the duration of one job.** ADR-0004's property 2 is replaced by:

1. **The machine still holds no long-lived credential.** It never receives an
   access key, a secret, a role, or a token that outlives the job. What it
   receives is a URL that is already an authorisation.
2. **The URL is scoped to exactly one object key**, the one this job's artifact
   is written to. It does not authorise reading, listing, deleting, or writing
   anywhere else. **The machine cannot enumerate storage**, which is the
   property ADR-0004 was actually protecting and which survives intact.
3. **The URL expires**, on a lifetime derived from the job's own duration
   ceiling rather than a round number, so an abandoned URL is not a standing
   grant.
4. **Everything inbound is unchanged.** Trainer sources and user datasets still
   travel as a binary archive down the control plane's connection, exactly as
   ADR-0004 specifies. This decision changes one direction only.
5. **The control plane still owns the artifact's identity.** It mints the URL,
   it decides the key, and it verifies the object landed. A job is not complete
   because the machine says so.

## Why this is not a reversal of ADR-0004

ADR-0004's reasoning was that transport should be *a control-plane concern
rather than a networking one*: no credentials distributed to the machine, no
bucket policy to scope, no egress path to audit. **All three still hold.** A
pre-signed URL distributes no credential, scopes no bucket policy (the scope is
in the URL), and adds one egress destination that the control plane itself
chose and can revoke by expiry.

What changes is narrower than "the machine talks to storage": the machine gains
the ability to *put one object it produced into one place it was told about*.
It still initiates nothing else, receives everything else, and enumerates
nothing.

## Alternatives considered

**Keep ADR-0004 as written and stream large artifacts through the control
plane.** Rejected on the measured sizes. It works, and it is what ships today
for a 132 MB adapter. At 13.6 GB it makes the control plane a proxy for data it
has no reason to read; at the sizes a 70B configuration implies it makes the
control plane's memory and link the bottleneck of every job, and it keeps a
billing machine alive for the length of the transfer. The cost of the wrong
answer here is paid per job, in money.

**Give the machine real object-storage credentials.** Rejected. This is the
thing ADR-0004 was right about. A credential on a machine that is destroyed
after one job is a credential whose blast radius is every object the credential
can reach, and it has to be rotated, scoped, and audited. A pre-signed URL is
the same capability with the blast radius already reduced to one key by
construction rather than by policy.

**Use a persistent filesystem instead.** Spike 8 confirmed the provider offers
persistent filesystems (`filesystems.create` / `edit` / `remove`, attachable at
create and at resume). Rejected for artifacts, and **kept open for model
weights** — a filesystem is provider-scoped storage, so it does not remove the
need to get the artifact into the platform's own storage, and it introduces a
resource that outlives the job and therefore needs its own reconciler. As a
*cache for base-model weights* it is attractive and is deliberately left as an
open option rather than decided here.

**Have the machine push over the SSH connection it already has.** Rejected as a
non-answer: that is what happens today, and it is the thing the measured sizes
made expensive.

## Consequences

- **The control plane needs an object store that mints pre-signed URLs.** MinIO
  in Phase B, S3-compatible by design, which the target architecture already
  specifies.
- **The reconciler's job grows slightly.** An object written by a machine whose
  run has since failed is an orphan in the same sense a machine is, and it is
  cheaper — but it is not free, and nothing currently deletes it.
- **Failure reporting is unchanged.** `result.json` is still written on every
  path including failure, and the orchestrator still never parses logs to learn
  what happened.
- **This is unexercised.** Nothing has yet written an artifact to a pre-signed
  URL from a JarvisLabs VM. Spike 2 established that outbound HTTPS is open from
  a VM, which is the prerequisite, and that is all. **Do not describe this as
  proven** — the same standard already applied to multi-GPU.

## Rollback

Stop minting URLs and have the packaging step pull over SSH again. The machine
side is one conditional: a URL was supplied, or it was not. Nothing else in the
job contract changes, because the artifact's identity was never the machine's to
decide.
