# ADR-0077 - A delivery format is offered only if the published image can make it

- **Status:** accepted
- **Date:** 2026-09-06
- **Supersedes:** nothing. Extends issue #74's delivery vocabulary
  (`temper_core.delivery`).
- **Issue:** none - found during the pre-submission hardware pass and
  recorded in `docs/final-pass-findings.md`.

## Context

A launch with both delivery formats ticked ran to completion on a real L4,
merged the base and the adapter, and then failed on the last step:

```
[trainer] ERROR: DeliveryFailure: the GGUF converter is not on the image's
PATH; this export cannot produce the quantised local format. Refusing
rather than shipping a file that is not the format its name claims.
```

The refusal itself was exactly right: it named the reason and declined to
ship a file that was not the format its name claims. What was wrong is
where it happened. `quantised` is offered in the launch wizard, described
in `GET /v1/jobs/spec` as "the run it on your own machine format", accepted
by `jobs.create`, and priced at nothing extra by the quote. Nothing between
the checkbox and the GPU asked whether the published image could honour the
request, so the run trained, merged, and billed INR 17.36 to discover
something the image already knew before it started.

`temper_core.delivery` is the vocabulary: what a format is called, what it
is for. That is a different fact from what one published digest can
actually produce, and the two had been conflated.

## Decision

**The published-image contract records which delivery formats it can
produce.** `packages/contracts/trainer-image.json` gains `delivery_formats`,
a list read by `trainer_build.producible_delivery_formats()`. The checked-in
contract says `["adapter", "merged"]` -- the two this repository's pinned
image can build -- and omits `quantised`, because the GGUF converter is not
on that image's PATH.

**Absence means every format is producible**, the same pass-open rule
`published_reference` already follows for the image itself. A control plane
reading an older contract, or one from before this field existed, must not
have a launch it used to allow start refusing for a reason it cannot see.

**The refusal happens at `jobs.create`, before anything is provisioned.**
Requesting a format the image cannot produce fails with 400
`delivery_format_not_producible`, naming the format and what the image can
make instead. This is the same shape as `unknown_delivery_format`, one layer
lower: that code means the format does not exist in the vocabulary at all;
this one means it exists but this digest cannot make one.

**`GET /v1/jobs/spec` publishes the same fact per format**, as a
`producible` boolean beside each format's `what_for`. The wizard reads it and
disables the checkbox for a format the image cannot produce, with the reason
in place of a click that would otherwise reach the launch and be refused
there. Absent again reads as producible, so an older control plane's spec
still offers every format the wizard has always shown.

**A republish carries the producibility record forward.**
`publish_trainer_image.write_contract` now reads the contract it is about to
replace and copies `delivery_formats` and its comment into the new one. The
pipeline's publish step measures nothing about producibility; it rebuilds
the same trainer source and records a new digest. Without this, every
republish would silently forget that `quantised` had been ruled out, and the
wizard would start offering it again until someone re-discovered the same
failure on a second paid run.

## Alternatives considered

- **Put the GGUF converter on the image's PATH.** The real fix, and still
  open. Rejected as the thing to do *here*: it is a Dockerfile and dependency
  change to the trainer image, orthogonal to closing the gap where a request
  for a format nobody can build reaches a GPU at all. Restoring `quantised`
  to `delivery_formats` is the one-line change once the converter exists and
  a run proves it.
- **Ask the image itself, at launch time, whether it can produce a format.**
  Rejected. That means provisioning a machine to ask the question the
  contract can answer for free, which is the exact cost this decision exists
  to avoid.
- **Leave the refusal where it was, on the GPU, and just fix the error
  code.** Rejected. A precise refusal that still costs a full training run
  and a merge is a better error message, not a fix.

## Consequences

- `quantised` renders disabled in the wizard today, with the reason stated,
  until a converter lands on the image and a run confirms it.
- A test suite that wants to exercise the `quantised` pipeline end-to-end
  (verification, download, manifest) against the simulated tier now has to
  say so: `trainer_build.producible_delivery_formats` is monkeypatched to
  `None` in those tests, since the checked-in contract's answer is a
  statement about one real digest, not about what the delivery machinery is
  capable of testing.
- The pipeline's publish workflow (`.github/workflows/image.yml`) does not
  need to know about delivery formats at all; `write_contract` reads the
  outgoing contract's answer forward on its own.
