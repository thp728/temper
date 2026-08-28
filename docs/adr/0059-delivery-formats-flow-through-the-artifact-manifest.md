# ADR-0059 — Delivery formats are produced off the correctly merged model and flow through the artifact manifest

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/011-evaluation-and-delivery.md`
- **Issue:** [#74](https://github.com/thp728/temper/issues/74)

## Context

A job finishes and hands back one form of its artifact: the adapter (or a fully
trained model). A user who wants to serve a single model, or run it on their own
machine, currently does the conversion themselves. The issue asks for two more
forms: **a merged single-file model** (base + trained change, for serving) and
**a quantised local-inference format** (from the correctly merged model, the
"run it on your own machine" story).

Two properties of this work are load-bearing and both are silent:

- **Order.** Merging into an already-quantised base compounds error, and a model
  merged in the wrong order still loads and answers, only slightly worse. The
  merge must happen at full precision, and quantisation exactly once, afterwards.
  The failure is silent, so an *assertion that the order held* is the deliverable
  (spec 011's testing decision).
- **The merged guarantee.** #70 (ADR-0054) shipped a provenance manifest that is
  *generated from the artifact record*, never hand-written beside it. Every
  additional format must flow through that same manifest generation, not beside
  it: a second, parallel description of what an artifact is would falsify
  ADR-0054 the day it merged.

This wave's boundaries set the rest: merge, precision, quantisation, format
verification and the template probe are this ticket's territory; the
orchestrator's retry/teardown, billing, the evaluation comparison and startup /
configuration wiring are not. The machine is warm with both models loaded, so
conversion happens there at export time — evaluation already runs on the warm
machine, and "format conversion is a step off the correctly merged model, not a
parallel pipeline" (spec 011).

## Decision

**Delivery formats are requested at launch, produced on the machine in a fixed
order (merge at full precision, then quantise once), each verified by loading
it, and each download flows through the same `temper_core.manifest.generate`.**
The order is a pure property in `temper_core.delivery`, asserted in code and by
tests over every requested subset.

### The delivery vocabulary is data, defined once

`packages/contracts/delivery-formats.json` is the single definition of the
delivery formats: id, the artifact kind each maps to, a plain-language "what it
is for" (criterion: a user chooses a format without knowing the internals), and
the conversion each requires. The trainer cannot import `packages/core`
(ADR-0010), so it reads this file as data, exactly as it reads
`trainer-defaults.json` and `fault-surface.json`; `temper_core.delivery` reads
the same file. One definition, two consumers, never retyped.

Three formats:

- `adapter` — the trained change as it is (the canonical artifact, kind
  `adapter` / `full_model`). No conversion.
- `merged` — base + trained change at full precision, kind `merged_model`.
  Conversion: merge.
- `quantised` — a quantised local-inference format from the correctly merged
  model, kind `quantised_local`. Conversion: quantise, **once, afterwards**.

### The merge-then-quantise order is a property

`temper_core.delivery.production_steps(requested)` returns the ordered
conversions for a requested set and *enforces* the invariant: requesting
`quantised` implies a preceding `merged` step, because quantisation consumes
the correctly merged model, never the adapter or a quantised base. The pure
function cannot produce an order that quantises before merging, and a property
test asserts the invariant across every subset of the vocabulary.

### Every format flows through the same manifest generation

`temper_core.manifest.generate` gains an optional `delivery_format`. When given,
it reads that format's verified artifact record from the job row and describes
it with the same generator — kind, members, bytes, checksum and loading from the
recorded delivery result, never transcribed. The download endpoint serves each
format's members with a manifest produced by this same function, so the
provenance of a merged-model download is the merged model's own record, and no
parallel description of an artifact exists anywhere.

### Production and verification

The trainer's export step, after the primary artifact and its template probe,
runs the requested conversions in the order `production_steps` returns:

- **merge** at full precision (bf16), never into the quantised base;
- **quantise** exactly once, from the merged output;
- each produced format is **verified by loading it** — on the machine this is the
  image's real loader (transformers for a merged safetensors model, the image's
  GGUF loader for the local format) — and an unloadable format fails the export
  with a stable code;
- the **template probe runs on each export** (already the rule for the primary
  artifact; now true per produced format);
- each format is uploaded through its own scoped write grant (ADR-0009's
  machinery, one grant per format).

The control plane mints one grant per requested format, verifies each object
that landed against the trainer's reported checksum, records a per-format
delivery record (kind, members, bytes, sha256) on the job row, and serves each
format at `GET /v1/jobs/{id}/artifact?format={id}` with the manifest generated
for that format. The published job record lists the produced formats with their
purpose and loading so the interface can offer a download with a plain-language
choice.

## Alternatives considered

**Merge/quantise in the control plane at download time.** Rejected: the control
plane has no GPU, no base model and no warm context; spec 011's own reason for
running evaluation on the warm machine applies to conversion twice over. The
machine produces the formats while it exists.

**A separate manifest generator per format.** Rejected: that is the exact
"second, parallel description" ADR-0054 forbids. One generator, parameterised by
which format's recorded artifact it describes.

**Produce the formats eagerly for every job.** Rejected: the issue says "on
request". A merged model of a 4B base is gigabytes; producing and storing it for
every adapter run would be waste a user did not ask for. The request rides the
job spec, frozen at launch like everything else.

**Quantise in parallel with merge.** Rejected: that is precisely the silent
double-error the issue names. Quantisation is one step, off the merged output,
and the order is a property.

**Rely on a comment to document the order.** Rejected: the issue states the
failure is silent, so the order must be asserted, not narrated. `production_steps`
cannot return an illegal order, and the property test proves it cannot.

## Consequences

- The web launch form gains a delivery-format choice (each format stated in
  plain language); the finished job page lists the produced formats with their
  purpose and a per-format download.
- The trainer image ships `delivery.py` and `delivery-formats.json`; both are in
  `TRAINER_SOURCES` and the Dockerfile `COPY`, pinned by `test_trainer_context.py`.
- A new job column records the frozen delivery request, and a second records the
  verified per-format delivery results; the published job record exposes the
  produced formats.
- `GET /v1/jobs/{id}/artifact?format=...` serves any produced format; the default
  (no format) keeps serving the canonical artifact exactly as before, so nothing
  existing breaks.
- Host tests cannot run the real model loaders (the dev environment has no
  torch/transformers/GGUF). The export path is tested with a loader seam that
  genuinely validates a produced file, and the PR body names the on-machine load
  as the outstanding hardware verification with the reason a fake load does not
  substitute.
- The manifest for a delivery-format download is generated by the same
  `temper_core.manifest.generate`, so ADR-0054's drift guarantee is preserved
  per format.

## Rollback

Revert the trainer's `delivery.py` wiring and the control plane's grant/verify/
serve changes; keep or drop the `delivery_formats.json` vocabulary and the
`temper_core.delivery` module as a unit. The canonical artifact path is
untouched (the `format` query parameter defaults to it), so removing delivery
formats removes an addition, not a replacement.
