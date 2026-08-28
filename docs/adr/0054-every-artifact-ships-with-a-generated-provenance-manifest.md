# ADR-0054 — Every artifact ships with a generated provenance manifest

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/011-evaluation-and-delivery.md`
- **Issue:** [#70](https://github.com/thp728/temper/issues/70)

## Context

The artifact ships with no record of what produced it. The base model's licence
obligations flow through to whatever the user does next, and a user who cannot
see them cannot comply. The gaps, in the order a reviewer meets them, are the
same ones the spec names: which base model at which pinned revision, which
dataset (with a fingerprint and counts), which configuration including any
overrides, the evaluation summary that says whether training worked, and which
checkpoint the result came from.

Nearly everything that answer needs already exists on the job record by the time
an artifact is verified. #32 makes the artifact declare its kind derived from
the method (`temper_core.artifacts.kind_for`). #50 and #37 write artifacts and
checkpoints through a scoped grant and verify by streaming back through a
checksum; #59 records an export-time template probe result with the artifact.
#62 records which checkpoint was chosen and why, stored on the job rather than
derived, which is exactly "the checkpoint the result came from". #53 holds the
split after deduplication and the evaluation loss, which is the evaluation
summary. #48 resolves model facts (including licence) through a seam, and #58
admits models from outside the catalog after a probe.

The manifest's job is therefore to **assemble, not to gather**. The sentence
that shaped the design is the one the issue quotes: "a provenance document that
can drift from the job it describes is worse than none, because it will be
believed." A hand-written manifest, or a generated one that fills a missing
field with `"unknown"`, will be believed and is therefore a liability.

Two siblings land this wave and set the boundary: #36 (divergence stops early),
#34 (teardown confirmed), #56 (finished page shows its outcome) and #65
(mixture-of-experts labelled untested). None of those surfaces is touched here.

## Decision

**The artifact ships with a provenance manifest that is generated from the job
record, fails on a missing required field, and is readable by a person.**

### What the manifest records, and where each field comes from

- **Base model and pinned revision** — `job.base_model` and
  `job.base_revision` as stored on the job row, plus the licence and
  licence URL as resolved through the same `temper_core.catalog` / admitted-model
  probe the predictor reads. Where a field can be supplied two ways (the job's
  stashed `_admitted_license` versus a catalog lookup) the job's recorded
  value wins. The original minimal manifest kept `base_model` as a string for
  backward compatibility; the detailed record lives under `base_model_info`.

- **Licence obligations that propagate** — stated explicitly, not left to a
  reader to infer. The catalog today is Apache-2.0 only; the obligations text
  for that licence is a constant in `temper_core.manifest` and is what
  appears in the manifest's `base_model_info.license_obligations`. Any other
  licence that later lands in the catalog extends that table rather than
  falling through to a generic placeholder. A missing or placeholder licence
  (`""` or `"unknown"`) fails generation rather than being rendered as
  `"unknown"`.

- **Dataset fingerprint with counts** — the trainer's recorded `held_out_split`
  (`rows_in`, `rows_removed_duplicates`, `train_rows`, `held_out_rows`,
  `fraction`, `seed`) plus the validation report's counts (`row_count`,
  `usable_rows`) and the token count when the counting phase has landed.
  The fingerprint's identity part is the dataset content hash when one is
  recorded; no hash is recorded for datasets today (see outstanding gap
  below), so until then the fingerprint is the split plus the dataset id and
  counts. Two different files with the same counts and id therefore
  fingerprint identically today, which is an honest outstanding gap rather
  than a hidden one. When a hash is recorded (e.g.
  `dataset.content_hash` / `sha256`) it is carried as `content_hash`.

- **The full configuration including overrides** — `job.hyperparameters` as the
  effective configuration (defaults plus method-specific table and any
  overrides, via `temper_core.hyperparams.effective`) and `job.overrides` as
  the frozen list of pinned decisions. An empty `overrides` is legitimate;
  a missing `hyperparameters` fails.

- **The evaluation summary** — the held-out split plus the best-checkpoint's
  reason and the export-time template probe result (`result.template_probe`,
  #59) when present. The summary is what lets a reviewer see whether the
  model learned or memorised.

- **The checkpoint the result came from** — `job.best_checkpoint` as stored by
  #62 (`step`, `basis`, `reason`, `held_out_loss`). The choice is recorded
  once at terminal time and never re-derived; the manifest reads it, so a
  retention eviction or a rule edit cannot move a job's answer.

- **The artifact itself** — kind derived from `job.method` via
  `temper_core.artifacts.kind_for`, members from the stored `artifact_record`,
  bytes and checksum from the verification, and loading instructions from
  `temper_core.artifacts.loading_instructions`. The members that travel in the
  zip are the ground truth; the manifest's `artifact.members` is forced to
  match them.

All of the above lives in `temper_core.manifest`: a pure module with no I/O.
The caller supplies `generated_at` so the module never reads the clock
(ADR-0010: `packages/core` is pure), and the one exception to purity is the
licence lookup through `temper_core.catalog`, which is a stable, versioned
source, not a network call. Generation is therefore testable as a table of
job records and expected manifests.

### Missing required field fails generation

`MissingField` is raised naming the field rather than producing a placeholder.
No `"unknown"`, no empty string, no `null` standing in for a fact nobody
recorded. The control plane's download path catches that failure and falls back
to the minimal legacy manifest only for rows written before provenance existed
-- a provenance-capable job never reaches that branch. The failure is the
enforcement of the drift sentence: a manifest that invents a value will be
believed.

### How it ships

The manifest travels **with** the artifact, not beside it. The control plane's
`GET /v1/jobs/{id}/artifact` builds the zip from the stored `artifact_record`
members, then appends two provenance files:

- `temper-artifact.json` — the machine-readable provenance, pretty-printed
  (`indent=2`, `sort_keys=True`) so it is not only a machine artifact. The
  original minimal manifest's flat keys (`kind`, `base_model`,
  `base_revision`, `members`, `bytes`, `loading`) are kept at the top level
  for backward compatibility; the new provenance lives alongside them.

- `PROVENANCE.md` — the human-readable provenance, rendered from the same
  dict. Headings, sentences and a table of hyperparameters let a reviewer skim
  without a JSON viewer, and the full JSON is appended as a fenced block so
  the two cannot drift.

Both are real members of the zip, not a second endpoint. The endpoint streams
the zip, so the provenance does not change the flat-memory guarantee.

### Human readability is a requirement

The Markdown contains the same facts as the JSON, in sentences and headings.
A job that cannot be understood from its manifest has not been explained, and
the platform's value is explaining jobs. The JSON's pretty-printing is the
second, smaller half of the same requirement: a reviewer who opens the JSON
directly sees indented, sorted keys, not a single line.

## Alternatives considered

**Hand-write the manifest on the machine or in the control plane's download
handler.** Rejected: a hand-written document can drift from the job it
describes and will be believed. The generation reads the job record whole,
so the manifest says what the job says.

**Produce `"unknown"` or `""` for a missing field so the zip still builds.**
Rejected: the criterion "a missing required field fails generation rather
than producing a placeholder" is the enforcement of the drift sentence.
A placeholder is a lie that survives a download. `MissingField` names the
field so the defect can be diagnosed without guessing.

**Store the kind in a new column and read it back.** Rejected for the same
reason #32 was: the method is frozen at creation and immutable, so the kind
derived from it is immutable too. Storing a second source of truth would let
a row whose method says `qlora` claim `full_model`.

**Ship the manifest as a second endpoint (`GET /v1/jobs/{id}/manifest`).**
Rejected: an artifact whose provenance is fetched separately is an artifact
that can be moved without its provenance. Shipping it inside the zip means a
moved artifact still carries how it was made.

**Make the manifest a single compact JSON string.** Rejected: a single-line
JSON file is a machine artifact that a person has to format before reading.
Pretty-printing and a Markdown rendering are cheap and are what make the
manifest a review surface.

**Add a new store for provenance or extend the database schema for it.**
Rejected: provenance needs no new store. The job record already carries every
field the manifest needs except the dataset content hash (outstanding, see
below); adding a new table would be a second source of truth that could drift
from the job it is meant to describe.

## Consequences

- `temper_core.manifest` is the single definition of the provenance shape;
  the control plane's download path, the interface and the artifact manifest
  cannot drift.
- A complete job's artifact zip contains `temper-artifact.json` and
  `PROVENANCE.md`; a legacy row written before provenance still downloads
  via the minimal manifest so the platform does not retroactively break an
  artifact, but the fallback is visibly marked as incomplete.
- The control plane's API contract (`openapi.json`) is unchanged: the
  provenance is part of the artifact download, not a new HTTP resource.
- Existing adapter/full-model download tests now assert both manifest files
  and the provenance fields; legacy rows that lack provenance hit the
  incomplete-manifest fallback in the download path.
- `generate()` is pure: the caller supplies `generated_at`, so the module
  never reads the clock and is testable as a table. ADR-0010's pure rule
  holds.

## Outstanding gap

Dataset content hash is not recorded anywhere today. The dataset's bytes are
stored (and #50 verified artifacts via checksum) but the dataset itself is
stored without a content hash (`dataset.content_hash` / `sha256` is absent
from the dataset row). Until a hash is recorded, the "fingerprint" is the
split plus the dataset id and counts, and two different files with the same
counts and id would fingerprint identically. The manifest carries
`content_hash` when present and omits it otherwise, rather than pretending
counts are identity.

## Rollback

Revert the download handler's provenance append, keep the original minimal
`temper-artifact.json` branch, and drop `temper_core/manifest.py` and its
tests. The `job_progress` / `job_output` and checkpoint selection behaviour
(#53, #62) is untouched. The original manifest's flat keys remain, so an
artifact downloaded after the revert still carries a kind and a load path, just
no provenance.
