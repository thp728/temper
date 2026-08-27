# ADR-0044 — The artifact is the deliverable, and the adapter is one kind of it

- **Status:** accepted
- **Date:** 2026-08-28
- **Supersedes:** nothing (changes the glossary, not a transport decision)
- **Spec:** `docs/specs/006-streaming-transport-progress-and-artifacts.md`
- **Issue:** [#32](https://github.com/thp728/temper/issues/32)

## Context

The glossary defined the adapter as the deliverable. Full fine-tuning produces
no adapter — it rewrites the whole model — so that definition is about to
become false, and eleven tickets were about to assume one or the other: some
that the deliverable is an adapter, some that it is not. The issue's own
argument, which this record accepts as the frame: **the vocabulary change is
cheap now and expensive after the code has assumed otherwise.**

Spec 006 fixes the vocabulary: the artifact is what a user downloads; an
adapter is one kind of it, a fully trained model is another, and a merged model
is a third. Loading one differs from loading another, so the manifest that
travels with an artifact records which kind it is.

Two things merged the same day this issue opened, and both shape the record:

- [ADR-0035](0035-the-control-plane-verifies-a-machine-written-artifact-by-streaming-it-back.md)
  is where the artifact record is assembled: `_collect_artifact` verifies what
  the machine wrote against the reported checksum and stores the config. That
  function is the seam this record extends — the artifact's *declared kind*
  belongs beside the verification, in the same record.
- [ADR-0041](0041-checkpoints-are-written-off-the-machine-as-they-are-produced.md)
  explicitly left its record shape open "so issue #32 can give stored objects a
  declared kind and classify checkpoints without rewriting this." This record
  takes that judgement: **a checkpoint is not a kind of artifact.**

## Decision

**The artifact is the canonical deliverable, and an artifact declares its
kind.** Three decisions make that concrete:

### 1. The kind is derived from the job's method, never stored

The job row's `method` column (`qlora | lora | full`) is frozen at creation and
immutable. The artifact kind is a pure function of it, defined once in
`temper_core.artifacts`:

- `qlora` and `lora` produce an **adapter** (a PEFT adapter).
- `full` produces a **fully trained model**.
- A **merged model** is named in the vocabulary but produced by no method yet —
  merging is an artifact-time step, spec 011's territory.

A pre-method row (method `NULL`) reads as an adapter, because every run before
the method column existed was a QLoRA adapter; treating history's absence as
"adapter" is a description of the past, not a default that could mask a future
full-model row.

**Why derived and not stored:** "existing adapter artifacts are described
correctly under the new model without migration surprises" is the criterion.
A stored `kind` column would need a backfill whose answer is exactly this
function, would leave every pre-existing row wrong until the backfill ran, and
would then be a second source of truth that could disagree with the method —
a row whose method says `qlora` and whose kind says `full_model` is a row
nobody wrote. Deriving makes the answer total over the methods that exist, free
on day one, and incapable of drifting from the method that produced the bytes.
The artifact *record* exposes the derived kind wherever the artifact is
described (the API's published record, the download manifest); it is computed,
never persisted. The members are the recorded half: the artifact record, stored
at packaging time, names the stored objects the artifact consists of, because
that is what the download path serves and it is different per kind.

### 2. The download path serves any kind by reading the artifact record

The download endpoint (renamed `/v1/jobs/{job_id}/adapter` →
`/v1/jobs/{job_id}/artifact`) carries no per-kind branch. It reads the
artifact record's members — `(arcname, storage key)` pairs recorded when the
artifact was verified — and streams them into a zip, plus a `temper-artifact.json`
manifest declaring the kind, the base model it was trained on, and the load
path that kind needs. A row written before the record existed (only
`artifact_key`) is served through the canonical adapter pair, so a legacy
adapter download behaves exactly as it always did.

Loading instructions differ by kind and live in `temper_core.artifacts`: an
adapter is applied to a base model with PEFT; a fully trained or merged model
is loaded directly. One definition, read by the API's published record and the
download manifest, so the interface says how to load what it offers without
re-spelling the instructions.

### 3. A checkpoint is not a kind of artifact

The classification ADR-0041 left open: a checkpoint is **recovery material,
not a deliverable.** It is never offered through the artifact download, has its
own lifecycle (a bounded retention ring, supersession by newer checkpoints, a
resumption consumer), and lives under its own `checkpoints/` key namespace
with its own `checkpoints_json` record. Making a checkpoint a kind of artifact
would drag the retention ring and the recovery consumer into the download path
for no user who downloads anything. ADR-0041's record shape is therefore left
as-is — classified, not rewritten.

## Why this is not a migration

No column is added to say what an artifact is. The only new stored data is
`artifact_json` — the members and verification outcome the download serves —
and it is written only when an artifact is produced; a row without it reads
through the derivation exactly as before. An existing database opens, its
complete adapter jobs describe themselves as adapters, and nothing needed a
backfill.

## Alternatives considered

**Store a `kind` column, backfilled from method.** Rejected on the criterion
and the drift argument above: the backfill is this derivation run once, it
leaves pre-existing rows wrong until it runs, and a stored kind is a second
truth that can disagree with the method. Derivation is the backfill that never
runs and never disagrees.

**Have the trainer declare the kind in `result.json`.** Rejected: the kind is
a property of the *method* the job trained with, which the control plane owns
and the trainer does not even receive (the trainer applies resolved
hyperparameters and never sees `method`). A machine-declared kind could also
be a lie the control plane would have to verify; the derived kind cannot.
The trainer's report keeps its own vocabulary (`adapter_path`, `adapter_sha256`)
because it describes what the machine produced.

**Special-case the download per kind** (an `if kind == adapter` branch).
Rejected: that is the exact assumption the issue says to remove, and it does
not generalise — a full model's member set is whatever the trainer reports,
not a fixed list a branch could know. Reading the record's members is the only
shape that stays true as kinds land.

**Treat a checkpoint as a kind of artifact.** Rejected (decision 3): the
deliverable framing is false for recovery material, and the artifact download
path would have to learn retention and resumption semantics it has no use for.

**Ship the kind only in the API record, not in the download.** Rejected: spec
006 names the manifest that travels with the artifact as the place the kind is
recorded, and a kind that survives the download is what a user who moved the
zip can still read. The manifest is deliberately minimal — kind, provenance,
members, load path — with evaluation and lineage left to spec 011.

## Consequences

- **The glossary now names the artifact as the deliverable** and the adapter
  as one kind of it; the checkpoint entry records that it is not a kind.
- **`JobRecord` publishes `artifact`** — kind, member names, size, load path —
  and never the storage keys, so the interface names the deliverable without
  learning where it lives.
- **The download is a generic zip of the record's members plus a manifest**,
  named `{job_id}-artifact.zip`, no longer `-adapter.zip`.
- **Existing rows need no migration**: kind is derived from the immutable
  `method`, and a record-less row serves the canonical adapter pair.
- **User-visible strings that described the deliverable as an adapter** ("no
  adapter will be produced", "Download the adapter", "Your adapter") now say
  artifact, including in the API's events and the interface copy.
- **The trainer's report vocabulary is unchanged** — it describes what it
  produced, which for every run today is an adapter.

## Rollback

Point the interface and the download at `/v1/jobs/{job_id}/adapter` again,
restore `ADAPTER_MEMBER_NAMES`-driven member handling in the download, drop
`artifact_json` (the column and its `ADDED_COLUMNS` entry), and revert the
glossary and this record's vocabulary. The verification path, the storage seam
and the grant mechanism are untouched by this change either way — it classifies
what is stored, not how.
