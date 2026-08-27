# ADR-0044 — Imported datasets are the same bytes, through the same validation path

- **Status:** accepted
- **Date:** 2026-08-28
- **Supersedes:** nothing
- **Spec:** `docs/specs/009-models-methods-and-advanced-mode.md`
- **Issue:** [#45](https://github.com/thp728/temper/issues/45)

## Context

A user should not have to prepare a file to start. Issue #45's user story is
*"import a dataset from a public repository, so that I can start without
preparing a file"* — and its central sentence is the constraint that shapes
every line of the implementation: *"Imported rows go through exactly the same
validation as an upload — same schema detection, same line-numbered errors,
same thinking-mode detection. Nothing gets a shortcut for arriving over a
network."*

The failure mode that sentence exists to prevent is a second, subtly different
validation path for imports. A "convenient" import path — one that fetched the
rows, skipped some checks, or reported differently — would be the one users
actually used, and the report is the product's best work before a run starts;
two ingest paths that report differently would drift until they contradicted
each other, and the drift would surface as an import whose rows were treated
more leniently than an upload's.

The platform already has the shape this wants. Spec 006's validator is
**streaming**: `validation.validate_chunks` accepts any iterable of bytes
chunks and decides each row before the next is read, so peak memory stays flat
as the dataset grows (ADR-0036). An upload no longer holds the file either —
it streams the body to object storage and validates the stored object in the
background. The `models` seam ([ADR-0028](0028-peak-memory-is-predicted-through-a-model-facts-seam.md))
and the provider seam already established the pattern for "network I/O lives in
the control plane, behind a seam whose double keeps the suite and the journeys
off real hardware".

What was missing was a way for a remote repository to become the same bytes an
upload would have carried, with the same guarantees.

## Decision

**A dataset imports by reference through a remote-dataset seam that produces
the same bytes an upload carries, and everything downstream is the upload
path, untouched.**

Concretely:

1. **The seam resolves the reference up front.** `remote_datasets.RESOLVER`
   (the control-plane equivalent of `storage.STORE`) takes `(repo, config,
   split)` and either returns a fetchable source or raises `RemoteDatasetError`
   with a stable code and a reason. A repository that cannot be fetched is
   `repo_not_found`; a repository with several configurations whose user
   named none is `config_required` (guessing which subset a user meant is the
   shortcut this feature exists to avoid); a named configuration or split that
   does not exist is `config_not_found` / `split_not_found`; and a split that
   resolves to no rows is `split_empty` — **a split resolving to nothing is
   refused with that reason, before anything is stored.** The `/size` refusal
   is the fast path; the source also refuses `split_empty` if a stream turns
   out to carry no rows, so the criterion holds even when the server reports
   no size. The import endpoint turns these into coded 400s. Every other way a
   fetch can die — a rate limit, a vanished server, a dropped connection, a
   body that is not JSON — is a coded `fetch_failed` 400 with the half-written
   import cleaned up, never an unhandled error mid-import.
2. **Rows are streamed, never materialised.** The Hugging Face datasets-server
   `/rows` endpoint serves one bounded page (100 rows, the server's own cap)
   at a time; the source serialises each page through `jsonl_chunks` into
   bounded byte chunks and hands them straight to storage's `put_stream`. Peak
   memory stays flat however many rows the split holds — the same guarantee
   the streaming validator keeps for an upload, and the reason a remote
   dataset (exactly as large as an uploaded one) is never fetched whole.
3. **The size ceiling applies by reading the same configuration an upload
   reads.** `config.MAX_DATASET_BYTES` (ADR-0036, derived from measured
   throughput, never a retyped number) is enforced mid-stream in the same
   `counted_chunks` wrapper the upload path uses for a lying Content-Length. A
   remote reference has no declared size to refuse on ahead of time, so the
   mid-stream refusal — the same `dataset_too_large` 413 — is the whole
   enforcement, and it is the same enforcement.
4. **The bytes go to object storage under the same key scheme, and validation
   and token counting run in the background through the very functions an
   upload uses.** `datasets.import_dataset` creates the dataset row, stores the
   fetched bytes, then calls the same `_validate_in_background` the upload
   path calls. **An imported dataset that fails validation is therefore stored
   with its report like any other** — the report names the rows to fix, and
   the stored object is the object the report describes.
5. **The suite and the browser journeys run against a local double.**
   `fake_remote_datasets.py` is the datasets-seam equivalent of
   `FakeProvider`: `new_remote_datasets()` returns the fake when
   `TEMPER_FAKE_PROVIDER` is set (so "this process cannot reach anything
   external" stays a single flag, and the journeys import without a network),
   and the pytest suite replaces `RESOLVER` outright in an autouse fixture so
   no test even routes through the network path.

### Why the seam lives in the control plane

Resolving an arbitrary public dataset means reading it over the network, and
`temper_core` carries zero I/O — the same rule that put the model-facts and
provider seams in the control plane. The validator needed no seam at all: it
already consumes any `Iterable[bytes]`. The only new seam is the one between a
remote repository and storage, and it is the only place that knows the word
"repository".

### Why resolution is synchronous and up front

The acceptance criteria distinguish a **refusal** ("a split resolving to
nothing is refused with that reason") from a **failed validation that is still
stored** ("an imported dataset that fails validation is still stored with its
report"). Those are two different outcomes with two different timings. A
refusal happens before a row exists, answered as a coded 400; a failed
validation happens after the bytes are stored, recorded as an `invalid`
dataset with its report. Resolving the reference synchronously in the request
is what lets the refusals be refusals.

## Alternatives considered

**A second, import-specific validation implementation.** Rejected outright —
it is the failure mode the issue exists to prevent. The seam produces bytes
and the existing validator consumes them; there is exactly one validator.

**Fetch the dataset whole into memory and validate the materialised bytes.**
Rejected. A remote dataset is exactly as large as an uploaded one, and the
flat-memory guarantee ([ADR-0036](0036-the-dataset-size-limit-is-derived-from-measured-throughput.md))
applies to imports with the same force — holding a gigabyte-scale import in
memory to re-validate it would reintroduce precisely the memory multiplier the
streaming rewrite removed.

**Make the whole import asynchronous and record every refusal — including a
split resolving to nothing — as a coded report on the dataset record.**
Rejected because it conflates the two outcomes the criteria separate: a split
that resolves to nothing is *refused* with its reason (before anything is
stored), while a split whose rows fail validation is *stored with its report*.
Recording a `split_empty` as a stored invalid dataset would turn a refusal
into a failed import, and a user retrying would see a new row every time.

**Surface a fetch failure as the generic `validation_failed` report the
background validator writes when it dies.** Rejected. The criterion is "a
repository that cannot be fetched surfaces **the reason**" — a generic
"validation stopped unexpectedly" is exactly the failure this ticket names.
Fetch failures carry their own codes and reasons, and they are answered
synchronously, where the reason is legible.

**Parse the repository's raw JSONL/Parquet files directly.** Rejected. The
datasets-server rows API serves the rows as objects with their column names,
which is what maps onto chat-format JSONL; re-serialising each row through
`jsonl_chunks` produces bytes byte-for-byte comparable to an upload, with one
bounded page held at a time. Raw-file parsing would re-implement schema
handling and file-type dispatch for no benefit, and Parquet is not JSONL.

## Consequences

- **Imports and uploads share one validation pass, one storage seam, one
  size ceiling, and one report shape.** The only difference is where the
  bytes came from; the tests pin this by importing a dataset and uploading the
  same JSONL and asserting the two reports are equal.
- **Refusals are stable-code refusals.** `repo_not_found`,
  `config_required`, `config_not_found`, `split_not_found`, `split_empty`,
  `fetch_failed` and the shared `dataset_too_large` are the import surface's
  machine-readable vocabulary, same contract as every other API refusal.
  `repo_not_found` is deliberately distinct from the orchestration path's
  `dataset_not_found` ("no dataset with this id"): one identifier, one
  meaning, or a client cannot tell them apart.
- **A reference's provenance is part of the record.** The imported dataset's
  filename is the reference including the configuration and split that were
  actually fetched (a bare repo shows the defaulted config and split), so the
  report page says what was imported, not just that something was.
- **The real fetch path is unexercised on real hardware**, exactly like the
  real provider path: the parsing is proven against mocked HTTP responses and
  the journeys run against the double. This mirrors ADR-0027's posture — the
  transport tier gets its own real-endpoint test — and is recorded rather
  than hidden.
- **No database or storage schema changed.** An import is stored as a dataset
  row with an object key, exactly as an upload is; there is no `source` column
  because nothing downstream needs to know.

## Rollback

Remove the `POST /v1/datasets/import` endpoint, `datasets.import_dataset`,
the `remote_datasets.py` / `fake_remote_datasets.py` modules, and the import
form and journeys. The validator, storage seam and size ceiling are untouched,
so the upload path is exactly what it was before this record.
