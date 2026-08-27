Closes #45

A dataset can be imported from a public repository, optionally with a
configuration and split, and the imported rows go through exactly the same
validation as an upload — same schema detection, same line-numbered errors,
same thinking-mode detection. Nothing gets a shortcut for arriving over a
network.

What a user does: on the upload page there is now a card to import from a
public repository (e.g. `open-r1/OpenR1-Math-220k`, with optional config and
split). Temper resolves the reference, streams the rows into storage, and
validates them in the background through the very same path an upload uses.
The validation report that comes back is the same report, so a failed import
is kept with its report naming the rows to fix.

## How

A new seam, `remote_datasets.py`, turns a repository reference into the same
bytes an upload would have carried. `validation.validate_chunks` already
consumes any iterable of bytes chunks, so there is exactly one validator; the
seam is the only place that knows the word "repository". `POST
/v1/datasets/import` answers 202 like an upload and the report lands on
`GET /v1/datasets/{id}`.

## Why

**Decisions**

- **The seam lives in the control plane, not `temper_core`.** Resolving an
  arbitrary public dataset is network I/O, and the core package carries zero
  I/O — the same rule that placed the model-facts seam (ADR-0028) and the
  provider seam. The seam resolves the reference against the Hugging Face
  datasets-server (validity, config/split catalogue, split size) before
  anything is stored, so a bad reference is refused with its reason rather
  than dying mid-fetch. Rows are fetched one bounded page (100 rows, the
  server's cap) at a time and serialised through `jsonl_chunks` into bounded
  byte chunks, so peak memory stays flat however many rows the split holds —
  the flat-memory guarantee applies to imports because a remote dataset is
  exactly as large as an uploaded one.
- **Imported bytes are stored, then validated by the identical background
  pass an upload uses.** `datasets.import_dataset` creates the row, stores the
  fetched bytes via the same `_store_counted` helper (size ceiling included),
  then calls the same `_validate_in_background` — and the same token-counting
  phase after it. The tests pin this byte-for-byte: importing a dataset and
  uploading the same JSONL produces identical validation reports.
- **Refusals are up front, with their reasons.** `repo_not_found` (a
  repository that cannot be fetched), `config_required` (a bare repo with
  several configurations — guessing which subset a user meant is the shortcut
  this feature exists to avoid), `config_not_found`, `split_not_found`, and
  `split_empty` are answered as coded 400s before anything is stored. A fetch
  that dies part-way through streaming — rate limit, vanished server, dropped
  connection, unparseable body — surfaces as a coded `fetch_failed` 400 with
  the half-written import cleaned up. `repo_not_found` is deliberately
  distinct from the orchestration path's `dataset_not_found` ("no dataset with
  this id"), because one stable code with two meanings is a code a client
  cannot trust.
- **The size ceiling is read, never retyped.** Imports enforce
  `config.MAX_DATASET_BYTES` (ADR-0036's 1.3 GB, derived from measured
  throughput) mid-stream via the same `_store_counted` wrapper an upload uses
  for a lying Content-Length. A remote reference has no declared size, so the
  mid-stream refusal is the whole enforcement — the same `dataset_too_large`
  413.
- **The suite and journeys run against a local double.**
  `fake_remote_datasets.py` is the datasets-seam equivalent of `FakeProvider`;
  `new_remote_datasets()` returns it when `TEMPER_FAKE_PROVIDER` is set, and
  an autouse conftest fixture replaces `RESOLVER` outright for pytest, so no
  test or journey reaches the network.

**Rejected alternatives**

- A second, import-specific validation implementation — rejected outright; it
  is the failure mode the issue exists to prevent ("nothing gets a shortcut
  for arriving over a network"). The seam produces bytes; the existing
  validator consumes them.
- Fetching the dataset whole into memory before validating — rejected; it
  reintroduces the memory multiplier the streaming rewrite removed, and an
  import is exactly as large as an upload.
- Making the whole import asynchronous and recording every refusal (including
  a split resolving to nothing) as a stored invalid dataset — rejected; the
  criteria separate a *refusal* (before anything is stored, with its reason)
  from a *failed validation that is still stored with its report*.
- Surfacing fetch failures as the generic background `validation_failed`
  report — rejected; the criterion is "surfaces **the reason**".
- Parsing the repository's raw files (JSONL/Parquet) — rejected; the
  datasets-server rows API maps rows to chat-format JSONL directly, with one
  bounded page held at a time, and needs no file-type dispatch.

**Which acceptance criterion each part satisfies**

- *A dataset can be imported by reference, optionally with a configuration
  and split* — `DatasetImportRequest {repo, config?, split?}`, the import
  form and journeys.
- *Imported rows go through the identical validation path, not a parallel or
  relaxed one* — the seam produces bytes; `_validate_in_background` is the
  same function; pinned by
  `test_import_and_upload_of_the_same_rows_report_identically`.
- *A repository that cannot be fetched surfaces the reason rather than
  failing mid-job* — `repo_not_found`/`fetch_failed` coded 400s,
  `test_an_unfetchable_repository_is_refused_with_the_reason` and
  `test_a_fetch_that_dies_mid_stream_surfaces_its_reason`.
- *A split resolving to nothing is refused with that reason* —
  `split_empty`, refused in `resolve` and guarded again when a stream yields
  no rows, `test_a_split_that_resolves_to_nothing_is_refused_with_that_reason`.
- *An imported dataset that fails validation is still stored with its report*
  — the bytes are stored and validated by the shared background pass,
  `test_an_import_that_fails_validation_is_stored_with_its_report`.
- *The size ceiling applies to imports as it does to uploads* —
  `_store_counted` reads `config.MAX_DATASET_BYTES`,
  `test_the_size_ceiling_applies_to_imports`.

ADR-0044 records the shape and its rejected alternatives.

## Notes

- The browser journeys import against the canned fake (booted with
  `TEMPER_FAKE_PROVIDER`), so they reach no network; the e2e ports moved to
  this issue's range (4530/4531) so this worktree's journeys cannot collide
  with a sibling's.
- The real Hugging Face fetch path is unexercised on real hardware, exactly
  like the real provider path: its parsing is proven against mocked HTTP
  responses and its double, recorded in ADR-0044.
