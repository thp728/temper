# ADR-0073 — A bulk import reads Parquet, not the preview endpoint

- **Status:** accepted
- **Date:** 2026-09-03
- **Spec:** `docs/specs/002-dataset-transport-and-limits.md`
- **Issue:** dataset import from HF (the seam is issue [#45](https://github.com/thp728/temper/issues/45))

## Context

The remote-dataset seam (issue #45) imports a public Hugging Face split by
resolving it through `/is-valid` → `/splits` → `/size` and then paginating
`GET /rows?...&offset=N&length=100` until the split runs out. That is the
pattern HF's own quickstart shows, and it is where this implementation got it.
It is also the wrong pattern for what this seam does, and following it
uncritically is the mistake being corrected here.

**What we measured today.** Importing a 25,000-row split means about 250
sequential `/rows` calls. It reproducibly returned HTTP 429 part-way through —
while `/is-valid`, `/splits` and huggingface.co itself all answered 200 in the
same window, from the same process, with the same token. So the throttle is not
an account-wide block, not an IP block, and not a slow day. `/rows` has its own,
much tighter bucket, and a bulk import walks straight into it.

**Why that is architectural rather than bad luck.** HF's dataset-viewer server
documentation (https://huggingface.co/docs/dataset-viewer/en/server) splits its
endpoints in two: `/splits`, `/first-rows` and `/parquet` are precomputed by
background workers and served from a Mongo cache, while `/rows` and `/search`
are the only ones *generated on demand, per request*. The endpoints that kept
answering are the cached ones. The one that throttled is the one that does work
per call. The docs also describe `/rows` as a way to "download slices of a
dataset", capped at 100 rows per request. It is a preview API. We were using it
as a bulk transfer API, 100 rows at a time, and the rate limit is the API
telling us so.

**The rate limits nobody publishes and the ones they do.** HF publishes limits
for huggingface.co only (hub-docs `docs/hub/rate-limits.md`): for a free
authenticated account, **API 1,000**, **Resolvers 5,000** and **Pages 200**
requests per fixed 5-minute window. There is **no published limit for
datasets-server `/rows` at all** — which means an import built on it is built on
a number nobody has committed to. Parquet URLs are
`https://huggingface.co/datasets/<repo>/resolve/refs%2Fconvert%2Fparquet/...`;
the `/resolve/` segment puts them in the **Resolvers** bucket, five times the
API bucket and vastly more headroom than whatever `/rows` allows. And the shape
of the traffic changes with it: one request per shard instead of one per
hundred rows.

Everything else about the seam was fine. Resolution answers a bad reference
with its reason before anything is stored, failures are coded, and validation
is the same streaming path an upload takes. The defect is one mechanism inside
`stream()`.

## Decision

**A bulk import reads the split's Parquet files. `/rows` stays only as the
fallback for a split HF has not converted yet.**

- **`GET /parquet?dataset=<repo>&config=<config>` lists the files; the shards
  for the resolved (config, split) are downloaded and read.** The listing is
  one of the precomputed, cached responses, so asking for it costs the
  throttled path nothing. A split is often several shards, and all of them are
  read, ordered by the shard's URL path with digit runs compared by value.

  **Neither `filename` nor the listing's own order is that order**, and both
  looked like they were until they were checked. A large split is broken into
  parts — `default/train-part0/0000.parquet`, `train-part1/0000.parquet` — and
  each part restarts its numbering, so filenames collide: the 27,468 shards of
  `HuggingFaceFW/fineweb` carry 10,000 distinct filenames, each repeated once
  per part. The listing then returns them interleaved across parts rather than
  running through them. Sorting by `filename` would have imported part 1's
  first shard immediately after part 0's first shard; trusting the listing
  would have done the same. Comparing digit runs by value also keeps `part10`
  after `part2`, which plain string order does not.

  Two more things about that request were measured today rather than assumed.
  **Asking by configuration is not an optimisation, it is what makes large
  repositories work at all:** datasets-server refuses to compute a response
  over 10 MB, so `?dataset=HuggingFaceFW/fineweb` answers `501 Not
  Implemented` while the same request with `&config=default` answers 200 with
  27,468 files. And **`&split=` is not honoured** —
  `?dataset=nyu-mll/glue&config=cola&split=train` came back carrying all three
  of cola's splits — so it is not sent, and the split filter lives on this
  side, where it actually happens. Reading the parameter list and assuming it
  filters would have concatenated a repository's test split onto its train
  split, silently, which is the same class of mistake as the one this record
  is correcting.

- **`resolve()` is untouched.** Same three calls, same shared deadline, same
  stable codes (`repo_not_found`, `config_required`, `config_not_found`,
  `split_not_found`, `split_empty`, `fetch_failed`), same `display_name()`. The
  listing is fetched lazily inside `stream()`, so a mechanism swap cannot
  change what a synchronous refusal says or how long resolution takes. Nothing
  below the seam changes at all: `stream()` still yields bounded JSONL byte
  chunks, `_store_counted` and the streaming validator never learn which
  mechanism produced them, and the API contract is byte-identical.

- **Flat memory survives, and it took a flag to keep it.** A shard is drained
  to a temporary file in 1 MiB blocks — pyarrow needs a seekable file to read a
  Parquet footer, and an HTTP response is not one — then read with
  `ParquetFile.iter_batches()`, 100 rows at a time, deleted in a `finally`
  whether the stream finished, failed or was abandoned. One shard on disk at a
  time, never the split.

  The part worth recording: **batching alone does not bound anything.**
  `pq.ParquetFile(path)` defaults to `pre_buffer=True` and reads the whole file
  into Arrow memory before the first batch comes out. Measured here on a 32 MB
  shard of 1,000,000 chat rows (pyarrow 25.0.1): **32.3 MB** of Arrow
  allocation with the default, **0.99 MB** with `pre_buffer=False`. Two shards
  a factor of ten apart then held **1,084,608 bytes each, the same number to
  the byte**. Arrow allocates outside Python's allocator, so `tracemalloc` sees
  none of this and a test written against it would have passed while the
  guarantee was broken; `test_parquet_memory_stays_flat_as_a_shard_grows`
  watches `pa.total_allocated_bytes()` instead.

- **The fallback is narrow and deliberate.** HF converts a dataset to Parquet
  in the background, so a repository published minutes ago genuinely has no
  conversion yet. When the listing *answers* and carries nothing for the
  resolved split — no files, or the split reported as `pending` or `failed` —
  the import takes the old `/rows` path rather than refusing. A listing that
  *fails* is a real failure and is coded as one: falling back onto the endpoint
  that throttles because a cached endpoint errored would be treating a symptom
  with the disease.

- **The token rides every request, including the downloads.** Built once in
  `_authorized()` and read by the JSON calls and the shard downloads alike. A
  token on the listing but not on the shard would leave the half that moves the
  data anonymous, and would fail outright on a gated repository. A shard URL
  redirects to HF's CDN (`us.aws.cdn.hf.co`) and `urllib` forwards the header
  across that hop; some object stores reject a request carrying both a bearer
  header and a signed URL, so it was checked rather than assumed — 200 with
  and without the header, `PAR1` on the wire either way (2026-09-03).

- **`pyarrow` becomes a control-plane dependency.** Checked first: it was not
  available transitively. It sits beside the other Hugging Face-facing
  dependencies, never in `temper_core`, which stays free of dependencies.

- **Every failure is still a coded `RemoteDatasetError`.** The download and the
  Parquet read are inside the same contract as the JSON calls: the 429 retry
  policy, the wall-clock deadline enforced by `_EXECUTOR` plus
  `future.result(timeout=...)` (because `urlopen(timeout=)` bounds one socket
  operation, not one call), and a corrupt shard that raises out of pyarrow.
  One new failure mode needed a decision: Parquet carries types JSON does not
  have. `/rows` handed back HF's own JSON encoding of them, so reading files
  directly means meeting real timestamps, decimals and binary. `json.dumps`
  would raise `TypeError` mid-stream on those, so a `default=` hook encodes
  them — ISO-8601 for timestamps, which is HF's own encoding, and base64 for
  binary, which is this seam's choice and not a claim of byte parity. A dataset
  whose columns are raw bytes is not fine-tuning text; what matters is that it
  gets refused by the validator with a line number instead of becoming a 500.

### What this changes in ADR-0047, and what it does not

[ADR-0047](0047-imported-datasets-are-the-same-bytes-through-the-same-validation-path.md)
considered and rejected *"parse the repository's raw JSONL/Parquet files
directly"*, on the grounds that raw-file parsing would re-implement schema
handling and file-type dispatch, and that "Parquet is not JSONL". **That
rejection was right about the thing it named and wrong about the thing being
done here, and the two got conflated.** A repository's own files are whatever
the author uploaded: CSV, JSON, archives, several formats in one repo. Reading
those really would mean file-type dispatch. The `refs/convert/parquet` branch
is not that. It is HF's own normalisation of every viewable dataset into one
format with the same column-named rows `/rows` hands back, produced by the same
conversion the viewer serves from. There is exactly one format to read and no
dispatch to write, which is why the objection does not survive contact with the
actual endpoint.

Everything else ADR-0047 decided stands unchanged and is load-bearing here:
one validation pass shared with uploads, `jsonl_chunks` producing bytes
comparable to an upload's, the size ceiling enforced mid-stream from
`config.MAX_DATASET_BYTES`, coded failures, and the double the journeys run
against. Only its clause 2 — "the `/rows` endpoint serves one bounded page at
a time" — is superseded, and only as to the mechanism. The property that clause
existed to guarantee is the one this record spends the most words keeping.

## Alternatives considered

**Keep `/rows` and back off harder.** Rejected. It treats a rate limit as a
timing problem when it is a capacity one: the endpoint is generated per
request, has no published limit, and needs one call per hundred rows however
politely they are spaced. Backing off further makes a 25,000-row import slower
without making it more likely to finish.

**Keep `/rows` and cache pages / parallelise them.** Rejected. Parallelising
requests against the endpoint that throttles is the fastest way to be throttled
harder, and a cache does nothing for a dataset imported once.

**HTTP range requests against the Parquet files instead of buffering to
disk.** Genuinely tempting: pyarrow can read from a seekable file object, so a
range-reading adapter would avoid the temporary file. Rejected for now — it
means implementing seek semantics over HTTP, the round trips multiply for a
read that ends up fetching the whole file anyway, and the failure modes are
ours to debug rather than the kernel's. The cost of the decision is temporary
disk for one shard, and it is stated in the consequences.

**The `huggingface_hub` or `datasets` library.** Rejected. `datasets` would
pull a large dependency tree to do what one listing call and pyarrow already
do, and its caching and Arrow-backed loading are the opposite of what this path
needs — it wants to materialise a dataset, and the whole guarantee here is that
nothing does. This seam's contract is bytes, not a Dataset object.

**Refuse an import when no Parquet conversion exists.** Rejected. The
conversion is HF-side and not instant, so this would refuse a valid public
dataset because of when it happened to be published. The slower path still
works; it is just no longer the default.

**Fall back to `/rows` when the listing call itself fails.** Rejected, and
narrowly. It would make an import survive a listing blip, but it silently sends
traffic to the throttled endpoint on any error from the cached one, which is
exactly the behaviour that would hide this defect returning.

## Consequences

- A 25,000-row import is one listing call plus one download per shard, instead
  of ~250 calls against an undocumented rate limit. The traffic moves from a
  bucket nobody publishes a number for into the published Resolvers bucket
  (5,000 per 5 minutes).
- An import needs temporary disk for one shard, so the process needs a writable
  temp directory. Cleanup is in a `finally` and is pinned by a test, including
  the abandoned-stream case, because on Windows an open handle makes the delete
  fail outright — which the corrupt-shard test caught during this work.
- `pyarrow` is a large wheel (~40 MB) in the control-plane image. One import
  path is a thin justification for that weight; reading Parquet correctly by
  hand is not a thing anyone should do.
- A repository so large that even its per-config listing exceeds
  datasets-server's 10 MB response ceiling gets a coded `fetch_failed` rather
  than the `/rows` fallback. Checked rather than waved at: `HuggingFaceFW/fineweb`
  is the case, its `/parquet` answers 501, and its `/rows` answers 501 as well,
  so nothing that used to import stops importing. A dataset of that size is
  orders of magnitude past `config.MAX_DATASET_BYTES` anyway.
- Serialised chunks now span shards and pages instead of being flushed once per
  page, so chunks are uniformly `_CHUNK_BYTES` rather than one short chunk per
  100 rows. Downstream is indifferent — it consumes bounded chunks — but two
  tests that assumed page-sized chunks were adapted, and that is the honest
  reason they changed.
- `partial: true` in the listing means HF converted only the first slice of a
  very large dataset. The import inherits that boundary. Understood from HF's
  docs rather than measured here: the viewer's `/rows` is served from the same
  conversion, so this is not believed to be a new truncation. If that turns out
  to be wrong, it is a truncation worth surfacing to the user and a follow-up.
- The `/rows` code path is still live, still tested, and now runs only for
  unconverted datasets. It is a fallback with a reason, not dead code.

## Rollback

Revert this issue's commits. `_HuggingFaceSource._preview_rows` is the original
mechanism unchanged, so restoring it means having `stream()` call it
unconditionally and dropping `_parquet_rows`, `_parquet_batches`,
`_parquet_files_for`, `_download_parquet` and the `pyarrow` dependency. The
`_fetch`/`_authorized` split and the `json.dumps` `default=` hook are
independently useful and can stay. Resolution, the coded errors and everything
below the seam were never touched.
