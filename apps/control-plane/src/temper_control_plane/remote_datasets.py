"""The remote-dataset seam: resolves a public repository reference to a
stream of the same bytes an upload would have carried.

Issue #45. A user should not have to prepare a file to start: give a public
dataset repository -- optionally a configuration and a split -- and Temper
pulls the rows down, stores them, and validates them through exactly the
streaming path an upload uses. This seam is how a remote repository becomes
those bytes. Imported rows go through the identical validation (`validation.
validate_chunks` consumes whatever chunks it is given; the fetch is this
module's problem and nothing below it knows or cares), so nothing gets a
shortcut for arriving over a network.

It lives in the control plane for the same reason the model-facts seam does
(`models.py`): resolving an arbitrary public dataset means reading it over the
network, and `temper_core` carries zero I/O. The shape both sides agree on --
what a resolved reference looks like, and what a fetch failure is -- is
defined here, and the double in `fake_remote_datasets.py` supplies the canned
references the test suite and the browser journeys run against, the same way
`fake_provider.py` keeps every journey off real hardware.

Fetch failures are `RemoteDatasetError`: a stable code and a reason the import
endpoint turns into a coded refusal. A repository that cannot be fetched and a
split that resolves to nothing are answered up front with their reason, before
anything is stored -- the failure mode this issue exists to prevent is a
second, subtly different validation path for imports, and the other failure
mode is a fetch that dies mid-stream with nobody told why.

Rows arrive as **Parquet files, not as `/rows` pages** -- see ADR-0073,
`docs/adr/0073-a-bulk-import-reads-parquet-not-the-preview-endpoint.md`.
`/rows` is the dataset viewer's preview endpoint -- capped at 100 rows per
request, generated on demand rather than served from HF's precomputed cache,
and rate limited far more tightly than anything else here. Importing a
25,000-row split through it means ~250 sequential calls, which is how this
seam earned a 429 while `/is-valid`, `/splits` and huggingface.co all answered
200 at the same moment. `/parquet` lists the same split's files, the files
live behind a `/resolve/` URL in the far roomier resolver bucket, and one
request per shard replaces hundreds per split. The `/rows` path is kept as the
fallback for a dataset HF has not converted yet.
"""

from __future__ import annotations

import base64
import concurrent.futures
import contextlib
import datetime
import decimal
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import pyarrow.parquet as pq

from . import config

_TIMEOUT_S = 10

# `urlopen(..., timeout=X)` bounds each individual blocking socket operation
# (connect, one `recv`) -- not the wall-clock time of the call as a whole. A
# server that trickles its response out in slow pieces, each arriving just
# under X, can make one `urlopen()` call run for many multiples of X: this is
# how a request measured at 33s got past a stated 20s budget (issue "dataset
# import from HF"). The fix is a deadline this process itself enforces:
# the blocking call runs on a worker thread, and `future.result(timeout=...)`
# is what actually gives up at the wall-clock deadline, regardless of what
# the socket is doing. The worker thread is not killable -- Python has no
# way to interrupt a blocking socket call from outside it -- so it is left to
# finish or hit its own `urlopen` timeout on its own; the caller does not
# wait for it either way.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="hf-fetch"
)

# `resolve()` makes three sequential calls (`/is-valid`, `/splits`, `/size`)
# before the import endpoint can answer at all -- issue "dataset import from
# HF" surfaced that on a slow day, three independently-timed-out calls at
# `_TIMEOUT_S` each summed past whatever sits in front of this process (the
# Next.js dev proxy locally; presumably something with its own ceiling in
# front of it in any real deployment), so the caller got that layer's own
# opaque 500 instead of the coded refusal this module had actually produced a
# few seconds later. One shared budget across the whole resolve() bounds the
# worst case to a single, known number instead of the sum of however many
# calls resolution happens to need. A judgment call, not a measurement: it
# must clear one real round trip per call plus HF's own occasional slowness,
# and stay well under the proxy timeouts observed locally.
_RESOLVE_BUDGET_S = 20

# The Hugging Face datasets-server. It serves public datasets by reference --
# which configs and splits exist, where the Parquet conversion of each split
# lives, and (for a preview) the rows themselves one bounded page at a time --
# without requiring a credential. Its `/rows` endpoint caps a page at 100
# rows, which is why it is a preview API and not a bulk one.
_DS_API = "https://datasets-server.huggingface.co"
_ROW_PAGE = 100

# How many rows one Parquet batch carries. The bound that replaces `/rows`'s
# page: `iter_batches` hands back this many rows at a time, so the amount of
# decoded data alive at once is a property of this number and the row width,
# never of how many rows the split holds. Same order as the page it replaces,
# for the same reason -- small enough to be nothing, large enough that the
# per-batch overhead disappears.
_PARQUET_BATCH_ROWS = 100

# How much of a Parquet download is read from the socket at a time. Larger
# than `_CHUNK_BYTES` because nothing downstream holds one of these: it is a
# raw drain into a file, and a bigger read means fewer syscalls per shard.
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# A Parquet shard is one file, not a page, so its transfer time scales with
# its size and a single fixed timeout would either strangle a large shard or
# let a stalled connection sit for the same generous span a large one needs.
# The listing declares each file's `size`, so the budget is derived from it
# against a deliberately pessimistic throughput floor plus a fixed grace for
# connection setup. Judgment, not measurement: the floor is set far below any
# working connection so that only a genuinely stalled transfer reaches it.
_DOWNLOAD_GRACE_S = 30.0
_MIN_DOWNLOAD_BYTES_PER_S = 256 * 1024
# What a shard is assumed to weigh when the listing does not say. HF's
# converter shards at roughly 500 MB, so this is the same floor applied to
# the largest file the conversion is expected to produce.
_ASSUMED_SHARD_BYTES = 500 * 1024 * 1024

# How many serialised rows are held before they are yielded as one chunk.
# Bounded like every other chunk size here: the size of the chunk is not the
# point, the point is that nothing scales with the dataset.
_CHUNK_BYTES = 64 * 1024

# The conventions a bare reference falls back to -- one configuration defaults
# to 'default', a bare split to 'default' then 'train'. Defined once here and
# read by the fake resolver too, so the real resolver and its double agree on
# what an unnamed reference resolves to ("a value two components must agree on
# is defined once and read, never retyped").
DEFAULT_CONFIG = "default"
DEFAULT_SPLITS = ("default", "train")


class RemoteDatasetError(Exception):
    """A remote reference that cannot be honoured, with the reason.

    `code` is the stable, machine-readable name a refusal wears (the same
    contract every API refusal honours); `message` is what a user reads.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_payload(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class _ResolvedSource:
    """The shared half of every resolved reference: what it is, and how it is
    named. The two implementations differ only in where their rows come from,
    so the naming lives once here rather than drifting apart."""

    repo: str
    config: str | None
    split: str | None

    def display_name(self) -> str:
        """The name the imported dataset goes by: the reference, including the
        configuration and split that were actually fetched. A user who asked
        for a bare repo sees which config and split were defaulted to; a user
        who named both sees exactly what they asked for. Ends in `.jsonl`
        because that is what gets stored."""
        parts = [self.repo]
        if self.config:
            parts.append(self.config)
        if self.split:
            parts.append(self.split)
        return "/".join(parts) + ".jsonl"


class RemoteDatasetSource(Protocol):
    """A resolved reference: the provenance, and a stream of JSONL bytes."""

    @property
    def repo(self) -> str: ...

    @property
    def config(self) -> str | None: ...

    @property
    def split(self) -> str | None: ...

    def stream(self) -> Iterator[bytes]:
        """The dataset's rows as JSONL bytes, in bounded chunks.

        Raises `RemoteDatasetError` with a coded reason if the fetch fails
        part-way. The stream never holds the dataset whole: at most one page
        of rows and one chunk of bytes exist at a time.
        """
        ...

    def display_name(self) -> str: ...


class RemoteDatasets(Protocol):
    """What every resolver promises. This protocol *is* the seam."""

    def resolve(
        self,
        repo: str,
        config: str | None = None,
        split: str | None = None,
    ) -> RemoteDatasetSource:
        """Resolve a reference to a fetchable source, or raise
        `RemoteDatasetError` with the reason -- a repository that cannot be
        fetched, a configuration that must be named, a split that does not
        exist, or a split that resolves to nothing. Never guesses."""
        ...


def _json_default(value: Any) -> Any:
    """The JSON form of a value Parquet can hold and JSON cannot.

    `/rows` handed back values HF had already encoded as JSON; Parquet hands
    back typed ones, so a timestamp or a binary column would otherwise raise
    `TypeError` from `json.dumps` mid-import and break the seam's promise that
    every failure is a coded reason. Passed as `json.dumps(default=...)`, so it
    is called only for the values the encoder cannot handle itself: the common
    path (strings, numbers, lists, dicts) never reaches here.

    Timestamps become ISO-8601, which is HF's own encoding of them. The rest is
    this seam's choice rather than a claim of byte parity with `/rows`: a
    dataset whose columns are decimals or raw bytes is not fine-tuning text,
    and the validator refuses it with a line number either way. Producing
    *something* is what keeps that refusal line-numbered instead of a 500.
    """
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, bytes | bytearray):
        return base64.b64encode(value).decode("ascii")
    return str(value)


def jsonl_chunks(
    rows: Iterable[dict[str, Any]], chunk_bytes: int = _CHUNK_BYTES
) -> Iterator[bytes]:
    """Serialise row dicts to JSONL bytes in bounded chunks.

    The one place a remote dataset becomes the same bytes an upload would have
    carried -- both fetch mechanisms end here, so a Parquet import and a
    `/rows` import cannot drift into two encodings. `ensure_ascii=False` keeps
    non-ASCII text as UTF-8 rather than escaping it, which is what a user's own
    file would have contained -- the validator's line-numbered errors and
    thinking-mode detection then read an import and an upload identically, byte
    for byte.
    """
    buffer = bytearray()
    for row in rows:
        buffer += (
            json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
        ).encode("utf-8")
        if len(buffer) >= chunk_bytes:
            yield bytes(buffer)
            buffer.clear()
    if buffer:
        yield bytes(buffer)


# The header HF's own rate limiter sends on a 429 is `Retry-After`, a plain
# count of seconds (RFC 9110) -- not the `RateLimit` header the resolver
# (file-download) endpoints send, which is a different service. When it is
# absent, a 429 still means "try again shortly", so a fixed fallback stands
# in rather than treating a missing header as a permanent refusal.
_DEFAULT_RETRY_AFTER_S = 2.0

# Live testing (issue: "dataset import from HF") found datasets-server 429s
# are short, self-resolving blips, not a standing block -- 41 back-to-back
# `/rows` calls for the exact dataset and page that had just failed all came
# back 200 minutes later. A single retry can still lose to one: two
# consecutive unlucky attempts is enough to fail an import outright. More
# attempts trade a higher worst case on a *genuinely* persistent failure
# (still bounded by `deadline` either way) for a much better chance of
# riding out a blip that clears in a few seconds.
_MAX_429_RETRIES = 3


def _retry_after_s(e: urllib.error.HTTPError) -> float:
    raw = e.headers.get("Retry-After") if e.headers else None
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return _DEFAULT_RETRY_AFTER_S


def _authorized(url: str) -> str | urllib.request.Request:
    """The request one URL is made as: a bare string, or a `Request` carrying
    the bearer token when one is configured.

    Every endpoint this module reads answers public data with no credential at
    all, but an anonymous request shares a low, IP-based rate limit where an
    authenticated one goes against the token owner's own, higher quota --
    `config.HF_TOKEN` unset (the default) behaves exactly as before it
    existed. Gated repositories need it outright. Defined once and read by the
    JSON calls and the Parquet downloads alike, because a header that only
    some of the requests in one import carry is a rate limit nobody can
    reason about. Both the double the test suite substitutes for `urlopen` and
    the real one accept either shape, so the bare string is kept for the
    no-token case rather than always wrapping.
    """
    if config.HF_TOKEN:
        # The scheme is not caller-controlled: every URL reaching here is
        # either built from `_DS_API` or read out of a listing served by it.
        return urllib.request.Request(  # noqa: S310
            url, headers={"Authorization": f"Bearer {config.HF_TOKEN}"}
        )
    return url


def _perform(
    target: str | urllib.request.Request,
    timeout: float,
    consume: Callable[[Any], Any],
) -> Any:
    """The blocking half of one request, run on `_EXECUTOR`'s worker thread so
    the caller can enforce a wall-clock deadline `urlopen`'s own `timeout`
    cannot (see `_EXECUTOR`). `consume` reads the response *on that thread* --
    a JSON body whole, or a Parquet file drained to disk a block at a time.
    Raises exactly what `urlopen` and `consume` would; `_fetch` owns turning
    those into a coded `RemoteDatasetError`."""
    with urllib.request.urlopen(target, timeout=timeout) as resp:  # noqa: S310
        return consume(resp)


# What `_fetch` hands back for a literal 404, kept distinct from `None`
# because a download's `consume` legitimately returns `None` on success and
# "the resource does not exist" is a different answer from "it is empty".
_MISSING = object()


def _fetch(url: str, *, deadline: float, consume: Callable[[Any], Any]) -> Any:
    """One Hugging Face request, under a wall-clock deadline and the 429
    policy, with every failure coded.

    Returns `_MISSING` only for a 404 -- the resource does not exist. Any
    other failure raises a coded `RemoteDatasetError` naming the reason,
    because absence and failure are different answers to a user and the two
    must not be conflated (the same distinction the model-facts resolver
    draws).

    The `except Exception` is deliberately broad: a fetch can fail at any
    point -- DNS, a dropped connection mid-read, a body that is not JSON, a
    Parquet file that will not parse -- and the seam's contract is that
    *every* fetch failure is a coded reason, never an unhandled exception
    that becomes a 500 mid-import.

    A token raises the rate-limit ceiling; it does not remove it, so a 429 is
    still a real, expected outcome and is retried (up to `_MAX_429_RETRIES`
    times) rather than treated as a hard refusal on first sight -- HF's own
    guidance, and what live testing here found, is that it is usually a
    short blip, not a permanent no. A retry re-runs `consume` from the start,
    which is why the download's `consume` reopens its file for writing rather
    than appending: half a shard from the attempt that 429'd must not survive
    into the one that succeeds.

    `deadline` is an absolute `time.monotonic()` value shared across every
    call one caller makes (see `_RESOLVE_BUDGET_S`), not a fixed per-call
    timeout: a caller with no time left gets the coded refusal immediately,
    rather than making a request that cannot matter, and a 429's retry only
    happens if the wait fits inside what is left.
    """
    target = _authorized(url)
    retries = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RemoteDatasetError(
                "fetch_failed", "the dataset server did not respond in time"
            )
        try:
            future = _EXECUTOR.submit(_perform, target, remaining, consume)
            return future.result(timeout=remaining)
        except concurrent.futures.TimeoutError:
            # The wall-clock deadline, not the socket's own timeout -- the
            # worker thread is left running; see `_EXECUTOR`'s docstring.
            raise RemoteDatasetError(
                "fetch_failed", "the dataset server did not respond in time"
            ) from None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return _MISSING
            if e.code == 429 and retries < _MAX_429_RETRIES:
                wait = _retry_after_s(e)
                if wait < deadline - time.monotonic():
                    retries += 1
                    time.sleep(wait)
                    continue
            reason = f"Hugging Face returned {e.code} {e.reason}"
            try:
                detail = json.loads(e.read())
                if isinstance(detail, dict) and detail.get("error"):
                    reason = str(detail["error"])
            except Exception:  # noqa: S110 - an unparseable error body keeps its default
                pass
            raise RemoteDatasetError("fetch_failed", reason) from None
        except RemoteDatasetError:
            # Already coded by `consume` -- a download that ran past its own
            # deadline names that, and re-wrapping would bury the reason.
            raise
        except Exception as e:  # noqa: BLE001 - every fetch failure is a coded reason
            raise RemoteDatasetError(
                "fetch_failed", f"could not fetch from the dataset server: {e}"
            ) from None


def _api(path: str, *, deadline: float) -> dict[str, Any] | None:
    """One datasets-server request, answered as JSON.

    Returns None only for a 404. A response body that carries the server's own
    `error` field is a failure, not an answer, and is raised as one -- HF
    reports several refusals that way with a 200 status.
    """
    raw = _fetch(
        _DS_API + path, deadline=deadline, consume=lambda resp: resp.read()
    )
    if raw is _MISSING:
        return None
    try:
        body = json.loads(raw)
    except Exception as e:  # noqa: BLE001 - every fetch failure is a coded reason
        raise RemoteDatasetError(
            "fetch_failed", f"could not fetch from the dataset server: {e}"
        ) from None
    if isinstance(body, dict) and body.get("error"):
        raise RemoteDatasetError("fetch_failed", str(body["error"]))
    return body


def _download_budget_s(size: int | None) -> float:
    """The wall-clock budget one shard's download gets, from its declared
    size (see `_DOWNLOAD_GRACE_S`)."""
    weight = (
        size if isinstance(size, int) and size > 0 else _ASSUMED_SHARD_BYTES
    )
    return _DOWNLOAD_GRACE_S + weight / _MIN_DOWNLOAD_BYTES_PER_S


def _download_parquet(url: str, path: str, *, size: int | None) -> None:
    """Drain one Parquet shard to `path`, in bounded blocks, under a budget
    derived from its declared size.

    pyarrow needs a seekable file to read a Parquet footer, and an HTTP
    response is not one, so the shard is buffered to disk. To *disk* is the
    point: peak memory stays one block regardless of how large the shard is,
    which is the guarantee the whole import path is built on. The caller owns
    the temporary file and deletes it.

    The worker thread checks the deadline between blocks so an abandoned
    transfer stops itself shortly after the caller has given up on it, rather
    than continuing to write hundreds of megabytes into a file nobody will
    read (the thread cannot be killed from outside -- see `_EXECUTOR`).
    """
    deadline = time.monotonic() + _download_budget_s(size)

    def drain(resp: Any) -> None:
        with open(path, "wb") as fh:
            while True:
                if time.monotonic() >= deadline:
                    raise RemoteDatasetError(
                        "fetch_failed",
                        "the dataset server did not respond in time",
                    )
                block = resp.read(_DOWNLOAD_CHUNK_BYTES)
                if not block:
                    return
                fh.write(block)

    if _fetch(url, deadline=deadline, consume=drain) is _MISSING:
        raise RemoteDatasetError(
            "fetch_failed",
            f"the parquet file '{url.rsplit('/', 1)[-1]}' is no longer published",
        )


def _configs_of(splits_body: dict[str, Any] | None) -> list[str]:
    """The distinct configurations a `/splits` response exposes, in order."""
    seen: list[str] = []
    for entry in (splits_body or {}).get("splits") or []:
        config = entry.get("config")
        if config and config not in seen:
            seen.append(config)
    return seen


def _splits_of(splits_body: dict[str, Any] | None, config: str) -> list[str]:
    """The splits one configuration exposes, in the server's order."""
    return [
        entry.get("split")
        for entry in (splits_body or {}).get("splits") or []
        if entry.get("config") == config and entry.get("split")
    ]


def _parquet_batches(path: str) -> Iterator[dict[str, Any]]:
    """The rows of one downloaded Parquet shard, a bounded batch at a time.

    Never `read_table` or `to_pandas`: both materialise the file, and flat
    memory is the guarantee the whole ingest path rests on (see the module
    docstring in `temper_core.validation` and ADR-0036). `iter_batches` hands
    back `_PARQUET_BATCH_ROWS` rows at a time; `to_pylist` turns one batch
    into plain dicts, which is what `jsonl_chunks` already knows how to
    serialise -- the same rows, in the same shape, that `/rows` used to hand
    back.

    **`pre_buffer=False` is what makes that true, and it is not the default.**
    Measured here (pyarrow 25.0.1, a 32 MB shard of 1,000,000 chat rows):
    `pq.ParquetFile(path)` peaks at 32.3 MB of Arrow allocation -- it reads
    the whole file up front -- and the same iteration with `pre_buffer=False`
    peaks at 0.99 MB. The batching alone does not bound anything; without
    this flag a shard is materialised whole, which is exactly the defect this
    module claims not to have. `test_parquet_memory_stays_flat_as_a_shard_grows`
    is what keeps it from being flipped back.

    A malformed or truncated file is a coded reason like any other fetch
    failure, because by the time it is read the import has already told the
    caller its dataset id exists.
    """
    try:
        # The file is opened here, not by pyarrow from the path, so that this
        # frame owns the handle and closes it deterministically -- including
        # when the constructor itself rejects the file. An open handle makes
        # the caller's delete fail outright on Windows, which leaks a whole
        # shard per failed import; the corrupt-shard test caught exactly that.
        with open(path, "rb") as fh:
            handle = pq.ParquetFile(fh, pre_buffer=False)
            try:
                for batch in handle.iter_batches(
                    batch_size=_PARQUET_BATCH_ROWS
                ):
                    yield from batch.to_pylist()
            finally:
                handle.close()
    except Exception as e:  # noqa: BLE001 - every fetch failure is a coded reason
        raise RemoteDatasetError(
            "fetch_failed", f"could not read the dataset's parquet file: {e}"
        ) from None


def _shard_order(entry: dict[str, Any]) -> str:
    """A sort key putting one split's shards in the order the dataset is in.

    The whole URL path, with every run of digits widened to a fixed width so
    it sorts by value: plain lexicographic order puts `train-part10` before
    `train-part2`, and the part number is the outer index. A padded string
    rather than a list of mixed ints and strings, because two paths of
    different shapes would make that list comparison raise `TypeError` from
    inside a `sorted()` -- an unhandled exception in a module whose contract
    is that every failure is a coded reason.
    """
    path = urllib.parse.urlparse(str(entry.get("url"))).path
    return re.sub(r"\d+", lambda m: m.group().zfill(12), path)


def _parquet_files_for(
    listing: dict[str, Any] | None, config: str, split: str
) -> list[dict[str, Any]]:
    """The Parquet shards one (config, split) is published as, in shard order.

    Returns an empty list when the conversion has not produced this split --
    an absent listing, one with no files for it, or one that reports it as
    still pending or as failed. That is the caller's signal to fall back to
    `/rows`: HF converts a dataset in the background, so a repository
    published minutes ago genuinely has no Parquet yet, and refusing the
    import for it would be refusing something the slower path can do.

    Sorted by `_shard_order`, not by `filename` and not in listing order.
    Both of those are wrong, measured against the live API on 2026-09-03: a
    large split is sharded into *parts*
    (`default/train-part0/0000.parquet`, `train-part1/0000.parquet`, ...) and
    every part restarts its numbering, so `filename` collides -- 27,468 files
    of `HuggingFaceFW/fineweb` carry only 10,000 distinct filenames, each
    repeated once per part. The listing's own order interleaves the parts
    (`part0/0000`, `part1/0000`, ...) rather than running through them. Either
    key would import the rows in an order no version of the dataset is in.
    """
    body = listing or {}
    for state in ("pending", "failed"):
        for entry in body.get(state) or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("config") == config and entry.get("split") == split:
                return []
    files = [
        f
        for f in body.get("parquet_files") or []
        if isinstance(f, dict)
        and f.get("config") == config
        and f.get("split") == split
        and f.get("url")
    ]
    return sorted(files, key=_shard_order)


class HuggingFaceDatasets:
    """Resolves public dataset references against the Hugging Face
    datasets-server. Requires no credential: every response it reads is
    public. Resolution is three reads -- validity, the configuration/split
    catalogue, and the split's size -- each answered before anything is
    stored, so a bad reference is refused with its reason rather than dying
    mid-fetch."""

    def resolve(
        self,
        repo: str,
        config: str | None = None,
        split: str | None = None,
    ) -> RemoteDatasetSource:
        q = urllib.parse.quote

        # One deadline shared across every call this method makes, not one
        # timeout per call (see `_RESOLVE_BUDGET_S`): a caller in front of
        # this process has its own ceiling, and the sum of several
        # independently-timed-out calls (or retries) can clear it even
        # though each one on its own looked bounded. `_api` checks it before
        # every request, including a 429's retry, and raises the coded
        # refusal itself once it is gone.
        deadline = time.monotonic() + _RESOLVE_BUDGET_S

        # 1. Does the repository exist and is it public at all? `_api`
        #    already turns an error-carrying body or a non-404 HTTP failure
        #    into a `fetch_failed`, so the only thing left to check here is
        #    the true-404 case it hands back as `None`. (HF's `/is-valid`
        #    dropped the `{"valid": bool}` shape this used to check --
        #    a success now reads as per-capability flags with no `valid`
        #    key at all, which made every real repository read as invalid.)
        if _api(f"/is-valid?dataset={q(repo)}", deadline=deadline) is None:
            raise RemoteDatasetError(
                "repo_not_found",
                f"the repository '{repo}' does not exist or is not public",
            )

        # 2. Which configuration? A bare repo with several is refused with
        #    the list, because guessing which subset a user meant is exactly
        #    the shortcut this feature exists to avoid.
        catalogue = _api(f"/splits?dataset={q(repo)}", deadline=deadline)
        configs = _configs_of(catalogue)
        if not configs:
            raise RemoteDatasetError(
                "repo_not_found",
                f"'{repo}' exposes no configurations to import.",
            )
        if config is None:
            if len(configs) > 1:
                raise RemoteDatasetError(
                    "config_required",
                    f"'{repo}' has several configurations "
                    f"({', '.join(configs)}); name the one to import.",
                )
            config = configs[0]
        elif config not in configs:
            raise RemoteDatasetError(
                "config_not_found",
                f"'{repo}' has no configuration '{config}'; "
                f"available: {', '.join(configs)}.",
            )

        # 3. Which split? A bare split defaults to the conventions every
        #    dataset uses (DEFAULT_SPLITS); anything else must be named.
        splits = _splits_of(catalogue, config)
        if split is None:
            split = next(
                (s for s in DEFAULT_SPLITS if s in splits),
                splits[0] if splits else None,
            )
        if not splits:
            raise RemoteDatasetError(
                "split_not_found",
                f"'{repo}' configuration '{config}' exposes no splits.",
            )
        if split not in splits:
            raise RemoteDatasetError(
                "split_not_found",
                f"'{repo}' configuration '{config}' has no split '{split}'; "
                f"available: {', '.join(splits)}.",
            )

        # 4. Does the split resolve to anything? An empty split is refused
        #    here with that reason when the server reports it; the source also
        #    refuses if a stream turns out to carry no rows, so the criterion
        #    holds even when `/size` reports nothing.
        size_body = _api(
            f"/size?dataset={q(repo)}&config={q(config)}&split={q(split)}",
            deadline=deadline,
        )
        info = (size_body or {}).get("size") or {}
        num_rows = info.get("num_rows")
        if num_rows == 0:
            raise RemoteDatasetError(
                "split_empty",
                f"Split '{split}' of '{repo}' resolves to no rows; "
                f"there is nothing to import.",
            )

        return _HuggingFaceSource(
            repo,
            config,
            split,
            num_rows=num_rows,
            num_bytes=info.get("num_bytes"),
        )


class _HuggingFaceSource(_ResolvedSource):
    """A resolved reference backed by the split's Parquet files, falling back
    to the `/rows` preview endpoint when it has none.

    Either way rows are read a bounded batch at a time and serialised through
    `jsonl_chunks`, so peak memory stays flat however many rows the split holds
    -- the same guarantee the streaming validator keeps for an upload, and the
    reason the fetch never materialises the dataset."""

    def __init__(
        self,
        repo: str,
        config: str | None,
        split: str | None,
        num_rows: int | None,
        num_bytes: int | None,
    ):
        super().__init__(repo, config, split)
        self._num_rows = num_rows
        self._num_bytes = num_bytes

    def stream(self) -> Iterator[bytes]:
        q = urllib.parse.quote
        if self.config is None or self.split is None:
            # Unreachable through resolve(), which always pins both; kept as a
            # named failure rather than a None formatting into a URL.
            raise RemoteDatasetError(
                "fetch_failed", "reference was not fully resolved"
            )

        # Which mechanism is decided here and nowhere else, and it is decided
        # once per stream. The listing is a precomputed, cached response
        # (ADR-0073) -- one call, and a failure of it is a real failure, not a
        # licence to fall back onto the endpoint that throttles.
        #
        # Asked for by configuration, because the unfiltered listing covers
        # the whole repository and datasets-server refuses to compute a
        # response over 10 MB: `?dataset=HuggingFaceFW/fineweb` alone answers
        # 501, and the same request with `&config=default` answers 200 with
        # 27,468 files (measured 2026-09-03). `&split=` is *not* honoured --
        # `?dataset=nyu-mll/glue&config=cola&split=train` came back with all
        # three of cola's splits -- so it is not sent, and the split filter
        # stays where it actually happens, in `_parquet_files_for`.
        listing = _api(
            f"/parquet?dataset={q(self.repo)}&config={q(self.config)}",
            deadline=time.monotonic() + _TIMEOUT_S,
        )
        files = _parquet_files_for(listing, self.config, self.split)

        # The fallback, deliberately narrow: only a listing that *answered*
        # and had nothing for this split takes it. HF's Parquet conversion
        # runs in the background, so a dataset published minutes ago has none
        # yet, and that must not be the difference between an import working
        # and not.
        rows = self._parquet_rows(files) if files else self._preview_rows()

        yielded = False
        for chunk in jsonl_chunks(rows):
            yielded = True
            yield chunk
        if not yielded:
            # The `/size` refusal in resolve is the fast path; this is the
            # ground truth. A split that streams no rows is refused with the
            # same reason even when the size endpoint reported nothing.
            raise RemoteDatasetError(
                "split_empty",
                f"Split '{self.split}' of '{self.repo}' resolves to no rows; "
                f"there is nothing to import.",
            )

    def _parquet_rows(
        self, files: list[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """Every row of the split's Parquet shards, in shard then row order.

        One shard is downloaded at a time and deleted before the next starts,
        so the temporary disk this costs is one file rather than the split.
        `iter_batches` decodes `_PARQUET_BATCH_ROWS` rows at a time from the
        row group it is reading, so what is alive in memory is a batch and the
        row group behind it -- both properties of the file's layout, never of
        how many rows the split holds.
        """
        for entry in files:
            fd, path = tempfile.mkstemp(prefix="temper-hf-", suffix=".parquet")
            os.close(fd)
            try:
                _download_parquet(entry["url"], path, size=entry.get("size"))
                yield from _parquet_batches(path)
            finally:
                # Always, on every exit including an abandoned generator: a
                # shard left behind is disk this process will never reclaim.
                with contextlib.suppress(OSError):
                    os.unlink(path)

    def _preview_rows(self) -> Iterator[dict[str, Any]]:
        """Every row of the split, one `/rows` page at a time.

        The fallback for a split with no Parquet conversion (ADR-0073). It is
        the mechanism this seam shipped with: correct, and rate limited hard
        enough that it is no longer the default.
        """
        q = urllib.parse.quote
        # Narrowing for the type checker only: `stream()` has already refused
        # a reference that is not fully resolved, and it is the sole caller.
        assert self.config is not None and self.split is not None
        offset = 0
        while True:
            body = _api(
                f"/rows?dataset={q(self.repo)}&config={q(self.config)}"
                f"&split={q(self.split)}&offset={offset}&length={_ROW_PAGE}",
                # Each page gets its own budget rather than sharing one across
                # the whole stream: unlike `resolve()`, the number of pages is
                # not bounded, so a fixed total would either be too small for
                # a large split or would cap the split size by proxy.
                deadline=time.monotonic() + _TIMEOUT_S,
            )
            rows: list[dict[str, Any]] = []
            for entry in (body or {}).get("rows") or []:
                if not isinstance(entry, dict):
                    continue
                row = entry.get("row")
                if isinstance(row, dict):
                    rows.append(row)
            if not rows:
                return
            yield from rows
            offset += len(rows)
            if len(rows) < _ROW_PAGE:
                return


def new_remote_datasets() -> RemoteDatasets:
    """The default resolver.

    Real unless `TEMPER_FAKE_PROVIDER` is set. Fetching a public dataset
    costs nothing and risks no billing account, but the browser journeys
    (`apps/web/e2e`) exist to run with no dependency on network reachability
    at all -- the same reason that flag already swaps in the fake provider and
    the fake model-facts resolver. Reusing it here rather than adding a second
    switch keeps "this process cannot reach anything external" a single flag.
    The pytest suite goes further still and replaces `RESOLVER` outright
    (`conftest.no_real_remote_datasets`), so no test even routes through this
    function.
    """
    from . import config

    if config.FAKE_PROVIDER:
        from .fake_remote_datasets import journeys_datasets

        return journeys_datasets()
    return HuggingFaceDatasets()


# The one resolver the import path reads, built once at import -- the
# control-plane equivalent of `storage.STORE`. Tests replace it with a fake.
RESOLVER: RemoteDatasets = new_remote_datasets()
