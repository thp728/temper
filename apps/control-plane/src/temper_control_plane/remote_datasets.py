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
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

_TIMEOUT_S = 10

# The Hugging Face datasets-server. It serves public datasets by reference --
# which configs and splits exist, and the rows themselves, one bounded page at
# a time -- without requiring a credential. Its `/rows` endpoint caps a page
# at 100 rows, which is exactly the bound the flat-memory guarantee needs.
_DS_API = "https://datasets-server.huggingface.co"
_ROW_PAGE = 100

# How many serialised rows are held before they are yielded as one chunk.
# Bounded like every other chunk size here: the size of the chunk is not the
# point, the point is that nothing scales with the dataset.
_CHUNK_BYTES = 64 * 1024


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


def jsonl_chunks(
    rows: Iterable[dict[str, Any]], chunk_bytes: int = _CHUNK_BYTES
) -> Iterator[bytes]:
    """Serialise row dicts to JSONL bytes in bounded chunks.

    The one place a remote dataset becomes the same bytes an upload would have
    carried. `ensure_ascii=False` keeps non-ASCII text as UTF-8 rather than
    escaping it, which is what a user's own file would have contained -- the
    validator's line-numbered errors and thinking-mode detection then read an
    import and an upload identically, byte for byte.
    """
    buffer = bytearray()
    for row in rows:
        buffer += (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        if len(buffer) >= chunk_bytes:
            yield bytes(buffer)
            buffer.clear()
    if buffer:
        yield bytes(buffer)


def _api(path: str) -> dict[str, Any] | None:
    """One datasets-server request, answered as JSON.

    Returns None only for a 404 -- the resource does not exist. Any other
    failure, including a response body that carries the server's own `error`
    field, raises a coded `RemoteDatasetError` naming the reason, because
    absence and failure are different answers to a user and the two must not
    be conflated (the same distinction the model-facts resolver draws).
    """
    url = _DS_API + path
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        reason = "datasets-server refused the request"
        try:
            detail = json.loads(e.read())
            if isinstance(detail, dict) and detail.get("error"):
                reason = str(detail["error"])
        except Exception:  # noqa: S110 - an unparseable error body keeps its default
            pass
        raise RemoteDatasetError("fetch_failed", reason) from None
    except (TimeoutError, urllib.error.URLError) as e:
        raise RemoteDatasetError(
            "fetch_failed", f"could not reach the dataset server: {e}"
        ) from None
    if isinstance(body, dict) and body.get("error"):
        raise RemoteDatasetError("fetch_failed", str(body["error"]))
    return body


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

        # 1. Does the repository exist and is it public at all?
        validity = _api(f"/is-valid?dataset={q(repo)}") or {}
        if not validity.get("valid"):
            reason = validity.get("reason") or (
                f"the repository '{repo}' does not exist or is not public"
            )
            raise RemoteDatasetError("dataset_not_found", str(reason))

        # 2. Which configuration? A bare repo with several is refused with
        #    the list, because guessing which subset a user meant is exactly
        #    the shortcut this feature exists to avoid.
        catalogue = _api(f"/splits?dataset={q(repo)}")
        configs = _configs_of(catalogue)
        if not configs:
            raise RemoteDatasetError(
                "dataset_not_found",
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

        # 3. Which split? A bare split defaults to 'default' then 'train',
        #    the conventions every dataset uses; anything else must be named.
        splits = _splits_of(catalogue, config)
        if split is None:
            split = next(
                (s for s in ("default", "train") if s in splits),
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
        #    here with that reason rather than importing nothing.
        size_body = _api(
            f"/size?dataset={q(repo)}&config={q(config)}&split={q(split)}"
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
    """A resolved reference backed by the datasets-server `/rows` endpoint.

    Rows are read one bounded page (100 rows, the server's cap) at a time and
    serialised through `jsonl_chunks`, so peak memory stays flat however many
    rows the split holds -- the same guarantee the streaming validator keeps
    for an upload, and the reason the fetch never materialises the dataset."""

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
        offset = 0
        while True:
            body = _api(
                f"/rows?dataset={q(self.repo)}&config={q(self.config)}"
                f"&split={q(self.split)}&offset={offset}&length={_ROW_PAGE}"
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
            yield from jsonl_chunks(rows)
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
