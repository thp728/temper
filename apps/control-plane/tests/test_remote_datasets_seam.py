"""The remote-dataset seam -- issue #45.

`FakeRemoteDatasets` is exercised the same way `FakeProvider` and `FakeModels`
are: constructed with the references a case needs, refusing anything it was
not given. `HuggingFaceDatasets` is exercised with `urllib.request.urlopen`
monkeypatched locally -- the one place that undoes
`conftest.no_real_remote_datasets`'s guard, because this is the file that owns
proving the resolution and the row fetch are correct.

The Parquet fixtures are built here, in memory, with pyarrow (ADR-0073). No
test reaches the network: the shards a case needs are written to a buffer and
handed back by the same `urlopen` stand-in that answers the JSON endpoints.
"""

from __future__ import annotations

import datetime
import io
import json
import os
import tempfile
import urllib.error
import urllib.request
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from temper_control_plane import remote_datasets
from temper_control_plane.fake_remote_datasets import (
    FakeRemoteDatasets,
    chat_rows,
)
from temper_control_plane.remote_datasets import (
    RemoteDatasetError,
    jsonl_chunks,
)

# --- FakeRemoteDatasets -------------------------------------------------------


def test_fake_resolves_exactly_the_references_it_was_given():
    rows = chat_rows(3)
    fake = FakeRemoteDatasets({("org/repo", None, "train"): rows})
    source = fake.resolve("org/repo", None, "train")
    assert source.display_name() == "org/repo/default/train.jsonl"
    assert b"".join(source.stream()) == b"".join(jsonl_chunks(rows))


def test_fake_refuses_an_unknown_reference_with_the_code_and_reason():
    fake = FakeRemoteDatasets({})
    with pytest.raises(RemoteDatasetError) as exc:
        fake.resolve("org/ghost")
    assert exc.value.code == "repo_not_found"
    assert "org/ghost" in exc.value.message


def test_fake_refuses_an_empty_split_with_that_reason():
    fake = FakeRemoteDatasets({("org/repo", None, "train"): []})
    with pytest.raises(RemoteDatasetError) as exc:
        fake.resolve("org/repo", None, "train")
    assert exc.value.code == "split_empty"
    assert "no rows" in exc.value.message


def test_jsonl_chunks_serialises_rows_the_way_an_upload_would_carry_them():
    rows = chat_rows(2)
    lines = b"".join(jsonl_chunks(rows)).decode("utf-8").splitlines()
    assert [json.loads(line) for line in lines] == rows


# --- HuggingFaceDatasets ------------------------------------------------------


def _url_of(target: str | urllib.request.Request) -> str:
    """The URL a `urlopen` stand-in was called with, whichever shape `_api`
    sent -- a bare string, or a `Request` when a developer's own `.env`
    happens to carry an `HF_TOKEN`. Real `urlopen` accepts both; every fake
    `urlopen` in this file goes through this so which shape arrived does not
    change what the suite proves."""
    return (
        target.full_url
        if isinstance(target, urllib.request.Request)
        else target
    )


def _opener(routes: dict[str, dict], status: int = 200):
    """A stand-in for `urllib.request.urlopen` that answers by URL suffix."""

    def opener(target, timeout=10):
        url = _url_of(target)
        for suffix, body in routes.items():
            if url.endswith(suffix):
                if body is None:
                    import urllib.error

                    raise urllib.error.HTTPError(
                        url, 404, "not found", {}, None
                    )
                resp = Mock()
                resp.read.return_value = json.dumps(body).encode()
                resp.__enter__ = Mock(return_value=resp)
                resp.__exit__ = Mock(return_value=False)
                return resp
        raise AssertionError(f"no mocked route for {url}")

    return opener


def _split_route(repo, config, split, num_rows, num_bytes=1000):
    return {
        "splits": [
            {"dataset": repo, "config": config, "split": s, "num_examples": n}
            for s, n in ((split, num_rows), ("test", 2))
        ]
    }


def test_resolves_a_bare_repo_to_its_single_config_and_default_split(
    monkeypatch,
):
    repo, config, split = "org/repo", "default", "train"
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _opener(
            {
                f"/is-valid?dataset={repo}": {"valid": True},
                f"/splits?dataset={repo}": _split_route(
                    repo, config, split, 5
                ),
                f"/size?dataset={repo}&config={config}&split={split}": {
                    "size": {"num_rows": 5, "num_bytes": 1000}
                },
            }
        ),
    )

    source = remote_datasets.HuggingFaceDatasets().resolve(repo)

    assert source.config == "default"
    assert source.split == "train"
    assert source.display_name() == "org/repo/default/train.jsonl"


def test_resolve_refuses_a_repository_the_server_reports_an_error_for(
    monkeypatch,
):
    # HF's `/is-valid` no longer answers with `{"valid": false, ...}` -- an
    # unfetchable repository comes back as a body carrying an `error` key
    # instead, which `_api` already turns into a `fetch_failed` before
    # `resolve` ever sees it.
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _opener(
            {
                "/is-valid?dataset=org/ghost": {
                    "error": "This dataset does not exist.",
                }
            }
        ),
    )

    with pytest.raises(RemoteDatasetError) as exc:
        remote_datasets.HuggingFaceDatasets().resolve("org/ghost")
    assert exc.value.code == "fetch_failed"
    assert "does not exist" in exc.value.message


def test_resolve_refuses_a_repository_that_404s_outright(monkeypatch):
    # The one case `resolve` still codes itself: `_api` hands back `None`
    # only for a literal 404, which HF's `/is-valid` doesn't currently send
    # for any reference we've observed, but the seam still answers it.
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _opener({"/is-valid?dataset=org/ghost": None}),
    )

    with pytest.raises(RemoteDatasetError) as exc:
        remote_datasets.HuggingFaceDatasets().resolve("org/ghost")
    assert exc.value.code == "repo_not_found"
    assert "org/ghost" in exc.value.message


def test_resolve_refuses_a_bare_repo_with_several_configs(monkeypatch):
    repo = "org/multi"
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _opener(
            {
                f"/is-valid?dataset={repo}": {"valid": True},
                f"/splits?dataset={repo}": {
                    "splits": [
                        {"config": "a", "split": "train", "num_examples": 1},
                        {"config": "b", "split": "train", "num_examples": 1},
                    ]
                },
            }
        ),
    )

    with pytest.raises(RemoteDatasetError) as exc:
        remote_datasets.HuggingFaceDatasets().resolve(repo)
    assert exc.value.code == "config_required"
    assert "a, b" in exc.value.message


def test_resolve_refuses_a_split_that_does_not_exist(monkeypatch):
    repo, config = "org/repo", "default"
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _opener(
            {
                f"/is-valid?dataset={repo}": {"valid": True},
                f"/splits?dataset={repo}": {
                    "splits": [
                        {"config": config, "split": "train", "num_examples": 1}
                    ]
                },
            }
        ),
    )

    with pytest.raises(RemoteDatasetError) as exc:
        remote_datasets.HuggingFaceDatasets().resolve(repo, config, "test")
    assert exc.value.code == "split_not_found"


def test_resolve_refuses_a_split_that_resolves_to_nothing(monkeypatch):
    repo, config, split = "org/repo", "default", "train"
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _opener(
            {
                f"/is-valid?dataset={repo}": {"valid": True},
                f"/splits?dataset={repo}": _split_route(
                    repo, config, split, 0
                ),
                f"/size?dataset={repo}&config={config}&split={split}": {
                    "size": {"num_rows": 0, "num_bytes": 0}
                },
            }
        ),
    )

    with pytest.raises(RemoteDatasetError) as exc:
        remote_datasets.HuggingFaceDatasets().resolve(repo)
    assert exc.value.code == "split_empty"
    assert "no rows" in exc.value.message


def test_stream_falls_back_to_paginated_rows_when_there_is_no_parquet(
    monkeypatch,
):
    """HF converts a dataset to Parquet in the background, so a repository
    published minutes ago genuinely has none yet (ADR-0073). A listing that
    answered and had nothing for this split takes the `/rows` path -- the
    mechanism this seam shipped with -- rather than failing the import."""
    repo, config, split = "org/repo", "default", "train"
    rows = [chat_rows(100), chat_rows(3)]  # a full page, then the tail

    def opener(target, timeout=10):
        url = _url_of(target)
        if "/parquet?" in url:
            body = _EMPTY_LISTING
        elif "offset=0" in url:
            body = {
                "rows": [
                    {"row_idx": i, "row": r} for i, r in enumerate(rows[0])
                ]
            }
        elif "offset=100" in url:
            body = {
                "rows": [
                    {"row_idx": 100 + i, "row": r}
                    for i, r in enumerate(rows[1])
                ]
            }
        else:
            body = {"rows": []}
        resp = Mock()
        resp.read.return_value = json.dumps(body).encode()
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        return resp

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    source = remote_datasets._HuggingFaceSource(repo, config, split, 103, 0)
    lines = b"".join(source.stream()).decode("utf-8").splitlines()
    assert [json.loads(line) for line in lines] == rows[0] + rows[1]


def test_stream_surfaces_a_mid_stream_fetch_failure(monkeypatch):
    # A page of deliberately fat rows: one 64 KiB chunk has to be delivered
    # before the failing page is reached, or the case is not mid-stream. The
    # serialised buffer now spans pages rather than being flushed per page,
    # so "a page" and "a chunk" are no longer the same size.
    fat = [
        {"messages": [{"role": "user", "content": "x" * 1000}]}
        for _ in range(100)
    ]

    def opener(target, timeout=10):
        url = _url_of(target)
        if "/parquet?" in url:
            body = _EMPTY_LISTING
        elif "offset=0" in url:
            body = {
                "rows": [{"row_idx": i, "row": r} for i, r in enumerate(fat)]
            }
        else:
            body = {"error": "Request rate limited: retry later."}
        resp = Mock()
        resp.read.return_value = json.dumps(body).encode()
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        return resp

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    source = remote_datasets._HuggingFaceSource(
        "org/repo", "default", "train", None, None
    )
    it = source.stream()
    assert next(it)  # the first (full) page streams...
    with pytest.raises(RemoteDatasetError) as exc:
        list(it)  # ...and the failure on the next page is a coded reason
    assert exc.value.code == "fetch_failed"
    assert "rate limited" in exc.value.message


def test_an_unparseable_fetch_failure_is_a_coded_reason(monkeypatch):
    """A fetch can fail at any point -- a dropped connection, a body that is
    not JSON -- and the seam's contract is that every failure is a coded
    reason, never an unhandled exception mid-import."""

    def opener(url, timeout=10):
        raise urllib.error.URLError("connection reset by peer")

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    with pytest.raises(RemoteDatasetError) as exc:
        remote_datasets.HuggingFaceDatasets().resolve("org/repo")
    assert exc.value.code == "fetch_failed"
    assert "connection reset" in exc.value.message


def test_a_stream_that_resolves_to_no_rows_is_refused_with_that_reason(
    monkeypatch,
):
    """The `/size` refusal in resolve is the fast path; this is the ground
    truth. When the size endpoint reported nothing and the stream carries no
    rows, the source still refuses with the same `split_empty` reason."""

    def opener(url, timeout=10):
        body = {"rows": []}
        resp = Mock()
        resp.read.return_value = json.dumps(body).encode()
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        return resp

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    source = remote_datasets._HuggingFaceSource(
        "org/repo", "default", "train", None, None
    )
    with pytest.raises(RemoteDatasetError) as exc:
        list(source.stream())
    assert exc.value.code == "split_empty"
    assert "no rows" in exc.value.message


# --- the Parquet fetch (ADR-0073) --------------------------------------------

# What `/parquet` answers for a dataset whose conversion has produced nothing
# for this split. The shape HF sends, minus the files.
_EMPTY_LISTING: dict = {
    "parquet_files": [],
    "pending": [],
    "failed": [],
    "partial": False,
}


def _parquet_bytes(
    rows: list[dict], row_group_size: int | None = None
) -> bytes:
    """One Parquet shard, built in memory. The fixture is the real format --
    written by the same library the seam reads with -- so a test that passes
    is a statement about Parquet, not about a mock's shape."""
    sink = io.BytesIO()
    pq.write_table(
        pa.Table.from_pylist(rows), sink, row_group_size=row_group_size
    )
    return sink.getvalue()


def _shard_url(repo: str, config: str, split: str, filename: str) -> str:
    """The URL shape HF publishes a converted shard at. The `/resolve/`
    segment is the point: it puts the download in the resolver rate-limit
    bucket rather than the API one (ADR-0073)."""
    return (
        f"https://huggingface.co/datasets/{repo}/resolve/"
        f"refs%2Fconvert%2Fparquet/{config}/{split}/{filename}"
    )


def _listing(repo: str, config: str, split: str, shards: list[tuple]) -> dict:
    return {
        "parquet_files": [
            {
                "dataset": repo,
                "config": config,
                "split": split,
                "url": _shard_url(repo, config, split, name),
                "filename": name,
                "size": len(payload),
            }
            for name, payload in shards
        ],
        "pending": [],
        "failed": [],
        "partial": False,
    }


def _bytes_response(payload: bytes):
    """A `urlopen` stand-in's response over `payload`, readable in blocks the
    way the downloader reads one -- `read(n)` at a time until it is empty."""
    buffer = io.BytesIO(payload)
    resp = Mock()
    resp.read.side_effect = lambda n=-1: buffer.read(
        n if n and n > 0 else None
    )
    resp.__enter__ = Mock(return_value=resp)
    resp.__exit__ = Mock(return_value=False)
    return resp


def _parquet_opener(
    listing: dict, shards: dict[str, bytes], seen: list = None
):
    """A `urlopen` stand-in that answers `/parquet` with `listing` and each
    shard URL with its bytes. Anything else is an assertion failure, which is
    how a test proves the `/rows` endpoint was never touched."""

    def opener(target, timeout=10):
        url = _url_of(target)
        if seen is not None:
            seen.append(url)
        if url in shards:
            return _bytes_response(shards[url])
        if "/parquet?" in url:
            return _bytes_response(json.dumps(listing).encode())
        raise AssertionError(f"no mocked route for {url}")

    return opener


def _source(repo="org/repo", config="default", split="train"):
    return remote_datasets._HuggingFaceSource(repo, config, split, None, None)


def test_stream_reads_the_splits_parquet_shard_rather_than_the_rows_endpoint(
    monkeypatch,
):
    """The whole point of ADR-0073: one request for the shard instead of one
    per 100 rows. The opener refuses every route it was not given, so a
    `/rows` call would fail the test rather than quietly pass it."""
    repo, config, split = "org/repo", "default", "train"
    rows = chat_rows(250)
    payload = _parquet_bytes(rows)
    listing = _listing(repo, config, split, [("0000.parquet", payload)])
    seen: list[str] = []
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _parquet_opener(
            listing,
            {_shard_url(repo, config, split, "0000.parquet"): payload},
            seen,
        ),
    )

    body = b"".join(_source(repo, config, split).stream())

    assert body == b"".join(jsonl_chunks(rows))
    assert not any("/rows?" in url for url in seen)
    assert len(seen) == 2  # the listing, and the one shard
    # The listing is asked for by configuration: the unfiltered one covers the
    # whole repository, and datasets-server refuses to compute one over 10 MB.
    # `split` is deliberately absent -- the server ignores it (ADR-0073).
    assert f"/parquet?dataset={repo}&config={config}" in seen[0]
    assert "split=" not in seen[0]


def test_stream_reads_every_shard_of_a_multi_file_split_in_shard_order(
    monkeypatch,
):
    """A split is often several files. Reading only the first would silently
    import a fraction of the dataset -- the worst failure this path has,
    because nothing about it looks like a failure. The listing here is given
    out of order, because shard order is a property of the dataset and must
    not depend on how the listing came back."""
    repo, config, split = "org/repo", "default", "train"
    first, second, third = chat_rows(4), chat_rows(3), chat_rows(2)
    payloads = {
        "0000.parquet": _parquet_bytes(first),
        "0001.parquet": _parquet_bytes(second),
        "0002.parquet": _parquet_bytes(third),
    }
    shuffled = [("0002.parquet", payloads["0002.parquet"])] + [
        (name, payloads[name]) for name in ("0000.parquet", "0001.parquet")
    ]
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _parquet_opener(
            _listing(repo, config, split, shuffled),
            {
                _shard_url(repo, config, split, name): payload
                for name, payload in payloads.items()
            },
        ),
    )

    body = b"".join(_source(repo, config, split).stream())

    assert body == b"".join(jsonl_chunks(first + second + third))


def test_stream_orders_a_part_sharded_split_by_part_then_shard(monkeypatch):
    """The case `filename` gets wrong, measured against the live API: a large
    split is broken into parts (`train-part0/`, `train-part1/`, ...) and each
    part restarts its numbering, so filenames collide -- 27,468 shards of
    `HuggingFaceFW/fineweb` carry 10,000 distinct filenames. The listing's own
    order interleaves the parts, so trusting it is wrong too. Part 10 is here
    because plain string order puts it before part 2.
    """
    repo, config, split = "org/repo", "default", "train"
    layout = [
        ("train-part0", "0000.parquet"),
        ("train-part0", "0001.parquet"),
        ("train-part2", "0000.parquet"),
        ("train-part10", "0000.parquet"),
    ]
    rows = {
        (part, name): [
            {"messages": [{"role": "user", "content": f"{part}/{name}#{i}"}]}
            for i in range(2)
        ]
        for part, name in layout
    }
    urls = {
        (part, name): _shard_url(repo, config, part, name)
        for part, name in layout
    }
    shards = {urls[key]: _parquet_bytes(rows[key]) for key in layout}
    # Listed the way HF lists one: interleaved across parts, and the `split`
    # field says `train` however the URL is laid out.
    interleaved = [layout[2], layout[3], layout[0], layout[1]]
    listing = {
        "parquet_files": [
            {
                "dataset": repo,
                "config": config,
                "split": split,
                "url": urls[key],
                "filename": key[1],
                "size": len(shards[urls[key]]),
            }
            for key in interleaved
        ],
        "pending": [],
        "failed": [],
    }
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _parquet_opener(listing, shards),
    )

    body = b"".join(_source(repo, config, split).stream())

    expected = [row for key in layout for row in rows[key]]
    assert body == b"".join(jsonl_chunks(expected))


def test_stream_reads_only_the_shards_of_the_resolved_config_and_split(
    monkeypatch,
):
    """`/parquet` lists the whole repository -- every config, every split. An
    import that read them all would silently concatenate the test split onto
    the train one."""
    repo = "org/repo"
    wanted, other = chat_rows(3), chat_rows(2)
    wanted_bytes, other_bytes = _parquet_bytes(wanted), _parquet_bytes(other)
    listing = {
        "parquet_files": [
            {
                "dataset": repo,
                "config": "default",
                "split": "test",
                "url": _shard_url(repo, "default", "test", "0000.parquet"),
                "filename": "0000.parquet",
                "size": len(other_bytes),
            },
            {
                "dataset": repo,
                "config": "other",
                "split": "train",
                "url": _shard_url(repo, "other", "train", "0000.parquet"),
                "filename": "0000.parquet",
                "size": len(other_bytes),
            },
            {
                "dataset": repo,
                "config": "default",
                "split": "train",
                "url": _shard_url(repo, "default", "train", "0000.parquet"),
                "filename": "0000.parquet",
                "size": len(wanted_bytes),
            },
        ],
        "pending": [],
        "failed": [],
    }
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _parquet_opener(
            listing,
            {
                _shard_url(
                    repo, "default", "train", "0000.parquet"
                ): wanted_bytes
            },
        ),
    )

    body = b"".join(_source(repo, "default", "train").stream())

    assert body == b"".join(jsonl_chunks(wanted))


def test_stream_falls_back_to_rows_when_the_conversion_is_still_pending(
    monkeypatch,
):
    """A split HF has listed as pending has no shard to read yet, and the
    listing carries files for the *other* splits -- so "the listing was not
    empty" is not the test. The reported state is."""
    repo, config, split = "org/repo", "default", "train"
    rows = chat_rows(3)
    listing = {
        "parquet_files": [
            {
                "dataset": repo,
                "config": config,
                "split": "test",
                "url": _shard_url(repo, config, "test", "0000.parquet"),
                "filename": "0000.parquet",
                "size": 10,
            }
        ],
        "pending": [{"dataset": repo, "config": config, "split": split}],
        "failed": [],
    }

    def opener(target, timeout=10):
        url = _url_of(target)
        if "/parquet?" in url:
            return _bytes_response(json.dumps(listing).encode())
        assert "/rows?" in url, url
        body = {"rows": [{"row_idx": i, "row": r} for i, r in enumerate(rows)]}
        return _bytes_response(json.dumps(body).encode())

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    body = b"".join(_source(repo, config, split).stream())

    assert body == b"".join(jsonl_chunks(rows))


def test_a_parquet_split_that_carries_no_rows_is_refused_with_that_reason(
    monkeypatch,
):
    """The same `split_empty` ground truth the `/rows` path has: a shard that
    exists but holds nothing is still nothing to import."""
    repo, config, split = "org/repo", "default", "train"
    empty = pa.Table.from_pylist(chat_rows(1)).slice(0, 0)
    sink = io.BytesIO()
    pq.write_table(empty, sink)
    payload = sink.getvalue()
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _parquet_opener(
            _listing(repo, config, split, [("0000.parquet", payload)]),
            {_shard_url(repo, config, split, "0000.parquet"): payload},
        ),
    )

    with pytest.raises(RemoteDatasetError) as exc:
        list(_source(repo, config, split).stream())
    assert exc.value.code == "split_empty"


def test_the_parquet_download_carries_the_bearer_token(monkeypatch):
    """The download is on huggingface.co, not datasets-server, and it is the
    request that moves the data -- a token on the listing but not on the
    shard would leave the expensive half anonymous, and would fail outright
    on a gated repository."""
    monkeypatch.setattr(remote_datasets.config, "HF_TOKEN", "hf_test_token")
    repo, config, split = "org/repo", "default", "train"
    payload = _parquet_bytes(chat_rows(2))
    url = _shard_url(repo, config, split, "0000.parquet")
    headers: list[str | None] = []

    def opener(target, timeout=10):
        assert isinstance(target, urllib.request.Request)
        headers.append(target.get_header("Authorization"))
        if _url_of(target) == url:
            return _bytes_response(payload)
        return _bytes_response(
            json.dumps(
                _listing(repo, config, split, [("0000.parquet", payload)])
            ).encode()
        )

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    list(_source(repo, config, split).stream())

    assert headers == ["Bearer hf_test_token", "Bearer hf_test_token"]


def test_a_shard_that_cannot_be_downloaded_is_a_coded_reason(monkeypatch):
    repo, config, split = "org/repo", "default", "train"
    payload = _parquet_bytes(chat_rows(2))
    url = _shard_url(repo, config, split, "0000.parquet")

    def opener(target, timeout=10):
        if _url_of(target) == url:
            raise urllib.error.HTTPError(url, 500, "boom", {}, None)
        return _bytes_response(
            json.dumps(
                _listing(repo, config, split, [("0000.parquet", payload)])
            ).encode()
        )

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    with pytest.raises(RemoteDatasetError) as exc:
        list(_source(repo, config, split).stream())
    assert exc.value.code == "fetch_failed"
    assert "500" in exc.value.message


def test_a_shard_that_is_not_valid_parquet_is_a_coded_reason(monkeypatch):
    """A truncated or corrupt shard raises out of pyarrow, and the seam's
    contract is that every failure is a coded reason -- by this point the
    import has already answered with a dataset id and has only the stored
    report left to fail into."""
    repo, config, split = "org/repo", "default", "train"
    payload = b"PAR1 this is not a parquet file"
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _parquet_opener(
            _listing(repo, config, split, [("0000.parquet", payload)]),
            {_shard_url(repo, config, split, "0000.parquet"): payload},
        ),
    )

    with pytest.raises(RemoteDatasetError) as exc:
        list(_source(repo, config, split).stream())
    assert exc.value.code == "fetch_failed"
    assert "parquet" in exc.value.message


def _temp_shards() -> set[str]:
    return {
        name
        for name in os.listdir(tempfile.gettempdir())
        if name.startswith("temper-hf-")
    }


def test_a_downloaded_shard_is_deleted_whether_the_stream_finishes_or_not(
    monkeypatch,
):
    """The shard is buffered to disk because pyarrow needs a seekable file.
    Buffered means owned: a stream that completes, one that fails on the next
    shard, and one the caller simply stops reading must all leave nothing
    behind."""
    repo, config, split = "org/repo", "default", "train"
    # Fat enough that the first shard fills a 64 KiB chunk on its own: the
    # abandoned-stream half of this test needs one chunk delivered before the
    # second shard is ever reached.
    good = _parquet_bytes(
        [{"messages": [{"role": "user", "content": "x" * 1000}]}] * 200
    )
    bad = b"not parquet at all"
    urls = {
        _shard_url(repo, config, split, "0000.parquet"): good,
        _shard_url(repo, config, split, "0001.parquet"): bad,
    }
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _parquet_opener(
            _listing(
                repo,
                config,
                split,
                [("0000.parquet", good), ("0001.parquet", bad)],
            ),
            urls,
        ),
    )
    before = _temp_shards()

    with pytest.raises(RemoteDatasetError):
        list(_source(repo, config, split).stream())
    assert _temp_shards() == before

    abandoned = _source(repo, config, split).stream()
    next(abandoned)
    abandoned.close()
    assert _temp_shards() == before


def test_typed_parquet_values_are_serialised_rather_than_crashing_the_import(
    monkeypatch,
):
    """Parquet carries types JSON has none of. `/rows` handed back HF's own
    JSON encoding of them; reading the file directly means meeting the real
    ones, and a `TypeError` out of `json.dumps` mid-stream would break the
    promise that every failure is a coded reason."""
    repo, config, split = "org/repo", "default", "train"
    table = pa.table(
        {
            "text": ["hello", "world"],
            "at": [
                datetime.datetime(2026, 9, 3, 12, 0),
                datetime.datetime(2026, 9, 3, 13, 30),
            ],
            "blob": [b"\x00\x01", b"\x02\x03"],
        }
    )
    sink = io.BytesIO()
    pq.write_table(table, sink)
    payload = sink.getvalue()
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _parquet_opener(
            _listing(repo, config, split, [("0000.parquet", payload)]),
            {_shard_url(repo, config, split, "0000.parquet"): payload},
        ),
    )

    lines = (
        b"".join(_source(repo, config, split).stream())
        .decode("utf-8")
        .splitlines()
    )

    first = json.loads(lines[0])
    assert first["text"] == "hello"
    assert first["at"] == "2026-09-03T12:00:00"
    assert isinstance(first["blob"], str)  # base64, not a crash


def test_a_download_budget_is_derived_from_the_declared_shard_size():
    """A fixed timeout would either strangle a large shard or give a stalled
    connection the span a large one needs. Both ends read from the same
    floor, so the budget grows with the file and nothing else."""
    small = remote_datasets._download_budget_s(1024)
    large = remote_datasets._download_budget_s(512 * 1024 * 1024)

    assert small >= remote_datasets._DOWNLOAD_GRACE_S
    assert large > small * 10
    # An undeclared size is treated as the largest shard the conversion is
    # expected to produce, not as no time at all.
    assert remote_datasets._download_budget_s(None) > small


def test_parquet_memory_stays_flat_as_a_shard_grows(monkeypatch):
    """The guarantee the whole ingest path rests on (ADR-0036): peak memory is
    a property of the batch, never of the dataset.

    Measured, not asserted by inspection. Arrow allocates outside Python's
    allocator, so `pa.total_allocated_bytes()` is what has to be watched. Two
    shards a factor of ten apart: reading them held 1,084,608 bytes each,
    the same number to the byte, because what is alive is a batch and the row
    group behind it. This is the test that catches `pre_buffer`, whose pyarrow
    default is `True` and reads the whole file up front -- with it the peak
    tracks the shard instead (32.3 MB on a 32 MB shard, pyarrow 25.0.1).

    Rows are random text so the shard cannot compress away to nothing, and
    both sizes are past the point where the steady state is reached: a shard
    smaller than the working set peaks below it and would make the comparison
    meaningless.
    """
    import hashlib

    repo, config, split = "org/repo", "default", "train"

    def noise_rows(n: int) -> list[dict]:
        # Hashed rather than random: incompressible, so the shard on disk is
        # the size the row count implies, and reproducible, so a failure here
        # is the same failure on the next run.
        return [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "".join(
                            hashlib.sha256(f"{i}-{k}".encode()).hexdigest()
                            for k in range(8)
                        ),
                    }
                ]
            }
            for i in range(n)
        ]

    def peak_while_streaming(rows: list[dict]) -> tuple[int, int]:
        payload = _parquet_bytes(rows, row_group_size=1000)
        monkeypatch.setattr(
            remote_datasets.urllib.request,
            "urlopen",
            _parquet_opener(
                _listing(repo, config, split, [("0000.parquet", payload)]),
                {_shard_url(repo, config, split, "0000.parquet"): payload},
            ),
        )
        base = pa.total_allocated_bytes()
        peak = 0
        streamed = 0
        # Counted, never retained -- retaining the stream would put the
        # dataset in the measurement and prove nothing.
        for chunk in _source(repo, config, split).stream():
            streamed += len(chunk)
            peak = max(peak, pa.total_allocated_bytes() - base)
        return peak, streamed

    small_peak, small_streamed = peak_while_streaming(noise_rows(2_000))
    large_peak, large_streamed = peak_while_streaming(noise_rows(20_000))

    # The payload really did grow by an order of magnitude...
    assert large_streamed > 8 * small_streamed
    assert large_streamed > 10 * 1024 * 1024
    # ...and the memory held while streaming it did not follow. The margin is
    # for the allocator, not for growth: the two peaks measured identical.
    assert large_peak < 1.25 * small_peak


# --- rate limiting and the token (issue: "dataset import from HF") -----------


def test_api_retries_after_a_429_using_retry_after(monkeypatch):
    """HF's own guidance: a 429 is usually load-shedding, not a permanent
    refusal. A retry waits exactly what `Retry-After` says."""
    calls = {"n": 0}

    def opener(target, timeout=10):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                _url_of(target),
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                None,
            )
        resp = Mock()
        resp.read.return_value = json.dumps({"valid": True}).encode()
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        return resp

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)
    sleeps: list[float] = []
    monkeypatch.setattr(remote_datasets.time, "sleep", sleeps.append)

    body = remote_datasets._api(
        "/is-valid?dataset=org/repo",
        deadline=remote_datasets.time.monotonic() + 5,
    )

    assert body == {"valid": True}
    assert calls["n"] == 2
    assert sleeps == [0.0]


def test_api_does_not_retry_a_429_whose_wait_would_clear_the_deadline(
    monkeypatch,
):
    """A retry that cannot land before the caller's own deadline is not worth
    starting -- it fails as the coded refusal instead of stalling past a
    budget the caller chose for a reason (see `_RESOLVE_BUDGET_S`)."""

    def opener(target, timeout=10):
        raise urllib.error.HTTPError(
            _url_of(target),
            429,
            "Too Many Requests",
            {"Retry-After": "9999"},
            None,
        )

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    with pytest.raises(RemoteDatasetError) as exc:
        remote_datasets._api(
            "/is-valid?dataset=org/repo",
            deadline=remote_datasets.time.monotonic() + 5,
        )
    assert exc.value.code == "fetch_failed"
    assert "429" in exc.value.message


def test_api_survives_several_consecutive_429s_within_the_retry_budget(
    monkeypatch,
):
    """A single retry can still lose to one unlucky pair of attempts --
    live testing against the real datasets-server found 429s are short,
    self-resolving blips (issue: "dataset import from HF"). Up to
    `_MAX_429_RETRIES` attempts gives a real blip room to clear."""
    calls = {"n": 0}

    def opener(target, timeout=10):
        calls["n"] += 1
        if calls["n"] <= remote_datasets._MAX_429_RETRIES:
            raise urllib.error.HTTPError(
                _url_of(target),
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                None,
            )
        resp = Mock()
        resp.read.return_value = json.dumps({"valid": True}).encode()
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        return resp

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)
    monkeypatch.setattr(remote_datasets.time, "sleep", lambda s: None)

    body = remote_datasets._api(
        "/is-valid?dataset=org/repo",
        deadline=remote_datasets.time.monotonic() + 5,
    )

    assert body == {"valid": True}
    assert calls["n"] == remote_datasets._MAX_429_RETRIES + 1


def test_api_gives_up_after_exhausting_the_429_retry_budget(monkeypatch):
    """A persistent 429 -- not a blip -- still ends in the coded refusal
    rather than retrying forever; `_MAX_429_RETRIES` is a bound, not a
    promise the server will ever say yes."""
    calls = {"n": 0}

    def opener(target, timeout=10):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            _url_of(target),
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            None,
        )

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)
    monkeypatch.setattr(remote_datasets.time, "sleep", lambda s: None)

    with pytest.raises(RemoteDatasetError) as exc:
        remote_datasets._api(
            "/is-valid?dataset=org/repo",
            deadline=remote_datasets.time.monotonic() + 5,
        )

    assert exc.value.code == "fetch_failed"
    assert "429" in exc.value.message
    assert calls["n"] == remote_datasets._MAX_429_RETRIES + 1


def test_api_sends_the_configured_token_as_a_bearer_header(monkeypatch):
    monkeypatch.setattr(remote_datasets.config, "HF_TOKEN", "hf_test_token")
    seen: dict[str, object] = {}

    def opener(target, timeout=10):
        seen["target"] = target
        resp = Mock()
        resp.read.return_value = json.dumps({"valid": True}).encode()
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        return resp

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    remote_datasets._api(
        "/is-valid?dataset=org/repo",
        deadline=remote_datasets.time.monotonic() + 5,
    )

    target = seen["target"]
    assert isinstance(target, urllib.request.Request)
    assert target.get_header("Authorization") == "Bearer hf_test_token"


def test_api_sends_no_authorization_header_when_no_token_is_configured(
    monkeypatch,
):
    monkeypatch.setattr(remote_datasets.config, "HF_TOKEN", None)
    seen: dict[str, object] = {}

    def opener(target, timeout=10):
        seen["target"] = target
        resp = Mock()
        resp.read.return_value = json.dumps({"valid": True}).encode()
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        return resp

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    remote_datasets._api(
        "/is-valid?dataset=org/repo",
        deadline=remote_datasets.time.monotonic() + 5,
    )

    assert isinstance(seen["target"], str)


def test_api_gives_up_at_the_wall_clock_deadline_even_if_urlopen_does_not(
    monkeypatch,
):
    """The bug this exists to catch: `urlopen`'s own `timeout=` bounds each
    socket operation, not the call as a whole, so a fake that blocks past the
    deadline without erroring stands in for a server that trickles its
    response out in slow pieces. `_api` must still return once `deadline`
    passes, not once the blocking call happens to finish."""
    import time as real_time

    def opener(target, timeout=10):
        real_time.sleep(0.3)  # longer than the 0.05s deadline below
        resp = Mock()
        resp.read.return_value = json.dumps({"valid": True}).encode()
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        return resp

    monkeypatch.setattr(remote_datasets.urllib.request, "urlopen", opener)

    started = remote_datasets.time.monotonic()
    with pytest.raises(RemoteDatasetError) as exc:
        remote_datasets._api(
            "/is-valid?dataset=org/repo",
            deadline=started + 0.05,
        )
    elapsed = remote_datasets.time.monotonic() - started
    assert exc.value.code == "fetch_failed"
    assert elapsed < 0.3
