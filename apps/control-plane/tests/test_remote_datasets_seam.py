"""The remote-dataset seam -- issue #45.

`FakeRemoteDatasets` is exercised the same way `FakeProvider` and `FakeModels`
are: constructed with the references a case needs, refusing anything it was
not given. `HuggingFaceDatasets` is exercised with `urllib.request.urlopen`
monkeypatched locally -- the one place that undoes
`conftest.no_real_remote_datasets`'s guard, because this is the file that owns
proving the resolution and the paginated stream are correct.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

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
    assert exc.value.code == "dataset_not_found"
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


def _opener(routes: dict[str, dict], status: int = 200):
    """A stand-in for `urllib.request.urlopen` that answers by URL suffix."""

    def opener(url, timeout=10):
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


def test_resolve_refuses_a_repository_that_cannot_be_fetched(monkeypatch):
    monkeypatch.setattr(
        remote_datasets.urllib.request,
        "urlopen",
        _opener(
            {
                "/is-valid?dataset=org/ghost": {
                    "valid": False,
                    "reason": "This dataset does not exist.",
                }
            }
        ),
    )

    with pytest.raises(RemoteDatasetError) as exc:
        remote_datasets.HuggingFaceDatasets().resolve("org/ghost")
    assert exc.value.code == "dataset_not_found"
    assert "does not exist" in exc.value.message


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


def test_stream_paginates_and_serialises_rows_in_order(monkeypatch):
    repo, config, split = "org/repo", "default", "train"
    rows = [chat_rows(100), chat_rows(3)]  # a full page, then the tail

    def opener(url, timeout=10):
        if "offset=0" in url:
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
    def opener(url, timeout=10):
        if "offset=0" in url:
            body = {
                "rows": [
                    {"row_idx": i, "row": r} for i, r in enumerate(chat_rows(100))
                ]
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
