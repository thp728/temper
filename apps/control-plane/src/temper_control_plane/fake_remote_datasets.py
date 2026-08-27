"""A remote-dataset resolver that reaches no network.

Kept in the package rather than in a test file, matching `fake_provider.py`
and `fake_models.py`: every ticket that touches the import path tests against
the same double instead of growing its own. Constructed with the references a
case needs -- there is no canned single scenario here because the refusals
(repository missing, split empty) matter as much as the happy path.

`resolve` mirrors the real resolver's refusals so a test asserts on behaviour,
not on which implementation produced it: a reference that was not seeded is
`repo_not_found`, a seeded reference with no rows is `split_empty`. Rows
stream through the same `jsonl_chunks` serialisation the real source uses, so
an import reads the same bytes whether it was fetched or faked.
"""

from __future__ import annotations

from typing import Any

from .remote_datasets import (
    DEFAULT_CONFIG,
    DEFAULT_SPLITS,
    RemoteDatasetError,
    RemoteDatasetSource,
    _ResolvedSource,
    jsonl_chunks,
)


class FakeRemoteDatasets:
    """Resolves exactly the `(repo, config, split)` references it was given.

    A seeded reference streams its rows; a reference whose rows are empty is
    refused as `split_empty`; a reference that was never seeded is refused as
    `repo_not_found` naming what it does know, rather than guessing or
    reaching for a network no test should depend on.
    """

    def __init__(
        self,
        sources: dict[
            tuple[str, str | None, str | None], list[dict[str, Any]]
        ],
    ) -> None:
        self._sources = dict(sources)

    def resolve(
        self,
        repo: str,
        config: str | None = None,
        split: str | None = None,
    ) -> RemoteDatasetSource:
        key = (repo, config, split)
        rows = self._sources.get(key)
        if rows is None:
            raise RemoteDatasetError(
                "repo_not_found",
                f"FakeRemoteDatasets has no source for {key!r}. "
                f"Known: {sorted(self._sources)}",
            )
        if not rows:
            raise RemoteDatasetError(
                "split_empty",
                f"Split '{split}' of '{repo}' resolves to no rows; "
                f"there is nothing to import.",
            )
        # Default the unnamed halves the way the real resolver does, so the
        # imported dataset is named identically whichever resolver produced it.
        resolved_config = config if config is not None else DEFAULT_CONFIG
        resolved_split = split if split is not None else DEFAULT_SPLITS[0]
        return _FakeSource(repo, resolved_config, resolved_split, rows)


class _FakeSource(_ResolvedSource):
    """A seeded reference's source: its rows are held (they are small, canned
    fixtures) and streamed through the same serialisation a real fetch uses."""

    def __init__(
        self,
        repo: str,
        config: str | None,
        split: str | None,
        rows: list[dict[str, Any]],
    ) -> None:
        super().__init__(repo, config, split)
        self._rows = rows

    def stream(self):
        return jsonl_chunks(self._rows)


def chat_rows(n: int) -> list[dict[str, Any]]:
    """`n` valid chat rows, the same shape the upload fixtures use."""
    return [
        {
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ]
        }
        for i in range(n)
    ]


def journeys_datasets() -> FakeRemoteDatasets:
    """The double the browser journeys import against when the control plane
    is booted with `TEMPER_FAKE_PROVIDER`: one reference that imports to a
    valid report, one that fails validation but is still stored with its
    report, and one whose split resolves to nothing. The journeys need a
    reference that validates so the happy path runs to a report, and one that
    fails so the "kept with its report" clause is journey-visible too.
    """
    return FakeRemoteDatasets(
        {
            ("acme/demo-chat", None, None): chat_rows(12),
            ("acme/demo-broken", None, None): [
                {"content": "no messages list at all"},
                {"messages": [{"role": "user", "content": "q"}]},
            ],
            ("acme/empty-split", None, None): [],
        }
    )
