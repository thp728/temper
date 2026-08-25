"""Catalog revisions are pinned to commit identifiers."""

from __future__ import annotations

import re

import pytest

from temper_core.catalog import CATALOG, is_pinned_revision

_BRANCH_LIKE = re.compile(r"^(main|master|develop|\d+\.\d+)$")


def test_every_catalog_entry_carries_a_pinned_commit_sha():
    for m in CATALOG.values():
        assert is_pinned_revision(m.revision), (
            f"{m.id} revision '{m.revision}' is not a 40-char commit SHA"
        )


def test_no_catalog_entry_carries_a_branch_name():
    for m in CATALOG.values():
        assert m.revision != "main"
        assert m.revision != "master"
        assert not _BRANCH_LIKE.match(m.revision)
        assert len(m.revision) == 40
        assert all(c in "0123456789abcdef" for c in m.revision.lower())


def test_catalog_with_branch_name_fails_a_check_rather_than_loading():
    from temper_core.catalog import BaseModel

    bad = BaseModel(
        id="bad",
        repo="Qwen/Qwen3-4B",
        revision="main",
        params_b=4.0,
        license="Apache-2.0",
        license_url="https://example.com",
        context_length=4096,
        good_for="test",
        min_gpu="L4",
        est_peak_vram_gb=5.0,
    )
    assert not is_pinned_revision(bad.revision)
    # The catalog module's import-time check would raise ValueError for such an entry
    with pytest.raises(ValueError, match="not a 40-character commit SHA"):
        if not is_pinned_revision(bad.revision):
            raise ValueError(
                f"Catalog entry '{bad.id}' has revision '{bad.revision}' which is not a "
                f"40-character commit SHA. Pin it to a resolved commit identifier so a "
                f"completed run cannot change underneath it."
            )


def test_is_pinned_revision_accepts_only_40_hex():
    assert is_pinned_revision("1cfa9a7208912126459214e8b04321603b3df60c")
    assert is_pinned_revision("b968826d9c46dd6066d109eabc6255188de91218")
    assert is_pinned_revision(
        "1CFA9A7208912126459214E8B04321603B3DF60C"
    )  # uppercase
    assert not is_pinned_revision("main")
    assert not is_pinned_revision("master")
    assert not is_pinned_revision(
        "1cfa9a7208912126459214e8b04321603b3df60"
    )  # 39 chars
    assert not is_pinned_revision(
        "1cfa9a7208912126459214e8b04321603b3df60c0"
    )  # 41 chars
    assert not is_pinned_revision(
        "Zcfa9a7208912126459214e8b04321603b3df60c"
    )  # non-hex
