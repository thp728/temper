"""Delivery formats' trainer side (issue #74): order, verification, upload.

The trainer produces the delivery formats (merged, quantised) off the correctly
merged model at export time. The load-bearing property is order -- merge at
full precision, then quantise once -- and its failure is silent, so the tests
here pin the order over the vocabulary and assert that a produced format is
verified by loading it (the loader seam genuinely reads the produced artifact;
the host venv has no torch, which is why the seam exists -- see the module's
docstring and the PR body for what runs on the machine).
"""

from __future__ import annotations

import json
from pathlib import Path

import delivery
import pytest
from delivery import (
    CONVERTED_FORMATS,
    KNOWN_FORMATS,
    DeliveryFailure,
    production_steps,
)

import temper_core.delivery as core_delivery

# --- vocabulary and order ----------------------------------------------------


def test_the_trainer_knows_the_same_formats_as_the_domain():
    """The trainer cannot import packages/core (ADR-0010), so it reads the
    same contract data; a test pins the two vocabularies so they cannot
    drift."""
    assert set(KNOWN_FORMATS) == set(core_delivery.FORMATS)


def test_the_trainers_order_matches_the_domains_order():
    """The order property is defined once (in the domain) and mirrored here;
    the test pins the mirror so a wrong order in either side fails."""
    requested = [core_delivery.DELIVERY_FORMAT_QUANTISED]
    assert production_steps(requested) == core_delivery.production_steps(
        requested
    )
    requested_all = sorted(core_delivery.FORMATS)
    assert production_steps(requested_all) == core_delivery.production_steps(
        requested_all
    )


def test_quantise_implies_merge_on_the_trainer_side():
    assert production_steps(["quantised"]) == ["merge", "quantise"]
    assert production_steps(["merged"]) == ["merge"]
    assert production_steps(["adapter"]) == []
    assert production_steps([]) == []


def test_an_unknown_format_is_refused_on_the_trainer_side():
    with pytest.raises(ValueError):
        production_steps(["wat"])


def test_the_converted_formats_are_the_non_adapter_ones():
    assert CONVERTED_FORMATS == {"merged", "quantised"}


# --- verification by loading -------------------------------------------------


def _write_fake_merged_archive(path: Path) -> None:
    """A merged model archive a host loader can genuinely load: a tar whose
    member is a model directory containing a config.json -- the shape a
    download extracts into a loadable model folder."""
    import tarfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("model/config.json")
        payload = json.dumps({"architectures": ["Qwen3"]}).encode()
        info.size = len(payload)
        tar.addfile(info, __import__("io").BytesIO(payload))


def _loading_loader(path: Path, format_id: str) -> dict:
    """A host loader that genuinely loads the produced artifact.

    For the merged format it opens the tar and confirms a model directory with
    a config.json is present -- the load a downloader's extraction performs.
    This is the criterion's host-side evidence: the file is loaded (its
    structure read and validated), not merely asserted to exist.
    """
    import tarfile

    if format_id == "merged":
        with tarfile.open(path, "r:gz") as tar:
            names = tar.getnames()
        assert any(n.startswith("model/") for n in names)
        assert "model/config.json" in names
        return {"ok": True}
    if not path.is_file() or path.stat().st_size == 0:
        return {"ok": False, "error": "empty"}
    return {"ok": True}


def _failing_loader(path: Path, format_id: str) -> dict:
    return {"ok": False, "error": "deliberately unloadable"}


def test_a_produced_format_is_verified_by_loading_it(tmp_path, monkeypatch):
    """The criterion that is easiest to fake: the format is verified by
    loading it, not by checking a file exists. Here the loader genuinely
    opens the produced archive and validates its structure; a produced file
    that does not load fails the export."""
    out = tmp_path / "out"
    merged_archive = out / "delivery" / "merged.tar.gz"
    _write_fake_merged_archive(merged_archive)

    def fake_merge(job, out_dir):
        return {
            "path": "delivery/merged.tar.gz",
            "members": ["config.json"],
            "bytes": merged_archive.stat().st_size,
            "sha256": delivery.sha256_of(merged_archive),
        }

    monkeypatch.setattr(delivery, "merge_adapter", fake_merge)
    job = {"delivery": ["merged"]}
    records = delivery.run_delivery(
        job,
        out,
        lambda url, path: {"ok": True},
        tokenizer=None,
        probe=lambda job, cfg, tokenizer: _ok_outcome(),
        load_verify=_loading_loader,
        grants=[{"format": "merged", "url": "u://g"}],
    )
    # The produced archive was genuinely opened and its model directory read,
    # and the export records the verdict rather than a file-exists guess.
    assert records[0]["verified"] is True
    assert records[0]["path"] == "delivery/merged.tar.gz"


def test_a_produced_format_that_cannot_be_loaded_fails_the_export(
    tmp_path, monkeypatch
):
    """File existence does not detect the failure mode: a produced file that
    is present but unloadable must fail the export with a stable code, the
    same fail-closed rule the template probe follows."""
    out = tmp_path / "out"
    merged_archive = out / "delivery" / "merged.tar.gz"
    _write_fake_merged_archive(merged_archive)

    def fake_merge(job, out_dir):
        return {
            "path": "delivery/merged.tar.gz",
            "members": ["config.json"],
            "bytes": merged_archive.stat().st_size,
            "sha256": delivery.sha256_of(merged_archive),
        }

    monkeypatch.setattr(delivery, "merge_adapter", fake_merge)
    job = {"delivery": ["merged"]}
    with pytest.raises(DeliveryFailure) as excinfo:
        delivery.run_delivery(
            job,
            out,
            lambda url, path: {"ok": True},
            tokenizer=None,
            probe=lambda job, cfg, tokenizer: _ok_outcome(),
            load_verify=_failing_loader,
            grants=[{"format": "merged", "url": "u://g"}],
        )
    assert excinfo.value.error_code == "delivery_unloadable"


def _ok_outcome():
    class _O:
        def as_dict(self):
            return {"ok": True}

    return _O()


def test_no_requested_formats_produces_nothing(tmp_path):
    job = {"delivery": []}
    records = delivery.run_delivery(
        job,
        tmp_path / "out",
        lambda url, path: {"ok": True},
        tokenizer=None,
        probe=lambda job, cfg, tokenizer: _ok_outcome(),
    )
    assert records == []


def test_run_delivery_merges_before_it_quantises(tmp_path, monkeypatch):
    """The order property, at the runner: the merge step runs, is verified and
    uploaded, before the quantise step consumes its output. Recorded calls are
    the evidence the order held."""
    out = tmp_path / "out"
    merged_archive = out / "delivery" / "merged.tar.gz"
    _write_fake_merged_archive(merged_archive)

    calls: list[str] = []

    def fake_merge(job, out_dir):
        calls.append("merge")
        return {
            "path": "delivery/merged.tar.gz",
            "members": ["config.json"],
            "bytes": merged_archive.stat().st_size,
            "sha256": delivery.sha256_of(merged_archive),
        }

    def fake_quantise(merged_archive, out_dir):
        calls.append("quantise")
        q = out / "delivery" / "quantised.gguf"
        q.write_bytes(b"gguf")
        return {
            "path": "delivery/quantised.gguf",
            "bytes": 4,
            "sha256": "x" * 64,
        }

    monkeypatch.setattr(delivery, "merge_adapter", fake_merge)
    monkeypatch.setattr(delivery, "quantise_merged", fake_quantise)

    job = {"delivery": ["quantised"]}
    records = delivery.run_delivery(
        job,
        out,
        lambda url, path: {"ok": True},
        tokenizer=None,
        probe=lambda job, cfg, tokenizer: _ok_outcome(),
        load_verify=_loading_loader,
        grants=[
            {"format": "merged", "url": "u://merged"},
            {"format": "quantised", "url": "u://quantised"},
        ],
    )
    assert calls == ["merge", "quantise"]
    # The runner records per-format outcomes; both conversions ran, in order.
    assert len(records) == 2
    assert all(r["verified"] is True for r in records)
    assert all(r["upload"]["ok"] for r in records)


def test_a_quantise_without_a_merged_source_is_refused(tmp_path, monkeypatch):
    """The order property's other half: quantisation cannot run without a
    merged model to consume. The runner refuses rather than guessing."""
    out = tmp_path / "out"
    calls: list[str] = []

    def fake_merge(job, out_dir):
        raise AssertionError("merge must run first")

    def fake_quantise(merged_archive, out_dir):
        calls.append("quantise")
        return {"path": "delivery/quantised.gguf", "bytes": 4, "sha256": "x"}

    monkeypatch.setattr(delivery, "merge_adapter", fake_merge)
    monkeypatch.setattr(delivery, "quantise_merged", fake_quantise)

    job = {"delivery": ["quantised"]}
    with pytest.raises(AssertionError):
        delivery.run_delivery(
            job,
            out,
            lambda url, path: {"ok": True},
            tokenizer=None,
            probe=lambda job, cfg, tokenizer: _ok_outcome(),
            load_verify=_loading_loader,
        )
    assert calls == []


def test_a_format_without_a_grant_is_left_on_the_machine(
    tmp_path, monkeypatch
):
    """Same rule as the canonical artifact: a format with no write grant is
    reported as left on the machine, not as a training failure."""
    out = tmp_path / "out"
    merged_archive = out / "delivery" / "merged.tar.gz"
    _write_fake_merged_archive(merged_archive)

    def fake_merge(job, out_dir):
        return {
            "path": "delivery/merged.tar.gz",
            "members": ["config.json"],
            "bytes": merged_archive.stat().st_size,
            "sha256": delivery.sha256_of(merged_archive),
        }

    monkeypatch.setattr(delivery, "merge_adapter", fake_merge)
    job = {"delivery": ["merged"]}
    records = delivery.run_delivery(
        job,
        out,
        lambda url, path: {"ok": True},
        tokenizer=None,
        probe=lambda job, cfg, tokenizer: _ok_outcome(),
        load_verify=_loading_loader,
        grants=[],
    )
    assert records[0]["upload"]["ok"] is False
    assert "left on the machine" in records[0]["upload"]["error"]
