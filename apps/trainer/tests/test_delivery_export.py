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
import struct
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
    For the quantised format it reads the GGUF file header: a real GGUF begins
    with the 4-byte magic 0x46554747, so a file that does not carry it is not
    the format its name claims. This is the criterion's host-side evidence:
    the file is loaded (its structure read and validated), not merely asserted
    to exist.
    """
    import struct
    import tarfile

    if format_id == "merged":
        with tarfile.open(path, "r:gz") as tar:
            names = tar.getnames()
        assert any(n.startswith("model/") for n in names)
        assert "model/config.json" in names
        return {"ok": True}
    with path.open("rb") as f:
        magic = f.read(4)
    if len(magic) != 4 or struct.unpack("<I", magic)[0] != 0x46554747:
        return {"ok": False, "error": "not a GGUF file (bad magic)"}
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


def test_a_quantised_file_that_is_not_a_real_gguf_fails_the_export(
    tmp_path, monkeypatch
):
    """THE criterion for the local format: a file that exists but is not the
    format its name claims is the silent failure the issue names. A produced
    'quantised.gguf' that does not carry the GGUF magic must fail the export,
    because the loader genuinely reads the header rather than checking size."""
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

    def fake_quantise(merged_archive, out_dir):
        q = out / "delivery" / "quantised.gguf"
        # A non-empty file with the wrong magic: file existence does not
        # detect this, the loader does.
        q.write_bytes(b"not a gguf at all")
        return {
            "path": "delivery/quantised.gguf",
            "bytes": 18,
            "sha256": "y" * 64,
        }

    monkeypatch.setattr(delivery, "merge_adapter", fake_merge)
    monkeypatch.setattr(delivery, "quantise_merged", fake_quantise)
    job = {"delivery": ["quantised"]}
    with pytest.raises(DeliveryFailure) as excinfo:
        delivery.run_delivery(
            job,
            out,
            lambda url, path: {"ok": True},
            tokenizer=None,
            probe=lambda job, cfg, tokenizer: _ok_outcome(),
            load_verify=_loading_loader,
            grants=[{"format": "quantised", "url": "u://g"}],
        )
    assert excinfo.value.error_code == "delivery_unloadable"


def test_quantise_fails_closed_when_the_image_has_no_converter(
    tmp_path, monkeypatch
):
    """The quantised local format is produced by the image's own GGUF tooling.
    A converter that is not present fails the export closed rather than
    shipping a file that is not the format its name claims."""
    out = tmp_path / "out"
    merged_archive = out / "delivery" / "merged.tar.gz"
    _write_fake_merged_archive(merged_archive)
    monkeypatch.setattr(delivery.shutil, "which", lambda name: None)
    with pytest.raises(DeliveryFailure) as excinfo:
        delivery.quantise_merged(merged_archive, out)
    assert excinfo.value.error_code == "delivery_quantise_unavailable"


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
        # A real GGUF begins with the 4-byte magic, so the host loader can
        # genuinely load it (read and validate the header), not merely see
        # that a file exists.
        q.write_bytes(struct.pack("<I", 0x46554747) + b"rest")
        return {
            "path": "delivery/quantised.gguf",
            "bytes": 8,
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


# --- the heartbeat: a slow-but-healthy export must not read as a stall ------


def test_a_slow_merge_and_upload_each_emit_a_heartbeat(tmp_path, monkeypatch):
    """Found on real hardware (2026-09-06): a 7B model's merge and upload
    each ran silent long enough to trip the control plane's 900s stall
    detector, which destroyed a healthy machine mid-export. The heartbeat
    must fire during both the conversion and the upload, or a large model's
    export is indistinguishable from a wedged one."""
    import time

    monkeypatch.setattr(delivery, "HEARTBEAT_INTERVAL_S", 0.02)
    out = tmp_path / "out"
    merged_archive = out / "delivery" / "merged.tar.gz"
    _write_fake_merged_archive(merged_archive)

    def slow_merge(job, out_dir):
        time.sleep(0.1)
        return {
            "path": "delivery/merged.tar.gz",
            "members": ["config.json"],
            "bytes": merged_archive.stat().st_size,
            "sha256": delivery.sha256_of(merged_archive),
        }

    def slow_upload(url, path):
        time.sleep(0.1)
        return {"ok": True}

    monkeypatch.setattr(delivery, "merge_adapter", slow_merge)
    messages: list[str] = []
    job = {"delivery": ["merged"]}
    delivery.run_delivery(
        job,
        out,
        slow_upload,
        tokenizer=None,
        probe=lambda job, cfg, tokenizer: _ok_outcome(),
        load_verify=_loading_loader,
        grants=[{"format": "merged", "url": "u://g"}],
        log=messages.append,
    )
    assert any("producing merged" in m for m in messages)
    assert any("uploading merged" in m for m in messages)


def test_no_heartbeat_fires_when_no_log_is_given(tmp_path, monkeypatch):
    """A standalone run passes no `log`; the heartbeat must be a no-op, not
    an error, so `run_delivery`'s existing callers are unaffected."""
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
    assert records[0]["verified"] is True
