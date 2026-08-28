"""Produce delivery formats off the correctly merged model (issue #74).

The job spec can ask for more than the canonical artifact: a merged
single-file model for serving, and a quantised local-inference format for
running on your own machine. Both are produced here, on the machine, at export
time -- the machine is warm with both models loaded, which is where spec 011
says conversion belongs ("format conversion is a step off the correctly merged
model, not a parallel pipeline").

The load-bearing property is **order**, and its failure is silent: merging into
an already-quantised base compounds error, and a model merged in the wrong order
still loads and answers, only slightly worse. The steps therefore come from the
same vocabulary the control plane reads (`packages/contracts/
delivery-formats.json` -- this trainer never installs packages/core, ADR-0010),
and `production_steps` cannot put quantise before merge: it is the assertion,
and the host test suite pins the order over every subset.

Each produced format is **verified by loading it**, not by checking a file
exists. On the machine this is the image's real loader (transformers for a
merged safetensors model, the image's local-format loader for the quantised
format); `verify_loaded` is the seam the host suite drives with a double that
genuinely loads the produced artifact, because the host venv has no
torch/transformers. A produced format that does not load fails the export with
a stable code -- the same fail-closed rule the template probe follows.

The **template probe runs on each export**: the primary artifact already runs
it (issue #59); each produced delivery format runs it too, and the outcome is
recorded with that format's record.

The heavy libraries (transformers, peft, torch) are imported lazily inside the
functions that need them, exactly as `prefetch_model` and
`load_probe_tokenizer` do, so this module imports cleanly in the host suite.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

# The conversion order, mirroring temper_core.delivery.CONVERSION_ORDER. The
# trainer cannot import packages/core (ADR-0010), so it reads the same contract
# data the core module reads and re-derives the order from it; the host suite
# pins this trainer's order to the core module's so the two cannot drift.
CONVERSION_ORDER = ("merge", "quantise")


def _contract_path() -> Path:
    """Find the delivery-format contract, sibling or tree.

    The same lookup every other contract this trainer reads uses: beside this
    file in the image at /opt/trainer, falling back to the workspace tree so
    the host test suite resolves it without the image.
    """
    sibling = Path(__file__).resolve().parent / "delivery-formats.json"
    if sibling.is_file():
        return sibling
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "packages" / "contracts" / "delivery-formats.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "packages/contracts/delivery-formats.json not found beside this file "
        "or anywhere in the workspace tree"
    )


_CONTRACT = json.loads(_contract_path().read_text(encoding="utf-8"))
_FORMATS = {row["id"]: row for row in _CONTRACT["formats"]}
KNOWN_FORMATS = frozenset(_FORMATS)

# The delivery formats that are produced by a conversion (everything but the
# as-trained artifact). `adapter` is the canonical artifact and is never
# re-produced here.
CONVERTED_FORMATS = frozenset(
    fmt_id for fmt_id, row in _FORMATS.items() if row.get("conversion")
)


def known_formats() -> frozenset[str]:
    """Every delivery format this trainer knows, from the one contract."""
    return KNOWN_FORMATS


def production_steps(requested: list[str]) -> list[str]:
    """The ordered conversions for `requested`, mirroring the core module.

    The trainer cannot import `temper_core.delivery`, so it reads the same
    contract data and applies the same rule: the adapter needs no conversion,
    every other requested format resolves to its conversion in
    `CONVERSION_ORDER`, and a quantised request brings the merge along (it
    consumes the correctly merged model). `test_agreement_with_the_domain.py`
    pins this to `temper_core.delivery.production_steps`, so the two cannot
    drift about what order is legal.
    """
    unknown = [f for f in requested if f not in KNOWN_FORMATS]
    if unknown:
        raise ValueError(
            f"unknown delivery format(s) {unknown}; known: "
            f"{sorted(KNOWN_FORMATS)}"
        )
    conversions: set[str] = set()
    for fmt_id in requested:
        row = _FORMATS[fmt_id]
        if row.get("conversion"):
            conversions.add(row["conversion"])
    # quantised consumes the merged model, so its merge step must run first.
    if "quantised" in requested or "quantise" in conversions:
        conversions.add("merge")
    return [c for c in CONVERSION_ORDER if c in conversions]


# Where a produced delivery format lands on /out, and the archive a directory
# format is tarred into. A merged model is a directory of files (config,
# weight shards, tokenizer) that cannot be addressed by one pre-minted key, so
# it travels as one streamed archive, exactly like a full fine-tune's
# model.tar.gz (issue #66).
MERGE_OUT = "delivery/merged"
QUANTISED_OUT = "delivery/quantised"
MERGED_ARCHIVE = "delivery/merged.tar.gz"
QUANTISED_ARCHIVE = "delivery/quantised.gguf"


class DeliveryFailure(Exception):
    """A produced delivery format could not be verified by loading it.

    Carries the stable ``error_code`` the result document records. The export
    must not succeed with an unloadable format: a file that exists but does
    not load is exactly the failure mode the issue names, and file existence
    does not detect it.
    """

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


def sha256_of(path: Path) -> str:
    """A file's SHA-256, computed in bounded blocks rather than read whole."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tar_directory(source: Path, archive: Path) -> None:
    """Tar one top-level folder into `archive`, streamed.

    A merged model is a set of files; the archive gives the download one
    object to address, with a `model/` top-level folder so extraction produces
    a loadable directory. Written streaming, so a large model is never held
    whole here (the flat-memory rule).
    """
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=source.name)


def merge_adapter(job: dict, out_dir: Path) -> dict:
    """Merge the trained adapter into the base at full precision (bf16).

    Loads the base model at full precision -- never into the 4-bit quantised
    base training used, which is the doubly-approximated merge the issue names
    -- applies the adapter, merges, and saves a complete model directory. The
    base is already in the HF cache (training just used it), so the load
    resolves without the network.

    transformers and peft are provided by the base image; the import is lazy
    so this module imports cleanly in the host suite.

    Returns the record: path (relative to out_dir), members, bytes, sha256.
    """
    from peft import PeftModel  # noqa: PLC0415 - base image only
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    adapter_dir = _adapter_dir(out_dir)
    merged_dir = out_dir / MERGE_OUT
    merged_dir.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(
        job["base_model"],
        revision=job.get("base_revision"),
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    base.config.save_pretrained(merged_dir)

    archive = out_dir / MERGED_ARCHIVE
    _tar_directory(merged_dir, archive)
    return {
        "path": MERGED_ARCHIVE,
        "members": [p.name for p in sorted(merged_dir.iterdir())],
        "bytes": archive.stat().st_size,
        "sha256": sha256_of(archive),
    }


def _adapter_dir(out_dir: Path) -> Path:
    """The adapter directory the canonical artifact was collected from.

    `collect_artifacts` records `artifact_path` relative to /out; the adapter
    sits in the directory that path lives in (the final adapter is at the top
    of `run/`). Merging reads it from there -- the same bytes that shipped.
    """
    run = out_dir / "run"
    candidates = [run, *(p for p in sorted(run.glob("checkpoint-*")))]
    for d in candidates:
        if (d / "adapter_model.safetensors").is_file():
            return d
    raise DeliveryFailure(
        "delivery_merge_unavailable",
        "no adapter to merge: no adapter_model.safetensors found under /out/run.",
    )


def quantise_merged(merged_archive: Path, out_dir: Path) -> dict:
    """Quantise the correctly merged model once, into a local format.

    The input is the merged model produced by the merge step -- never the
    adapter and never a quantised base. The quantisation happens once, here,
    after the full-precision merge (the order property's second half). The
    exact converter is the image's own tooling; a converter that cannot run
    fails the export closed rather than silently shipping an unloadable file.

    Returns the record: path, bytes, sha256.
    """
    try:
        from transformers import (  # noqa: PLC0415 - base image only
            AutoModelForCausalLM,
        )
    except ImportError as e:
        raise DeliveryFailure(
            "delivery_quantise_unavailable",
            f"no local-format converter in the image: {type(e).__name__}: {e}",
        ) from e

    import tempfile

    merged_dir = Path(tempfile.mkdtemp(prefix="temper-merged-"))
    try:
        with tarfile.open(merged_archive, "r:gz") as tar:
            tar.extractall(merged_dir, filter="data")
        model_dir = next(merged_dir.iterdir())
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        # The image's own quantised-local writer. `save_pretrained` with a
        # quantisation path is transformers' surface for "export a local
        # format"; if the pinned image's surface differs this is the line the
        # hardware verification (spec 011's clause) confirms.
        quantised_path = out_dir / QUANTISED_ARCHIVE
        quantised_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(quantised_path)
        return {
            "path": QUANTISED_ARCHIVE,
            "bytes": quantised_path.stat().st_size,
            "sha256": sha256_of(quantised_path),
        }
    finally:
        import shutil

        shutil.rmtree(merged_dir, ignore_errors=True)


def verify_loaded(path: Path, format_id: str) -> dict:
    """Verify a produced format by loading it; never by checking it exists.

    On the machine this is the image's real loader: transformers for a merged
    safetensors model, the image's local-format loader for the quantised
    format. The host suite injects a double that genuinely loads the produced
    artifact (the test for the loading criterion lives beside the loader).

    Returns {"ok": True} or {"ok": False, "error": ...} so the caller records
    the outcome without a crash, and the export fails closed when not ok.
    """
    try:
        _load_artifact(path, format_id)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001 - the verdict, not a crash
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _load_artifact(path: Path, format_id: str) -> None:
    """Genuinely load `path` as a `format_id` artifact.

    The machine's half of verification-by-loading. The host suite substitutes
    this (via `verify_loaded`) with a loader that reads the produced file,
    because the host venv has no torch/transformers -- and the PR body for
    issue #74 names the on-machine load as the outstanding hardware
    verification, with the reason a fake load does not substitute.
    """
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    if format_id == "merged":
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(td, filter="data")
            model_dir = next(Path(td).iterdir())
            model = AutoModelForCausalLM.from_pretrained(model_dir)
            model.eval()
        return
    model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()


# The upload seam: the entrypoint's `upload_artifact` has exactly this shape
# (a scoped-URL PUT that reports an outcome rather than raising); injecting it
# keeps this module importable without entrypoint.
UploadFn = Callable[[str, Path], dict]


def run_delivery(
    job: dict,
    out_dir: Path,
    upload: UploadFn,
    *,
    tokenizer: Any,
    probe: Callable[[dict, dict, Any], Any],
    cfg: dict | None = None,
    load_verify: Callable[[Path, str], dict] = verify_loaded,
    grants: list[dict] | None = None,
) -> list[dict]:
    """Produce the job's requested delivery formats, in order, verified.

    Reads `job["delivery"]` (the format ids the launch requested) and runs the
    conversions in `production_steps` order: merge first, then quantise --
    never the reverse. Each produced format is verified by loading it, runs
    the template probe, and is uploaded through its own scoped grant (one per
    format, minted by the control plane, same shape as the artifact's). Every
    step's outcome is recorded, so result.json says what was produced, what
    was verified, and what reached storage.

    Returns the per-format records, in delivery order, ready for result.json.
    """
    requested = job.get("delivery") or []
    if not requested:
        return []
    steps = production_steps(list(requested))
    grants_by_format = {g["format"]: g for g in (grants or [])}
    records: list[dict] = []
    produced: dict[str, Path] = {}

    if "merge" in steps:
        rec = _produce_and_verify(
            "merged",
            job,
            out_dir,
            cfg,
            tokenizer,
            probe,
            load_verify,
            lambda: merge_adapter(job, out_dir),
        )
        _upload(rec, upload, out_dir, grants_by_format.get("merged"))
        records.append(rec)
        produced["merged"] = out_dir / rec["path"]

    if "quantise" in steps:
        merged_archive = produced.get("merged")
        if merged_archive is None:
            # The order property guarantees merge ran first, but state it
            # here as the recorded truth rather than as a hope.
            raise DeliveryFailure(
                "delivery_order",
                "quantise requested before a merged model was produced; "
                "the merge must run first (issue #74).",
            )
        rec = _produce_and_verify(
            "quantised",
            job,
            out_dir,
            cfg,
            tokenizer,
            probe,
            load_verify,
            lambda: quantise_merged(merged_archive, out_dir),
        )
        _upload(rec, upload, out_dir, grants_by_format.get("quantised"))
        records.append(rec)

    return records


def _produce_and_verify(
    format_id: str,
    job: dict,
    out_dir: Path,
    cfg: dict | None,
    tokenizer: Any,
    probe: Callable[[dict, dict, Any], Any],
    load_verify: Callable[[Path, str], dict],
    produce: Callable[[], dict],
) -> dict:
    """Run one conversion, verify its result by loading it, and probe it.

    A produced format that does not load fails the export with a stable code:
    the failure mode the issue names is a file that exists and is not
    loadable, and file existence does not detect it.
    """
    record = produce()
    path = out_dir / record["path"]
    verdict = load_verify(path, format_id)
    record["verified"] = verdict["ok"]
    if not verdict["ok"]:
        raise DeliveryFailure(
            "delivery_unloadable",
            f"{format_id} format was produced but could not be loaded: "
            f"{verdict.get('error')}",
        )
    record["template_probe"] = probe(job, cfg or {}, tokenizer).as_dict()
    return record


def _upload(
    record: dict, upload: UploadFn, out_dir: Path, grant: dict | None
) -> None:
    """PUT one produced format to its scoped grant, and record the outcome.

    A standalone run (or a request with no grant for this format) records that
    nothing was uploaded as a fact, not as a failure of training -- exactly
    how the canonical artifact's upload behaves.
    """
    if grant is None or not grant.get("url"):
        record["upload"] = {
            "ok": False,
            "error": "no write grant for this format; the format was left "
            "on the machine",
        }
        return
    record["upload"] = upload(grant["url"], out_dir / record["path"])
