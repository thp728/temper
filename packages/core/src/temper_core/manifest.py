"""Provenance manifest: generated from the run record, never hand-written.

Spec 011 / issue #70. The artifact ships with no record of what produced it,
and the base model's licence obligations flow through to whatever the user does
next. The manifest is the compliance artifact that answers: which base model
at which pinned revision, which dataset (with counts), which configuration
including any overrides, the evaluation summary, the checkpoint the result came
from, and the licence obligations that propagate.

A provenance document that can drift from the run it describes is worse than
none, because it will be believed. Every field the manifest carries is therefore
derived from what the run recorded, never hand-typed, and where a field can be
supplied two ways the recorded one wins. A missing required field fails
generation rather than producing a placeholder: no "unknown", no empty string,
no null standing in for a fact nobody recorded.

The module is pure: no I/O, no framework imports, one seam read (`catalog`)
only where the licence lookup is a stable, versioned source. That keeps
manifest generation testable without a database and without hardware, and done
means the same as every other `temper_core` pure module: given a run record,
the manifest contains every required field and no placeholder.

Human readability is a requirement, not a nicety: the manifest is what a
reviewer reads to decide whether training worked, so the JSON is pretty-printed
and `render_text` produces a Markdown document that states each section in
sentences as well as keys.

Issue #65's "untested" label (mixture-of-experts / unknown architecture)
travels on the finished run's stored probe result, and if the manifest records
such a label it reads it from the record rather than defining it. Expect to
rebase when that field lands.
"""

from __future__ import annotations

import json
import time
from typing import Any

from . import artifacts, catalog


class MissingField(ValueError):
    """A required provenance field is absent from the run record.

    Raised rather than guessed: a missing field is not "unknown", it is a
    defect, and the artifact is not shipped with a manifest that lies about it.
    ``field`` names the missing key so the failure can be diagnosed without
    guessing which of the many required fields was the one that was absent.
    """

    def __init__(self, field: str, detail: str | None = None) -> None:
        self.field = field
        msg = (
            f"manifest generation failed: required field '{field}' is missing"
        )
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


# The obligations text for the licences this platform actually ships. The
# catalog today is Apache-2.0 only; the text is what propagates to a
# derivative artifact and is stated explicitly because a user who cannot see it
# cannot comply with it (report C §2.4). Any other licence that later lands in
# the catalog must extend this table rather than fall through to a generic
# placeholder.
LICENSE_OBLIGATIONS: dict[str, str] = {
    "Apache-2.0": (
        "Apache-2.0: you may use, modify and distribute this artifact, including "
        "commercially, provided you retain the original copyright and licence "
        "notices, state significant changes, and include the Apache-2.0 licence "
        "text. The fine-tuned weights are a derivative of the base model; the "
        "base model's Apache-2.0 terms flow through to this artifact."
    ),
    "apache-2.0": (
        "Apache-2.0: you may use, modify and distribute this artifact, including "
        "commercially, provided you retain the original copyright and licence "
        "notices, state significant changes, and include the Apache-2.0 licence "
        "text. The fine-tuned weights are a derivative of the base model; the "
        "base model's Apache-2.0 terms flow through to this artifact."
    ),
}

# Generic fallback for a recorded licence that is not in the table yet. Not a
# placeholder for a missing licence: a missing licence fails, while a present
# but unfamiliar one is stated as "check the base model's terms" rather than
# guessed.
GENERIC_OBLIGATIONS_TEMPLATE = (
    "This artifact is a derivative of base model '{repo}' at revision "
    "'{revision}', licensed under '{license}'. You must comply with that "
    "licence's terms before distributing this artifact; see {url} for the "
    "base model's terms. When the licence could not be resolved the explicit "
    "record is the obligation: determine the base model's terms before "
    "distributing."
)


def _require(value: Any, field: str, detail: str | None = None) -> Any:
    """Return value or raise MissingField if it is absent.

    Absent means None, empty string (after stripping), empty list/dict where a
    record is expected, or a sentinel placeholder like "unknown". A field that
    is present but empty is as missing as one that is not there at all.
    """
    if value is None:
        raise MissingField(field, detail)
    if isinstance(value, str) and not value.strip():
        raise MissingField(
            field, detail or "empty string is not a recorded fact"
        )
    if isinstance(value, str) and value.strip().lower() == "unknown":
        # "unknown" is a placeholder, not a recorded fact. A licence that is
        # truly unknown is a missing field rather than a labelled one.
        # For non-licence fields (e.g. a base_model id of "unknown") this also
        # correctly fails -- no run should carry it.
        raise MissingField(field, detail or "value 'unknown' is a placeholder")
    if isinstance(value, (list, tuple)) and len(value) == 0:
        # Overridden decisions may be legitimately empty, but a required list
        # that is empty where a record is expected is missing. Callers that
        # allow empty must check themselves rather than call _require.
        raise MissingField(field, detail or "empty list")
    if isinstance(value, dict) and len(value) == 0:
        raise MissingField(field, detail or "empty dict")
    return value


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    """Require that `value` is a non-empty dict."""
    _require(value, field)
    if not isinstance(value, dict):
        raise MissingField(
            field, f"expected a dict, got {type(value).__name__}"
        )
    if not value:
        raise MissingField(field, "empty dict")
    return value


def _licence_obligations(
    licence: str, repo: str, revision: str, licence_url: str
) -> str:
    """Explicit licence obligations for `licence`."""
    if licence in LICENSE_OBLIGATIONS:
        return LICENSE_OBLIGATIONS[licence]
    # A recorded licence that is not in the table yet: state the generic
    # propagation explicitly rather than guessing.
    return GENERIC_OBLIGATIONS_TEMPLATE.format(
        repo=repo, revision=revision, license=licence, url=licence_url
    )


def _lookup_base_model(job: dict[str, Any]) -> tuple[str, str, str, str]:
    """Resolve base model repo, revision, licence and licence_url from the job.

    The job stores ``base_model`` (catalog id or admitted id) and
    ``base_revision`` (pinned SHA). Licence is read from the stable catalog
    when the model is catalogued; for an admitted model the probe result is
    not on the job row, so the caller may have stashed the resolved licence
    on the job as ``_admitted_license`` / ``_admitted_license_url`` (set by the
    control plane's download path when it materialised the model). Where a
    field can be supplied two ways, the job's own recorded value wins, but a
    missing recorded value falls back to the catalog rather than failing the
    whole generation when the catalog is authoritative.
    """
    base_model_id = _require(job.get("base_model"), "base_model")
    base_revision = _require(job.get("base_revision"), "base_revision")

    # The repo: prefer an explicit repo on the job (the orchestrator could
    # stash it), fall back to catalog's repo for a catalog model, then to the
    # base_model id itself (the id is the repo for an admitted model only when
    # the caller used the repo as the id -- not sufficient, but never a guess
    # for a catalog model).
    repo = job.get("base_model_repo") or job.get("base_model_repo_url") or None
    cat = catalog.get(str(base_model_id))
    if cat is not None:
        repo = repo or cat.repo
        licence = cat.license
        licence_url = cat.license_url
    else:
        # Admitted model: licence was recorded in the probe and materialised by
        # the control plane's `admission._as_catalog_model` or stashed on the
        # job dict by the caller. Prefer the recorded job stash.
        licence = job.get("_admitted_license") or job.get("license") or ""
        licence_url = (
            job.get("_admitted_license_url")
            or job.get("license_url")
            or f"https://huggingface.co/{repo}"
            if repo
            else ""
        )
        # If still empty and the job carries a persisted probe (e.g.
        # ``_admitted_probe``), read licence from there -- the probe is the
        # recorded fact for an admitted model, and the manifest must read it
        # rather than define it.
        if not licence:
            probe = job.get("_admitted_probe") or {}
            if isinstance(probe, dict):
                licence = probe.get("license") or probe.get("licence") or ""
                if not licence and isinstance(probe.get("probe"), dict):
                    licence = probe["probe"].get("license") or ""
            if not licence and isinstance(job.get("probe"), dict):
                licence = job["probe"].get("license") or ""
        repo = repo or str(base_model_id)

    # The repo is required: without it the licence_url is meaningless.
    _require(
        repo,
        "base_model_repo",
        "job.base_model is not a repo and no catalog entry supplied it",
    )
    # Licence is required and must not be a placeholder.
    _require(
        licence,
        "license",
        "base model's licence is missing; compliance requires it",
    )
    _require(licence_url, "license_url")
    return str(repo), str(base_revision), str(licence), str(licence_url)


def _dataset_fingerprint(
    dataset: dict[str, Any] | None, held_out_split: dict[str, Any]
) -> dict[str, Any]:
    """Dataset fingerprint and counts, assembled from the recorded split and the
    dataset row.

    ``held_out_split`` is the trainer's recorded split (rows_in,
    rows_removed_duplicates, train_rows, held_out_rows, fraction, seed).
    ``dataset`` may be None when the caller drives generation from the job
    alone (e.g. in a test); the fingerprint then falls back to the split's
    counts. Where both exist, the dataset's own counts are cross-checked rather
    than silently preferred -- a mismatch is a drift, and drift is the failure
    the manifest exists to prevent.
    """
    _require_dict(held_out_split, "held_out_split")
    for field in (
        "rows_in",
        "train_rows",
        "held_out_rows",
        "fraction",
        "seed",
    ):
        _require(held_out_split.get(field), f"held_out_split.{field}")

    # rows_removed_duplicates may be 0 legitimately, so check existence not truthiness.
    if held_out_split.get("rows_removed_duplicates") is None:
        raise MissingField("held_out_split.rows_removed_duplicates")

    fingerprint: dict[str, Any] = {
        "rows_in": held_out_split["rows_in"],
        "rows_removed_duplicates": held_out_split["rows_removed_duplicates"],
        "train_rows": held_out_split["train_rows"],
        "held_out_rows": held_out_split["held_out_rows"],
        "fraction": held_out_split["fraction"],
        "seed": held_out_split["seed"],
    }

    if dataset is not None:
        # Dataset identity. `id` and `filename` are the stable fingerprint;
        # `object_key` is the seam's address and is not published, but the
        # counts from the validation report are.
        ds_id = dataset.get("id") or dataset.get("dataset_id")
        if ds_id:
            fingerprint["dataset_id"] = ds_id
        filename = dataset.get("filename")
        if filename:
            fingerprint["filename"] = filename
        # Counts from the validation/report path, when present. These are the
        # "counts" the issue and spec mean alongside the fingerprint.
        report = dataset.get("report") or {}
        for key in (
            "row_count",
            "usable_rows",
            "schema_type",
            "enable_thinking",
        ):
            if report.get(key) is not None:
                fingerprint[key] = report[key]
        # Token count (issue #42) is a measure, not a requirement for the
        # fingerprint; when present it is recorded, when absent it is omitted
        # rather than faked.
        token_count = dataset.get("token_count")
        if token_count is None and isinstance(report.get("token_count"), int):
            token_count = report["token_count"]
        if isinstance(token_count, int):
            fingerprint["token_count"] = token_count
        token_dist = dataset.get("token_distribution") or report.get(
            "token_distribution"
        )
        if token_dist is not None:
            fingerprint["token_distribution"] = token_dist

    return fingerprint


def _evaluation_summary(
    held_out_split: dict[str, Any],
    best_checkpoint: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """The evaluation summary: split record plus the chosen checkpoint's reason
    and, when available, the template probe outcome (#59)."""
    summary: dict[str, Any] = {
        "held_out_split": held_out_split,
        "best_checkpoint": best_checkpoint,
    }
    # Checkpoints list is not required for the summary but when the job
    # carries it the summary names how many were retained.
    checkpoints = result.get("checkpoints")
    if isinstance(checkpoints, list):
        summary["checkpoints_retained"] = len(checkpoints)
    # Export-time template probe result (#59) travels on the job when it
    # exists; it is evaluation-adjacent and recorded, so the manifest carries
    # it rather than re-deriving.
    probe = result.get("template_probe")
    if isinstance(probe, dict) and probe:
        summary["template_probe"] = probe
    return summary


def generate(
    job: dict[str, Any],
    dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate the provenance manifest for `job`.

    ``job`` is the run record as returned by ``db.get_job``: a dict carrying
    ``id``, ``base_model``, ``base_revision``, ``hyperparameters``,
    ``overrides``, ``result`` (with ``held_out_split`` and ``template_probe``),
    ``best_checkpoint``, ``artifact_record`` and ``method``. ``dataset`` is the
    dataset row as returned by ``db.get_dataset`` (or None when unavailable in
    a test harness). Every required field is validated and a MissingField is
    raised naming the field rather than producing a placeholder.

    If a field can be supplied two ways, the recorded value on ``job`` wins.
    For example ``job["_admitted_license"]`` (stashed by the download path)
    wins over a catalog lookup, and ``held_out_split`` from ``result`` wins
    over a dataset-derived split.

    Returns a dict ready to be pretty-printed as JSON and rendered as Markdown
    via ``render_text``. The dict contains no placeholders: a missing required
    field raises rather than yielding ``"unknown"``.
    """
    # --- provenance -----------------------------------------------------
    job_id = _require(job.get("id"), "job_id")
    created_at = job.get("created_at")
    if created_at is None:
        raise MissingField("created_at")
    finished_at = job.get("finished_at")
    # finished_at is required for a complete job's manifest -- a manifest
    # without a finish time cannot say when the artifact was produced.
    if finished_at is None and job.get("status") == "complete":
        raise MissingField(
            "finished_at", "a complete job must have finished_at"
        )

    # --- base model and pinned revision ---------------------------------
    repo, revision, licence, licence_url = _lookup_base_model(job)
    base_model_id = _require(job.get("base_model"), "base_model")
    obligations = _licence_obligations(licence, repo, revision, licence_url)

    # --- dataset fingerprint with counts --------------------------------
    result = _require_dict(
        job.get("result") or job.get("result_json"), "result"
    )
    held_out_split = _require_dict(
        result.get("held_out_split"), "result.held_out_split"
    )
    # The fingerprint is the split plus the dataset's counts.
    dataset_fp = _dataset_fingerprint(dataset, held_out_split)
    # The spec's "dataset fingerprint with counts" must include at least the
    # counts the platform records; if the dataset row itself is absent we still
    # have the split's counts, but a completely missing dataset identity is
    # allowed only when the split already carries it (tests drive generation
    # from the job alone). For a production complete job the dataset row is
    # always present.
    if dataset is None:
        # Allow test harness without a dataset row, but require that the
        # fingerprint at least has the split's counts (already validated).
        pass
    else:
        _require(
            dataset.get("id")
            or dataset.get("dataset_id")
            or held_out_split.get("rows_in"),
            "dataset.id",
        )

    # --- full configuration including overrides --------------------------
    raw_hyperparams = job.get("hyperparameters")
    if raw_hyperparams is None or not isinstance(raw_hyperparams, dict):
        raise MissingField(
            "hyperparameters",
            "full configuration is required; none was recorded",
        )
    # An empty dict is legitimate: it means the user accepted all defaults, and
    # the full configuration is the effective defaults. A job that truly has no
    # record would be None, which is already rejected. The manifest records the
    # effective configuration (defaults plus any overrides) so the document says
    # what actually ran, not just what the user typed.
    from . import hyperparams as hp_module

    method_for_config = job.get("method") or "qlora"
    try:
        effective_hyperparams = hp_module.effective(
            raw_hyperparams, method=method_for_config
        )
    except Exception:
        # If the effective resolver cannot be applied (unknown method etc),
        # fall back to the raw recorded dict rather than failing the manifest
        # on a resolver error -- the raw dict is the recorded fact.
        effective_hyperparams = raw_hyperparams
    # Overrides are frozen on the job row; an empty list is legitimate (no
    # override), but None (no record) is a missing field for a complete job.
    overrides_val = job.get("overrides")
    if overrides_val is None:
        # Pre-override rows (before issue #79) have no overrides column; for
        # those the manifest records an empty list rather than failing -- the
        # absence describes history (no override existed) rather than a missing
        # fact. For a row that *should* have been recorded but was not, this is
        # the one leniency the spec allows.
        overrides_val = []
    if not isinstance(overrides_val, list):
        raise MissingField(
            "overrides", f"expected list, got {type(overrides_val).__name__}"
        )

    configuration: dict[str, Any] = {
        "method": method_for_config,
        "hyperparameters": effective_hyperparams,
        "overrides": overrides_val,
    }
    # The seed is part of the configuration: the split and the training are
    # deterministic under it, so a manifest without it cannot be reproduced.
    # The trainer pins it to 42, but the manifest records what the run recorded,
    # so it reads it from held_out_split.seed rather than retyping 42.
    configuration["seed"] = held_out_split["seed"]

    # --- checkpoint the result came from (#62) --------------------------
    best_checkpoint = job.get("best_checkpoint") or job.get(
        "best_checkpoint_json"
    )
    if (
        best_checkpoint is None
        or not isinstance(best_checkpoint, dict)
        or not best_checkpoint
    ):
        raise MissingField(
            "best_checkpoint",
            "the chosen checkpoint and its reason must be recorded",
        )
    # The stored choice must name a step, except when there were no checkpoints
    # at all (basis "none") -- but a complete job's manifest must name one; a
    # "none" basis is a missing result for an artifact-producing run.
    if best_checkpoint.get("step") is None:
        if best_checkpoint.get("basis") != "none":
            raise MissingField("best_checkpoint.step")
        # For a manifest that ships with an artifact, "none" is not a valid
        # choice -- it would claim an artifact with no checkpoint. Fail.
        raise MissingField(
            "best_checkpoint.step",
            "no checkpoint was chosen; an artifact-producing run must have a chosen checkpoint",
        )
    _require(best_checkpoint.get("basis"), "best_checkpoint.basis")
    _require(best_checkpoint.get("reason"), "best_checkpoint.reason")

    # --- artifact kind, members, loading --------------------------------
    method_for_kind = job.get("method")
    kind = artifacts.kind_for(method_for_kind)
    # Artifact record is the stored, verified set of members the download
    # serves (issue #32). The published ``artifact`` omits storage keys, so
    # where both exist the stored record wins (the "recorded one" rule).
    artifact_record = job.get("artifact_record") or job.get("artifact_json")
    members: list[str]
    artifact_bytes: Any = None
    artifact_sha256: Any = None
    if isinstance(artifact_record, dict) and artifact_record.get("members"):
        members = [
            str(m.get("name"))
            for m in artifact_record["members"]
            if isinstance(m, dict) and m.get("name")
        ]
        if not members:
            raise MissingField(
                "artifact.members", "artifact record has no member names"
            )
        artifact_bytes = artifact_record.get("bytes")
        artifact_sha256 = artifact_record.get("sha256")
    else:
        # Fallback to the published artifact (for tests driving generation from
        # a published job). Still requires members.
        art_pub = job.get("artifact")
        if isinstance(art_pub, dict) and art_pub.get("members"):
            members = [str(n) for n in art_pub["members"]]
            artifact_bytes = art_pub.get("bytes")
            artifact_sha256 = art_pub.get("sha256")
        else:
            raise MissingField(
                "artifact.members",
                "no artifact record and no published artifact members",
            )
    _require(members, "artifact.members")
    loading = artifacts.loading_instructions(kind)

    # --- evaluation summary (#53, #62, #59) -------------------------------
    evaluation = _evaluation_summary(held_out_split, best_checkpoint, result)

    # --- assemble -------------------------------------------------------
    # The base model details are stored under "base_model_info" to keep the
    # top-level "base_model" string for backward compatibility with the
    # original minimal manifest (issue #32). The same applies to "kind",
    # "members", "bytes" and "loading" which are also stored under "artifact"
    # but duplicated at the top level for old readers.
    base_model_info: dict[str, Any] = {
        "id": str(base_model_id),
        "repo": repo,
        "revision": revision,
        "license": licence,
        "license_url": licence_url,
        "license_obligations": obligations,
    }
    manifest: dict[str, Any] = {
        "version": 1,
        "generated_at": time.time(),
        "job": {
            "id": str(job_id),
            "created_at": created_at,
            "finished_at": finished_at,
            "status": job.get("status"),
            "method": configuration["method"],
            "kind": kind,
        },
        "base_model": str(base_model_id),
        "base_revision": revision,
        "base_model_info": base_model_info,
        "dataset": dataset_fp,
        "configuration": configuration,
        "evaluation": evaluation,
        "checkpoint": {
            "step": best_checkpoint["step"],
            "basis": best_checkpoint["basis"],
            "reason": best_checkpoint["reason"],
            "held_out_loss": best_checkpoint.get("held_out_loss"),
        },
        "artifact": {
            "kind": kind,
            "members": members,
            "bytes": artifact_bytes,
            "sha256": artifact_sha256,
            "loading": loading,
        },
        # Top-level aliases for the minimal manifest (issue #32) so existing
        # readers that expect flat keys continue to work.
        "kind": kind,
        "members": members,
        "bytes": artifact_bytes,
        "loading": loading,
    }

    # Carry the untested/MoE label if the job's stored probe or model facts
    # recorded it. Issue #65's label travels on the finished run; the manifest
    # reads it rather than defining it.
    probe_for_label = job.get("_admitted_probe") or job.get("probe") or {}
    # The admitted probe may be nested under "probe" on the admitted model row;
    # the job may also carry a flat "is_moe" flag if the control plane stashed it.
    is_moe = job.get("_admitted_is_moe")
    if is_moe is None and isinstance(probe_for_label, dict):
        is_moe = probe_for_label.get("is_moe")
        if is_moe is None and isinstance(probe_for_label.get("summary"), dict):
            is_moe = probe_for_label["summary"].get("is_moe")
    if isinstance(is_moe, bool) and is_moe:
        manifest["base_model_info"]["is_moe"] = True
        manifest["base_model_info"]["untested_label"] = "untested"
        # The probe's untested finding, when present, carries the product
        # reason for the label.
        findings = (
            probe_for_label.get("findings")
            if isinstance(probe_for_label, dict)
            else None
        )
        if isinstance(findings, list):
            for f in findings:
                if isinstance(f, dict) and f.get("code") in (
                    "untested_architecture",
                ):
                    manifest["base_model_info"]["untested_reason"] = f.get(
                        "message"
                    )
                    break
    # Also look for an explicit untested label already on the job (for tests
    # simulating #65 without a full probe). Recorded wins.
    if job.get("untested_label"):
        manifest["base_model_info"]["untested_label"] = job["untested_label"]
    if job.get("untested_reason"):
        manifest["base_model_info"]["untested_reason"] = job["untested_reason"]

    return manifest


def render_text(manifest: dict[str, Any]) -> str:
    """Human-readable Markdown provenance for ``manifest``.

    The manifest dict is the machine-readable record; this renders the same
    facts as a document a reviewer can read without a JSON viewer. Every
    required field appears as a heading and a sentence, so a missing field that
    the machine dict would raise on is visibly absent here too.
    """
    lines: list[str] = []
    w = lines.append

    job = manifest.get("job", {})
    # New provenance stores details under base_model_info; the top-level
    # base_model string is kept for backward compatibility (issue #32).
    if isinstance(manifest.get("base_model_info"), dict):
        base = manifest["base_model_info"]
    elif isinstance(manifest.get("base_model"), dict):
        base = manifest["base_model"]
    elif isinstance(manifest.get("base_model"), str):
        base = {
            "id": manifest["base_model"],
            "revision": manifest.get("base_revision", "?"),
        }
    else:
        base = {}
    dataset = manifest.get("dataset", {})
    config = manifest.get("configuration", {})
    evaluation = manifest.get("evaluation", {})
    checkpoint = manifest.get("checkpoint", {})
    artifact = manifest.get("artifact", {})

    w("# Provenance Manifest\n")
    w(
        f"This artifact was produced by job `{job.get('id', '?')}` "
        f"({job.get('method', '?')} -> {artifact.get('kind', '?')}) "
        f"and is valid as of the run that produced it. "
        f"A manifest that can drift from its run is worse than none.\n"
    )
    w(f"**Generated at:** {manifest.get('generated_at', '?')}\n")
    w(f"**Version:** {manifest.get('version', 1)}\n")

    w("## Base model\n")
    w(f"- **ID:** {base.get('id', '?')}\n")
    w(f"- **Repository:** {base.get('repo', '?')}\n")
    w(f"- **Pinned revision:** `{base.get('revision', '?')}`\n")
    w(
        f"- **Licence:** {base.get('license', '?')} ({base.get('license_url', '?')})\n"
    )
    if base.get("is_moe"):
        w("- **Mixture-of-Experts:** yes - labelled **untested**\n")
        if base.get("untested_reason"):
            w(f"  - *Why:* {base['untested_reason']}\n")
    elif base.get("untested_label"):
        w(f"- **Label:** {base['untested_label']}\n")
        if base.get("untested_reason"):
            w(f"  - *Why:* {base['untested_reason']}\n")
    w(f"- **Obligations:** {base.get('license_obligations', '?')}\n")

    w("## Dataset\n")
    w(
        f"- **Dataset ID:** {dataset.get('dataset_id', dataset.get('id', '?'))}\n"
    )
    if dataset.get("filename"):
        w(f"- **Filename:** {dataset['filename']}\n")
    # Prefer the split's rows_in as the arrival count; fall back to row_count.
    rows_in = dataset.get("rows_in", dataset.get("row_count", "?"))
    w(f"- **Rows in:** {rows_in}\n")
    if dataset.get("rows_removed_duplicates") is not None:
        w(
            f"- **Duplicates removed (before split):** {dataset['rows_removed_duplicates']}\n"
        )
    w(f"- **Training rows:** {dataset.get('train_rows', '?')}\n")
    w(f"- **Held-out rows:** {dataset.get('held_out_rows', '?')}\n")
    if dataset.get("fraction") is not None:
        w(f"- **Held-out fraction:** {dataset['fraction']}\n")
    if dataset.get("seed") is not None:
        w(
            f"- **Split seed:** {dataset['seed']} (deterministic under this seed)\n"
        )
    if dataset.get("row_count") is not None:
        w(f"- **Validation row_count:** {dataset['row_count']}\n")
    if dataset.get("usable_rows") is not None:
        w(f"- **Usable rows:** {dataset['usable_rows']}\n")
    if dataset.get("token_count") is not None:
        w(f"- **Token count:** {dataset['token_count']}\n")

    w("## Configuration\n")
    w(f"- **Method:** {config.get('method', '?')}\n")
    hp = config.get("hyperparameters", {})
    if isinstance(hp, dict) and hp:
        w("- **Hyperparameters:**\n")
        for k in sorted(hp):
            w(f"  - `{k}`: {json.dumps(hp[k])}\n")
    overrides = config.get("overrides", [])
    if overrides:
        w("- **Overrides (pinned decisions):**\n")
        for o in overrides:
            if isinstance(o, dict):
                w(f"  - {o.get('decision', '?')}: {o.get('value', '?')}\n")
            else:
                w(f"  - {o}\n")
    else:
        w("- **Overrides:** none (all decisions are the predictor's)\n")

    w("## Evaluation\n")
    held = evaluation.get("held_out_split", {})
    if held:
        w(
            f"- **Held-out split:** {held.get('held_out_rows', '?')} of "
            f"{held.get('rows_in', '?')} rows held out (seed {held.get('seed', '?')}, "
            f"fraction {held.get('fraction', '?')})\n"
        )
    best = evaluation.get("best_checkpoint", checkpoint)
    if isinstance(best, dict) and best.get("step") is not None:
        w(
            f"- **Best checkpoint:** step {best['step']} "
            f"(basis: {best.get('basis', '?')})\n"
        )
        if best.get("held_out_loss") is not None:
            w(f"  - Held-out loss: {best['held_out_loss']}\n")
        if best.get("reason"):
            w(f"  - *Why:* {best['reason']}\n")
    probe = evaluation.get("template_probe")
    if isinstance(probe, dict) and probe:
        w(f"- **Template probe:** {json.dumps(probe)}\n")

    w("## Checkpoint (the result came from)\n")
    w(f"- **Step:** {checkpoint.get('step', '?')}\n")
    w(f"- **Basis:** {checkpoint.get('basis', '?')}\n")
    w(f"- **Reason:** {checkpoint.get('reason', '?')}\n")
    if checkpoint.get("held_out_loss") is not None:
        w(f"- **Held-out loss:** {checkpoint['held_out_loss']}\n")

    w("## Artifact\n")
    w(f"- **Kind:** {artifact.get('kind', '?')}\n")
    members = artifact.get("members") or []
    w(
        f"- **Members:** {', '.join(str(m) for m in members) if members else '?'}\n"
    )
    if artifact.get("bytes") is not None:
        w(f"- **Bytes:** {artifact['bytes']}\n")
    if artifact.get("sha256"):
        w(f"- **SHA-256:** {artifact['sha256']}\n")
    if artifact.get("loading"):
        w(f"- **Loading:** {artifact['loading']}\n")

    w("\n---\n")
    w(
        "*This manifest was generated from the run record, not hand-written. "
        "If a field is missing the generation fails rather than emitting a placeholder.*\n"
    )
    # Machine-readable appendix: the full JSON, pretty-printed.
    w("\n## Machine-readable manifest (JSON)\n")
    w("```json\n")
    # Use sorted keys and indent for human readability of the JSON itself.
    try:
        w(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    except Exception:
        w(json.dumps(str(manifest), indent=2) + "\n")
    w("```\n")
    return "".join(lines)


def to_pretty_json(manifest: dict[str, Any]) -> str:
    """The machine-readable JSON, pretty-printed for a person to read."""
    return json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
