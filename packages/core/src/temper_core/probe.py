"""The admission probe (Spec 009 / issue #58).

The catalog is a curated, tested path and a promise about what has been run --
and a default, not a boundary. Any model repository may be used at a pinned
revision once it has passed a compatibility probe. The probe runs *in front of
the user* and its result is **shown**, not only enforced: a model that passes
with warnings is usable, and the user knows what they took on.

The probe reads its facts through the same `models` seam the predictor uses --
`temper_core.models.ModelFacts` is resolved once, by the seam, and both the
peak-memory arithmetic (`temper_core.memory`) and this probe consume that
shape. There is deliberately no second resolver: two implementations of *how
many parameters does this model have* would drift, and the drift would surface
as a model the probe accepted and the predictor mispriced (ADR-0010).

The checks, and which failures block and which warn (product decisions, with
the reasons kept beside them):

* **The revision resolves.** A pinned reference that cannot be resolved cannot
  be trained against at all, so this blocks. It is the seam's own refusal,
  surfaced as `revision_unresolvable` (see `unresolvable`).
* **A chat template is present.** Without one, chat data cannot be formatted
  and the alternative is hand-writing role delimiters, which becomes the modal
  silent failure. Blocks (`missing_chat_template`).
* **Padding differs from end-of-sequence.** A tokenizer whose pad token is the
  same as its EOS token, unmasked, teaches the model never to stop. Blocks
  (`padding_collides_with_eos`). "The tokenizer loads" is the seam's own
  concern and every failure of it maps to a block here: a missing
  `tokenizer_config.json` surfaces as no chat template (below), and a file
  that cannot be resolved at all surfaces as `revision_unresolvable`. Nothing
  deeper is checkable at admission -- instantiating the tokenizer is a
  machine-side concern, and the padding/EOS distinction is the proxy that is
  load-bearing for training correctness.
* **The licence resolves.** A licence that cannot be resolved is shown as
  unknown rather than guessed -- a wrong licence is worse than a visibly
  missing one (the posture the models seam already takes). Warns
  (`license_unresolvable`).
* **The architecture is tested here.** Mixture-of-experts and unknown
  architectures change target-module selection, memory scaling and packing
  behaviour, so they are labelled untested. They warn rather than block,
  because curation is a default and not a boundary -- blocking on the third
  would make "any model, including large ones" untrue in practice. Warns
  (`untested_architecture`).
* **The predicted memory fits something available.** This is the predictor's
  own memory arithmetic, reused whole: the probe calls `memory.predict_peak`
  -- the exact function `selection.select_hardware` and the creation-time
  refusal (#54) are built on -- at the default configuration, and compares it
  against the cards this platform can provision (`gpus.CAPACITY_GB`), taking
  the smallest card that holds it. "Available" here means *provisionable*:
  current availability is a fact about the moment, and a persisted probe must
  not record a moment as if it were a property of the model -- the live
  current-availability search still runs at job creation (#54), against the
  real configuration, where it belongs. There is deliberately no second peak
  computation: if the memory model is refined, the probe and the refusal
  refine together because they are one function (ADR-0010, ADR-0043). Per
  ADR-0043, memory *blocks at creation*; at admission it warns with the same
  arithmetic (`memory_does_not_fit`), so the user sees the numbers the launch
  would refuse with, and the refusal still fires at the moment of commitment
  where the exact configuration is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from . import gpus, hyperparams, memory, selection
from .models import ModelFacts

# Stable finding codes. A reader branches on the code; the message carries the
# specifics, with the product reason kept beside it.
FINDING_REVISION_UNRESOLVABLE = "revision_unresolvable"
FINDING_MISSING_CHAT_TEMPLATE = "missing_chat_template"
FINDING_PAD_EOS_COLLIDE = "padding_collides_with_eos"
FINDING_LICENSE_UNRESOLVABLE = "license_unresolvable"
FINDING_UNTESTED_ARCHITECTURE = "untested_architecture"
FINDING_MEMORY_DOES_NOT_FIT = "memory_does_not_fit"

Severity = Literal["block", "warn"]


@dataclass(frozen=True)
class ProbeFinding:
    """One check's outcome. `severity` is what the caller may branch on: a
    `block` means the model is not usable, a `warn` means it is usable and the
    user is told what they took on."""

    code: str
    severity: Severity
    message: str
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.details:
            out["details"] = self.details
        return out


@dataclass(frozen=True)
class ProbeSummary:
    """The facts the probe resolved, snapshotted for display and for
    materialising an admitted model without re-resolving the seam."""

    architecture: str
    params_b: float
    context_length: int
    license: str
    is_moe: bool


@dataclass(frozen=True)
class MemoryCheck:
    """The predicted-memory half of the probe, with the arithmetic shown.

    `fits` is whether the predicted peak at the default configuration fits a
    card this platform can provision; `card` is the *smallest* card that holds
    it, so the listing can say "needs at least an L4" the way the catalog
    does. The peak comes from `memory.predict_peak` -- the same function the
    creation-time refusal (#54) is built on -- so a launch that would be
    refused is refused with numbers the user already saw here, and the two
    cannot drift. `fits` is never False against a stale availability snapshot,
    because availability is not part of this check; the live search runs at
    creation.
    """

    fits: bool
    peak_gb: float | None = None
    card: str | None = None
    capacity_gb: float | None = None
    headroom_gb: float | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fits": self.fits,
            "peak_gb": _round2(self.peak_gb),
            "card": self.card,
            "capacity_gb": _round2(self.capacity_gb),
            "headroom_gb": _round2(self.headroom_gb),
            "note": self.note,
        }


def _round2(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


@dataclass(frozen=True)
class ProbeResult:
    """The probe's verdict, shaped for persistence and display.

    `ok` is the only field a caller may branch on for usability: False means
    at least one block. `verdict` is the human-facing label ("blocked",
    "usable_with_warnings", "usable") derived from `ok` and the findings, so
    the two cannot disagree. `memory` is the predicted-memory line, always
    present when the probe could check it.
    """

    repo: str
    revision: str
    summary: ProbeSummary
    findings: tuple[ProbeFinding, ...] = field(default_factory=tuple)
    memory: MemoryCheck | None = None

    @property
    def blocking_findings(self) -> tuple[ProbeFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "block")

    @property
    def ok(self) -> bool:
        return not self.blocking_findings

    @property
    def verdict(self) -> str:
        if not self.ok:
            return "blocked"
        if self.findings:
            return "usable_with_warnings"
        return "usable"

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "ok": self.ok,
            "verdict": self.verdict,
            "architecture": self.summary.architecture,
            # The resolved facts an admitted model is materialised from and a
            # listing shows; carried beside the verdict so neither is
            # re-resolved through the seam to display the other.
            "params_b": self.summary.params_b,
            "context_length": self.summary.context_length,
            "license": self.summary.license,
            "is_moe": self.summary.is_moe,
            "findings": [f.as_dict() for f in self.findings],
            "memory": self.memory.as_dict()
            if self.memory is not None
            else None,
        }


def summary_of(facts: ModelFacts) -> ProbeSummary:
    """The facts a result carries, in the shape a listing needs."""
    return ProbeSummary(
        architecture=facts.architecture,
        params_b=facts.params / 1e9,
        context_length=facts.context_length,
        license=facts.license,
        is_moe=facts.is_moe,
    )


def unresolvable(repo: str, revision: str, detail: str) -> ProbeResult:
    """The blocking result when a pinned reference does not resolve.

    Resolution is the `models` seam's own job (it refuses a reference that
    cannot be answered); this turns that refusal into a probe verdict so the
    admission path persists and shows one shape whether the failure happened
    resolving the revision or checking the model.
    """
    return ProbeResult(
        repo=repo,
        revision=revision,
        summary=ProbeSummary(
            architecture="unknown",
            params_b=0.0,
            context_length=0,
            license="",
            is_moe=False,
        ),
        findings=(
            ProbeFinding(
                FINDING_REVISION_UNRESOLVABLE,
                "block",
                f"the repository '{repo}' at pinned revision '{revision}' "
                f"could not be resolved ({detail}). A revision that does not "
                "resolve cannot be trained against, so this model is blocked.",
            ),
        ),
    )


def _check_memory(
    facts: ModelFacts,
    *,
    lora_r: int,
    sequence_len: int,
    micro_batch_size: int,
) -> MemoryCheck:
    """The predicted-memory line: the product's own peak arithmetic at the
    default configuration, against the cards this platform can provision.

    This is a **floor**, not a prediction of what a launch will choose.
    Admission happens before any configuration is chosen, so the honest thing
    to report is the least a model can be run on: the lightest executable
    method's peak, and the smallest card that holds it, so the listing can say
    "needs at least an L4" the way the catalog does.

    It is therefore the *minimum* across `EXECUTABLE_METHODS`, not the first of
    them. When this was written qlora was the only executable method and
    `EXECUTABLE_METHODS[0]` said the same thing; issue #66 added full
    fine-tuning at the front of that tuple, which silently turned the floor
    into the most demanding method and advertised an A100 for models that run
    on an L4.
    """
    peak = min(
        (
            memory.predict_peak(
                facts,
                method=method,
                lora_r=lora_r,
                sequence_len=sequence_len,
                micro_batch_size=micro_batch_size,
            )
            for method in selection.EXECUTABLE_METHODS
        ),
        key=lambda p: p.total_gb,
    )
    for card in sorted(gpus.CAPACITY_GB, key=lambda c: gpus.CAPACITY_GB[c]):
        capacity = gpus.CAPACITY_GB[card]
        if peak.total_gb <= capacity:
            return MemoryCheck(
                fits=True,
                peak_gb=peak.total_gb,
                card=card,
                capacity_gb=capacity,
                headroom_gb=capacity - peak.total_gb,
                note=None,
            )
    card = max(gpus.CAPACITY_GB, key=lambda c: gpus.CAPACITY_GB[c])
    capacity = gpus.CAPACITY_GB[card]
    return MemoryCheck(
        fits=False,
        peak_gb=peak.total_gb,
        card=card,
        capacity_gb=capacity,
        headroom_gb=capacity - peak.total_gb,
        note=(
            f"the predicted peak of {peak.total_gb:.1f} GB exceeds the "
            f"{capacity:.0f} GB {card}, the largest card this platform "
            "provisions; memory blocks at launch (ADR-0043), so creating a "
            "job would be refused until a configuration fits -- a smaller "
            "model, a shorter sequence, or a lighter method."
        ),
    )


def probe(
    facts: ModelFacts,
    *,
    repo: str,
    revision: str,
    lora_r: int | None = None,
    sequence_len: int | None = None,
    micro_batch_size: int | None = None,
) -> ProbeResult:
    """Probe one resolved model at one pinned revision.

    Pure over its inputs -- facts arrive through the `models` seam, exactly as
    the predictor reads them -- so the probe is testable without a network.
    `lora_r`/`sequence_len`/`micro_batch_size` default to the product
    defaults, because admission happens before any dataset or configuration is
    chosen.
    """
    lora_r = lora_r if lora_r is not None else hyperparams.DEFAULTS["lora_r"]
    sequence_len = (
        sequence_len
        if sequence_len is not None
        else hyperparams.DEFAULTS["sequence_len"]
    )
    micro_batch_size = (
        micro_batch_size
        if micro_batch_size is not None
        else hyperparams.DEFAULTS["micro_batch_size"]
    )

    findings: list[ProbeFinding] = []

    if not facts.has_chat_template:
        findings.append(
            ProbeFinding(
                FINDING_MISSING_CHAT_TEMPLATE,
                "block",
                "this model publishes no chat template, so chat data cannot "
                "be formatted; the alternative is hand-writing role "
                "delimiters, which this platform does not do. Blocked.",
            )
        )

    if not facts.pad_eos_distinct:
        findings.append(
            ProbeFinding(
                FINDING_PAD_EOS_COLLIDE,
                "block",
                "this tokenizer's padding token is the same as its "
                "end-of-sequence token; unmasked, shared padding teaches the "
                "model never to stop. Blocked.",
            )
        )

    if not facts.license:
        findings.append(
            ProbeFinding(
                FINDING_LICENSE_UNRESOLVABLE,
                "warn",
                "the licence could not be resolved and is recorded as "
                "unknown; a visibly missing licence is safer than a guessed "
                "one. Know your obligations before shipping what you train.",
            )
        )

    if facts.is_moe:
        findings.append(
            ProbeFinding(
                FINDING_UNTESTED_ARCHITECTURE,
                "warn",
                "this model is a mixture-of-experts architecture, which is "
                "untested here: expert routing changes LoRA target-module "
                "selection, memory scales with total rather than active "
                "parameters, and routing interacts poorly with small-batch "
                "adapters. It is usable and labelled untested -- curation is "
                "a default, not a boundary.",
            )
        )
    elif facts.architecture == "unknown":
        findings.append(
            ProbeFinding(
                FINDING_UNTESTED_ARCHITECTURE,
                "warn",
                "this model's architecture is not one this platform has "
                "trained, so the configuration is labelled untested. It is "
                "usable, with the untested label -- curation is a default, "
                "not a boundary.",
            )
        )

    memory_check = _check_memory(
        facts,
        lora_r=lora_r,
        sequence_len=sequence_len,
        micro_batch_size=micro_batch_size,
    )
    if memory_check.fits is False:
        findings.append(
            ProbeFinding(
                FINDING_MEMORY_DOES_NOT_FIT,
                "warn",
                memory_check.note
                or "the predicted memory does not fit anything available.",
                details={
                    "peak_gb": memory_check.peak_gb,
                    "capacity_gb": memory_check.capacity_gb,
                    "card": memory_check.card,
                    "headroom_gb": memory_check.headroom_gb,
                },
            )
        )

    return ProbeResult(
        repo=repo,
        revision=revision,
        summary=summary_of(facts),
        findings=tuple(findings),
        memory=memory_check,
    )
