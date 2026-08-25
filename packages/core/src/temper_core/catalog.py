"""The base-model catalog.

An allow-list, not a limitation. The question "why can't I use any Hugging Face
model?" has a real answer: detection is easy, *support* is not. Every family
brings its own chat-template quirks, tokenizer edge cases and packing
compatibility. The catalog is a promise about what has been tested, and
curation is the feature.

Revisions are pinned. An unpinned `main` means a model can change under a
completed run, which breaks the claim that a run is reproducible.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

# A pinned revision is a full 40-character hex commit SHA. Branch names like
# "main" are not pinned: the model can change underneath a completed run.
# Case-insensitive: git SHAs are hex and may appear in either case.
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def is_pinned_revision(revision: str) -> bool:
    """Whether a revision is a resolved commit SHA rather than a branch name."""
    return bool(_REVISION_RE.match(revision))


@dataclass(frozen=True)
class BaseModel:
    id: str
    repo: str
    revision: str  # pinned; never "main"
    params_b: float
    license: str
    license_url: str
    context_length: int
    good_for: str
    min_gpu: str
    est_peak_vram_gb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Both are dense and Apache-2.0. Qwen3.5 was deliberately NOT chosen: it is
# Gated Delta Networks plus sparse MoE, which brings in the two operational
# fault lines the research says to exclude -- MoE changes LoRA target-module
# selection and makes memory scale with total rather than active parameters,
# and non-standard attention breaks packing assumptions. Qwen3 dense is the
# most recent generation where the whole stack is boring, and boring is the
# requirement.
#
# Revisions are pinned to the commit that was current at catalog curation
# (2025-07-26). Resolved via https://huggingface.co/api/models/<repo>/revision/main.
CATALOG: dict[str, BaseModel] = {
    m.id: m
    for m in [
        BaseModel(
            id="qwen3-4b",
            repo="Qwen/Qwen3-4B",
            revision="1cfa9a7208912126459214e8b04321603b3df60c",
            params_b=4.0,
            license="Apache-2.0",
            license_url="https://huggingface.co/Qwen/Qwen3-4B",
            context_length=40960,
            good_for="Fast iteration and smaller datasets. The default.",
            min_gpu="L4 (24 GB)",
            # Measured, not estimated: 5.31 GB observed on an L4 under QLoRA
            # r=16 all-linear at sequence length 2048.
            est_peak_vram_gb=5.3,
        ),
        BaseModel(
            id="qwen3-8b",
            repo="Qwen/Qwen3-8B",
            revision="b968826d9c46dd6066d109eabc6255188de91218",
            params_b=8.2,
            license="Apache-2.0",
            license_url="https://huggingface.co/Qwen/Qwen3-8B",
            context_length=32768,
            good_for="Higher quality when the dataset justifies it.",
            min_gpu="L4 (24 GB)",
            # Extrapolated from the 4B measurement by parameter ratio, not
            # observed. Flagged as such until an 8B run confirms it.
            est_peak_vram_gb=9.5,
        ),
    ]
}

# Fail fast if any entry is not pinned. Loading a branch name would break the
# reproducibility claim that pinning exists to support.
for _m in CATALOG.values():
    if not is_pinned_revision(_m.revision):
        raise ValueError(
            f"Catalog entry '{_m.id}' has revision '{_m.revision}' which is not a "
            f"40-character commit SHA. Pin it to a resolved commit identifier so a "
            f"completed run cannot change underneath it."
        )

DEFAULT_MODEL = "qwen3-4b"


def get(model_id: str) -> BaseModel | None:
    return CATALOG.get(model_id)


def listing() -> list[dict[str, Any]]:
    return [m.to_dict() for m in CATALOG.values()]
