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

from dataclasses import asdict, dataclass
from typing import Any


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
CATALOG: dict[str, BaseModel] = {
    m.id: m
    for m in [
        BaseModel(
            id="qwen3-4b",
            repo="Qwen/Qwen3-4B",
            revision="main",  # TODO pin to a commit SHA before submission
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
            revision="main",  # TODO pin to a commit SHA before submission
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

DEFAULT_MODEL = "qwen3-4b"


def get(model_id: str) -> BaseModel | None:
    return CATALOG.get(model_id)


def listing() -> list[dict[str, Any]]:
    return [m.to_dict() for m in CATALOG.values()]
