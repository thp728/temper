"""Disk sizing: the space a job's machine needs, not a constant equal to the
platform minimum.

Replaces `apps/control-plane/src/temper_control_plane/orchestrator.py`'s
`STORAGE_GB = 100` (issue #64). A 70B model's weights alone are larger than
that constant, so a user who selected one got a machine that failed partway
through its own download. Sized the same way peak VRAM is
(`temper_core.memory`): arithmetic over `temper_core.models.ModelFacts`, no
I/O, exercised the same way whether the facts came from a live resolve or a
test's double.

**What is summed**, per spec 005's "Disk is computed, not constant":

* **Weights at download precision.** The trainer downloads the base model in
  bf16 regardless of training method -- QLoRA quantises to NF4 on-device
  (`apps/trainer/entrypoint.py`'s `load_in_4bit=True`); it never downloads an
  already-quantised checkpoint. So this pool is always `facts.params * 2
  bytes`, not `memory.WEIGHT_BYTES_PER_PARAM[method]`, which prices the
  *training* dtype rather than the *download* one.
* **Retained checkpoints.** Axolotl keeps the newest `save_total_limit`
  checkpoint directories and deletes older ones as training goes
  (`apps/trainer/entrypoint.py`), so disk must hold that many copies of the
  trained weights at once, not the run's cumulative total. A checkpoint is
  the adapter's own size for LoRA/QLoRA -- measured at fp32, 33M trainable
  params -> 132.2 MB, `apps/trainer/README.md` -- and the full model's for a
  full fine-tune, which this trainer has never executed (spike 6, ADR-0029),
  so its bf16 estimate is labelled rather than measured.
* **Merged output.** Zero for every method this trainer runs: LoRA and QLoRA
  ship the adapter itself as the deliverable, and full fine-tuning's trained
  checkpoint already *is* the full model, so neither produces a second,
  separately-sized artifact. The term is kept in the arithmetic rather than
  omitted, because a method that does merge (spec 009's territory) must not
  need a second formula to account for one more pool.
* **Image and working space.** A fixed allowance: 8.5 GB is the trainer
  image's own measured build size (spike 4, `apps/trainer/README.md`,
  2026-08-18); the rest is headroom for the dataset copy, HF's staging
  directory during download, and Axolotl's own scratch and logs, none of it
  measured, so it is labelled as an assumption rather than presented as a
  figure spike 5 produced.

**Floored at the platform minimum, capped at the measured ceiling** -- both
from spike 5 (`spike/findings-spike5.json`): 100 GB is the smallest disk the
provider will create; 7200 GB is the largest, named by the API itself when it
refused a request for 8000. A job whose required disk exceeds the ceiling
cannot be provisioned at any price and is refused before launch, with the
shortfall named, rather than discovered as a `create` call that fails after
every other decision has already been made.

**Storage is billed on its own line, not folded into the GPU-hour.** Spike 5
found the GPU-hour rate unchanged by a 40x larger disk (41.31 INR/hr at both
100 GB and 4000 GB) -- so a quote that derives everything from GPU-hours
understates a large-disk job by exactly the amount storage costs. The rate
here is the one spike 5 could find documented, not one it could measure in a
run lasting minutes against a bill that resolves over a month, and it is
carried in USD because no live, per-account storage price exists to convert
it against -- labelled, not silently assumed to be the account's currency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import memory
from .models import ModelFacts

# spike 5 (`spike/findings-spike5.json`): "storage_parameter.default_storage_gb"
# is the provider's smallest disk; "storage_ceiling.ceiling_gb" is its largest,
# stated by the API itself refusing 8000 GB. Both measured, not estimated.
PLATFORM_MIN_DISK_GB = 100
PLATFORM_MAX_DISK_GB = 7200

# bf16: what the trainer actually downloads, for every method -- see the
# module docstring. Kept distinct from `memory.WEIGHT_BYTES_PER_PARAM`, which
# answers a different question (what a method trains with, not what it
# fetched).
WEIGHTS_DOWNLOAD_BYTES_PER_PARAM = 2.0

# What one retained checkpoint costs per saved parameter, by method.
# qlora/lora: measured -- `apps/trainer/README.md`'s adapter is fp32
# (33,030,144 trainable params at r=16 -> 132.2 MB, 4.00 bytes/param, matching
# the safetensors header's "504 tensors, every one F32"). full: not measured,
# because full fine-tuning has never executed on this trainer (ADR-0029);
# assumed at bf16, the same dtype `memory.py` trains it at.
CHECKPOINT_BYTES_PER_PARAM: dict[str, float] = {
    "qlora": 4.0,
    "lora": 4.0,
    "full": 2.0,
}

# Measured: the pinned trainer image's own build (spike 4, 2026-08-18,
# `apps/trainer/README.md`, "Build from pinned digest | 183s, 8.5 GB").
IMAGE_GB = 8.5

# NOT measured: headroom for the dataset copy, HF's staging directory during
# download, and Axolotl's own scratch and logs. No spike sized this pool; it
# is carried as a labelled assumption rather than left at zero, because
# omitting it would understate every job by the same fixed amount.
WORKING_SPACE_GB = 10.0

# Documented, not measured (spike 5's own caveat: "MEASURED: the rate. NOT
# MEASURED: the storage line itself -- this spike ran for minutes and a
# GB-month bill does not resolve in minutes"). Carried in USD because no live
# storage price exists behind the provider seam to convert it against.
STORAGE_USD_PER_GB_MONTH = 0.10
HOURS_PER_MONTH = 730  # 365.25 * 24 / 12, the standard billing approximation

BYTES_PER_GB = 1e9


class DiskExceedsCeilingError(Exception):
    """The disk this job needs is larger than the provider can create at any
    price -- refused before anything is provisioned, never discovered as a
    `create` call that fails after every other decision has already been
    made."""

    def __init__(self, required_gb: float, ceiling_gb: float):
        self.required_gb = required_gb
        self.ceiling_gb = ceiling_gb
        self.shortfall_gb = required_gb - ceiling_gb
        super().__init__(
            f"This job needs {required_gb:.0f} GB of disk, which is "
            f"{self.shortfall_gb:.0f} GB more than the {ceiling_gb:.0f} GB "
            "the provider can create at any price."
        )


@dataclass(frozen=True)
class DiskPlan:
    """The disk a job needs, broken into the pools it was summed from.

    Not hidden behind the final number: spec 005's whole premise is that a
    prediction shows the arithmetic that produced it.
    """

    weights_gb: float
    checkpoints_gb: float
    merged_output_gb: float
    image_gb: float
    working_space_gb: float
    required_gb: float  # the raw sum, before the floor and the cap
    provisioned_gb: int  # floored at the platform minimum, what is requested
    storage_cost_usd_per_hour: float


def required_disk(
    facts: ModelFacts,
    *,
    method: str,
    lora_r: int,
    retained_checkpoints: int,
) -> DiskPlan:
    """The disk `facts` needs to train with `method`, floored and capped.

    Raises `DiskExceedsCeilingError` when the raw requirement is larger than
    the provider can create at any price -- checked before the floor is
    applied, because a job whose weights alone exceed the ceiling is refused
    for that reason, not quietly rounded down to something provisionable.
    """
    if method not in CHECKPOINT_BYTES_PER_PARAM:
        raise ValueError(
            f"unknown method {method!r}; expected one of "
            f"{sorted(CHECKPOINT_BYTES_PER_PARAM)}"
        )
    if retained_checkpoints < 0:
        raise ValueError(
            f"retained_checkpoints must be >= 0, got {retained_checkpoints}"
        )

    weights_gb = facts.params * WEIGHTS_DOWNLOAD_BYTES_PER_PARAM / BYTES_PER_GB
    saved_params = (
        facts.params
        if method == "full"
        else memory.trainable_params(facts, lora_r)
    )
    one_checkpoint_gb = (
        saved_params * CHECKPOINT_BYTES_PER_PARAM[method] / BYTES_PER_GB
    )
    checkpoints_gb = one_checkpoint_gb * retained_checkpoints
    # No method this trainer runs produces a second, separately-sized
    # artifact -- see the module docstring's "Merged output".
    merged_output_gb = 0.0

    required_gb = (
        weights_gb
        + checkpoints_gb
        + merged_output_gb
        + IMAGE_GB
        + WORKING_SPACE_GB
    )
    if required_gb > PLATFORM_MAX_DISK_GB:
        raise DiskExceedsCeilingError(required_gb, PLATFORM_MAX_DISK_GB)

    provisioned_gb = max(PLATFORM_MIN_DISK_GB, math.ceil(required_gb))
    storage_cost_usd_per_hour = (
        provisioned_gb * STORAGE_USD_PER_GB_MONTH / HOURS_PER_MONTH
    )
    return DiskPlan(
        weights_gb=weights_gb,
        checkpoints_gb=checkpoints_gb,
        merged_output_gb=merged_output_gb,
        image_gb=IMAGE_GB,
        working_space_gb=WORKING_SPACE_GB,
        required_gb=required_gb,
        provisioned_gb=provisioned_gb,
        storage_cost_usd_per_hour=storage_cost_usd_per_hour,
    )
