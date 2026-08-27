"""Hardware selection: the cheapest configuration that fits.

Replaces `apps/control-plane/src/temper_control_plane/orchestrator.py`'s
`GPU_PREFERENCE` -- three names, taken in order, first with free capacity
winning, which considered neither what a job needed nor what anything cost
(issue #55). Selection instead searches every (method, GPU type, device
count) triple the provider currently has free, keeps the ones
`temper_core.memory.predict_peak` predicts will fit, and picks the cheapest.
Ties break toward fewer, larger devices: a multi-GPU job pays interconnect
overhead a single card does not, and asking for hardware a job does not need
is worse than being slower -- particularly in front of a reader who sells
the hardware.

Method is not chosen separately from the GPU and then reconciled. The search
ranges over the whole (method, gpu_type, device_count) space at once and
takes whichever triple is cheapest, because fixing the method ahead of time
would be exactly the kind of preference list this ticket deletes -- and
because price never depends on method, so at a tied price the more capable
method that still fits is a strictly better answer, not a different one.

Only QLoRA is executable today: `apps/trainer/entrypoint.py` hard-codes
`adapter="qlora", load_in_4bit=True`, and multi-GPU sharding has only ever
been proven at small scale (spike 6, `docs/adr/0028-...`). Teaching the
trainer to run LoRA, full fine-tuning or more than one device is spec 005's
"further notes" and spec 009's job. `select_hardware`'s `methods` parameter
therefore defaults to the executable set alone; a caller building a quote
that must describe (not run) the alternatives can pass a wider one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import gpus, memory
from .models import ModelFacts

# Best-to-worst: at a tied price, the search keeps the first of these that
# fits, because price never depends on method and a free upgrade in quality
# should never be turned down.
METHODS_BEST_FIRST: tuple[str, ...] = ("full", "lora", "qlora")

# What the trainer can actually run today (see the module docstring). Product
# code should not need to name "qlora" as a literal -- it reads this instead,
# so the day spec 009 teaches the trainer another method there is exactly one
# place that says so.
EXECUTABLE_METHODS: tuple[str, ...] = ("qlora",)


@dataclass(frozen=True)
class GpuAvailability:
    """One provider row: a GPU type, how many sit free on that node right
    now, and what it costs.

    `num_free_devices` is per node, not summed across the fleet: devices on
    two different nodes cannot be attached to one machine, so a row is the
    ceiling on how many of that type a single job can request together
    (spike 6, `spike/spike6.py`). Two rows may share a `gpu_type` when the
    provider has free capacity on more than one node of that type -- they
    are kept separate on purpose, never merged.
    """

    gpu_type: str
    price_per_hour: float
    num_free_devices: int


@dataclass(frozen=True)
class HardwarePlan:
    """The chosen configuration: method, card, count, and what it costs."""

    method: str
    gpu_type: str
    device_count: int
    price_per_hour: float  # total across every device, not per device
    currency: str
    peak: memory.PeakMemory
    headroom_gb: float


class NoFittingHardwareError(Exception):
    """No (method, GPU type, device count) the provider has free right now
    is predicted to fit this job."""


def _cheaper(
    current: HardwarePlan | None, candidate: HardwarePlan
) -> HardwarePlan:
    """Whichever of the two is the better answer: lower price wins; a tied
    price prefers fewer devices, because interconnect overhead is real and a
    slower single card is a better answer than a faster pair at the same
    cost."""
    if current is None:
        return candidate
    if candidate.price_per_hour < current.price_per_hour:
        return candidate
    if (
        candidate.price_per_hour == current.price_per_hour
        and candidate.device_count < current.device_count
    ):
        return candidate
    return current


def select_hardware(
    facts: ModelFacts,
    *,
    lora_r: int,
    sequence_len: int,
    micro_batch_size: int,
    availability: Sequence[GpuAvailability],
    currency: str,
    methods: Sequence[str] = EXECUTABLE_METHODS,
) -> HardwarePlan:
    """The cheapest (method, GPU, device count) predicted to fit `facts`.

    `availability` is expected to already be filtered to machine-capable rows
    -- the provider seam's job, since "machine-capable" (`workload_type ==
    "vm"`) is a fact about the provider's API, not about this arithmetic.
    `currency` comes from the account, read live, and travels with the price
    rather than being assumed.

    Raises `NoFittingHardwareError` when nothing available, at any method or
    device count, predicts a fit -- refused before anything is provisioned,
    never discovered on a billing machine.
    """
    best: HardwarePlan | None = None
    for row in availability:
        if row.num_free_devices <= 0:
            continue
        capacity = gpus.CAPACITY_GB.get(row.gpu_type)
        if capacity is None:
            # A type this platform has no capacity figure for is one it
            # cannot predict a fit for -- and an unpriced fit is not a fit,
            # it is a guess. Skipped, not defaulted.
            continue
        for device_count in range(1, row.num_free_devices + 1):
            price = row.price_per_hour * device_count
            if best is not None and price > best.price_per_hour:
                # Every larger device_count on this row only costs more; the
                # cheapest fit already found at a lower or equal price on
                # this row cannot be beaten by asking for more of it.
                break
            for method in methods:
                peak = memory.predict_peak(
                    facts,
                    method=method,
                    lora_r=lora_r,
                    sequence_len=sequence_len,
                    micro_batch_size=micro_batch_size,
                    device_count=device_count,
                )
                headroom = memory.headroom_gb(peak, capacity)
                if headroom < 0:
                    continue
                candidate = HardwarePlan(
                    method=method,
                    gpu_type=row.gpu_type,
                    device_count=device_count,
                    price_per_hour=price,
                    currency=currency,
                    peak=peak,
                    headroom_gb=headroom,
                )
                best = _cheaper(best, candidate)
                # The first fitting method, in best-to-worst order, is the
                # best answer this price can buy -- a cheaper method at the
                # same price is not a better one.
                break
    if best is None:
        raise NoFittingHardwareError(
            "No available GPU predicts a fit for this job, at any method or "
            "device count the provider currently has free."
        )
    return best
