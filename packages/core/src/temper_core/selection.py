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
class HardwareAlternative:
    """A configuration the search found that fits and did not choose.

    Every alternative is real: it passed the same fit check the winner did,
    and lost on the selection rule (higher price, or a tied price with more
    devices). Returning them is what lets a quote say what the user gave up
    to take the chosen card -- spec 005's "alternatives with what each would
    have cost", discovered at the one place they are discovered.
    """

    gpu_type: str
    device_count: int
    price_per_hour: float  # total across every device, not per device
    headroom_gb: float


@dataclass(frozen=True)
class HardwarePlan:
    """The chosen configuration: method, card, count, and what it costs.

    `alternatives` are the fitting configurations the same search rejected,
    so the selection's reasoning travels with its answer rather than being
    re-derived by whatever shows the plan.
    """

    method: str
    gpu_type: str
    device_count: int
    price_per_hour: float  # total across every device, not per device
    currency: str
    peak: memory.PeakMemory
    headroom_gb: float
    alternatives: tuple[HardwareAlternative, ...] = ()


class NoFittingHardwareError(Exception):
    """No (method, GPU type, device count) the provider has free right now
    is predicted to fit this job.

    When the caller pinned a configuration (issue #79's overrides), or when a
    card is free but nothing fits at any method or device count (the user's
    hyperparameter overrides can make that the case, issue #80), the error
    carries the arithmetic that refused it -- `peak_gb` against `capacity_gb`
    on `gpu_type`, `method` and `device_count` -- so the refuser can show the
    user the same numbers `memory.headroom_gb` used rather than a bare
    "nothing fits". An unpinned refusal with no free machine-capable device
    carries none of it: there was no single configuration to price.
    """

    def __init__(
        self,
        message: str,
        *,
        method: str | None = None,
        gpu_type: str | None = None,
        device_count: int | None = None,
        peak_gb: float | None = None,
        capacity_gb: float | None = None,
    ) -> None:
        self.method = method
        self.gpu_type = gpu_type
        self.device_count = device_count
        self.peak_gb = peak_gb
        self.capacity_gb = capacity_gb
        super().__init__(message)


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


def _no_fitting_error(
    facts: ModelFacts,
    *,
    lora_r: int,
    sequence_len: int,
    micro_batch_size: int,
    availability: Sequence[GpuAvailability],
    methods: Sequence[str],
    method: str | None,
    gpu_type: str | None,
    device_count: int | None,
) -> NoFittingHardwareError:
    """The refusal for an empty search, with the arithmetic when it was pinned.

    An unpinned search that still finds nothing gets a bare "nothing fits"
    only when no machine-capable device is free at all -- there was no single
    configuration to price. When a card *is* free but nothing fits, the
    search's own arithmetic is named (the cheapest executable method at the
    smallest device count against the largest free card): the configuration
    was described by the caller (issue #80's hyperparameter overrides), and a
    bare refusal would not say which peak lost against which capacity.
    """
    plain = NoFittingHardwareError(
        "No available GPU predicts a fit for this job, at any method or "
        "device count the provider currently has free."
    )
    eff_method = method or (methods[0] if methods else EXECUTABLE_METHODS[0])
    eff_count = device_count or 1
    peak = memory.predict_peak(
        facts,
        method=eff_method,
        lora_r=lora_r,
        sequence_len=sequence_len,
        micro_batch_size=micro_batch_size,
        device_count=eff_count,
    )
    if gpu_type is not None:
        capacity = gpus.CAPACITY_GB.get(gpu_type)
        if capacity is None:
            return plain
        if peak.total_gb > capacity:
            return NoFittingHardwareError(
                f"{eff_method} on a {gpu_type} predicts {peak.total_gb:.1f} GB "
                f"peak, which the {capacity:.0f} GB {gpu_type} cannot hold",
                method=eff_method,
                gpu_type=gpu_type,
                device_count=eff_count,
                peak_gb=peak.total_gb,
                capacity_gb=capacity,
            )
        return NoFittingHardwareError(
            f"{eff_method} on a {gpu_type} fits ({peak.total_gb:.1f} GB peak "
            f"within its {capacity:.0f} GB), but no {gpu_type} is currently "
            "free to provision",
            method=eff_method,
            gpu_type=gpu_type,
            device_count=eff_count,
            peak_gb=peak.total_gb,
            capacity_gb=capacity,
        )
    free = [
        r
        for r in availability
        if r.num_free_devices > 0 and r.gpu_type in gpus.CAPACITY_GB
    ]
    if not free:
        if method is None and gpu_type is None and device_count is None:
            # Nothing free and nothing pinned: there was no single
            # configuration to price, so none is named (the bare shape an
            # unpinned plan expects).
            return plain
        return NoFittingHardwareError(
            f"{eff_method} predicts {peak.total_gb:.1f} GB peak, but the "
            "provider reports no free machine-capable device at all",
            method=eff_method,
            device_count=eff_count,
            peak_gb=peak.total_gb,
        )
    biggest = max(free, key=lambda r: gpus.CAPACITY_GB[r.gpu_type])
    capacity = gpus.CAPACITY_GB[biggest.gpu_type]
    if eff_count > biggest.num_free_devices:
        return NoFittingHardwareError(
            f"{eff_method} needs {eff_count} devices, but the most any "
            f"currently-free node offers is {biggest.num_free_devices}",
            method=eff_method,
            gpu_type=biggest.gpu_type,
            device_count=eff_count,
            peak_gb=peak.total_gb,
            capacity_gb=capacity,
        )
    if peak.total_gb > capacity:
        return NoFittingHardwareError(
            f"{eff_method} predicts {peak.total_gb:.1f} GB peak, beyond the "
            f"{capacity:.0f} GB of the largest card currently available "
            f"({biggest.gpu_type})",
            method=eff_method,
            gpu_type=biggest.gpu_type,
            device_count=eff_count,
            peak_gb=peak.total_gb,
            capacity_gb=capacity,
        )
    return NoFittingHardwareError(
        f"{eff_method} predicts {peak.total_gb:.1f} GB peak, which fits a "
        f"{biggest.gpu_type} ({capacity:.0f} GB), but none is currently free",
        method=eff_method,
        gpu_type=biggest.gpu_type,
        device_count=eff_count,
        peak_gb=peak.total_gb,
        capacity_gb=capacity,
    )


def select_hardware(
    facts: ModelFacts,
    *,
    lora_r: int,
    sequence_len: int,
    micro_batch_size: int,
    availability: Sequence[GpuAvailability],
    currency: str,
    methods: Sequence[str] = EXECUTABLE_METHODS,
    method: str | None = None,
    gpu_type: str | None = None,
    device_count: int | None = None,
) -> HardwarePlan:
    """The cheapest (method, GPU, device count) predicted to fit `facts`.

    `availability` is expected to already be filtered to machine-capable rows
    -- the provider seam's job, since "machine-capable" (`workload_type ==
    "vm"`) is a fact about the provider's API, not about this arithmetic.
    `currency` comes from the account, read live, and travels with the price
    rather than being assumed.

    `method`/`gpu_type`/`device_count` pin one or more of the decision to a
    caller-chosen value (issue #79's overrides). A pinned decision is a hard
    constraint, never a preference: the search considers only configurations
    matching it, and refuses rather than falling back to the predictor's own
    pick, because a run that silently ignored an override would be lying
    about what it did. A pinned method is searched even outside `methods` --
    the caller that pinned it decides executability -- and when nothing fits
    the refusal carries the arithmetic that refused it (see
    `NoFittingHardwareError`).

    Raises `NoFittingHardwareError` when nothing available, at any method or
    device count, predicts a fit -- refused before anything is provisioned,
    never discovered on a billing machine.
    """
    if device_count is not None and device_count < 1:
        raise ValueError(f"device_count must be >= 1, got {device_count}")
    if method is not None and method not in memory.WEIGHT_BYTES_PER_PARAM:
        raise ValueError(
            f"unknown method {method!r}; expected one of "
            f"{sorted(memory.WEIGHT_BYTES_PER_PARAM)}"
        )
    search_methods = (method,) if method is not None else methods
    best: HardwarePlan | None = None
    # Every configuration the search found that fits, whether it won or not.
    # One per (method, gpu_type, device_count) at its cheapest price; the
    # loser set is what a quote turns into "the alternatives with what each
    # would have cost".
    found: dict[tuple[str, str, int], HardwarePlan] = {}
    for row in availability:
        if gpu_type is not None and row.gpu_type != gpu_type:
            continue
        if row.num_free_devices <= 0:
            continue
        capacity = gpus.CAPACITY_GB.get(row.gpu_type)
        if capacity is None:
            # A type this platform has no capacity figure for is one it
            # cannot predict a fit for -- and an unpriced fit is not a fit,
            # it is a guess. Skipped, not defaulted.
            continue
        if device_count is not None:
            counts: range = range(device_count, device_count + 1)
        else:
            counts = range(1, row.num_free_devices + 1)
        for count in counts:
            if count > row.num_free_devices:
                continue
            price = row.price_per_hour * count
            for search_method in search_methods:
                peak = memory.predict_peak(
                    facts,
                    method=search_method,
                    lora_r=lora_r,
                    sequence_len=sequence_len,
                    micro_batch_size=micro_batch_size,
                    device_count=count,
                )
                headroom = memory.headroom_gb(peak, capacity)
                if headroom < 0:
                    continue
                candidate = HardwarePlan(
                    method=search_method,
                    gpu_type=row.gpu_type,
                    device_count=count,
                    price_per_hour=price,
                    currency=currency,
                    peak=peak,
                    headroom_gb=headroom,
                )
                best = _cheaper(best, candidate)
                # The first fitting method, in best-to-worst order, is the
                # best answer this price can buy -- a cheaper method at the
                # same price is not a better one, so the rest of this row's
                # price point is not worth a fit check.
                existing = found.get((search_method, row.gpu_type, count))
                if existing is None or price < existing.price_per_hour:
                    found[(search_method, row.gpu_type, count)] = candidate
                break
    if best is None:
        raise _no_fitting_error(
            facts,
            lora_r=lora_r,
            sequence_len=sequence_len,
            micro_batch_size=micro_batch_size,
            availability=availability,
            methods=search_methods,
            method=method,
            gpu_type=gpu_type,
            device_count=device_count,
        )
    alternatives = tuple(
        HardwareAlternative(
            gpu_type=c.gpu_type,
            device_count=c.device_count,
            price_per_hour=c.price_per_hour,
            headroom_gb=c.headroom_gb,
        )
        for c in found.values()
        if not (
            c.method == best.method
            and c.gpu_type == best.gpu_type
            and c.device_count == best.device_count
        )
    )
    return HardwarePlan(
        method=best.method,
        gpu_type=best.gpu_type,
        device_count=best.device_count,
        price_per_hour=best.price_per_hour,
        currency=best.currency,
        peak=best.peak,
        headroom_gb=best.headroom_gb,
        alternatives=alternatives,
    )
