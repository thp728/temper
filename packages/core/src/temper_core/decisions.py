"""Decision records: every choice the predictor made, with why and the cost
of the alternatives.

Spec 005's product thesis, as data: a configuration that silently picks an L4
is not obviously better than one that asks. A record that says which card,
because of which constraint, and what the alternatives would have cost has
taught the user something and earned the right to decide for them. These
records are deliberately the same shape as the project's own ADRs -- decision,
value chosen, the constraint that forced it, alternatives with what each would
have cost -- turned into a product surface, and structured rather than rendered
so the API, the interface and the artifact manifest all read from one source.

Six decisions are recorded, in the order issue #76's acceptance criteria name
them: method, hardware, device count, disk, precision and sequence length.
All six are produced from the same inputs the quote is built from -- the model
facts, the effective hyperparameters, the chosen hardware plan and the disk
plan -- so a reason can never drift from the configuration it explains. The
records ride on the quote: frozen into the job spec at launch, shown before it
and after it, never regenerated on read, because an explanation that only
exists while the page is open is the black box this is built to remove.

The memory figures are `temper_core.memory`'s own arithmetic over the model's
facts -- the same numbers that decided whether the chosen card fits -- so an
alternative costs exactly what the predictor would have priced it at. Prices
travel with the account's currency, never assumed.

**Assumption, labelled:** the precision decision's chosen value ("nf4 (4-bit)")
and its NF4 arithmetic assume the executable method (qlora today, per
`selection.EXECUTABLE_METHODS`). This module is only called from the quote
path, which selects from that set, so the assumption cannot silently bite; the
day the trainer runs a second method, this decision grows a second honest
answer rather than a wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from temper_core import disk, gpus, memory, selection
from temper_core.models import ModelFacts


@dataclass(frozen=True)
class DecisionAlternative:
    """One configuration the predictor considered and did not choose.

    `cost` is what that alternative would have cost, with its unit and its
    measured-or-derived basis where the figure is not exact: a memory
    requirement for alternatives that change the footprint, a per-hour price
    for alternatives that change the hardware. `constraint` is why it lost.
    """

    value: str
    cost: str
    constraint: str


@dataclass(frozen=True)
class Decision:
    """One decision the predictor made on the user's behalf.

    The shape of an ADR turned into a product surface: the decision, the value
    chosen, the constraint that forced it, and the alternatives with what each
    would have cost. `alternatives` may be empty when nothing else was
    available -- an honest "there was no other option that fit" is itself a
    reason.
    """

    decision: str
    chosen: str
    constraint: str
    alternatives: tuple[DecisionAlternative, ...]


def to_dict(decision: Decision) -> dict[str, object]:
    """`decision` as the dict the contract and the frozen quote both carry.

    One serialization, so the API response, the browser and the artifact
    manifest cannot drift apart (spec 005: structured rather than rendered).
    """
    return {
        "decision": decision.decision,
        "chosen": decision.chosen,
        "constraint": decision.constraint,
        "alternatives": [
            {
                "value": a.value,
                "cost": a.cost,
                "constraint": a.constraint,
            }
            for a in decision.alternatives
        ],
    }


def _gb(gb: float) -> str:
    return f"{gb:.1f} GB"


def _price(price_per_hour: float, currency: str) -> str:
    """A per-hour price in the account's own currency. The provider prices in
    hundredths, so two decimals is the honest precision, never a formatting
    choice applied somewhere it can be forgotten."""
    return f"{currency} {price_per_hour:.2f}/hr"


def decide(
    facts: ModelFacts,
    *,
    hyperparameters: dict[str, Any],
    plan: selection.HardwarePlan,
    disk_plan: disk.DiskPlan,
) -> tuple[Decision, ...]:
    """The six decision records for one configuration, computed from the same
    seams the quote reads.

    Pure arithmetic over the inputs, no I/O: whether the chosen card fits, and
    what each alternative would have cost, are the same numbers `memory` and
    `disk` already produce -- this module arranges them as reasons, it does not
    re-derive them differently. The alternatives for hardware and device count
    are the fitting configurations `selection` considered and rejected
    (`plan.alternatives`); the alternatives for method, precision and sequence
    length are computed here from the model's facts, because those decisions
    cost in memory (which hardware you would need) rather than in a price the
    search already considered.
    """
    capacity = gpus.CAPACITY_GB[plan.gpu_type]
    lora_r = int(hyperparameters["lora_r"])
    sequence_len = int(hyperparameters["sequence_len"])
    micro_batch_size = int(hyperparameters["micro_batch_size"])

    # --- method ------------------------------------------------------------
    peak_lora = memory.predict_peak(
        facts,
        method="lora",
        lora_r=lora_r,
        sequence_len=sequence_len,
        micro_batch_size=micro_batch_size,
        device_count=plan.device_count,
    )
    peak_full = memory.predict_peak(
        facts,
        method="full",
        lora_r=lora_r,
        sequence_len=sequence_len,
        micro_batch_size=micro_batch_size,
        device_count=plan.device_count,
    )
    method_constraint = (
        f"the trainer can execute {plan.method} today, and selection picks the "
        f"cheapest executable method that fits; price never depends on method, "
        f"so a method that loses here loses on capability (spec 009 teaches "
        f"the trainer the rest), not on price"
    )
    method_alternatives = []
    lora_headroom = memory.headroom_gb(peak_lora, capacity)
    if lora_headroom >= 0:
        lora_cost = (
            f"needs {_gb(peak_lora.total_gb)} peak, fitting the {plan.gpu_type} "
            f"with {_gb(lora_headroom)} to spare"
        )
    else:
        lora_cost = (
            f"needs {_gb(peak_lora.total_gb)} peak, beyond the "
            f"{_gb(capacity)} {plan.gpu_type}"
        )
    method_alternatives.append(
        DecisionAlternative(
            value="lora",
            cost=lora_cost,
            constraint=(
                f"same {_price(plan.price_per_hour, plan.currency)} as "
                f"{plan.method}, but the trainer cannot execute LoRA yet"
            ),
        )
    )
    method_alternatives.append(
        DecisionAlternative(
            value="full fine-tune",
            cost=f"needs {_gb(peak_full.total_gb)} peak",
            constraint=(
                f"beyond the {_gb(capacity)} {plan.gpu_type}, and not "
                f"executable today even where a bigger card would hold it"
            ),
        )
    )

    # --- hardware ----------------------------------------------------------
    # The alternatives are the fitting configurations selection rejected,
    # cheapest first per card: the price the user gave up to take the chosen
    # card is the whole point of showing them.
    cheapest_per_type: dict[str, selection.HardwareAlternative] = {}
    for alt in sorted(
        plan.alternatives, key=lambda a: (a.price_per_hour, a.device_count)
    ):
        if alt.gpu_type == plan.gpu_type:
            continue
        cheapest_per_type.setdefault(alt.gpu_type, alt)
    hardware_constraint = (
        f"predicted peak is {_gb(plan.peak.total_gb)}; the {plan.gpu_type} "
        f"({_gb(capacity)}) is the cheapest card currently available that "
        f"holds it with {_gb(plan.headroom_gb)} to spare"
    )
    hardware_alternatives = [
        DecisionAlternative(
            value=alt.gpu_type,
            cost=_price(alt.price_per_hour, plan.currency),
            constraint=(
                f"fits ({_gb(alt.headroom_gb)} headroom on its "
                f"{_gb(gpus.CAPACITY_GB[alt.gpu_type])}), but costs more than "
                f"the {plan.gpu_type}"
            ),
        )
        for alt in cheapest_per_type.values()
    ]

    # --- device count ------------------------------------------------------
    # Each alternative carries its own total price from the node the search
    # found it on -- never derived from the chosen card's rate, which would
    # guess a price for hardware the provider may price differently.
    more_devices = sorted(
        (
            a
            for a in plan.alternatives
            if a.gpu_type == plan.gpu_type
            and a.device_count > plan.device_count
        ),
        key=lambda a: a.device_count,
    )
    device_constraint = (
        f"one {plan.gpu_type} holds the predicted peak of {_gb(plan.peak.total_gb)}. "
        f"{plan.method} has never run sharded, so extra cards do not shrink the "
        f"per-device footprint -- they only add interconnect overhead and cost, "
        f"and a configuration that asks for hardware the job does not need is "
        f"worse than a slower one"
    )
    device_alternatives = [
        DecisionAlternative(
            value=f"{alt.device_count} × {plan.gpu_type}",
            cost=_price(alt.price_per_hour, plan.currency),
            constraint=(
                f"{plan.method} has never run sharded, so the extra cards "
                f"replicate the same footprint rather than shrinking it"
            ),
        )
        for alt in more_devices[:2]
    ]

    # --- disk --------------------------------------------------------------
    disk_alternatives = []
    if disk_plan.required_gb < disk_plan.provisioned_gb:
        disk_alternatives.append(
            DecisionAlternative(
                value=f"the raw need ({_gb(disk_plan.required_gb)})",
                cost=(
                    "a smaller disk bills less on the separate USD storage line, "
                    "but the provider will not create one"
                ),
                constraint=(
                    f"the platform's smallest provisionable disk is "
                    f"{disk.PLATFORM_MIN_DISK_GB} GB (spike 5); the job's need "
                    f"is smaller, so the floor, not the need, is what bills"
                ),
            )
        )
    disk_constraint = (
        f"weights at download precision ({_gb(disk_plan.weights_gb)}) + retained "
        f"checkpoints ({_gb(disk_plan.checkpoints_gb)}) + the trainer image "
        f"({_gb(disk_plan.image_gb)}) + working space "
        f"({_gb(disk_plan.working_space_gb)}, an assumed allowance) = "
        f"{_gb(disk_plan.required_gb)} raw, floored at the platform minimum of "
        f"{disk.PLATFORM_MIN_DISK_GB} GB"
    )

    # --- precision ---------------------------------------------------------
    # The chosen method's weights pool is the NF4 one (qlora is the executable
    # method today, so this decision's "nf4 (4-bit)" is its only honest
    # answer); the alternative's pool is the bf16 base's, already computed for
    # the method decision. Both are `memory`'s own pools, read not retyped.
    nf4_weights_gb = plan.peak.weights_gb
    bf16_weights_gb = peak_lora.weights_gb
    precision_constraint = (
        f"{plan.method} quantises the frozen base to NF4 at "
        f"{memory.WEIGHT_BYTES_PER_PARAM[plan.method]:.1f} bytes/param, cutting "
        f"the weights pool from {_gb(bf16_weights_gb)} (bf16) to "
        f"{_gb(nf4_weights_gb)} -- the difference between fitting the "
        f"{plan.gpu_type} and renting something bigger"
    )
    precision_alternatives = [
        DecisionAlternative(
            value="bf16 (no quantisation)",
            cost=(
                f"holds the base at 2 bytes/param -- the weights pool grows "
                f"to {_gb(bf16_weights_gb)} and peak to "
                f"{_gb(peak_lora.total_gb)}, {_gb(bf16_weights_gb - nf4_weights_gb)} "
                f"more than NF4"
            ),
            constraint=(
                "4× the weights memory, which is what the unquantised method "
                "pays -- and the trainer does not execute it yet"
            ),
        )
    ]

    # --- sequence length ---------------------------------------------------
    longer_len = sequence_len * 2
    peak_longer = memory.predict_peak(
        facts,
        method=plan.method,
        lora_r=lora_r,
        sequence_len=longer_len,
        micro_batch_size=micro_batch_size,
        device_count=plan.device_count,
    )
    longer_headroom = memory.headroom_gb(peak_longer, capacity)
    if longer_headroom >= 0:
        longer_cost_suffix = f"still fits the {plan.gpu_type} with {_gb(longer_headroom)} to spare"
    else:
        longer_cost_suffix = f"beyond the {_gb(capacity)} {plan.gpu_type}"
    sequence_constraint = (
        f"the trainer default ({sequence_len}, trainer-defaults.json), well "
        f"inside the model's {facts.context_length} context so no row is "
        f"truncated against the context limit; activation memory "
        f"({_gb(plan.peak.activations_gb)} at this length) is the memory budget "
        f"the hardware was chosen against -- whether a row exceeds it is "
        f"token-counting (spec 006), not a number this predictor has"
    )
    sequence_alternatives = [
        DecisionAlternative(
            value=str(longer_len),
            cost=(
                f"doubles activation memory to {_gb(peak_longer.activations_gb)} "
                f"({_gb(peak_longer.total_gb)} peak) -- {longer_cost_suffix}"
            ),
            constraint=(
                "nothing in the dataset has asked for a longer window, and "
                "longer sequences cost more per pass"
            ),
        )
    ]

    return (
        Decision(
            "method",
            plan.method,
            method_constraint,
            tuple(method_alternatives),
        ),
        Decision(
            "hardware",
            plan.gpu_type,
            hardware_constraint,
            tuple(hardware_alternatives),
        ),
        Decision(
            "device count",
            str(plan.device_count),
            device_constraint,
            tuple(device_alternatives),
        ),
        Decision(
            "disk",
            f"{disk_plan.provisioned_gb} GB",
            disk_constraint,
            tuple(disk_alternatives),
        ),
        Decision(
            "precision",
            "nf4 (4-bit)",
            precision_constraint,
            tuple(precision_alternatives),
        ),
        Decision(
            "sequence length",
            str(sequence_len),
            sequence_constraint,
            tuple(sequence_alternatives),
        ),
    )
