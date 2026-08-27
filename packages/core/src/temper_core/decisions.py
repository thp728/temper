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

from temper_core import disk, gpus, hyperparams, memory, overrides, selection
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
    reason. `overridden` (issue #79) marks a decision the user changed rather
    than one the predictor made, so the plan can show the difference.
    """

    decision: str
    chosen: str
    constraint: str
    alternatives: tuple[DecisionAlternative, ...]
    overridden: bool = False


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
        "overridden": decision.overridden,
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
    overridden: frozenset[str] = frozenset(),
) -> tuple[Decision, ...]:
    """The six decision records for one configuration, computed from the same
    seams the quote reads.

    Pure arithmetic over the inputs, no I/O: whether the chosen card fits, and
    what each alternative would have cost, are the same numbers `memory` and
    `disk` already produce -- this module arranges them as reasons, it does
    not re-derive them differently. The alternatives for hardware and device
    count are the fitting configurations `selection` considered and rejected
    (`plan.alternatives`); the alternatives for method, precision and sequence
    length are computed here from the model's facts, because those decisions
    cost in memory (which hardware you would need) rather than in a price the
    search already considered.

    `overridden` (issue #79) is the set of decision names the user pinned
    rather than the predictor chose; each such record still explains its
    configuration honestly, but the constraint acknowledges the override and
    the `overridden` flag travels with the record so the plan can mark it.
    """
    capacity = gpus.CAPACITY_GB[plan.gpu_type]
    lora_r = int(hyperparameters["lora_r"])
    sequence_len = int(hyperparameters["sequence_len"])
    micro_batch_size = int(hyperparameters["micro_batch_size"])
    default_sequence_len = int(hyperparams.DEFAULTS["sequence_len"])

    def peak_for(method: str, seq: int) -> memory.PeakMemory:
        return memory.predict_peak(
            facts,
            method=method,
            lora_r=lora_r,
            sequence_len=seq,
            micro_batch_size=micro_batch_size,
            device_count=plan.device_count,
        )

    peak_lora = peak_for("lora", sequence_len)

    # --- method ------------------------------------------------------------
    # The alternatives are the other methods the user could have trained with,
    # each costing in memory -- the reason the chosen method wins. Price never
    # depends on method, so a method that loses here loses on capability
    # (spec 009 teaches the trainer the rest), not on price.
    method_alternatives = []
    for m, label in (
        ("full", "full fine-tune"),
        ("lora", "lora"),
        ("qlora", "qlora"),
    ):
        if m == plan.method:
            continue
        peak_m = peak_for(m, sequence_len)
        headroom_m = memory.headroom_gb(peak_m, capacity)
        if headroom_m >= 0:
            cost_m = (
                f"needs {_gb(peak_m.total_gb)} peak, fitting the "
                f"{plan.gpu_type} with {_gb(headroom_m)} to spare"
            )
        else:
            cost_m = (
                f"needs {_gb(peak_m.total_gb)} peak, beyond the "
                f"{_gb(capacity)} {plan.gpu_type}"
            )
        if m == "full":
            if headroom_m >= 0:
                constraint_m = (
                    "fits, but the trainer does not execute a full "
                    "fine-tune yet (spec 009)"
                )
            else:
                constraint_m = (
                    f"beyond the {_gb(capacity)} {plan.gpu_type}, and not "
                    f"executable today even where a bigger card would hold it"
                )
        elif m == "lora":
            constraint_m = (
                f"same {_price(plan.price_per_hour, plan.currency)} as "
                f"{plan.method}, but the trainer cannot execute LoRA yet"
            )
        else:  # qlora
            constraint_m = (
                f"the trainer executes it today -- NF4 at "
                f"{memory.WEIGHT_BYTES_PER_PARAM['qlora']:.1f} bytes/param is "
                f"the cheapest footprint that fits"
            )
        method_alternatives.append(
            DecisionAlternative(label, cost_m, constraint_m)
        )
    if "method" in overridden:
        method_constraint = (
            f"you chose {plan.method}. The trainer executes only "
            f"{overrides.executable_names()} today (spec 009), so a launch of "
            f"this override will be refused until it teaches the rest; price "
            f"never depends on method"
        )
    else:
        method_constraint = (
            f"the trainer can execute {plan.method} today, and selection picks "
            f"the cheapest executable method that fits; price never depends "
            f"on method, so a method that loses here loses on capability "
            f"(spec 009 teaches the trainer the rest), not on price"
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
    if "hardware" in overridden:
        hardware_constraint = (
            f"you chose the {plan.gpu_type}; it holds the predicted peak of "
            f"{_gb(plan.peak.total_gb)} with {_gb(plan.headroom_gb)} to spare "
            f"at {_price(plan.price_per_hour, plan.currency)}. The predictor "
            f"would have picked the cheapest card that holds it"
        )
    else:
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
    # guess a price for hardware the provider may price differently. When the
    # count was overridden, selection searched that count alone, so the
    # natural alternative -- the predictor's single card -- is derived from
    # the chosen configuration's own per-card rate.
    other_counts = sorted(
        (
            a
            for a in plan.alternatives
            if a.gpu_type == plan.gpu_type
            and a.device_count != plan.device_count
        ),
        key=lambda a: a.device_count,
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
        for alt in other_counts[:2]
    ]
    if "device count" in overridden and plan.device_count > 1:
        device_alternatives.insert(
            0,
            DecisionAlternative(
                value=f"1 × {plan.gpu_type}",
                cost=_price(
                    plan.price_per_hour / plan.device_count, plan.currency
                ),
                constraint=(
                    f"the predictor would choose one card: {plan.method} has "
                    f"never run sharded, so the extra cards only add "
                    f"interconnect overhead and cost"
                ),
            ),
        )
    if "device count" in overridden:
        device_constraint = (
            f"you chose {plan.device_count} × {plan.gpu_type}. One {plan.gpu_type} "
            f"holds the predicted peak of {_gb(plan.peak.total_gb)}; "
            f"{plan.method} has never run sharded, so extra cards do not "
            f"shrink the per-device footprint, and a launch of more than one "
            f"device is spec 009's territory"
        )
    else:
        device_constraint = (
            f"one {plan.gpu_type} holds the predicted peak of "
            f"{_gb(plan.peak.total_gb)}. {plan.method} has never run sharded, "
            f"so extra cards do not shrink the per-device footprint -- they "
            f"only add interconnect overhead and cost, and a configuration "
            f"that asks for hardware the job does not need is worse than a "
            f"slower one"
        )

    # --- disk --------------------------------------------------------------
    disk_alternatives = []
    if disk_plan.required_gb < disk_plan.provisioned_gb:
        if "disk" in overridden:
            # The user chose a disk above the job's need; the alternative is
            # the predictor's own floor, recomputed without the override.
            try:
                default_disk = disk.required_disk(
                    facts,
                    method=plan.method,
                    lora_r=lora_r,
                    retained_checkpoints=int(
                        hyperparameters["save_total_limit"]
                    ),
                )
            except Exception:  # noqa: BLE001 - a reason must never raise
                default_disk = disk_plan
            disk_alternatives.append(
                DecisionAlternative(
                    value=f"the predictor's floor ({_gb(default_disk.provisioned_gb)})",
                    cost=(
                        "a smaller disk bills less on the separate USD storage "
                        "line, but the provider will not create one below "
                        f"{disk.PLATFORM_MIN_DISK_GB} GB"
                    ),
                    constraint=(
                        f"your {_gb(disk_plan.provisioned_gb)} is what you "
                        f"asked for; the job's raw need is only "
                        f"{_gb(disk_plan.required_gb)}, so the extra is your "
                        f"choice, billed on the USD line"
                    ),
                )
            )
        else:
            disk_alternatives.append(
                DecisionAlternative(
                    value=f"the raw need ({_gb(disk_plan.required_gb)})",
                    cost=(
                        "a smaller disk bills less on the separate USD storage "
                        "line, but the provider will not create one"
                    ),
                    constraint=(
                        f"the platform's smallest provisionable disk is "
                        f"{disk.PLATFORM_MIN_DISK_GB} GB (spike 5); the job's "
                        f"need is smaller, so the floor, not the need, is what "
                        f"bills"
                    ),
                )
            )
    if "disk" in overridden:
        disk_constraint = (
            f"you chose {_gb(disk_plan.provisioned_gb)}. The job's raw need "
            f"is {_gb(disk_plan.required_gb)} (weights at download precision "
            f"{_gb(disk_plan.weights_gb)} + retained checkpoints "
            f"{_gb(disk_plan.checkpoints_gb)} + the trainer image "
            f"{_gb(disk_plan.image_gb)} + working space "
            f"{_gb(disk_plan.working_space_gb)}, an assumed allowance), so "
            f"the extra bills on the separate USD storage line"
        )
    else:
        disk_constraint = (
            f"weights at download precision ({_gb(disk_plan.weights_gb)}) + "
            f"retained checkpoints ({_gb(disk_plan.checkpoints_gb)}) + the "
            f"trainer image ({_gb(disk_plan.image_gb)}) + working space "
            f"({_gb(disk_plan.working_space_gb)}, an assumed allowance) = "
            f"{_gb(disk_plan.required_gb)} raw, floored at the platform "
            f"minimum of {disk.PLATFORM_MIN_DISK_GB} GB"
        )

    # --- precision ---------------------------------------------------------
    # The precision is the chosen method's own: qlora quantises the frozen
    # base to NF4, lora and full hold it at bf16 (`overrides.METHOD_PRECISION`,
    # one definition, read not retyped). The alternative is the other
    # precision, whose weights pool `memory` has already computed. The NF4
    # pool is qlora's own arithmetic regardless of the chosen method -- the
    # frozen base always costs 0.5 bytes/param when quantised, never a
    # bf16-sized pool dressed up as NF4.
    precision_chosen = overrides.METHOD_PRECISION[plan.method]
    peak_qlora = peak_for("qlora", sequence_len)
    nf4_weights_gb = peak_qlora.weights_gb
    bf16_weights_gb = peak_lora.weights_gb
    if precision_chosen == overrides.PRECISION_NF4:
        if "precision" in overridden:
            precision_constraint = (
                f"you chose NF4 ({memory.WEIGHT_BYTES_PER_PARAM['qlora']:.1f} "
                f"bytes/param), the quantised footing that fits the "
                f"{plan.gpu_type} -- the trainer's own default"
            )
        else:
            precision_constraint = (
                f"{plan.method} quantises the frozen base to NF4 at "
                f"{memory.WEIGHT_BYTES_PER_PARAM[plan.method]:.1f} "
                f"bytes/param, cutting the weights pool from "
                f"{_gb(bf16_weights_gb)} (bf16) to {_gb(nf4_weights_gb)} -- "
                f"the difference between fitting the {plan.gpu_type} and "
                f"renting something bigger"
            )
        precision_alternatives = [
            DecisionAlternative(
                value=overrides.PRECISION_BF16,
                cost=(
                    f"holds the base at 2 bytes/param -- the weights pool "
                    f"grows to {_gb(bf16_weights_gb)} and peak to "
                    f"{_gb(peak_lora.total_gb)}, "
                    f"{_gb(bf16_weights_gb - nf4_weights_gb)} more than NF4"
                ),
                constraint=(
                    "4× the weights memory, which is what the unquantised "
                    "method pays -- and the trainer does not execute it yet"
                ),
            )
        ]
    else:  # bf16 chosen (lora or full method)
        if "precision" in overridden:
            precision_constraint = (
                "you chose bf16 (2 bytes/param), the unquantised footing "
                "that costs 4× the weights memory of NF4 -- LoRA's price, "
                "and the trainer does not execute it yet"
            )
        else:
            precision_constraint = (
                f"{plan.method} holds the base at bf16 (2 bytes/param) -- the "
                f"unquantised price of training every weight, "
                f"{_gb(bf16_weights_gb - nf4_weights_gb)} more weights memory "
                f"than NF4"
            )
        precision_alternatives = [
            DecisionAlternative(
                value=overrides.PRECISION_NF4,
                cost=(
                    f"quantises the frozen base to NF4 at "
                    f"{memory.WEIGHT_BYTES_PER_PARAM['qlora']:.1f} "
                    f"bytes/param -- the weights pool shrinks to "
                    f"{_gb(nf4_weights_gb)}"
                ),
                constraint=(
                    f"the quantised footing the trainer executes today -- it "
                    f"is the cheapest footprint that fits the {plan.gpu_type}"
                ),
            )
        ]

    # --- sequence length ---------------------------------------------------
    # The alternative is the trainer default at the other end of the same
    # activation arithmetic: doubling the window doubles activation memory,
    # so the reason names the pool the hardware was chosen against.
    if sequence_len == default_sequence_len:
        other_len = default_sequence_len * 2
        alternative_peak = peak_for(plan.method, other_len)
        other_headroom = memory.headroom_gb(alternative_peak, capacity)
        if other_headroom >= 0:
            longer_cost_suffix = (
                f"still fits the {plan.gpu_type} with "
                f"{_gb(other_headroom)} to spare"
            )
        else:
            longer_cost_suffix = f"beyond the {_gb(capacity)} {plan.gpu_type}"
        sequence_alternatives = [
            DecisionAlternative(
                value=str(other_len),
                cost=(
                    f"doubles activation memory to "
                    f"{_gb(alternative_peak.activations_gb)} "
                    f"({_gb(alternative_peak.total_gb)} peak) -- "
                    f"{longer_cost_suffix}"
                ),
                constraint=(
                    "nothing in the dataset has asked for a longer window, "
                    "and longer sequences cost more per pass"
                ),
            )
        ]
        sequence_constraint = (
            f"the trainer default ({sequence_len}, trainer-defaults.json), "
            f"well inside the model's {facts.context_length} context so no "
            f"row is truncated against the context limit; activation memory "
            f"({_gb(plan.peak.activations_gb)} at this length) is the memory "
            f"budget the hardware was chosen against -- whether a row exceeds "
            f"it is token-counting (spec 006), not a number this predictor "
            f"has"
        )
    else:
        default_peak = peak_for(plan.method, default_sequence_len)
        sequence_alternatives = [
            DecisionAlternative(
                value=str(default_sequence_len),
                cost=(
                    f"cuts activation memory to "
                    f"{_gb(default_peak.activations_gb)} "
                    f"({_gb(default_peak.total_gb)} peak), back to the "
                    f"trainer default"
                ),
                constraint=(
                    f"you overrode the trainer default ({default_sequence_len}) "
                    f"to {sequence_len}; reverting is what this would cost"
                ),
            )
        ]
        sequence_constraint = (
            f"you chose {sequence_len}; the trainer default is "
            f"{default_sequence_len} (trainer-defaults.json), still inside the "
            f"model's {facts.context_length} context. Activation memory at "
            f"this length is {_gb(plan.peak.activations_gb)} -- the memory "
            f"budget the hardware was chosen against"
        )

    return (
        Decision(
            "method",
            plan.method,
            method_constraint,
            tuple(method_alternatives),
            overridden="method" in overridden,
        ),
        Decision(
            "hardware",
            plan.gpu_type,
            hardware_constraint,
            tuple(hardware_alternatives),
            overridden="hardware" in overridden,
        ),
        Decision(
            "device count",
            str(plan.device_count),
            device_constraint,
            tuple(device_alternatives),
            overridden="device count" in overridden,
        ),
        Decision(
            "disk",
            f"{disk_plan.provisioned_gb} GB",
            disk_constraint,
            tuple(disk_alternatives),
            overridden="disk" in overridden,
        ),
        Decision(
            "precision",
            precision_chosen,
            precision_constraint,
            tuple(precision_alternatives),
            overridden="precision" in overridden,
        ),
        Decision(
            "sequence length",
            str(sequence_len),
            sequence_constraint,
            tuple(sequence_alternatives),
            overridden="sequence length" in overridden,
        ),
    )
