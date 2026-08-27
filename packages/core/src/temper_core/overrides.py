"""Plan overrides: pin one decision, recompute the rest.

Issue #79. Spec 005's product thesis turned interactive: the plan a first-time
user reads as an explanation is the plan an experienced user edits. A new
choice beside a stale value is a silent quality bug -- the same reason a rank
change recomputes its scale -- so an override never mutates the plan it landed
on; it is re-requested, and the rest recomputes around it with the recomputation
rules in one place (`temper_core.selection` for hardware, `temper_core.disk`
for disk, `temper_core.memory` for the fit check).

This module owns only the vocabulary and the coupling -- what each of the six
decisions can be overridden to, and how method and precision move together. It
turns the user's `{decision, value}` pairs into the typed pins `selection` and
`disk` accept plus the hyperparameters with `sequence length` applied. It does
no arithmetic of its own: whether the pinned configuration fits is
`selection.select_hardware`'s refusal, computed from the same `memory` pools
the unpinned search uses, so taking control cannot lose the safety property.

**The override value is the plan's own vocabulary.** Each decision's `chosen`
string in the quote (`temper_core.decisions`) is exactly the value this module
accepts for that decision, so the control sits beside the explanation it
edits (spec 005: one surface, not a beginner mode and an expert mode).

**Precision and method are one coupled decision, named twice.** QLoRA
quantises the frozen base to NF4; LoRA and full fine-tuning hold it at bf16
(`temper_core.memory.WEIGHT_BYTES_PER_PARAM`). Overriding precision to "bf16"
therefore *means* "train LoRA", and overriding method to "lora" *means*
"bf16". Naming one and letting the other be derived is honest; naming both and
having them disagree is refused (`InconsistentOverridesError`), never silently
resolved, because a configuration that contradicts itself would describe a job
nobody intended.

**Executability is separate from feasibility.** `executable` answers whether
the trainer can run a configuration *today* (QLoRA, single device -- spec
005's "running them is Spec 009"); the plan may still describe a configuration
the trainer cannot run, because a predictor that cannot describe it cannot
refuse it honestly. The launch is where that description stops being a quote.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import gpus, selection

# The six decisions, in the order #76's acceptance criteria name them. This is
# the one place the override surface is named; `decisions.decide` produces a
# record per name and this module accepts a value per name.
DECISIONS: tuple[str, ...] = (
    "method",
    "hardware",
    "device count",
    "disk",
    "precision",
    "sequence length",
)

# The two training precisions the predictor can describe. The strings are the
# same ones `decisions.decide` shows as a precision decision's chosen value,
# so the control and the explanation share a vocabulary.
PRECISION_NF4 = "nf4 (4-bit)"
PRECISION_BF16 = "bf16 (no quantisation)"

# method -> the precision that method trains at. qlora quantises the frozen
# base to NF4 (`memory.WEIGHT_BYTES_PER_PARAM["qlora"] == 0.5`); lora and full
# hold it at bf16 (2.0). The coupling is load-bearing, not cosmetic: overriding
# one side and leaving the other stale is exactly the silent quality bug issue
# #79 exists to refuse.
METHOD_PRECISION: dict[str, str] = {
    "qlora": PRECISION_NF4,
    "lora": PRECISION_BF16,
    "full": PRECISION_BF16,
}

# Device-count and sequence-length values are integers as strings; disk is "N GB".
_INTEGER_DECISIONS = ("device count", "sequence length")


class OverrideError(Exception):
    """A refusal to apply an override, carrying the stable code and the
    structured fields a refusal surface (the API's error detail, an inline
    plan control) can render. Raised only for refusals that need no provider:
    an override that cannot be *described* is refused here; one that cannot
    *fit* is refused by `selection`/`disk` arithmetic."""

    code = "override_error"

    def __init__(self, message: str, **fields: object) -> None:
        self.message = message
        self.fields = fields
        super().__init__(message)

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, **self.fields}


class UnknownDecisionError(OverrideError):
    """A decision name that is not one the predictor makes. Mirrors the
    `unknown_hyperparameter` rule for the hyperparameter surface: a key the
    caller believes is in effect but is not is worse than a refusal."""

    code = "unknown_decision"

    def __init__(self, decision: str) -> None:
        self.decision = decision
        super().__init__(
            f"'{decision}' is not a decision the predictor makes. Known "
            f"decisions: {', '.join(DECISIONS)}.",
            decision=decision,
            known=list(DECISIONS),
        )


class UnknownOverrideValueError(OverrideError):
    """A value that is not in the decision's vocabulary. The allowed set rides
    on the error so the surface can offer the legal values rather than a bare
    refusal."""

    code = "unknown_override_value"

    def __init__(
        self, decision: str, value: str, allowed: Sequence[str] | None = None
    ) -> None:
        self.decision = decision
        self.value = value
        fields: dict[str, object] = {"decision": decision, "value": value}
        message = (
            f"'{value}' is not a value '{decision}' can be overridden to."
        )
        if allowed is not None:
            message += f" Allowed: {', '.join(allowed)}."
            fields["allowed"] = list(allowed)
        super().__init__(message, **fields)


class InvalidOverrideValueError(OverrideError):
    """A value that is in the right shape but out of range (a device count or
    sequence length below one)."""

    code = "invalid_override_value"

    def __init__(self, decision: str, value: str, reason: str) -> None:
        self.decision = decision
        self.value = value
        super().__init__(
            f"'{value}' is not a valid '{decision}': {reason}.",
            decision=decision,
            value=value,
            reason=reason,
        )


class InconsistentOverridesError(OverrideError):
    """Precision and method overrides that contradict each other. Refused
    rather than resolved, because a configuration that contradicts itself
    describes a job nobody intended (see the module docstring)."""

    code = "inconsistent_overrides"

    def __init__(self, method: str, precision: str) -> None:
        self.method = method
        self.precision = precision
        super().__init__(
            f"{method} trains at {METHOD_PRECISION[method]}, but you also "
            f"overrode precision to {precision}. Name one, and the other "
            f"follows.",
            method=method,
            precision=precision,
        )


@dataclass(frozen=True)
class Override:
    """One user change to one decision: which decision, and the value to pin
    it to. The value is the decision record's own vocabulary (see
    `resolve`)."""

    decision: str
    value: str


@dataclass(frozen=True)
class ResolvedOverrides:
    """What a set of overrides pins, and what the rest must recompute from.

    `method` is the effective training method after the precision<->method
    coupling is applied (None when neither was overridden, so the predictor
    keeps choosing). `hyperparameters` is the base specification with any
    `sequence length` override applied. `overridden` is the set of decision
    names the user changed, so the plan can mark exactly those as overridden.
    """

    method: str | None
    gpu_type: str | None
    device_count: int | None
    disk_gb: int | None
    precision: str | None
    hyperparameters: dict[str, Any]
    overridden: frozenset[str]


def to_dict(override: Override) -> dict[str, str]:
    """`override` as the dict the contract and the frozen job spec carry."""
    return {"decision": override.decision, "value": override.value}


def from_dict(data: dict[str, Any]) -> Override:
    return Override(decision=data["decision"], value=str(data["value"]))


def resolve(
    hyperparameters: dict[str, Any], overrides: Sequence[Override]
) -> ResolvedOverrides:
    """The typed pins and effective specification for `overrides`.

    `hyperparameters` is the *effective* base specification the predictor
    would have used (see `temper_core.hyperparams.effective`); a `sequence
    length` override is applied on top, so the rest of the plan recomputes
    around it. Raises an `OverrideError` subclass for every refusal that needs
    no provider; whether the pinned configuration then *fits* is
    `selection`'s/`disk`'s arithmetic, not this module's.
    """
    hp = dict(hyperparameters)
    method: str | None = None
    gpu_type: str | None = None
    device_count: int | None = None
    disk_gb: int | None = None
    precision: str | None = None
    overridden: set[str] = set()

    for ov in overrides:
        d, v = ov.decision, ov.value
        if d not in DECISIONS:
            raise UnknownDecisionError(d)
        if d == "method":
            if v not in METHOD_PRECISION:
                raise UnknownOverrideValueError(
                    d, v, allowed=list(METHOD_PRECISION)
                )
            method = v
        elif d == "hardware":
            if v not in gpus.CAPACITY_GB:
                raise UnknownOverrideValueError(
                    d, v, allowed=sorted(gpus.CAPACITY_GB)
                )
            gpu_type = v
        elif d in _INTEGER_DECISIONS:
            try:
                n = int(v)
            except ValueError:
                raise UnknownOverrideValueError(
                    d, v, allowed=["a positive integer"]
                ) from None
            if n < 1:
                raise InvalidOverrideValueError(d, v, "it must be at least 1")
            if d == "device count":
                device_count = n
            else:
                hp["sequence_len"] = n
        elif d == "disk":
            match = re.fullmatch(r"\s*(\d+)\s*GB\s*", v)
            if match is None:
                raise UnknownOverrideValueError(d, v, allowed=["'N GB'"])
            disk_gb = int(match.group(1))
        elif d == "precision":
            if v not in (PRECISION_NF4, PRECISION_BF16):
                raise UnknownOverrideValueError(
                    d, v, allowed=[PRECISION_NF4, PRECISION_BF16]
                )
            precision = v
        overridden.add(d)

    # The precision<->method coupling, applied after every override is read so
    # an inconsistent pair is caught whole rather than half-applied.
    if method is not None and precision is not None:
        if METHOD_PRECISION[method] != precision:
            raise InconsistentOverridesError(method, precision)
    elif precision is not None and method is None:
        # The user named the precision, so the method follows: NF4 is QLoRA's
        # footing, bf16 is LoRA's. "full" is never inferred from a precision
        # -- a full fine-tune is a different decision, not a dtype.
        method = "qlora" if precision == PRECISION_NF4 else "lora"

    return ResolvedOverrides(
        method=method,
        gpu_type=gpu_type,
        device_count=device_count,
        disk_gb=disk_gb,
        precision=precision,
        hyperparameters=hp,
        overridden=frozenset(overridden),
    )


def executable(method: str, device_count: int) -> bool:
    """Whether the trainer can run this configuration today.

    QLoRA alone is executable (`selection.EXECUTABLE_METHODS`); multi-GPU
    sharding has never shipped (spike 6), so more than one device is spec
    009's territory even where the method itself would run. Separated from
    feasibility on purpose: `executable` names what can be *run*, and
    `selection`/`memory` name what can *fit*.
    """
    return method in selection.EXECUTABLE_METHODS and device_count == 1


def executable_names() -> str:
    """The executable methods as an English list, read not retyped -- one
    rendering, shared by the reason records and the launch refusal."""
    names = [f"'{m}'" for m in selection.EXECUTABLE_METHODS]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def executable_default() -> str:
    """The method a job with no method override runs: the executable one."""
    return selection.EXECUTABLE_METHODS[0]


def options_for(decision: str) -> list[str] | None:
    """The legal values a select-style control can offer for `decision`, or
    None for the free-form numeric decisions (device count, disk, sequence
    length), which accept any positive integer / any 'N GB'.

    The values are the decision's own vocabulary (the strings `resolve`
    accepts and `decisions.decide` shows as `chosen`), read from one place --
    the plan control is generated from this, never hand-listed in the
    interface.
    """
    if decision == "method":
        return list(METHOD_PRECISION)
    if decision == "hardware":
        return sorted(gpus.CAPACITY_GB)
    if decision == "precision":
        return [PRECISION_NF4, PRECISION_BF16]
    return None
