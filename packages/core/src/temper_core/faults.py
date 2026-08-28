"""The fault surface's vocabulary (issue #24), read from one data file.

The fault surface is how a deliberately broken run is made, so that a recovery
path can be proven rather than argued: a reviewer turns on a fault and watches
what the product does. Its vocabulary -- the six faults, the code each carries
and the side responsible for making it real -- is data in
`packages/contracts/fault-surface.json`, because two very different code paths
must agree on it: the fake provider and the orchestrator (control plane) and
the trainer (which reads the data file directly, never this module -- the
trainer image does not install `temper_core`, ADR-0010).

The split of responsibility is the spec's constraint: provider-side faults are
behaviour of the provider seam (only the fake provider honours them), and
trainer-side faults are an environment switch the trainer reads. `side` in the
data is that split, written down so the two sides cannot drift about who does
what.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The one hyperparameter key the fault surface travels under. It was the
# reserved test affordance that asked the simulated machine to end early with
# a named code (ADR-0026); the fault surface is that mechanism grown (issue
# #24): a string value still ends the machine early with that code, and a
# dict value names a fault from the surface. Defined here and read everywhere,
# so `surface.PLATFORM_INTERNAL_KEYS` and the fake provider cannot drift from
# what the orchestrator looks for.
HYPERPARAMETER_KEY = "simulated_failure_code"

# The environment variable the *trainer* reads its fault from. Trainer-side
# faults are an environment switch (the spec's constraint): the control plane
# writes the job's fault spec into this variable on the machine, and the
# trainer reads it. The trainer image cannot import this package (ADR-0010),
# so the trainer declares its own constant with the same name; a trainer test
# pins the two equal, which is how a value two components must agree on stays
# defined once.
FAULT_ENV = "TEMPER_FAULT_SPEC"


def _contract_path() -> Path:
    """Find `fault-surface.json` by walking up to the workspace root.

    The same lookup every other contract reader in this package uses
    (`surface.py`, `hyperparams.py`): a missing file stops the process rather
    than defaulting, so a broken checkout is a boot failure, not a silently
    narrower surface.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "packages" / "contracts" / "fault-surface.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "packages/contracts/fault-surface.json not found above " + __file__
    )


_DOC: dict[str, Any] = json.loads(_contract_path().read_text(encoding="utf-8"))
# name -> the fault record, computed once.
_FAULTS: dict[str, dict[str, Any]] = {
    entry["name"]: entry for entry in _DOC["faults"]
}


def names() -> tuple[str, ...]:
    """Every fault name, in the contract's order."""
    return tuple(_FAULTS)


def fault(name: str) -> dict[str, Any] | None:
    """The fault record, or None when the name is not part of the surface."""
    return _FAULTS.get(name)


def is_known(name: str) -> bool:
    return name in _FAULTS


def code_for(name: str) -> str | None:
    """The stable error code a run broken by `name` carries, or None.

    Every injected fault is named in the run's history so a deliberately
    broken run can never be mistaken for a real one; for the faults that
    produce their own result document (oom, divergence) that naming includes
    a `simulated_` code in the run's record. The other faults fail through
    the platform's ordinary machinery -- the stall detector, a missing result
    document, teardown -- so they carry the platform's ordinary codes and are
    named by the history instead; their `code` is deliberately absent from
    the contract.
    """
    entry = _FAULTS.get(name)
    if entry is None:
        raise ValueError(f"unknown simulated fault {name!r}")
    code = entry.get("code")
    return str(code) if code is not None else None


def describe(name: str) -> str:
    """The plain-language description, for the history and the docs."""
    entry = _FAULTS.get(name)
    if entry is None:
        raise ValueError(f"unknown simulated fault {name!r}")
    return str(entry["description"])


def side_of(name: str) -> str:
    """Which side makes the fault real: 'trainer' or 'provider'."""
    entry = _FAULTS.get(name)
    if entry is None:
        raise ValueError(f"unknown simulated fault {name!r}")
    return str(entry["side"])


def params_for(name: str) -> tuple[str, ...]:
    """The parameter names `name` accepts, in the contract's order."""
    entry = _FAULTS.get(name)
    if entry is None:
        raise ValueError(f"unknown simulated fault {name!r}")
    return tuple(entry.get("params") or ())


def spec_error(spec: dict[str, Any]) -> str | None:
    """A reason this fault spec is invalid, or None when it is well-formed.

    One validator for the create path and the provider seam: an unknown fault
    name, an unknown parameter, or a malformed parameter is refused loudly
    rather than half-honoured -- a fault spec the caller believes is in
    effect but is not is worse than a refusal. The trainer runs the same
    checks against the same contract data (it cannot import this module,
    ADR-0010), so the two sides cannot drift about what a spec may carry.
    """
    name = spec.get("name")
    if not isinstance(name, str) or not is_known(name):
        return f"'{name}' is not a fault the surface knows."
    allowed = {*params_for(name), "name"}
    unknown = sorted(k for k in spec if k not in allowed)
    if unknown:
        return (
            f"fault '{name}' does not take parameter(s) {unknown}; a "
            "parameter the caller believes is in effect but is not is worse "
            "than a refusal."
        )
    for key in ("after_line", "times", "machine_id"):
        if key in spec:
            try:
                int(spec[key])
            except (TypeError, ValueError):
                return (
                    f"fault parameter '{key}' must be a whole number, got "
                    f"{spec[key]!r}"
                )
    if "delay_s" in spec:
        try:
            delay = float(spec["delay_s"])
        except (TypeError, ValueError):
            return (
                f"fault parameter 'delay_s' must be a number, got "
                f"{spec['delay_s']!r}"
            )
        if delay <= 0:
            return (
                f"fault parameter 'delay_s' must be positive, got "
                f"{spec['delay_s']!r}"
            )
    return None


def from_hyperparameters(
    hyperparameters: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """The dict fault spec a job carries, or None when the surface is off.

    The single definition of how the surface is switched on per job: the
    platform-internal hyperparameter holds either a string (the reserved
    early-exit affordance, unchanged) or a dict naming a fault. The
    orchestrator reads this to name the fault in the job's history and to
    gate whether the trainer's environment is told about it.
    """
    requested = (hyperparameters or {}).get(HYPERPARAMETER_KEY)
    if isinstance(requested, dict):
        return requested
    return None
