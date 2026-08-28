"""The effective job specification, resolved here and nowhere else.

The create-job page shows the user exactly what their job will freeze. Since
#83 that promise is also what launches: the orchestrator writes
`effective(overrides)` into the job spec whole, and the trainer applies what
it is given without resolving anything -- there is one resolver, this one, and
its answer is visible in the job's record. Alpha recomputes from rank when rank
moves alone, and rsLoRA is inferred at rank >= 32, both before launch.

The defaults are **not declared here**. They are data in
`packages/contracts/trainer-defaults.json` -- the one definition (#82), read
through this module and shipped into the trainer image at build time because
the image never installs this package (ADR-0010). The same file carries the
**per-method table** (`by_method`, issue #66): the values that key off the
selected method -- a full fine-tune's lower learning rate -- so a method is a
second *footing* for the same defaults, not a second defaults file. The
*reachable* set -- which keys a user may override -- is no longer declared here
either: since #33 it is the exposed tier of the generated advanced surface
(`temper_core.surface.overrideable_keys`), so the refusal vocabulary and the
surface cannot drift apart. This module owns only the resolution rules around
the data, so an edit to either data file reaches the page and the run together
or not at all.

**Labelled assumption:** the read happens once at import and a missing file
stops the process -- the config.py rule that a value which cannot be honoured
must not be silently defaulted. That is only correct while this package lives
in the workspace tree it walks; if temper_core ever ships as a wheel outside
the monorepo, this lookup is the thing to revisit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import surface


def _contract_path() -> Path:
    """Find the contract by walking up to the workspace root.

    Marked on `pyproject.toml` beside `packages/` rather than depth-coded:
    a counted `parents[N]` breaks silently when this file moves, which is
    exactly how the control plane's own root lookup broke once already.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = (
            parent / "packages" / "contracts" / ("trainer-defaults.json")
        )
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"packages/contracts/trainer-defaults.json not found above "
        f"{__file__}; the workspace tree is incomplete"
    )


CONTRACT_PATH = _contract_path()
_loaded = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

DEFAULTS: dict[str, Any] = _loaded["defaults"]
# The per-method table (issue #66): the values that key off the selected
# method rather than a global constant. A full fine-tune updates every weight
# and trains at a lower learning rate than a QLoRA adapter does, and the
# difference is data in the one contract both the resolver and the trainer
# read (ADR-0010) -- never a second table, never a branch at the use site.
# Every key here is also a key in `defaults`, which is what keeps the
# trainer's required-key set (built from `defaults`) complete for every method.
BY_METHOD: dict[str, dict[str, Any]] = _loaded.get("by_method", {})
# The reachable set comes from the generated surface (issue #33): the exposed
# tier plus the platform's own internal keys (e.g. simulated_failure_code).
# `validate_overrides` in `temper_core.surface` is the gate; this set is what
# `effective` applies, so a key the gate rejects never reaches the spec.
ALLOWED_OVERRIDES: set[str] = set(surface.overrideable_keys())


def effective(
    overrides: dict[str, Any] | None, *, method: str | None = None
) -> dict[str, Any]:
    """Resolve overrides against the defaults, exactly as the trainer will.

    `method` selects the per-method table (`BY_METHOD`) when given, so a full
    fine-tune resolves its own learning rate rather than the adapter's. The
    method is chosen by the predictor at provisioning, so the caller that
    knows it (the orchestrator writing the job spec) passes it; callers that
    only shape a quote read the method-agnostic base.

    Returns the full specification including the inferred `lora_use_rslora`,
    because rsLoRA is inferred from the rank and never exposed as a choice --
    but a user reading the frozen spec should still see that it is in force.
    """
    cfg = dict(DEFAULTS)
    if method:
        method_defaults = BY_METHOD.get(method)
        if method_defaults:
            cfg.update(method_defaults)
    applied = {
        k: v for k, v in (overrides or {}).items() if k in ALLOWED_OVERRIDES
    }
    # Coerced to the schema's type (issue #80): an override arrives as a
    # string from a browser form, and the schema knows `lora_r` is an int and
    # `learning_rate` a float. The resolver is the single place a value is
    # normalised, so the quote, the orchestrator and the trainer's spec all
    # see the same typed value.
    cfg.update({k: surface.coerce_value(k, v) for k, v in applied.items()})

    # α is mechanically tied to r: a new rank never pairs with a stale scale.
    if "lora_r" in applied and "lora_alpha" not in applied:
        cfg["lora_alpha"] = 2 * int(cfg["lora_r"])

    # rsLoRA above rank 32: plain α/r scaling over-shrinks high-rank adapters.
    cfg["lora_use_rslora"] = int(cfg["lora_r"]) >= 32
    return cfg
