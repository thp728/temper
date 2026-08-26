"""The effective job specification, resolved here and nowhere else.

The create-job page shows the user exactly what their job will freeze. Since
#83 that promise is also what launches: the orchestrator writes
`effective(overrides)` into the job spec whole, and the trainer applies what
it is given without resolving anything -- there is one resolver, this one, and
its answer is visible in the job's record. Alpha recomputes from rank when rank
moves alone, and rsLoRA is inferred at rank >= 32, both before launch.

The defaults and the overridable-key list are **not declared here**. They are
data in `packages/contracts/trainer-defaults.json` -- the one definition
(#82), read through this module and shipped into the trainer image at build
time because the image never installs this package (ADR-0010). This module
owns only the resolution rules around the data, so an edit to the JSON reaches
the page and the run together or not at all.

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
ALLOWED_OVERRIDES: set[str] = set(_loaded["allowed_overrides"])


def effective(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve overrides against the defaults, exactly as the trainer will.

    Returns the full specification including the inferred `lora_use_rslora`,
    because rsLoRA is inferred from the rank and never exposed as a choice --
    but a user reading the frozen spec should still see that it is in force.
    """
    cfg = dict(DEFAULTS)
    applied = {
        k: v for k, v in (overrides or {}).items() if k in ALLOWED_OVERRIDES
    }
    cfg.update(applied)

    # α is mechanically tied to r: a new rank never pairs with a stale scale.
    if "lora_r" in applied and "lora_alpha" not in applied:
        cfg["lora_alpha"] = 2 * int(cfg["lora_r"])

    # rsLoRA above rank 32: plain α/r scaling over-shrinks high-rank adapters.
    cfg["lora_use_rslora"] = int(cfg["lora_r"]) >= 32
    return cfg
