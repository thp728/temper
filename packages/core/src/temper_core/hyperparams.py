"""The effective job specification, resolved here and nowhere else.

The create-job page shows the user exactly what their job will freeze. Since
#83 that promise is also what launches: the orchestrator writes
`effective(overrides)` into the job spec whole, and the trainer applies what
it is given without resolving anything -- there is one resolver, this one, and
its answer is visible in the job's record. Alpha recomputes from rank when rank
moves alone, and rsLoRA is inferred at rank >= 32, both before launch.

This table was once a hand-maintained mirror of defaults in
`apps/trainer/entrypoint.py`; those copies are gone now, so these literals are
the definition until #82 moves them into `packages/contracts/`. The trainer's
required-key set is pinned to `effective({})` by
`apps/trainer/tests/test_agreement_with_the_domain.py`, which is what stops a
default added here from failing every launch on the machine.
"""

from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, Any] = {
    "lora_r": 16,
    "lora_alpha": 32,  # α = 2r; recompute if r changes
    "lora_dropout": 0.0,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 8,  # effective batch 8; see wiki
    "sequence_len": 2048,
    "warmup_ratio": 0.1,
    "lr_scheduler": "cosine",
    "val_set_size": 0.05,
}

ALLOWED_OVERRIDES = {
    "lora_r",
    "lora_alpha",
    "learning_rate",
    "num_epochs",
    "max_steps",
    "sequence_len",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "val_set_size",
    "save_steps",
}


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
