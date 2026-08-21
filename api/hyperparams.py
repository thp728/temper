"""The effective job specification, as the control plane can state it before launch.

The create-job page shows the user exactly what their job will freeze. That
promise is only honest if it resolves overrides the way the trainer does, so
this module mirrors `trainer/entrypoint.py`: the same defaults, the same
allowed-override set, alpha recomputed from rank when rank moves alone, and
rsLoRA inferred at rank >= 32.

**The trainer's copy is authoritative.** The control plane does not import
trainer code -- the same rule that has feasibility.py duplicating
DEFAULT_EPOCHS -- so `test_hyperparams.py` pins this mirror against the
original instead. Change a default in the trainer and that test fails until
the mirror follows; change it only here and the page would describe a job the
trainer will not run.
"""

from __future__ import annotations

DEFAULTS = {
    "lora_r": 16,
    "lora_alpha": 32,          # α = 2r; recompute if r changes
    "lora_dropout": 0.0,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 8,   # effective batch 8; see wiki
    "sequence_len": 2048,
    "warmup_ratio": 0.1,
    "lr_scheduler": "cosine",
    "val_set_size": 0.05,
}

ALLOWED_OVERRIDES = {
    "lora_r", "lora_alpha", "learning_rate", "num_epochs", "max_steps",
    "sequence_len", "micro_batch_size", "gradient_accumulation_steps",
    "val_set_size", "save_steps",
}


def effective(overrides: dict | None) -> dict:
    """Resolve overrides against the defaults, exactly as the trainer will.

    Returns the full specification including the inferred `lora_use_rslora`,
    because rsLoRA is inferred from the rank and never exposed as a choice --
    but a user reading the frozen spec should still see that it is in force.
    """
    cfg = dict(DEFAULTS)
    applied = {k: v for k, v in (overrides or {}).items()
               if k in ALLOWED_OVERRIDES}
    cfg.update(applied)

    # α is mechanically tied to r: a new rank never pairs with a stale scale.
    if "lora_r" in applied and "lora_alpha" not in applied:
        cfg["lora_alpha"] = 2 * int(cfg["lora_r"])

    # rsLoRA above rank 32: plain α/r scaling over-shrinks high-rank adapters.
    cfg["lora_use_rslora"] = int(cfg["lora_r"]) >= 32
    return cfg
