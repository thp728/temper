"""VRAM capacity of the GPU types this platform provisions.

A card's capacity is a physical constant, not something that varies with
what the provider currently has free -- so unlike price and availability
(`temper_control_plane.provider.GpuChoice`, read live at provisioning time),
it belongs here as data rather than behind the provider seam.

L4 is measured, not read from a spec sheet: the trainer's real run
(2026-08-19, `apps/trainer/README.md`) confirms 24 GB from the card itself.
The rest come from `spike/spike.py`'s own pricing-pull comment
(2026-08-23) rather than a fresh source, and are labelled accordingly.
"""

from __future__ import annotations

CAPACITY_GB: dict[str, float] = {
    "L4": 24.0,  # measured: the 4B QLoRA run's own card
    "A100-80GB": 80.0,  # spike/spike.py, 2026-08-23 pricing pull
    "RTX-PRO6000": 96.0,  # spike/spike.py, 2026-08-23 pricing pull
    "H100": 80.0,  # spike/spike.py, 2026-08-23 pricing pull
    "H200": 141.0,  # spike/spike.py, 2026-08-23 pricing pull
}
