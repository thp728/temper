"""The peak-VRAM measurement (`entrypoint.max_vram_gb_from_smi`).

The measured half of issue #77's memory comparison: the trainer samples the
machine's per-GPU used VRAM while axolotl runs, and records the peak in
result.json for the control plane to freeze as the actual. The sampler itself
shells out to nvidia-smi (present only on a GPU machine), so the tests pin the
one pure piece -- the parse of nvidia-smi's lines into decimal GB -- against
canned output, including the shapes a real machine has been seen to emit.
"""

from __future__ import annotations

from entrypoint import max_vram_gb_from_smi


def test_single_device_used_memory_is_converted_to_decimal_gb():
    # 5310 MiB is the 5.31 GB the first real run measured; decimal GB is the
    # predictor's convention (temper_core.memory.BYTES_PER_GB = 1e9).
    assert max_vram_gb_from_smi(["0, 5310"]) == 5310 * (1024 * 1024 / 1e9)


def test_the_peak_is_the_maximum_across_devices_and_samples():
    lines = ["0, 1000", "1, 4200", "0, 999"]
    peak = max_vram_gb_from_smi(lines)
    assert peak == 4200 * (1024 * 1024 / 1e9)


def test_the_MiB_suffix_is_accepted():
    assert max_vram_gb_from_smi(["0, 2048 MiB"]) == 2048 * (1024 * 1024 / 1e9)


def test_a_malformed_line_is_skipped_not_fatal():
    assert max_vram_gb_from_smi(["not a row", "0, 1024", ""]) == 1024 * (
        1024 * 1024 / 1e9
    )


def test_no_lines_measures_nothing():
    assert max_vram_gb_from_smi([]) is None
