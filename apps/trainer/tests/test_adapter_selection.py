"""Which artifact the trainer ships.

Moved out of the control plane's suite: it asserts the trainer entrypoint's
behaviour, and an application reaching into another application to test it is
the coupling the reorg was meant to remove.
"""

from __future__ import annotations

import json

import entrypoint


def test_adapter_selection_prefers_final_over_checkpoints(
    tmp_path, monkeypatch
):
    """`checkpoint-10` must beat `checkpoint-9`, and the final adapter beats both.

    The original `sorted(rglob(...))[-1]` got this wrong twice over: it sorts
    step numbers as strings, and it ranked any checkpoint above the
    end-of-training adapter.
    """
    out = tmp_path / "out"
    run = out / "run"
    for step in (2, 9, 10):
        d = run / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "adapter_model.safetensors").write_bytes(f"ckpt{step}".encode())
        (d / "adapter_config.json").write_text(json.dumps({"r": step}))
    monkeypatch.setattr(entrypoint, "OUT_DIR", out)

    # Only checkpoints so far: the numerically highest wins, not "9".
    info = entrypoint.collect_artifacts("qlora")
    assert info["artifact_source"] == "checkpoint-10"
    assert info["adapter_config"] == {"r": 10}
    assert info["checkpoints"] == [
        "checkpoint-2",
        "checkpoint-9",
        "checkpoint-10",
    ]

    # Once training writes the final adapter, that is the one shipped.
    (run / "adapter_model.safetensors").write_bytes(b"final")
    (run / "adapter_config.json").write_text(json.dumps({"r": 16}))
    info = entrypoint.collect_artifacts("qlora")
    assert info["artifact_source"] == "final"
    assert info["adapter_config"] == {"r": 16}
