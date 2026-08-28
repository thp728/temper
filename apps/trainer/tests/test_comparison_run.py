"""The machine half of the side-by-side comparison (issue #69).

The pure decisions live in ``comparison.py`` and are tested there; this
module pins how the entrypoint applies them on the machine: the prompts come
from the trainer's own held-out file, the tuned side is the checkpoint the
run's selection rule actually chose (never the last one written), and
evaluation failure -- including a failure to load the models -- never fails
the run. ``run_machine_comparison`` takes its model loader as an injected
seam, so the whole flow is exercised on the host without torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import entrypoint

from temper_core import hyperparams


def complete_job(**hp_changes) -> dict:
    hp = hyperparams.effective({})
    hp.update(hp_changes)
    return {
        "job_id": "example-0001",
        "base_model": "Qwen/Qwen3-4B",
        "hyperparameters": hp,
    }


def chat_rows(n: int) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(n)
    ]


def write_checkpoint(
    out_dir: Path, step: int, *, loss: float, eval_loss: float
):
    """A checkpoint the uploader would call complete, with a recorded loss."""
    directory = out_dir / "run" / f"checkpoint-{step}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "log_history": [
                    {"step": step, "loss": loss, "eval_loss": eval_loss}
                ],
            }
        )
    )


class StubGenerator:
    def __init__(self, text: str = "generated", exc: Exception | None = None):
        self._text = text
        self._exc = exc

    def generate(self, conversation) -> str:
        if self._exc is not None:
            raise self._exc
        return self._text


def stub_loaders(base_exc=None, tuned_exc=None):
    """A fake `make_generators`: returns stub generators, or raises."""

    def load(job, cfg, chosen_dir, method):
        return (
            StubGenerator("base", base_exc),
            StubGenerator("tuned", tuned_exc),
        )

    return load


def test_the_comparison_compares_the_chosen_checkpoint_not_the_last_written(
    tmp_path: Path,
):
    """The checkpoint with the best held-out loss is the one compared, exactly
    as the run's own selection rule (issue #62) would choose -- the last one
    written is deliberately NOT the best here."""
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text("\n".join(json.dumps(r) for r in chat_rows(10)))
    # Step 20 has the best held-out loss; step 30 (the last) is worse. The
    # comparison must pick step 20.
    write_checkpoint(tmp_path, 20, loss=0.35, eval_loss=0.39)
    write_checkpoint(tmp_path, 30, loss=0.31, eval_loss=0.52)

    result = entrypoint.run_machine_comparison(
        complete_job(), {}, tmp_path, make_generators=stub_loaders()
    )

    assert result["ok"] is True
    assert result["selection"]["step"] == 20
    assert result["selection"]["basis"] == "best_held_out_loss"
    assert (
        len(result["rows"])
        == entrypoint.comparison_step.COMPARISON_PROMPT_COUNT
    )
    for row in result["rows"]:
        # The held-out answer never reaches the models: the prompt stops at
        # the user turn.
        assert all(m["role"] != "assistant" for m in row["prompt"])


def test_the_selection_is_the_shared_rule_not_a_reimplementation(
    tmp_path: Path,
):
    """The comparison uses the same pure function the control plane records
    the choice with (ADR-0010), so a drift cannot silently compare a
    different model than the run answers with."""
    from checkpoint import select_best_checkpoint

    (tmp_path / "eval.jsonl").write_text(
        "\n".join(json.dumps(r) for r in chat_rows(10))
    )
    write_checkpoint(tmp_path, 10, loss=0.41, eval_loss=0.44)
    write_checkpoint(tmp_path, 20, loss=0.35, eval_loss=0.39)
    write_checkpoint(tmp_path, 30, loss=0.31, eval_loss=0.52)

    records = entrypoint.checkpoint_records(tmp_path)
    expected = select_best_checkpoint(records).to_dict()

    result = entrypoint.run_machine_comparison(
        complete_job(), {}, tmp_path, make_generators=stub_loaders()
    )
    assert result["selection"] == expected


def test_a_failure_to_load_the_models_never_fails_the_run(tmp_path: Path):
    """The negative test that protects a paid run: make the model loading
    throw, and the run's result records ok: false with the reason rather than
    the whole run failing."""
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text("\n".join(json.dumps(r) for r in chat_rows(10)))
    write_checkpoint(tmp_path, 20, loss=0.35, eval_loss=0.39)

    result = entrypoint.run_machine_comparison(
        complete_job(),
        {},
        tmp_path,
        make_generators=stub_loaders(base_exc=RuntimeError("out of memory")),
    )

    assert result["ok"] is False
    assert "out of memory" in result["reason"]
    assert result["selection"]["step"] == 20


def test_a_generator_failure_records_the_reason_and_the_run_continues(
    tmp_path: Path,
):
    """The same guarantee one level down: a generation that throws mid-run is
    recorded, never fatal."""
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text("\n".join(json.dumps(r) for r in chat_rows(10)))
    write_checkpoint(tmp_path, 20, loss=0.35, eval_loss=0.39)

    result = entrypoint.run_machine_comparison(
        complete_job(),
        {},
        tmp_path,
        make_generators=stub_loaders(tuned_exc=RuntimeError("tuned died")),
    )

    assert result["ok"] is False
    assert "tuned died" in result["reason"]


def test_nothing_held_out_means_no_comparison_and_no_failure(tmp_path: Path):
    result = entrypoint.run_machine_comparison(
        complete_job(), {}, tmp_path, make_generators=stub_loaders()
    )
    assert result["ok"] is False
    assert "nothing was held out" in result["reason"]


def test_no_checkpoints_on_the_machine_is_recorded_not_a_failure(
    tmp_path: Path,
):
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text("\n".join(json.dumps(r) for r in chat_rows(10)))

    result = entrypoint.run_machine_comparison(
        complete_job(), {}, tmp_path, make_generators=stub_loaders()
    )
    assert result["ok"] is False
    assert result["selection"]["basis"] == "none"
