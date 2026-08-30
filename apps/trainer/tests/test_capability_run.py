"""The machine half of the general-capability slice (issue #73).

The pure decisions live in ``capability.py`` and are tested there; this
module pins how the entrypoint applies them on the machine: the tuned side is
the checkpoint the run's selection rule actually chose (never the last one
written), the same fixed slice answers both models, and evaluation failure --
including a failure to load the models -- never fails the run.
``run_machine_capability`` takes its model loader as an injected seam, so the
whole flow is exercised on the host without torch, exactly as
``test_comparison_run.py`` exercises the comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import capability
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


def correct_letters() -> list[str]:
    return [q["answer"] for q in capability.CAPABILITY_QUESTIONS]


class StubGenerator:
    def __init__(self, answers: list[str] | None = None, exc=None):
        self._answers = answers
        self._exc = exc
        self.calls = 0

    def generate(self, conversation) -> str:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        if self._answers is None:
            return "generated"
        i = min(self.calls - 1, len(self._answers) - 1)
        return self._answers[i]


def stub_loaders(
    base_answers=None, tuned_answers=None, base_exc=None, tuned_exc=None
):
    """A fake `make_generators`: returns stub generators, or raises."""

    def load(job, cfg, chosen_dir, method):
        return (
            StubGenerator(base_answers, base_exc),
            StubGenerator(tuned_answers, tuned_exc),
        )

    return load


def test_the_capability_slice_answers_through_the_chosen_checkpoint(
    tmp_path: Path,
):
    """The checkpoint with the best held-out loss is the one the slice answers
    through, exactly as the run's own selection rule (issue #62) would choose
    -- the last one written is deliberately NOT the best here."""
    write_checkpoint(tmp_path, 20, loss=0.35, eval_loss=0.39)
    write_checkpoint(tmp_path, 30, loss=0.31, eval_loss=0.52)

    result = entrypoint.run_machine_capability(
        complete_job(),
        {},
        tmp_path,
        make_generators=stub_loaders(
            base_answers=correct_letters(), tuned_answers=correct_letters()
        ),
    )

    assert result["ok"] is True
    assert result["selection"]["step"] == 20
    assert result["selection"]["basis"] == "best_held_out_loss"
    assert result["total"] == len(capability.CAPABILITY_QUESTIONS)
    assert result["base_correct"] == result["total"]
    assert result["tuned_correct"] == result["total"]
    assert result["delta"] == 0.0


def test_the_capability_selection_is_the_shared_rule_not_a_reimplementation(
    tmp_path: Path,
):
    """The slice uses the same pure function the control plane records the
    choice with (ADR-0010), so a drift cannot silently evaluate a different
    model than the run answers with."""
    from checkpoint import select_best_checkpoint

    write_checkpoint(tmp_path, 10, loss=0.41, eval_loss=0.44)
    write_checkpoint(tmp_path, 20, loss=0.35, eval_loss=0.39)
    write_checkpoint(tmp_path, 30, loss=0.31, eval_loss=0.52)

    records = entrypoint.checkpoint_records(tmp_path)
    expected = select_best_checkpoint(records).to_dict()

    result = entrypoint.run_machine_capability(
        complete_job(),
        {},
        tmp_path,
        make_generators=stub_loaders(
            base_answers=correct_letters(), tuned_answers=correct_letters()
        ),
    )
    assert result["selection"] == expected


def test_a_large_regression_surfaces_on_the_record(tmp_path: Path):
    """A tuned model that loses general capability is flagged on the record so
    the interface can surface it prominently (issue #73's acceptance
    criterion)."""
    write_checkpoint(tmp_path, 20, loss=0.35, eval_loss=0.39)
    # The tuned model can only manage half the slice's correct letters.
    half = correct_letters()
    for i in range(2, 8):
        half[i] = "A" if half[i] != "A" else "B"

    result = entrypoint.run_machine_capability(
        complete_job(),
        {},
        tmp_path,
        make_generators=stub_loaders(
            base_answers=correct_letters(), tuned_answers=half
        ),
    )
    assert result["ok"] is True
    assert result["large_regression"] is True
    assert result["delta"] < 0


def test_a_failure_to_load_the_models_never_fails_the_run(tmp_path: Path):
    """The negative test that protects a paid run: make the model loading
    throw, and the run's result records ok: false with the reason rather than
    the whole run failing."""
    write_checkpoint(tmp_path, 20, loss=0.35, eval_loss=0.39)

    result = entrypoint.run_machine_capability(
        complete_job(),
        {},
        tmp_path,
        make_generators=stub_loaders(base_exc=RuntimeError("out of memory")),
    )

    assert result["ok"] is False
    assert "out of memory" in result["reason"]
    assert result["selection"]["step"] == 20
    assert result["large_regression"] is False


def test_a_generator_failure_records_the_reason_and_the_run_continues(
    tmp_path: Path,
):
    """The same guarantee one level down: a generation that throws mid-run is
    recorded, never fatal."""
    write_checkpoint(tmp_path, 20, loss=0.35, eval_loss=0.39)

    result = entrypoint.run_machine_capability(
        complete_job(),
        {},
        tmp_path,
        make_generators=stub_loaders(tuned_exc=RuntimeError("tuned died")),
    )

    assert result["ok"] is False
    assert "tuned died" in result["reason"]


def test_no_checkpoints_on_the_machine_is_recorded_not_a_failure(
    tmp_path: Path,
):
    result = entrypoint.run_machine_capability(
        complete_job(), {}, tmp_path, make_generators=stub_loaders()
    )
    assert result["ok"] is False
    assert result["selection"]["basis"] == "none"


def test_the_loaded_pair_is_shared_between_the_two_eval_steps(tmp_path: Path):
    """The entrypoint loads the two models once and hands the same pair to the
    comparison and the capability slice (Spec 011's eval runs where the
    weights already are -- one load, two steps). This pins the shape the
    entrypoint relies on: when `loaded` is passed, the slice does not load
    again and still records the selection."""
    write_checkpoint(tmp_path, 20, loss=0.35, eval_loss=0.39)
    models, reason, selection = entrypoint._load_warm_models(
        complete_job(), {}, tmp_path, make_generators=stub_loaders()
    )
    assert models is not None
    assert reason is None

    base_gen, tuned_gen, selection_record, chosen_step = models
    assert chosen_step == 20
    result = entrypoint.run_machine_capability(
        complete_job(), {}, tmp_path, loaded=models
    )
    assert result["ok"] is True
    assert result["selection"] == selection_record
    # The injected generators answered; a real load would have doubled the
    # generator count -- the shared pair is the point.
    assert base_gen.calls == len(capability.CAPABILITY_QUESTIONS)
    assert tuned_gen.calls == len(capability.CAPABILITY_QUESTIONS)
