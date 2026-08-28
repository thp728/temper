"""The trainer's half of the held-out split (issue #53).

The split itself lives in the domain module and is tested there; this module
pins the trainer's use of it: it is applied before the config is built, the
held-out rows never enter the training file, and the config hands the split to
Axolotl as its test dataset rather than letting Axolotl carve its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import entrypoint
import pytest

from temper_core import hyperparams


def complete_job(**hp_changes) -> dict:
    """A job spec shaped like the one the control plane writes at launch."""
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


def test_the_split_writes_separate_train_and_eval_files(tmp_path: Path):
    rows = chat_rows(12)
    eval_path, record = entrypoint.prepare_held_out_split(
        rows, complete_job(), tmp_path
    )

    train = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text().splitlines()
    ]
    held = [
        json.loads(line)
        for line in (tmp_path / "eval.jsonl").read_text().splitlines()
    ]
    # The held-out rows never enter the training file -- the two cannot
    # overlap because they are different files.
    train_keys = {json.dumps(r["messages"], sort_keys=True) for r in train}
    held_keys = {json.dumps(r["messages"], sort_keys=True) for r in held}
    assert train_keys.isdisjoint(held_keys)
    assert len(train) + len(held) == 12
    assert eval_path == tmp_path / "eval.jsonl"
    assert record["rows_in"] == 12
    assert record["train_rows"] == len(train)
    assert record["held_out_rows"] == len(held)


def test_duplicates_are_removed_before_the_split(tmp_path: Path):
    rows = chat_rows(11) + [chat_rows(1)[0]]  # one exact duplicate
    eval_path, record = entrypoint.prepare_held_out_split(
        rows, complete_job(), tmp_path
    )
    assert record["rows_removed_duplicates"] == 1
    assert record["rows_in"] == 12
    train_count = sum(1 for _ in (tmp_path / "train.jsonl").open())
    held_count = sum(1 for _ in (tmp_path / "eval.jsonl").open())
    assert train_count + held_count == 11


def test_nothing_held_out_means_no_eval_file_and_no_test_dataset(
    tmp_path: Path,
):
    rows = chat_rows(10)
    eval_path, record = entrypoint.prepare_held_out_split(
        rows, complete_job(val_set_size=0.0), tmp_path
    )
    assert record["held_out_rows"] == 0
    assert eval_path is None
    assert not (tmp_path / "eval.jsonl").exists()


def test_the_config_points_training_at_the_split_and_eval_at_the_test_dataset():
    eval_path = Path("/out/eval.jsonl")
    cfg, _rejected = entrypoint.build_config(
        complete_job(), eval_path=eval_path
    )
    # The training file is the platform's split, and Axolotl is not allowed to
    # carve a second one out of it (it accepts test_datasets or val_set_size,
    # not both).
    assert cfg["datasets"][0]["path"] == str(
        entrypoint.OUT_DIR / "train.jsonl"
    )
    assert cfg["val_set_size"] == 0.0
    assert cfg["test_datasets"] == [
        {
            "path": str(eval_path),
            "type": "chat_template",
            "field_messages": "messages",
        }
    ]


def test_without_a_split_there_is_no_test_dataset():
    cfg, _rejected = entrypoint.build_config(complete_job())
    assert "test_datasets" not in cfg
    assert cfg["val_set_size"] == 0.0


def test_the_eval_cadence_is_pinned_not_left_to_a_default():
    """Held-out loss must be measured *during* the run (issue #53): the
    chart and the plateau need more than one point. Pinning eval_strategy to
    epoch-end keeps that from depending on an unpinned Axolotl default."""
    cfg, _rejected = entrypoint.build_config(complete_job())
    assert cfg["eval_strategy"] == "epoch"


def test_a_spec_missing_val_set_size_fails_as_incomplete(tmp_path: Path):
    """The split reads the effective val_set_size from the spec, so a spec
    that omits it fails the named way (`spec_incomplete`) rather than as a
    generic training anomaly."""
    job = complete_job()
    del job["hyperparameters"]["val_set_size"]
    with pytest.raises(entrypoint.IncompleteJobSpec) as excinfo:
        entrypoint.prepare_held_out_split(chat_rows(12), job, tmp_path)
    assert "val_set_size" in str(excinfo.value)


def test_the_split_fraction_is_the_effective_val_set_size(tmp_path: Path):
    """A user who raises val_set_size raises the fraction the platform holds
    out -- the value in the resolved spec is the one that splits, never a
    trainer default."""
    rows = chat_rows(100)
    _eval_path, record = entrypoint.prepare_held_out_split(
        rows, complete_job(val_set_size=0.5), tmp_path
    )
    assert record["fraction"] == 0.5
    assert record["held_out_rows"] == 50
