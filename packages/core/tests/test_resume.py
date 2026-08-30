"""Resumption decisions: pure functions over checkpoints and specs (issue #60).

The control plane's resumption path is verified without hardware, so the
decision logic is a pure module with a table of tests: which checkpoint a
resumption comes back to, what a resumed spec looks like, and when
resumption must stop. What is NOT here -- downloading a checkpoint, naming an
interruption from a live stream -- is the orchestrator's, and stays there.
"""

from __future__ import annotations

import pytest

from temper_core import resume
from temper_core.resume import (
    INTERRUPTED_CODE,
    RESUME_EXHAUSTED_CODE,
    RESUME_RETRY_CAP,
    is_interruption,
    latest_checkpoint,
    may_resume,
    resume_path,
    resumed_spec,
)


def ckpt(step, verified=True):
    """One checkpoint record, shaped as the control plane stores it."""
    return {"step": step, "verified": verified}


def test_an_interruption_is_its_own_code_and_no_training_failure():
    """A run that ended without a result document is named as such, and a
    client can branch on the code rather than on prose."""
    assert is_interruption(INTERRUPTED_CODE)
    for other in ("training_failed", "training_oom", "training_diverged", None):
        assert not is_interruption(other), other


def test_the_latest_verified_checkpoint_is_chosen():
    """Resumption comes back to the highest step among the verified,
    retained checkpoints -- the one that trained furthest, so the least work
    is lost."""
    chosen = latest_checkpoint([ckpt(10), ckpt(20), ckpt(30)])
    assert chosen["step"] == 30


def test_unverified_checkpoints_are_never_resume_candidates():
    """Bytes that are not in storage are nothing to resume from; only
    verified records count, whatever step they claim."""
    assert latest_checkpoint([ckpt(10), ckpt(20, verified=False)])["step"] == 10
    assert latest_checkpoint([ckpt(10, verified=False)]) is None
    assert latest_checkpoint([]) is None


def test_a_record_without_a_step_is_refused_not_skipped():
    """A checkpoint that cannot be named cannot be resumed from, and a
    silently skipped record would let the choice pretend it did not exist."""
    with pytest.raises(ValueError):
        latest_checkpoint([{"verified": True}])


def test_the_resumed_spec_is_the_frozen_spec_plus_the_directive():
    """A resumption must not quietly change the optimisation: the resumed
    attempt runs the same frozen spec, with only `resume_from_checkpoint`
    added, and nothing the trainer does not know."""
    spec = {"job_id": "j", "hyperparameters": {"learning_rate": 9e-5}}
    resumed = resumed_spec(spec, resume_path(20))
    assert resumed == {
        **spec,
        resume.RESUME_FROM_KEY: "/out/run/checkpoint-20",
    }
    # The input spec is not mutated.
    assert resume.RESUME_FROM_KEY not in spec


def test_the_resume_path_names_the_checkpoint_under_the_trainers_out_dir():
    """The container path axolotl resumes from is under `/out/run` -- the
    same path space the trainer's own output_dir uses -- and names the step,
    which is how axolotl recognises a checkpoint at all."""
    assert resume_path(20) == "/out/run/checkpoint-20"
    assert resume_path(2) == "/out/run/checkpoint-2"


def test_resumption_is_bounded():
    """Resumption provisions a fresh machine each time, so it must stop: the
    cap bounds a pathological repeat while leaving the single-interruption
    recovery the adversarial tier demonstrates comfortably inside it."""
    assert may_resume(0)
    assert may_resume(RESUME_RETRY_CAP - 1)
    assert not may_resume(RESUME_RETRY_CAP)
    assert not may_resume(RESUME_RETRY_CAP + 1)


def test_a_negative_resumption_count_is_refused():
    with pytest.raises(ValueError):
        may_resume(-1)


def test_the_codes_are_stable_and_distinct():
    """Every failure carries a stable code; `interrupted` (nothing to resume
    from) and `resume_retries_exhausted` (kept being interrupted past the
    cap) are different reasons and must not collapse into one."""
    assert INTERRUPTED_CODE == "interrupted"
    assert RESUME_EXHAUSTED_CODE == "resume_retries_exhausted"
    assert RESUME_EXHAUSTED_CODE != INTERRUPTED_CODE
