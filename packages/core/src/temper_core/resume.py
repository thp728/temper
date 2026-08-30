"""Resumption from an interrupted run's last checkpoint (issue #60).

An interrupted job resumes from its last checkpoint on a fresh machine,
restoring not only the weights but the optimiser, the scheduler and the step
position -- a resume that restores only weights produces a run that silently
differs from an uninterrupted one, and the whole checkpoint directory (which
is what the trainer uploads, issue #37) carries all four.

This module holds the *pure* decisions, with no I/O, so a test can pin them
without touching storage or a provider:

* what an interruption is (the stable code and the human sentence);
* which checkpoint a resumption comes back to (the highest step among the
  verified, retained checkpoints);
* what a resumed attempt's job spec looks like (the frozen spec plus
  `resume_from_checkpoint`, never a different spec -- the optimisation must
  not change because the machine did);
* when resumption must stop, so a configuration the infrastructure keeps
  killing cannot bill forever.

Resumption is a **new attempt against the same job** (spec 010's
implementation decision): the job record already distinguishes the job from
its executions, so a resumed run is a second attempt carrying its own machine,
rate and outcome. The attempts record is where this module's vocabulary
lands; this module does not touch the record itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# The interruption: a run that ended without a result document.
# ---------------------------------------------------------------------------
#
# A worker killed mid-training (the fault surface's `worker_kill`, or a real
# process death) produces no result.json -- the stream just ends. That is a
# different thing from a run that produced a result document saying it failed
# (which is a *training failure* with its own code), and the history must not
# call them the same. A client branches on the stable code, never on prose.

INTERRUPTED_CODE: str = "interrupted"
INTERRUPTED_MESSAGE: str = (
    "The run ended before producing a result: the worker or the machine was "
    "interrupted mid-training. Any checkpoint that survived was recorded on "
    "this job, and a resumption was attempted when one could."
)


def is_interruption(code: str | None) -> bool:
    """Whether a stable failure code names an interruption rather than a
    training failure. Every other code -- a result document that says the run
    failed, a budget stop, a divergence abort -- is not resumable in the
    interruption sense, and a caller that treats it as one would resume runs
    that never stopped for the reason resumption exists for."""
    return code == INTERRUPTED_CODE


# ---------------------------------------------------------------------------
# The retry cap -- a named constant with its derivation beside it, the way
# `memory_retry.MEMORY_RETRY_CAP` records its own.
# ---------------------------------------------------------------------------

# How many times a job may resume after an interruption. Each resumption
# provisions a fresh machine, so an unbounded loop bills forever on a
# configuration the infrastructure keeps killing. The memory recovery's cap
# is 4 (the whole escalation ladder plus one); resumption has no ladder -- it
# is the same run continued from the same kind of checkpoint each time -- so
# the cap exists only to bound a pathological repeat, and two is enough to
# prove that resumption recovers from a single interruption (the adversarial
# tier's scenario) while keeping the bound smaller than the memory ladder's.
# **A judgment, not a measured figure**: the number to revisit if a
# deployment observes repeated legitimate infrastructure kills is this one.
RESUME_RETRY_CAP: int = 2

# The stable code a job fails with when it kept being interrupted past the
# cap even though a checkpoint survived -- distinct from a bare `interrupted`,
# which is what a job with nothing to resume from fails with. Both are
# recorded beside the attempts, so "how many times and from what" is on the
# record rather than inferred.
RESUME_EXHAUSTED_CODE: str = "resume_retries_exhausted"
RESUME_EXHAUSTED_MESSAGE: str = (
    "The run was interrupted repeatedly, and each resumption was interrupted "
    "again before it could finish. The attempts are recorded on this job. "
    "The infrastructure that hosts this job is not staying up long enough to "
    "complete a run; check the machine lifetime before retrying."
)

# The code carried by the resumption's own events, so a resumption can be
# told apart from a memory recovery or a divergence abort in the history.
RESUME_RECOVERY_CODE: str = "resume"


def may_resume(resumptions_used: int, cap: int = RESUME_RETRY_CAP) -> bool:
    """Whether another resumption is permitted after `resumptions_used` have
    already happened. Negative counts are refused -- a count that cannot be
    explained is the same class of bug as a retry that never stops."""
    if resumptions_used < 0:
        raise ValueError("resumptions_used cannot be negative")
    return resumptions_used < cap


# ---------------------------------------------------------------------------
# Which checkpoint a resumption comes back to.
# ---------------------------------------------------------------------------


def latest_checkpoint(
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """The checkpoint a resumption comes back to: the highest step among the
    verified, retained checkpoints, or None when there is nothing usable.

    `checkpoints` is the job's recorded checkpoint records -- each an integer
    `step`, and `verified: True` only for one whose bytes are in storage. Only
    verified records are candidates: resuming from bytes that are not there
    is resuming from nothing. The latest (highest step) is chosen because it
    is the one that trained furthest, so the resumption loses the least work
    -- the whole point of recovering an interrupted run rather than starting
    over.

    A record with no usable `step` raises, mirroring
    `checkpoint.select_best_checkpoint`: a checkpoint that cannot be named
    cannot be resumed from, and a silently skipped record would let the
    choice pretend it did not exist.
    """
    candidates: list[dict[str, Any]] = []
    for record in checkpoints:
        step = record.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            raise ValueError(
                f"a checkpoint record carried no integer step: {record!r}"
            )
        if record.get("verified") is True:
            candidates.append(dict(record))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["step"])


# ---------------------------------------------------------------------------
# The resumed attempt's spec: the frozen spec, plus the resume directive.
# ---------------------------------------------------------------------------

# The job-spec key the trainer already reads to resume (its guard and its
# config pass-through exist -- issue #37's uploader wrote the checkpoint, the
# trainer's `build_config` applies the directive). Defined here once so the
# control plane writes the same key the trainer reads.
RESUME_FROM_KEY: str = "resume_from_checkpoint"


def resume_path(step: int) -> str:
    """The machine-local path axolotl resumes from for `step`.

    The control plane streams the checkpoint archive to the machine and the
    remote script extracts it under the container's `/out/run`, so the
    trainer's `resume_from_checkpoint` names a container path (the same
    path space the trainer's own `output_dir` uses). The step names the
    directory inside the archive, which is how axolotl recognises a
    checkpoint at all.
    """
    return f"/out/run/checkpoint-{step}"


def resumed_spec(
    spec: Mapping[str, Any],
    resume_from: str,
) -> dict[str, Any]:
    """The job spec a resumed attempt runs: the frozen spec plus the resume
    directive, nothing else changed.

    The frozen spec is the user's request and a resumption must not quietly
    change it -- the optimisation is the same because the machine is not
    supposed to change it. `resume_from` is a machine-local path (see
    `resume_path`); the returned spec is what the trainer applies, and the
    trainer already passes the key through to axolotl. The resumed step
    itself is recorded on the attempt, never smuggled into the trainer spec
    as a key it does not know.
    """
    out = dict(spec)
    out[RESUME_FROM_KEY] = resume_from
    return out
