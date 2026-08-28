"""The memory-failure recovery's domain logic (issue #35).

A job that runs out of device memory retries **automatically**, and the
retry preserves the effective batch so the optimisation does not change: the
per-step batch halves and the accumulation doubles, and the product of the
two -- the number of rows one optimiser step is taken over -- is fixed at
launch and maintained by every step of the escalation ladder. That invariant
is what makes the recovery a recovery rather than a restart: a user who did
not choose the batch size should not see their training change because of it.

Pure, no I/O, no framework imports -- the same rule that governs the rest of
`temper_core` (ADR-0010). The control plane reads this module to decide the
next attempt's hyperparameters; the trainer image cannot import it, so the
codes that cross to the trainer are pinned equal by a trainer test, exactly
as `faults.FAULT_ENV` is.

The escalation ladder is ordered and bounded (spec 010): halve the per-step
batch down to its floor, then gradient checkpointing, then sequence length,
then more capable hardware, then surface the failure with the attempts
recorded. A memory retry is **automatic**; a divergence retry (ADR-0055) is
a *choice* offered to the user -- the two decisions must never look alike in
the record or in the database, which is why every code this module emits is
asserted disjoint from the divergence codes by a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import faults

# ---------------------------------------------------------------------------
# The hyperparameter keys the recovery transforms. They are the resolved
# spec's own keys -- read, never redeclared as values -- so the recovery
# cannot drift from the trainer's vocabulary.
# ---------------------------------------------------------------------------

MICRO_BATCH_KEY = "micro_batch_size"
ACCUMULATION_KEY = "gradient_accumulation_steps"
SEQUENCE_LEN_KEY = "sequence_len"
GRADIENT_CHECKPOINTING_KEY = "gradient_checkpointing"

# ---------------------------------------------------------------------------
# The floors and the retry cap -- each a named constant with its derivation
# beside it, the way orchestrator.py records TEARDOWN_CONFIRM_SAMPLES.
# ---------------------------------------------------------------------------

# A per-step batch must hold at least one row; below that the batch rung is
# exhausted and the ladder escalates. The resolved default `micro_batch_size`
# is already 1 (packages/contracts/trainer-defaults.json), so the batch rung
# exists for specs above this floor and the default configuration skips it.
PER_STEP_BATCH_FLOOR: int = 1

# Sequence length below this is too short to be a useful conversational
# context for the data this platform trains, so halving stops here and the
# ladder escalates to hardware. **A judgement, not a measured figure**: the
# default is 2048 and halving reaches 512 after two reductions; the number to
# revisit when the catalog carries a model that cannot fit even truncated to
# 512 is this one, which is why it is configuration-shaped rather than a
# literal scattered at use sites.
SEQUENCE_LEN_FLOOR: int = 512

# How many automatic re-attempts a job may make after an out-of-memory
# failure. The ladder is finite (batch halving to its floor, gradient
# checkpointing, sequence length to its floor, one hardware escalation), so
# the cap exists to bound a pathological repeat -- a "more capable" card that
# still OOMs -- from provisioning machines forever. Four leaves room for the
# default configuration's whole climb (batch floor already reached, gradient
# checkpointing always on: sequence length, hardware, then one more) while
# keeping the bound small enough that a failing configuration cannot burn an
# evening unattended.
MEMORY_RETRY_CAP: int = 4

# ---------------------------------------------------------------------------
# The ladder's rungs, in order. A memory retry advances one rung at a time;
# the batch and sequence-length rungs repeat to their floors, gradient
# checkpointing and hardware fire once, and exhausting the ladder surfaces.
# ---------------------------------------------------------------------------

RUNG_HALVE_BATCH = "halve_batch"
RUNG_GRADIENT_CHECKPOINTING = "gradient_checkpointing"
RUNG_SEQUENCE_LENGTH = "sequence_length"
RUNG_HARDWARE = "hardware"
RUNG_SURFACE = "surface"

# ---------------------------------------------------------------------------
# Codes. A memory retry is automatic and a divergence retry is a choice
# (ADR-0055): the codes keep the two decisions distinguishable in the record
# and in the database, which a test asserts (never a divergence code, never
# the divergence retry).
# ---------------------------------------------------------------------------

# The trainer's name for a genuine out-of-memory failure (not the fault
# surface's deliberate `simulated_oom`). The trainer image cannot import this
# package, so it declares its own constant with the same value and a trainer
# test pins the two equal -- the FAULT_ENV pattern.
REAL_OOM_CODE = "training_oom"

# The codes a failed run's result can carry that mean "device memory was
# exhausted; retry with the effective batch preserved": the fault surface's
# own oom (the deliberate tier) and a genuine OOM.
MEMORY_FAILURE_CODES = frozenset({faults.code_for("oom"), REAL_OOM_CODE})

# The stable code the job fails with when the retry cap or the ladder is
# exhausted and memory still does not fit. Deliberately not `training_diverged`
# and not `training_failed`: a memory recovery that gave up is its own reason.
MEMORY_RETRIES_EXHAUSTED_CODE = "memory_retries_exhausted"
MEMORY_RETRIES_EXHAUSTED_MESSAGE = (
    "The job ran out of device memory repeatedly, and every memory reduction "
    "the platform can apply has been tried: the per-step batch, gradient "
    "checkpointing, sequence length, and more capable hardware. The last "
    "attempt still did not fit. The attempts are recorded on this job. Try a "
    "smaller configuration, a shorter sequence length, or a model that "
    "predicts a smaller peak."
)

# The code carried by the recovery's own events, so a memory recovery can be
# told apart from a divergence abort or instability warning in the history.
MEMORY_RECOVERY_CODE = "memory_recovery"


def effective_batch(hyperparameters: dict[str, Any]) -> int:
    """The number of rows one optimiser step is taken over: the product the
    recovery must preserve. Defined once here so the transformation and every
    assertion read the same arithmetic."""
    return int(hyperparameters[MICRO_BATCH_KEY]) * int(
        hyperparameters[ACCUMULATION_KEY]
    )


def halve_per_step_batch(
    batch: int, accumulation: int, floor: int = PER_STEP_BATCH_FLOOR
) -> tuple[int, int] | None:
    """The next (per-step batch, accumulation): batch halved, accumulation
    recomputed so the effective batch is preserved exactly.

    Returns None when the per-step batch is at its floor, or when it cannot be
    halved exactly -- an odd batch whose accumulation cannot compensate would
    change the effective batch, which is the very thing the recovery exists to
    prevent, so the rung escalates rather than quietly altering the run.

    The product ``batch * accumulation`` is unchanged whenever a pair is
    returned; `test_memory_retry.py` asserts that as a property over the whole
    reachable state space, not as one hand-picked sequence.
    """
    if batch <= floor:
        return None
    new_batch = batch // 2
    if new_batch < floor:
        new_batch = floor
    product = batch * accumulation
    if product % new_batch != 0:
        return None
    return (new_batch, product // new_batch)


@dataclass(frozen=True)
class MemoryRetryStep:
    """One rung of the escalation ladder, as the next attempt should run it.

    `hyperparameters` is the full resolved spec the next attempt runs --
    unchanged except for the keys the rung transformed. `rung` names the step
    taken, `action` says it in plain words, `notes` carry the rungs that were
    skipped (at their floor, or already in force) so the narrative is honest
    about what was considered, and `changed` is the structured
    ``{key: (old, new)}`` the record keeps. `effective_batch` is the preserved
    product, carried so the invariant travels with every step rather than
    being re-derived.
    """

    hyperparameters: dict[str, Any]
    rung: str
    action: str
    changed: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    effective_batch: int | None = None


class MemoryEscalator:
    """Walks the escalation ladder one out-of-memory retry at a time.

    Holds the resolved spec of the current attempt; each `step()` returns the
    next attempt's spec with the ladder advanced one rung, or None when the
    ladder is exhausted and the job must surface the failure.

    The ladder is linear and never revisits a rung: halve the per-step batch
    to its floor, gradient checkpointing (already in force on this platform --
    the calculated tier -- so it is recorded as skipped and passed over),
    halve the sequence length to its floor, escalate to more capable hardware
    once, then surface. Batch and sequence length each repeat within their own
    rung, which is what bounds the number of reductions while keeping the
    order the spec mandates.
    """

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        *,
        batch_floor: int = PER_STEP_BATCH_FLOOR,
        sequence_len_floor: int = SEQUENCE_LEN_FLOOR,
    ) -> None:
        self._hp = dict(hyperparameters)
        self._batch_floor = batch_floor
        self._sequence_len_floor = sequence_len_floor
        self._phase = RUNG_HALVE_BATCH

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """The current attempt's resolved spec, as a copy."""
        return dict(self._hp)

    def step(self) -> MemoryRetryStep | None:
        """The next rung's step, or None when the ladder is exhausted."""
        notes: list[str] = []
        while True:
            if self._phase == RUNG_HALVE_BATCH:
                batch = int(self._hp[MICRO_BATCH_KEY])
                accumulation = int(self._hp[ACCUMULATION_KEY])
                if batch <= self._batch_floor:
                    notes.append(
                        f"the per-step batch is already at its floor "
                        f"({self._batch_floor}); nothing to reduce there"
                    )
                    self._phase = RUNG_GRADIENT_CHECKPOINTING
                    continue
                pair = halve_per_step_batch(
                    batch, accumulation, floor=self._batch_floor
                )
                if pair is None:
                    notes.append(
                        f"the per-step batch ({batch}) cannot be halved "
                        "exactly without changing the effective batch"
                    )
                    self._phase = RUNG_GRADIENT_CHECKPOINTING
                    continue
                new_batch, new_accumulation = pair
                product = batch * accumulation
                self._hp[MICRO_BATCH_KEY] = new_batch
                self._hp[ACCUMULATION_KEY] = new_accumulation
                return MemoryRetryStep(
                    hyperparameters=dict(self._hp),
                    rung=RUNG_HALVE_BATCH,
                    action=(
                        f"the per-step batch was halved ({batch}->{new_batch}) "
                        f"and accumulation doubled "
                        f"({accumulation}->{new_accumulation}); the effective "
                        f"batch ({batch}x{accumulation}={product}) is unchanged"
                    ),
                    changed={
                        MICRO_BATCH_KEY: (batch, new_batch),
                        ACCUMULATION_KEY: (accumulation, new_accumulation),
                    },
                    notes=tuple(notes),
                    effective_batch=product,
                )

            if self._phase == RUNG_GRADIENT_CHECKPOINTING:
                current = self._hp.get(GRADIENT_CHECKPOINTING_KEY)
                if current in (None, True):
                    # The trainer always enables gradient checkpointing (the
                    # calculated tier, issue #33) and the resolved spec never
                    # carries the key; the rung is recorded as already in
                    # force rather than silently skipped.
                    notes.append(
                        "gradient checkpointing is already enabled by the "
                        "platform; nothing to reduce there"
                    )
                    self._phase = RUNG_SEQUENCE_LENGTH
                    continue
                self._hp[GRADIENT_CHECKPOINTING_KEY] = True
                self._phase = RUNG_SEQUENCE_LENGTH
                return MemoryRetryStep(
                    hyperparameters=dict(self._hp),
                    rung=RUNG_GRADIENT_CHECKPOINTING,
                    action="gradient checkpointing enabled",
                    changed={GRADIENT_CHECKPOINTING_KEY: (current, True)},
                    notes=tuple(notes),
                    effective_batch=effective_batch(self._hp),
                )

            if self._phase == RUNG_SEQUENCE_LENGTH:
                sequence_len = int(self._hp[SEQUENCE_LEN_KEY])
                if sequence_len <= self._sequence_len_floor:
                    notes.append(
                        f"the sequence length is already at its floor "
                        f"({self._sequence_len_floor}); nothing to reduce there"
                    )
                    self._phase = RUNG_HARDWARE
                    continue
                new_len = max(sequence_len // 2, self._sequence_len_floor)
                self._hp[SEQUENCE_LEN_KEY] = new_len
                return MemoryRetryStep(
                    hyperparameters=dict(self._hp),
                    rung=RUNG_SEQUENCE_LENGTH,
                    action=(
                        f"the sequence length was halved "
                        f"({sequence_len}->{new_len})"
                    ),
                    changed={SEQUENCE_LEN_KEY: (sequence_len, new_len)},
                    notes=tuple(notes),
                    effective_batch=effective_batch(self._hp),
                )

            if self._phase == RUNG_HARDWARE:
                # The control plane re-provisions against more capable
                # hardware (a card with more memory than the one that OOMed);
                # the next call finds the ladder exhausted and surfaces.
                self._phase = RUNG_SURFACE
                return MemoryRetryStep(
                    hyperparameters=dict(self._hp),
                    rung=RUNG_HARDWARE,
                    action="escalating to more capable hardware",
                    notes=tuple(notes),
                    effective_batch=effective_batch(self._hp),
                )

            # RUNG_SURFACE: every reduction the ladder can express is spent.
            return None
