"""The memory-failure recovery's domain logic (issue #35).

The recovery halves the per-step batch and doubles accumulation so the
effective batch -- `micro_batch_size * gradient_accumulation_steps` -- is
preserved, which is what makes it a recovery rather than a restart: a user
who did not choose the batch size does not see their training change. The
spec's own testing decision is that this is asserted as an **invariant, not
as a sequence**: the property below walks every reachable state (including
the floor case) and asserts the product never changes, rather than pinning
the particular sequence 8/1 -> 4/2 -> 2/4.

The escalation ladder is ordered and bounded: halve the per-step batch down
to its floor, then gradient checkpointing, then sequence length, then more
capable hardware, then surface the failure. A memory retry is automatic; a
divergence retry (ADR-0055) is a *choice* -- the codes and the retry path
must stay distinguishable in the record, which is why the memory codes below
are asserted disjoint from the divergence codes.
"""

from __future__ import annotations

from temper_core import faults
from temper_core.divergence import DIVERGED_CODE, INSTABILITY_CODE
from temper_core.memory_retry import (
    ACCUMULATION_KEY,
    GRADIENT_CHECKPOINTING_KEY,
    MEMORY_FAILURE_CODES,
    MEMORY_RECOVERY_CODE,
    MEMORY_RETRIES_EXHAUSTED_CODE,
    MEMORY_RETRY_CAP,
    MICRO_BATCH_KEY,
    PER_STEP_BATCH_FLOOR,
    REAL_OOM_CODE,
    RUNG_GRADIENT_CHECKPOINTING,
    RUNG_HALVE_BATCH,
    RUNG_HARDWARE,
    RUNG_SEQUENCE_LENGTH,
    SEQUENCE_LEN_FLOOR,
    SEQUENCE_LEN_KEY,
    MemoryEscalator,
    effective_batch,
    halve_per_step_batch,
)


def hp(batch: int, accumulation: int, sequence_len: int = 2048) -> dict:
    """A resolved hyperparameter dict shaped like `hyperparams.effective`'s."""
    return {
        MICRO_BATCH_KEY: batch,
        ACCUMULATION_KEY: accumulation,
        SEQUENCE_LEN_KEY: sequence_len,
    }


# ---------------------------------------------------------------------------
# the effective-batch invariant
# ---------------------------------------------------------------------------


def test_effective_batch_is_the_product():
    assert effective_batch(hp(8, 1)) == 8
    assert effective_batch(hp(1, 8)) == 8
    assert effective_batch(hp(4, 2)) == 8


def test_halving_halves_the_batch_and_doubles_accumulation():
    assert halve_per_step_batch(8, 1) == (4, 2)
    assert halve_per_step_batch(4, 2) == (2, 4)
    assert halve_per_step_batch(2, 4) == (1, 8)


def test_halving_preserves_the_effective_batch():
    for batch, accumulation in ((8, 1), (4, 2), (16, 4), (6, 2), (10, 8)):
        result = halve_per_step_batch(batch, accumulation)
        assert result is not None
        new_batch, new_acc = result
        assert new_batch * new_acc == batch * accumulation


def test_halving_stops_at_the_floor():
    # The per-step batch cannot go below one row; at the floor the rung is
    # exhausted and the ladder escalates rather than changing the batch.
    assert halve_per_step_batch(1, 8) is None
    assert halve_per_step_batch(2, 4) == (1, 8)
    assert halve_per_step_batch(1, 8) is None


def test_an_odd_batch_that_cannot_halve_exactly_escalates():
    # 5 rows per step cannot halve exactly while preserving 5*3=15 as an
    # integer product; changing the effective batch is the very thing the
    # recovery exists to prevent, so the rung escalates instead.
    assert halve_per_step_batch(5, 3) is None
    # But when the accumulation can compensate exactly, it does halve.
    assert halve_per_step_batch(5, 2) == (2, 5)


def test_the_effective_batch_is_invariant_across_every_reachable_state():
    """THE property: for every reachable (per-step batch, accumulation), the
    product is equal before and after each reduction, all the way to the
    floor -- asserted as an invariant over a grid of states, not as one
    hand-picked sequence. This is the criterion that is easiest to fake by
    walking 8/1 -> 4/2 -> 2/4 and checking each step; this walks the whole
    reachable state space instead."""
    for batch in range(1, 33):
        for accumulation in range(1, 17):
            launch = batch * accumulation
            current_b, current_a = batch, accumulation
            steps = 0
            while True:
                next_pair = halve_per_step_batch(current_b, current_a)
                if next_pair is None:
                    break
                new_b, new_a = next_pair
                # The invariant, asserted on the transition.
                assert new_b * new_a == current_b * current_a == launch
                current_b, current_a = new_b, new_a
                steps += 1
                assert steps < 64  # must terminate, and quickly
            # The walk ends at the floor: the per-step batch can go no lower.
            assert current_b >= PER_STEP_BATCH_FLOOR


# ---------------------------------------------------------------------------
# the escalation ladder
# ---------------------------------------------------------------------------


def test_a_default_config_escalates_batch_floor_then_checkpoint_then_sequence():
    # Defaults: micro_batch_size 1 (already at the floor), accumulation 8,
    # sequence_len 2048. The first OOM retry skips the batch rung (at its
    # floor) and gradient checkpointing (always on) and halves sequence length
    # -- the first rung that can actually reduce memory for this config.
    ladder = MemoryEscalator(hp(1, 8, 2048))
    step = ladder.step()
    assert step is not None
    assert step.rung == RUNG_SEQUENCE_LENGTH
    assert step.changed[SEQUENCE_LEN_KEY] == (2048, 1024)
    assert step.effective_batch == 8
    assert any("floor" in n for n in step.notes)
    assert any("already enabled" in n for n in step.notes)


def test_a_large_batch_is_halved_first():
    ladder = MemoryEscalator(hp(8, 1, 2048))
    step = ladder.step()
    assert step is not None
    assert step.rung == RUNG_HALVE_BATCH
    assert step.changed[MICRO_BATCH_KEY] == (8, 4)
    assert step.changed[ACCUMULATION_KEY] == (1, 2)
    assert step.effective_batch == 8


def test_the_batch_rung_reduces_repeatedly_to_the_floor():
    ladder = MemoryEscalator(hp(8, 1, 2048))
    reduced = []
    step = ladder.step()
    while step is not None and step.rung == RUNG_HALVE_BATCH:
        reduced.append(step.changed)
        assert step.effective_batch == 8
        step = ladder.step()
    # 8/1 -> 4/2 -> 2/4 -> 1/8, then the rung is exhausted.
    assert reduced == [
        {MICRO_BATCH_KEY: (8, 4), ACCUMULATION_KEY: (1, 2)},
        {MICRO_BATCH_KEY: (4, 2), ACCUMULATION_KEY: (2, 4)},
        {MICRO_BATCH_KEY: (2, 1), ACCUMULATION_KEY: (4, 8)},
    ]
    assert step is not None and step.rung == RUNG_SEQUENCE_LENGTH


def test_the_ladder_preserves_the_effective_batch_at_every_step():
    """The ladder-level invariant: walking a spec through the whole ladder --
    batch halving to the floor, sequence length to its floor, hardware -- the
    effective batch never changes, including at the floors. This is the
    "including the floor case" half of the invariant."""
    for batch in (1, 2, 8, 16):
        for accumulation in (1, 2, 8):
            for seq in (512, 1024, 2048):
                ladder = MemoryEscalator(hp(batch, accumulation, seq))
                launch = batch * accumulation
                while True:
                    step = ladder.step()
                    if step is None:
                        break
                    assert step.effective_batch == launch
                    assert effective_batch(step.hyperparameters) == launch


def test_gradient_checkpointing_when_off_is_enabled():
    spec = hp(1, 8, 2048)
    spec[GRADIENT_CHECKPOINTING_KEY] = False
    ladder = MemoryEscalator(spec)
    step = ladder.step()
    assert step is not None
    assert step.rung == RUNG_GRADIENT_CHECKPOINTING
    assert step.changed[GRADIENT_CHECKPOINTING_KEY] == (False, True)


def test_the_ladder_ends_at_more_capable_hardware_and_then_surfaces():
    # A config already at every floor: batch 1, sequence_len at its floor.
    ladder = MemoryEscalator(hp(1, 8, SEQUENCE_LEN_FLOOR))
    step = ladder.step()
    assert step is not None
    assert step.rung == RUNG_HARDWARE
    # After the single hardware escalation there is nothing left: surface.
    assert ladder.step() is None


def test_sequence_length_stops_at_its_floor():
    ladder = MemoryEscalator(hp(1, 8, 1024))
    first = ladder.step()
    assert first is not None and first.rung == RUNG_SEQUENCE_LENGTH
    assert first.changed[SEQUENCE_LEN_KEY] == (1024, 512)
    second = ladder.step()
    assert second is not None and second.rung == RUNG_HARDWARE
    assert ladder.step() is None


# ---------------------------------------------------------------------------
# the retry cap and the codes
# ---------------------------------------------------------------------------


def test_the_retry_cap_is_a_named_bound():
    # The ladder is finite, so the cap exists to bound a pathological repeat
    # from billing forever; it must be a positive named constant, never a
    # literal scattered at use sites.
    assert MEMORY_RETRY_CAP >= 1


def test_the_memory_failure_codes_carry_the_fault_and_the_real_failure():
    # A memory failure is either the fault surface's own oom (the deliberate
    # tier) or a genuine out-of-memory the trainer names; both mean "retry
    # with the effective batch preserved".
    assert faults.code_for("oom") == "simulated_oom"
    assert MEMORY_FAILURE_CODES == frozenset({"simulated_oom", REAL_OOM_CODE})


def test_a_memory_retry_never_looks_like_a_divergence_retry():
    # Divergence (ADR-0055) offers a single half-rate retry as a choice; a
    # memory retry is automatic. The codes must be disjoint so a client and
    # the database can tell the two decisions apart.
    divergence_codes = {DIVERGED_CODE, INSTABILITY_CODE}
    assert MEMORY_FAILURE_CODES.isdisjoint(divergence_codes)
    assert MEMORY_RETRIES_EXHAUSTED_CODE != DIVERGED_CODE
    assert MEMORY_RECOVERY_CODE not in divergence_codes
    assert MEMORY_RECOVERY_CODE not in MEMORY_FAILURE_CODES


def test_the_exhaustion_code_is_stable_and_the_message_is_plain():
    assert MEMORY_RETRIES_EXHAUSTED_CODE == "memory_retries_exhausted"
    assert REAL_OOM_CODE == "training_oom"
