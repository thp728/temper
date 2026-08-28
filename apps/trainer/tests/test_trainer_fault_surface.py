"""The trainer's half of the fault surface (issue #24).

Trainer-side faults are an environment switch the trainer reads: `main` reads
`TEMPER_FAULT_SPEC` and makes the fault real -- exhausting device memory,
driving the loss to a meaningless value, or killing the worker. These tests
pin the pure, host-runnable parts: reading and validating the spec, applying
the divergence override to the config, and the two agreements that keep the
trainer and the control plane from drifting (the environment variable's name
and the fault vocabulary itself). The runtime halves (`_exhaust_device_memory`,
`_kill_worker`) deliberately have no host test: they need a GPU or would kill
the test process, and they only ever run inside the pinned image.
"""

import json

import entrypoint
import pytest

from temper_core import faults as fault_surface


def test_the_environment_variable_name_is_defined_once():
    """The wire name the control plane writes and this trainer reads cannot
    drift; the trainer cannot import `temper_core` in the image (ADR-0010), so
    the agreement is pinned by a test instead."""
    assert entrypoint.FAULT_ENV == fault_surface.FAULT_ENV


def test_the_trainer_reads_the_same_vocabulary_as_the_control_plane():
    """Both sides validate fault names against the one contract file
    (`packages/contracts/fault-surface.json`), read as data on each side."""
    from entrypoint import _FAULTS_BY_NAME

    assert set(_FAULTS_BY_NAME) == set(fault_surface.names())


def test_the_surface_is_off_when_the_environment_is_unset():
    """Off by default is a safety property: no environment, no fault."""
    assert entrypoint.read_fault_spec(None) is None
    assert entrypoint.read_fault_spec("") is None
    assert entrypoint.read_fault_spec("   ") is None


def test_a_valid_fault_spec_is_parsed():
    spec = entrypoint.read_fault_spec(
        json.dumps({"name": "oom", "delay_s": 5})
    )
    assert spec == {"name": "oom", "delay_s": 5}


def test_an_unknown_fault_name_is_refused_loudly():
    with pytest.raises(ValueError, match="not a fault the surface knows"):
        entrypoint.read_fault_spec(json.dumps({"name": "not_a_fault"}))


def test_malformed_environment_is_refused_loudly():
    with pytest.raises(ValueError, match="not JSON"):
        entrypoint.read_fault_spec("definitely not json")
    with pytest.raises(ValueError, match="must be an object"):
        entrypoint.read_fault_spec(json.dumps(["not", "a", "dict"]))


def test_a_parameter_a_fault_does_not_take_is_refused_loudly():
    """A fault spec the caller believes is in effect but is not is worse than
    a refusal: divergence takes no `delay_s`, so asking for one must fail
    rather than be silently dropped."""
    with pytest.raises(ValueError, match="does not take parameter"):
        entrypoint.read_fault_spec(
            json.dumps({"name": "divergence", "delay_s": 5})
        )


def test_a_bad_delay_is_refused_loudly():
    with pytest.raises(ValueError, match="delay_s"):
        entrypoint.read_fault_spec(
            json.dumps({"name": "oom", "delay_s": "soon"})
        )
    with pytest.raises(ValueError, match="delay_s"):
        entrypoint.read_fault_spec(json.dumps({"name": "oom", "delay_s": 0}))


def test_divergence_overrides_the_resolved_learning_rate():
    """A loss driven to a meaningless value is made by handing the optimiser
    a learning rate no run could survive; the sabotage is recorded in the
    config the run writes, so the run's record says what actually trained."""
    cfg = {"learning_rate": 2e-4}
    entrypoint.apply_fault(cfg, {"name": "divergence"})
    assert cfg["learning_rate"] == 2e-4 * 1e6


def test_the_other_trainer_faults_do_not_touch_the_config():
    cfg = {"learning_rate": 2e-4}
    entrypoint.apply_fault(cfg, {"name": "oom"})
    entrypoint.apply_fault(cfg, {"name": "worker_kill"})
    assert cfg["learning_rate"] == 2e-4
