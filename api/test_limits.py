"""Runtime limits, exercised without waiting for them.

Every test here runs against a clock the test controls, so a fifteen-minute
silence and a twenty-four-hour run both happen in milliseconds. That is the
whole reason time is a parameter: a limit whose test sleeps is a limit whose
test gets skipped.
"""

import threading

import pytest

from api.errors import OrchestratorError
from api.fake_provider import FakeClock
from api.limits import RunLimits, guard

# Small enough that the guard's queue wait never dominates a test, and
# unrelated to the limits themselves -- the clock decides when a limit trips,
# this only decides how often the guard looks.
SLICE = 0.005


def limits(clock, *, stall=900.0, maximum=86400.0):
    return RunLimits(stall_timeout_s=stall, max_duration_s=maximum,
                     now=clock, poll_interval_s=SLICE)


def test_output_passes_through_untouched():
    clock = FakeClock(step=1.0)
    lines = ["one", "two", "three"]
    assert list(guard(iter(lines), limits(clock), 0.0)) == lines


def test_silence_beyond_the_stall_timeout_kills_the_job():
    """A source that stops producing output, without ending."""
    release = threading.Event()

    def hangs():
        yield "training started"
        release.wait(30)          # never set: the guard must give up first

    clock = FakeClock(step=60.0)      # a simulated minute per look
    with pytest.raises(OrchestratorError) as e:
        list(guard(hangs(), limits(clock, stall=900.0), 0.0))

    assert e.value.code == "gpu_stalled"
    assert "900" in str(e.value)
    release.set()


def test_exceeding_the_maximum_duration_kills_the_job():
    """Output keeps arriving; the job is simply too long to be sane.

    The clock steps less than the stall timeout per line, so the stall
    detector keeps resetting and this can only be the ceiling firing.
    """
    clock = FakeClock(step=300.0)
    source = iter(f"line {i}" for i in range(1000))
    with pytest.raises(OrchestratorError) as e:
        list(guard(source, limits(clock, stall=900.0, maximum=3600.0), 0.0))

    assert e.value.code == "gpu_max_duration_exceeded"
    assert "3600" in str(e.value)


def test_the_ceiling_counts_from_the_start_of_the_job_not_the_stream():
    """Provisioning is part of a job's duration, so it is part of the budget."""
    clock = FakeClock(step=1.0)
    with pytest.raises(OrchestratorError) as e:
        list(guard(iter(["a", "b"]), limits(clock, maximum=3600.0),
                   started=-7200.0))
    assert e.value.code == "gpu_max_duration_exceeded"


def test_a_long_silence_that_ends_in_time_is_reported_not_punished():
    """The mechanism is observable: a survived silence says so.

    Not one event per line -- the event log is the user's view of their run,
    and a bookkeeping line per training step would bury the output it exists
    to make legible.
    """
    resets = []
    clock = FakeClock(step=400.0)     # each line arrives after a simulated 400s
    out = list(guard(iter(["a", "b", "c"]), limits(clock, stall=900.0),
                     0.0, on_reset=resets.append))

    assert out == ["a", "b", "c"]
    assert resets and all(gap >= 400.0 for gap in resets)


def test_short_gaps_do_not_report():
    resets = []
    clock = FakeClock(step=1.0)
    list(guard(iter(["a", "b", "c"]), limits(clock, stall=900.0), 0.0,
               on_reset=resets.append))
    assert resets == []


def test_an_error_from_the_source_reaches_the_caller():
    """The guard is a pass-through, including for failure."""
    def explodes():
        yield "a"
        raise RuntimeError("ssh died")

    clock = FakeClock(step=1.0)
    with pytest.raises(RuntimeError, match="ssh died"):
        list(guard(explodes(), limits(clock), 0.0))
