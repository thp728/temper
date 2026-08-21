"""Runtime limits, exercised without waiting for them.

Every test here runs against a clock the test controls, so a fifteen-minute
silence and a twenty-four-hour run both happen in milliseconds. That is the
whole reason time is a parameter: a limit whose test sleeps is a limit whose
test gets skipped.
"""

import threading
from dataclasses import replace

import pytest

from api.errors import OrchestratorError
from api.fake_provider import simulated_limits
from api.limits import guard

def test_output_passes_through_untouched():
    lines = ["one", "two", "three"]
    assert list(guard(iter(lines), simulated_limits(step=1.0))) == lines


def test_silence_beyond_the_stall_timeout_kills_the_job():
    """A source that stops producing output, without ending."""
    release = threading.Event()

    def goes_quiet():
        yield "training started"
        release.wait(30)          # never set: the guard must give up first

    with pytest.raises(OrchestratorError) as e:                # a minute a look
        list(guard(goes_quiet(), simulated_limits(step=60.0, stall=900.0)))

    assert e.value.code == "gpu_stalled"
    assert "900" in str(e.value)
    release.set()


def test_exceeding_the_maximum_duration_kills_the_job():
    """Output keeps arriving; the job is simply too long to be sane.

    The clock steps less than the stall timeout per line, so the stall
    detector keeps resetting and this can only be the ceiling firing.
    """
    source = iter(f"line {i}" for i in range(1000))
    with pytest.raises(OrchestratorError) as e:
        list(guard(source, simulated_limits(step=300.0, stall=900.0,
                                            maximum=3600.0)))

    assert e.value.code == "gpu_max_duration_exceeded"
    assert "3600" in str(e.value)


def test_the_ceiling_counts_from_the_start_of_the_job_not_the_stream():
    """Provisioning is part of a job's duration, so it is part of the budget.

    The origin is two simulated hours before the stream opened, which is a job
    that spent its whole budget before producing a line. Measuring from the
    first line instead would call this a healthy run.
    """
    limits = replace(simulated_limits(step=1.0, maximum=3600.0),
                     started=-7200.0)
    with pytest.raises(OrchestratorError) as e:
        list(guard(iter(["a", "b"]), limits))
    assert e.value.code == "gpu_max_duration_exceeded"


def test_the_stall_budget_counts_from_the_stream_not_the_job():
    """The other origin, and it is the other way round on purpose.

    Charging four minutes of provisioning against the first line's grace period
    would quietly shorten the stall timeout by however long the machine took to
    arrive -- so a slow provision would look like a stall.
    """
    limits = replace(simulated_limits(step=1.0, stall=900.0), started=-7200.0)
    assert list(guard(iter(["a", "b"]), limits)) == ["a", "b"]


def test_a_long_silence_that_ends_in_time_is_reported_not_punished():
    """The mechanism is observable: a survived silence says so.

    Not one event per line -- the event log is the user's view of their run,
    and a bookkeeping line per training step would bury the output it exists
    to make legible.
    """
    resets = []
    # Each line arrives after a simulated 400s -- long, and not yet a stall.
    out = list(guard(iter(["a", "b", "c"]),
                     simulated_limits(step=400.0, stall=900.0),
                     on_reset=resets.append))

    assert out == ["a", "b", "c"]
    assert resets and all(gap >= 400.0 for gap in resets)


def test_short_gaps_do_not_report():
    resets = []
    list(guard(iter(["a", "b", "c"]), simulated_limits(step=1.0, stall=900.0),
               on_reset=resets.append))
    assert resets == []


def test_an_error_from_the_source_reaches_the_caller():
    """The guard is a pass-through, including for failure."""
    def explodes():
        yield "a"
        raise RuntimeError("ssh died")

    with pytest.raises(RuntimeError, match="ssh died"):
        list(guard(explodes(), simulated_limits(step=1.0)))


# --- the limits are configuration, and a misconfiguration is refused --------

def test_a_limit_read_from_the_environment_is_used(monkeypatch):
    from api import config

    monkeypatch.setenv("TEMPER_STALL_TIMEOUT_S", "120")
    assert config._seconds("TEMPER_STALL_TIMEOUT_S", 900) == 120


def test_an_unset_or_empty_limit_falls_back_to_the_default(monkeypatch):
    from api import config

    monkeypatch.delenv("TEMPER_STALL_TIMEOUT_S", raising=False)
    assert config._seconds("TEMPER_STALL_TIMEOUT_S", 900) == 900
    monkeypatch.setenv("TEMPER_STALL_TIMEOUT_S", "   ")
    assert config._seconds("TEMPER_STALL_TIMEOUT_S", 900) == 900


@pytest.mark.parametrize("raw", ["15m", "fifteen", "0", "-5"])
def test_a_limit_that_cannot_be_honoured_stops_the_process(monkeypatch, raw):
    """Refused loudly, not quietly replaced by the default.

    `TEMPER_STALL_TIMEOUT_S=15m` is the realistic typo, and falling back would
    leave an operator believing a limit was in force that was not — the same
    shape as the unread constant this work deleted, and less visible, because
    there would be nothing wrong in the source to find.
    """
    from api import config

    monkeypatch.setenv("TEMPER_STALL_TIMEOUT_S", raw)
    with pytest.raises(ValueError, match="TEMPER_STALL_TIMEOUT_S"):
        config._seconds("TEMPER_STALL_TIMEOUT_S", 900)
