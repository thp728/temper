"""Runtime limits, exercised without waiting for them.

Every test here runs against a clock the test controls, so a fifteen-minute
silence and a twenty-four-hour run both happen in milliseconds. That is the
whole reason time is a parameter: a limit whose test sleeps is a limit whose
test gets skipped.
"""

import threading
from dataclasses import replace

import pytest

from temper_control_plane.fake_provider import simulated_limits
from temper_control_plane.limits import guard
from temper_core.errors import OrchestratorError


def test_output_passes_through_untouched():
    lines = ["one", "two", "three"]
    assert list(guard(iter(lines), simulated_limits(step=1.0))) == lines


def test_silence_beyond_the_stall_timeout_kills_the_job():
    """A source that stops producing output, without ending."""
    release = threading.Event()

    def goes_quiet():
        yield "training started"
        release.wait(30)  # never set: the guard must give up first

    with pytest.raises(OrchestratorError) as e:  # a minute a look
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
        list(
            guard(
                source,
                simulated_limits(step=300.0, stall=900.0, maximum=3600.0),
            )
        )

    assert e.value.code == "gpu_max_duration_exceeded"
    assert "3600" in str(e.value)


def test_the_ceiling_counts_from_the_start_of_the_job_not_the_stream():
    """Provisioning is part of a job's duration, so it is part of the budget.

    The origin is two simulated hours before the stream opened, which is a job
    that spent its whole budget before producing a line. Measuring from the
    first line instead would call this a healthy run.
    """
    limits = replace(
        simulated_limits(step=1.0, maximum=3600.0), started=-7200.0
    )
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
    out = list(
        guard(
            iter(["a", "b", "c"]),
            simulated_limits(step=400.0, stall=900.0),
            on_reset=resets.append,
        )
    )

    assert out == ["a", "b", "c"]
    assert resets and all(gap >= 400.0 for gap in resets)


def test_short_gaps_do_not_report():
    resets = []
    list(
        guard(
            iter(["a", "b", "c"]),
            simulated_limits(step=1.0, stall=900.0),
            on_reset=resets.append,
        )
    )
    assert resets == []


def test_an_error_from_the_source_reaches_the_caller():
    """The guard is a pass-through, including for failure."""

    def explodes():
        yield "a"
        raise RuntimeError("ssh died")

    with pytest.raises(RuntimeError, match="ssh died"):
        list(guard(explodes(), simulated_limits(step=1.0)))


# --- the spend ceiling (issue #46) -------------------------------------------
#
# The spend ceiling is a third control beside the stall and the duration
# ceiling, and it is measured the same way the other two are: from the clock,
# in the control plane, never from anything the trainer reports. A job that
# passes the ceiling fails with budget_exhausted, distinguishable from a stall
# and from an over-long run.


def _spend_limits(
    step=300.0, ceiling_minor=100, price_per_hour=41.31, **kwargs
):
    return replace(
        simulated_limits(step=step, **kwargs),
        spend_ceiling_minor=ceiling_minor,
        price_per_hour=price_per_hour,
        currency="INR",
    )


def test_the_guard_stops_a_job_that_exceeds_the_spend_ceiling():
    """The ceiling is enforced on the same read of the clock as the duration
    ceiling, so it fires on the control plane's own measurement of elapsed
    time times the frozen price -- not on anything the source reports."""
    source = iter(["a", "b", "c", "d"])
    with pytest.raises(OrchestratorError) as e:
        list(guard(source, _spend_limits()))

    assert e.value.code == "budget_exhausted"
    assert "spend ceiling" in str(e.value)
    assert "INR" in str(e.value)


def test_a_job_under_the_spend_ceiling_is_untouched():
    source = iter(["a", "b", "c", "d"])
    out = list(
        guard(source, _spend_limits(ceiling_minor=1_000_000, step=60.0))
    )
    assert out == ["a", "b", "c", "d"]


def test_the_spend_ceiling_fires_before_the_stall_on_a_silent_source():
    """The point of enforcing spend outside the training process: a source
    that stops producing output (a wedged trainer) is still billed for, and
    the money ceiling catches it before the stall detector has any reason to.
    The stall budget is far beyond anything this test reaches, so the only
    thing that can have stopped it is the ceiling."""
    release = threading.Event()

    def goes_quiet():
        yield "training started"
        release.wait(30)  # never set: the ceiling must fire first

    with pytest.raises(OrchestratorError) as e:
        list(
            guard(
                goes_quiet(),
                _spend_limits(step=60.0, stall=90000.0),
            )
        )
    assert e.value.code == "budget_exhausted"
    release.set()


def test_the_spend_deadline_shrinks_as_the_machines_price_rises():
    """The ceiling is a cost, so the same ceiling is exhausted sooner on an
    expensive machine than a cheap one: the deadline is derived from the job's
    own frozen price, not from a fixed number of minutes."""
    cheap = _spend_limits(price_per_hour=41.31, ceiling_minor=1000)
    dear = _spend_limits(price_per_hour=250.0, ceiling_minor=1000)
    assert cheap.spend_deadline > dear.spend_deadline


def test_with_spend_refuses_an_unenforceable_combination():
    """A ceiling that cannot fire is worse than none: an unknown currency, a
    non-positive rate or a non-positive ceiling is refused loudly before
    anything is provisioned."""
    limits = simulated_limits(step=1.0)
    with pytest.raises(OrchestratorError) as e:
        limits.with_spend(100, 41.31, "BTC")
    assert e.value.code == "budget_unconfigurable"
    with pytest.raises(OrchestratorError):
        limits.with_spend(100, 0, "INR")
    with pytest.raises(OrchestratorError):
        limits.with_spend(0, 41.31, "INR")


def test_the_spend_ceiling_is_off_until_with_spend_configures_it():
    """The default RunLimits has no spend ceiling: every existing caller is
    unchanged, and check_spend is a no-op until the orchestrator freezes the
    price after provisioning."""
    from temper_control_plane.limits import RunLimits

    limits = RunLimits(stall_timeout_s=900.0, max_duration_s=86400.0).start()
    limits.check_spend()  # must not raise
    assert limits.spend_ceiling_minor is None


# --- the limits are configuration, and a misconfiguration is refused --------


def test_a_limit_read_from_the_environment_is_used(monkeypatch):
    from temper_control_plane import config

    monkeypatch.setenv("TEMPER_STALL_TIMEOUT_S", "120")
    assert config._seconds("TEMPER_STALL_TIMEOUT_S", 900) == 120


def test_an_unset_or_empty_limit_falls_back_to_the_default(monkeypatch):
    from temper_control_plane import config

    monkeypatch.delenv("TEMPER_STALL_TIMEOUT_S", raising=False)
    assert config._seconds("TEMPER_STALL_TIMEOUT_S", 900) == 900
    monkeypatch.setenv("TEMPER_STALL_TIMEOUT_S", "   ")
    assert config._seconds("TEMPER_STALL_TIMEOUT_S", 900) == 900


@pytest.mark.parametrize("raw", ["15m", "fifteen", "0", "-5"])
def test_a_limit_that_cannot_be_honoured_stops_the_process(monkeypatch, raw):
    """Refused loudly, not quietly replaced by the default.

    `TEMPER_STALL_TIMEOUT_S=15m` is the realistic typo, and falling back would
    leave an operator believing a limit was in force that was not; the same
    shape as the unread constant this work deleted, and less visible, because
    there would be nothing wrong in the source to find.
    """
    from temper_control_plane import config

    monkeypatch.setenv("TEMPER_STALL_TIMEOUT_S", raw)
    with pytest.raises(ValueError, match="TEMPER_STALL_TIMEOUT_S"):
        config._seconds("TEMPER_STALL_TIMEOUT_S", 900)


# --- the check hook: somewhere for a decision that is not a limit -----------


def test_the_check_runs_on_every_iteration_and_can_stop_the_stream():
    """The guard's loop is where anything that wants to interrupt a run looks.

    It is the only code running while a job trains, so cancellation borrows it
    rather than growing a second loop beside it. The guard stays ignorant of
    what the check means: it raises, and whoever supplied it decides what the
    job's outcome is.
    """

    class Stop(Exception):
        pass

    calls = []

    def check():
        calls.append(len(calls))
        if len(calls) > 2:
            raise Stop()

    with pytest.raises(Stop):
        list(
            guard(
                iter(["one", "two", "three", "four"]),
                simulated_limits(step=1.0),
                check=check,
            )
        )
    assert len(calls) == 3, "the check is asked once per trip round the loop"


def test_the_check_is_heard_even_when_no_output_is_arriving():
    """A cancelled job that has gone quiet must still stop promptly.

    This is the case that decides where the check belongs. Asking once per
    arriving line would mean a wedged machine could not be cancelled at all --
    the request would wait out the stall timeout on a billing GPU, which is
    the exact bill cancelling exists to stop.
    """
    release = threading.Event()

    def goes_quiet():
        yield "training started"
        release.wait(30)  # never set: the check must fire first

    class Stop(Exception):
        pass

    seen = []

    def check():
        seen.append(1)
        if len(seen) > 3:
            raise Stop()

    # A stall budget far beyond anything this test reaches, so the only thing
    # that can end it is the check.
    with pytest.raises(Stop):
        list(
            guard(
                goes_quiet(),
                simulated_limits(step=1.0, stall=90000.0),
                check=check,
            )
        )
    release.set()
