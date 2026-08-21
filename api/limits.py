"""Circuit breakers for a job that has stopped making progress.

Two controls with different meanings and different outcomes:

* **A stall** — no output for the configured period. The job is not obviously
  broken; it has gone quiet, and a quiet job on a billing machine is
  indistinguishable from a wedged one until somebody looks.
* **The duration ceiling** — the job has run longer than any legitimate run on
  this catalog should, whether or not it is still talking.

They are kept apart because the remedies are not the same. A stall points at
the machine or the training process; hitting the ceiling points at the job
being too large for the limit, which is a policy conversation. Collapsing them
into one "job took too long" would hand the user the wrong question, in the
same way collapsing *unreachable* and *authentication failed* did.

**These are not spend policy.** They exist to stop a job that is no longer
doing anything, not to cap what a user may legitimately train. The distinction
matters because it decides what the defaults are: a spend cap would be set from
a budget, and these are set from the longest silence and the longest run that
are still plausible.

The guard sits between the provider's line iterator and everything that reads
it, so it works for any transport — the enforcement lives here rather than in
the SSH implementation, and a push transport inherits it unchanged.

**Time is a parameter.** The clock is injected so that a fifteen-minute silence
and a twenty-four-hour run can both be exercised in milliseconds. A limit whose
test has to sleep is a limit whose test gets skipped, and then the limit is
back to being a constant nobody reads.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterator

from . import config
from .errors import OrchestratorError

# A silence worth reporting, as a fraction of the stall budget. The literal
# reading of "report every reset" is one event per output line, which would
# more than double the event log with bookkeeping and bury the training output
# it exists to make visible. A quarter of the budget is the point at which a
# gap is interesting: it is far longer than the gaps a healthy run produces,
# and it is the first warning that the next one may not come back.
REPORT_FRACTION = 0.25


@dataclass(frozen=True)
class RunLimits:
    """The two limits, plus how the guard perceives time.

    `poll_interval_s` is not a limit — it is how long the guard is willing to
    block before looking at the clock again. It exists because the clock is
    injected and the queue's wait is not: with real time the two agree, and a
    test that drives a fake clock needs the guard to come up for air.
    """

    stall_timeout_s: float
    max_duration_s: float
    now: Callable[[], float] = time.monotonic
    poll_interval_s: float = 1.0
    # When the job began, by this object's own clock. `None` until `start()`
    # stamps it. It lives here rather than travelling alongside as a second
    # argument so that the origin and the clock measuring from it cannot come
    # from two different places — which is the one way this could silently
    # measure nothing at all.
    started: float | None = None

    @classmethod
    def from_config(cls, now: Callable[[], float] = time.monotonic) -> "RunLimits":
        return cls(stall_timeout_s=config.STALL_TIMEOUT_S,
                   max_duration_s=config.MAX_JOB_DURATION_S, now=now)

    def start(self) -> "RunLimits":
        """Stamp the job's origin. The ceiling counts from here."""
        return replace(self, started=self.now())

    @property
    def deadline(self) -> float:
        if self.started is None:
            raise ValueError("limits were never started")
        return self.started + self.max_duration_s

    def check_duration(self) -> None:
        """Raise if the ceiling has passed. Callable between stages.

        The guard below can only notice the ceiling while it is reading lines.
        A job spends time before that — provisioning, waiting for SSH, pushing
        sources — so the boundary between stages is checked too. What is still
        not interruptible is the inside of a single provider call; those carry
        their own timeouts, and saying so is more useful than implying this
        covers them.
        """
        if self.now() >= self.deadline:
            raise OrchestratorError("gpu_max_duration_exceeded",
                                    self.max_duration_message())

    def max_duration_message(self) -> str:
        return (f"The job ran longer than the maximum permitted duration of "
                f"{self.max_duration_s:.0f}s and was stopped. Its machine is "
                f"being destroyed; the teardown confirmation follows in this "
                f"job's events.")

    def stall_message(self) -> str:
        return (f"The job produced no output for {self.stall_timeout_s:.0f}s "
                f"and was treated as stalled. Its machine is being destroyed; "
                f"the teardown confirmation follows in this job's events.")


def guard(source: Iterator[str], limits: RunLimits,
          on_reset: Callable[[float], None] | None = None,
          check: Callable[[], None] | None = None) -> Iterator[str]:
    """Yield lines from `source`, refusing to wait forever for the next one.

    `check` is called once per iteration and may raise to abandon the run. It
    is how cancellation gets a hearing: this loop is the only thing running
    while a job trains, and it comes round whether or not a line arrived, so a
    request to stop is noticed within a poll interval even on a silent stream.
    The guard does not know what the check is for — it does not import the
    database, and it does not decide what stopping means. That keeps the two
    limits here and the user's decision elsewhere, which is right, because one
    of the three ends the job `failed` and the other does not.

    The two limits are measured from different origins on purpose. The ceiling
    counts from when the *job* began, because provisioning and waiting for SSH
    are part of a job's elapsed time. The stall budget counts from **here**,
    because it is a statement about the stream: charging four minutes of
    provisioning against the first line's grace period would quietly shorten
    the stall timeout by however long the machine took to arrive.

    The source is drained on its own thread because reading it is what blocks —
    there is no way to ask a plain iterator whether the next item is late. When
    a limit trips, that thread is left where it is rather than reached into:
    closing a generator another thread is executing is not something Python
    permits, and it is unnecessary here. The caller's next act on this path is
    to destroy the machine, which drops the connection the thread is blocked
    on, and the thread is a daemon in any case.
    """
    items: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            for line in source:
                items.put(("line", line))
        except BaseException as e:                # relayed, not swallowed
            items.put(("error", e))
            return
        items.put(("done", None))

    threading.Thread(target=pump, daemon=True, name="limit-guard").start()

    report_after = limits.stall_timeout_s * REPORT_FRACTION
    deadline = limits.deadline
    last_line = limits.now()

    while True:
        # First, because a user who has asked to stop is owed that answer
        # before the machine is given another poll interval to bill for.
        if check is not None:
            check()
        # Read exactly once per iteration: every check below is answering a
        # question about the same instant, and a clock read twice is a clock
        # that can move between two halves of one decision.
        t = limits.now()
        if t >= deadline:
            raise OrchestratorError("gpu_max_duration_exceeded",
                                    limits.max_duration_message())
        if t - last_line >= limits.stall_timeout_s:
            raise OrchestratorError("gpu_stalled", limits.stall_message())

        try:
            kind, payload = items.get(timeout=limits.poll_interval_s)
        except queue.Empty:
            continue

        if kind == "done":
            return
        if kind == "error":
            raise payload

        # `t` was read before the wait below returned, so a reported gap can
        # understate the real silence by up to one poll interval. That is a
        # second against a budget of minutes, and it buys one clock reading per
        # iteration, which is what keeps simulated time deterministic.
        gap, last_line = t - last_line, t
        if on_reset is not None and gap >= report_after:
            on_reset(gap)
        yield payload
