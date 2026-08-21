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
from dataclasses import dataclass
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

    @classmethod
    def from_config(cls, now: Callable[[], float] = time.monotonic) -> "RunLimits":
        return cls(stall_timeout_s=config.STALL_TIMEOUT_S,
                   max_duration_s=config.MAX_JOB_DURATION_S, now=now)


def guard(source: Iterator[str], limits: RunLimits, started: float,
          on_reset: Callable[[float], None] | None = None) -> Iterator[str]:
    """Yield lines from `source`, refusing to wait forever for the next one.

    `started` is when the *job* began, not when the stream did: provisioning
    and waiting for SSH are part of a job's elapsed time, so they are part of
    its budget.

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
    deadline = started + limits.max_duration_s
    last_line = started

    while True:
        # Read exactly once per iteration: every check below is answering a
        # question about the same instant, and a clock read twice is a clock
        # that can move between two halves of one decision.
        t = limits.now()
        if t >= deadline:
            raise OrchestratorError(
                "gpu_max_duration_exceeded",
                f"The job ran longer than the maximum permitted duration of "
                f"{limits.max_duration_s:.0f}s and was stopped. The machine "
                f"has been destroyed.")
        if t - last_line >= limits.stall_timeout_s:
            raise OrchestratorError(
                "gpu_stalled",
                f"The job produced no output for {limits.stall_timeout_s:.0f}s "
                f"and was treated as stalled. The machine has been destroyed.")

        try:
            kind, payload = items.get(timeout=limits.poll_interval_s)
        except queue.Empty:
            continue

        if kind == "done":
            return
        if kind == "error":
            raise payload

        gap, last_line = t - last_line, t
        if on_reset is not None and gap >= report_after:
            on_reset(gap)
        yield payload
