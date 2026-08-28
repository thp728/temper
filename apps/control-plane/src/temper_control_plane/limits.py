"""Circuit breakers for a job that has stopped making progress.

Three controls with different meanings and different outcomes:

* **A stall** — no output for the configured period. The job is not obviously
  broken; it has gone quiet, and a quiet job on a billing machine is
  indistinguishable from a wedged one until somebody looks.
* **The duration ceiling** — the job has run longer than any legitimate run on
  this catalog should, whether or not it is still talking.
* **The spend ceiling** — the job has cost more than any single job should,
  however fast it is talking. This is the one issue #46 adds, and it is a
  different control from the other two: it is a cap on *money*, so an
  expensive machine is stopped sooner than a cheap one at the same ceiling.

They are kept apart because the remedies are not the same. A stall points at
the machine or the training process; hitting the duration ceiling points at
the job being too large for the limit, which is a policy conversation;
hitting the spend ceiling points at the job costing too much, which is a
money conversation. Collapsing them into one "job took too long" would hand
the user the wrong question, in the same way collapsing *unreachable* and
*authentication failed* did.

**These are not spend policy beyond the cap itself.** The stall and duration
controls exist to stop a job that is no longer doing anything; the spend
ceiling exists to stop a runaway before it consumes the account. The
distinction matters because it decides what the defaults are: the stall and
duration limits are set from the longest silence and the longest run that are
still plausible, and the spend ceiling is set from what a legitimate run can
cost.

**The spend ceiling is enforced from outside the training process.** The
training process is the container on the machine; these checks run in the
control plane, and the spend they measure is elapsed wall clock times the
price the job froze at provisioning — never anything the trainer reports. A
process that has stopped responding cannot enforce its own limit, so a
machine that goes silent (and keeps billing) still trips the ceiling, because
the measurement does not depend on it answering.

The guard sits between the provider's line iterator and everything that reads
it, so it works for any transport — the enforcement lives here rather than in
the SSH implementation, and a push transport inherits it unchanged.

**Time is a parameter.** The clock is injected so that a fifteen-minute
silence and a twenty-four-hour run can both be exercised in milliseconds. A
limit whose test has to sleep is a limit whose test gets skipped, and then the
limit is back to being a constant nobody reads.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

from temper_core import quote
from temper_core.errors import OrchestratorError

from . import config

# A silence worth reporting, as a fraction of the stall budget. The literal
# reading of "report every reset" is one event per output line, which would
# more than double the event log with bookkeeping and bury the training output
# it exists to make visible. A quarter of the budget is the point at which a
# gap is interesting: it is far longer than the gaps a healthy run produces,
# and it is the first warning that the next one may not come back.
REPORT_FRACTION = 0.25

# The stable code a job stopped by the spend ceiling carries (issue #46). The
# same vocabulary as the reference architecture's `budget_exhausted` terminal
# code, so a run stopped by money is distinguishable from one stopped by time
# (`gpu_max_duration_exceeded`) or by silence (`gpu_stalled`), which is the
# point of the user story: a safety-limit stop is a failure with a specific
# reason, not something the user chose and not something that just broke.
BUDGET_EXHAUSTED_CODE = "budget_exhausted"


@dataclass(frozen=True)
class RunLimits:
    """The limits, plus how the guard perceives time.

    `poll_interval_s` is not a limit — it is how long the guard is willing to
    block before looking at the clock again. It exists because the clock is
    injected and the queue's wait is not: with real time the two agree, and a
    test that drives a fake clock needs the guard to come up for air.

    The spend ceiling is `None` until `with_spend` configures it, because the
    ceiling is a *cost* and a cost can only be enforced against a price. The
    price is chosen at provisioning, after this object is constructed, so the
    orchestrator freezes it here the moment the plan is selected.
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
    # The spend ceiling (issue #46), as a cost in the account currency's
    # minor unit, and the rate to derive elapsed spend from. Both are `None`
    # until `with_spend` configures them after provisioning, which is also
    # what makes the default "no spend ceiling" — every existing caller
    # constructs a `RunLimits` without them and is unchanged.
    spend_ceiling_minor: int | None = None
    price_per_hour: float | None = None
    currency: str | None = None

    @classmethod
    def from_config(
        cls, now: Callable[[], float] = time.monotonic
    ) -> RunLimits:
        return cls(
            stall_timeout_s=config.STALL_TIMEOUT_S,
            max_duration_s=config.MAX_JOB_DURATION_S,
            now=now,
        )

    def start(self) -> RunLimits:
        """Stamp the job's origin. The ceilings count from here."""
        return replace(self, started=self.now())

    def with_spend(
        self,
        ceiling_minor: int,
        price_per_hour: float,
        currency: str,
    ) -> RunLimits:
        """Return a copy carrying the spend ceiling and the rate to measure it.

        Called once provisioning chose a machine and froze its price, so the
        ceiling becomes enforceable as a wall-clock deadline derived from the
        job's own rate — the same derivation `temper_core.actuals` uses for
        the measured cost. An unenforceable combination (a currency the quote
        has no minor unit for, a non-positive rate or ceiling) is refused with
        a coded error before anything is provisioned, rather than silently
        becoming a ceiling that never fires.
        """
        if not isinstance(price_per_hour, (int, float)) or price_per_hour <= 0:
            raise OrchestratorError(
                "budget_unconfigurable",
                f"A spend ceiling cannot be enforced against price "
                f"{price_per_hour!r}; refusing to provision a job whose "
                "ceiling could never fire.",
            )
        try:
            quote.minor_unit_for(currency)
        except ValueError as e:
            raise OrchestratorError(
                "budget_unconfigurable",
                f"A spend ceiling cannot be enforced in currency "
                f"{currency!r}; refusing to provision a job whose ceiling "
                "could never fire.",
            ) from e
        if int(ceiling_minor) <= 0:
            raise OrchestratorError(
                "budget_unconfigurable",
                f"A spend ceiling of {ceiling_minor!r} minor units is not a "
                "ceiling; refusing to provision a job under a cap that "
                "cannot fire.",
            )
        return replace(
            self,
            spend_ceiling_minor=int(ceiling_minor),
            price_per_hour=float(price_per_hour),
            currency=currency,
        )

    @property
    def deadline(self) -> float:
        if self.started is None:
            raise ValueError("limits were never started")
        return self.started + self.max_duration_s

    def _resolved_spend(self) -> tuple[float, int]:
        """(started, minor unit) for the configured spend ceiling, or raise.

        The one place the spend ceiling's configuration is resolved: the
        deadline and the message both need the same "is it configured, and
        what is the currency's minor unit" answer, and a value read once is a
        value that cannot disagree with itself between the two readers.
        """
        started = self.started
        ceiling = self.spend_ceiling_minor
        price = self.price_per_hour
        currency = self.currency
        if (
            started is None
            or ceiling is None
            or price is None
            or currency is None
        ):
            raise ValueError("spend ceiling was never configured")
        return started + ceiling * (
            quote.SECONDS_PER_HOUR / (price * quote.minor_unit_for(currency))
        ), quote.minor_unit_for(currency)

    @property
    def spend_deadline(self) -> float:
        """The moment elapsed spend reaches the ceiling, on this object's clock.

        Derived, never guessed: the ceiling is a cost in minor units, and the
        rate the job froze at provisioning says how fast those units accrue,
        so `ceiling_minor * (3600 / (price_per_hour * minor_unit))` is the
        budget in seconds. An expensive machine exhausts the same cost sooner
        than a cheap one, which is the whole point of a money ceiling rather
        than a minute ceiling.
        """
        deadline, _minor = self._resolved_spend()
        return deadline

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
            raise OrchestratorError(
                "gpu_max_duration_exceeded", self.max_duration_message()
            )

    def check_spend(self, at: float | None = None) -> None:
        """Raise if the spend ceiling has passed. Callable between stages.

        The same boundary role as `check_duration`, for the money ceiling: a
        machine that costs more than the cap while the control plane is still
        setting it up is a runaway as surely as one that overruns in time. A
        no-op when no spend ceiling has been configured. `at` is the instant
        to judge, for callers that already read the clock once (the guard
        passes the same `t` it uses for the duration ceiling, so the two
        cannot disagree about the moment); a boundary call reads it fresh.
        """
        if self.spend_ceiling_minor is None:
            return
        instant = self.now() if at is None else at
        if instant >= self.spend_deadline:
            raise OrchestratorError(
                BUDGET_EXHAUSTED_CODE, self.budget_message()
            )

    def max_duration_message(self) -> str:
        return (
            f"The job ran longer than the maximum permitted duration of "
            f"{self.max_duration_s:.0f}s and was stopped. Its machine is "
            f"being destroyed; the teardown confirmation follows in this "
            f"job's events."
        )

    def budget_message(self) -> str:
        _deadline, minor = self._resolved_spend()
        ceiling = self.spend_ceiling_minor
        assert ceiling is not None  # _resolved_spend just confirmed it
        amount = ceiling / minor
        currency = self.currency
        assert currency is not None  # _resolved_spend just confirmed it
        return (
            f"The job exceeded the spend ceiling of {amount:.2f} {currency} "
            f"and was stopped. Its checkpoints were saved where possible; its "
            f"machine is being destroyed, and the teardown confirmation "
            f"follows in this job's events."
        )

    def stall_message(self) -> str:
        return (
            f"The job produced no output for {self.stall_timeout_s:.0f}s "
            f"and was treated as stalled. Its machine is being destroyed; "
            f"the teardown confirmation follows in this job's events."
        )


def guard(
    source: Iterator[str],
    limits: RunLimits,
    on_reset: Callable[[float], None] | None = None,
    check: Callable[[], None] | None = None,
) -> Iterator[str]:
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
        except BaseException as e:  # relayed, not swallowed
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
            raise OrchestratorError(
                "gpu_max_duration_exceeded", limits.max_duration_message()
            )
        # The spend ceiling is checked on the same read of the clock as the
        # duration ceiling, so the two cannot disagree about the instant. It
        # is checked before the stall: a machine going silent is often the
        # same moment its spend is still accumulating, and the money ceiling
        # is the reason that names the bill.
        limits.check_spend(at=t)
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
