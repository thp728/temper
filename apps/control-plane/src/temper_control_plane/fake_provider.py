"""A provider that spends no money.

Kept in the package rather than in a test file so that every ticket which
touches the orchestration path tests against the same double instead of growing
its own. It implements the `Provider` protocol and records what was asked of it.

What it can be told to do, because these are the paths worth testing:

* yield a scripted sequence of output lines;
* stop producing output part-way through;
* space its lines out in time, so that a test can tell output arriving while
  a job is working from output arriving once it has finished;
* fail at any stage, with a chosen error code — or with an exception nobody
  anticipated;
* fail the destroy call a chosen number of times, and go on being listed
  afterwards;
* go silent mid-stream without ending, which is what a stalled job looks like
  from here and is not the same thing as a stream that stops;
* hold still at a named stage or after a chosen number of lines, so that a
  test can act on a job at a known point in its run rather than sleeping and
  hoping — the difference between cancelling during the image build and
  cancelling during training is minutes on a real machine and microseconds
  here, and only a pause makes the two distinguishable.

It ships with a clock for the same reason: the limits that catch a silent job
are minutes and hours long, and a suite that waits for them is a suite nobody
runs.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Iterator, Sequence

from temper_core.errors import OrchestratorError

from .limits import RunLimits
from .provider import GpuChoice, Machine

STAGES = ("select_gpu", "create", "await_ready", "push", "fetch", "stream")

# Fixed rather than configurable: tests assert against these, and a knob no
# test turns is a knob that only makes the double harder to read.
MACHINE_ID = 4242
GPU = GpuChoice("L4", 41.31, "INR")

# How long a silent stream stays silent. Long enough that no limit under test
# can lose the race, short enough that a suite which somehow reaches it ends.
SILENCE_S = 10.0

# How long a pause waits to be released, and how long a test waits to reach
# one. Both are ceilings on a wait that normally ends in microseconds: they
# exist so that a mistake in a test fails it rather than hanging the suite.
PAUSE_S = 10.0
REACH_S = 5.0


class FakeClock:
    """Monotonic time under the test's control.

    It advances a fixed step on every *reading* rather than per second, which
    is what makes a stall test deterministic: the guard reads the clock exactly
    once per iteration, so the step says how much simulated time one trip round
    its loop costs, and the answer is the same on every machine.
    """

    def __init__(self, step: float = 0.0) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def simulated_limits(
    *, step: float, stall: float = 900.0, maximum: float = 86400.0
) -> RunLimits:
    """Limits whose minutes and hours pass in milliseconds.

    `step` is how much simulated time one trip round the guard's loop costs.
    `poll_interval_s` is how long the guard will really block before looking at
    the clock again — small here only so that simulated time, which moves when
    the clock is read, moves quickly in real time too. It is not a limit.

    Already started, because a test asking about a limit is asking about a job
    that is already running.
    """
    return RunLimits(
        stall_timeout_s=stall,
        max_duration_s=maximum,
        now=FakeClock(step=step),
        poll_interval_s=0.005,
    ).start()


class FakeProvider:
    def __init__(
        self,
        *,
        lines: Sequence[str] = (),
        result: dict | None = None,
        fail_at: str | None = None,
        fail_code: str = "training_failed",
        fail_unexpectedly: bool = False,
        stop_after: int | None = None,
        silent_after: int | None = None,
        line_delay: float = 0.0,
        destroy_failures: int = 0,
        stays_listed: bool = False,
        adapter_bytes: bytes = b"weights",
        pause_at_stage: str | None = None,
        pause_at_line: int | None = None,
    ) -> None:
        if fail_at is not None and fail_at not in STAGES:
            raise ValueError(f"unknown stage {fail_at!r}")
        if pause_at_stage is not None and pause_at_stage not in STAGES:
            raise ValueError(f"unknown stage {pause_at_stage!r}")
        self._lines = list(lines)
        self._result = result
        self._fail_at = fail_at
        self._fail_code = fail_code
        self._fail_unexpectedly = fail_unexpectedly
        self._stop_after = stop_after
        self._silent_after = silent_after
        self._line_delay = line_delay
        self._destroy_failures = destroy_failures
        self._stays_listed = stays_listed
        self._adapter_bytes = adapter_bytes
        self._pause_at_stage = pause_at_stage
        self._pause_at_line = pause_at_line

        # A pause the test drives: `paused` is set when the job reaches the
        # chosen point, and it stays there until the test sets `resume`. The
        # job's own thread is the one held, so whatever the test does in
        # between happens at a known point in the run rather than a hoped-for
        # one.
        self.paused = threading.Event()
        self.resume = threading.Event()

        # Observable afterwards.
        self.calls: list[str] = []
        self.created: list[Machine] = []
        self.destroy_attempts = 0
        self.destroyed = False
        self.pushed: list[tuple[str, bytes]] = []
        self.script: bytes | None = None
        self.closed = False

    # -- protocol -----------------------------------------------------------

    def select_gpu(self, preference: Sequence[str]) -> GpuChoice:
        self._enter("select_gpu")
        return GPU

    def create(self, gpu_type: str, storage_gb: int, name: str) -> Machine:
        self._enter("create")
        machine = Machine(MACHINE_ID, handle=f"fake://{name}")
        self.created.append(machine)
        return machine

    def await_ready(self, machine: Machine) -> str:
        self._enter("await_ready")
        return "SSH ready after 0s"

    def push(self, machine: Machine, payload: bytes, dest: str) -> None:
        self._enter("push")
        self.pushed.append((dest, payload))

    def push_stream(
        self, machine: Machine, chunks: Iterable[bytes], dest: str
    ) -> None:
        """The streaming push. Enters `push` and records into `pushed` exactly
        as the buffered one, so fail_at and the recorded calls read the same
        whichever way a caller feeds its bytes."""
        self._enter("push")
        self.pushed.append((dest, b"".join(chunks)))

    def fetch(self, machine: Machine, path: str) -> bytes:
        self._enter("fetch")
        return self._adapter_bytes

    def fetch_stream(self, machine: Machine, path: str) -> Iterator[bytes]:
        """The streaming fetch. One chunk; a double holding test-sized bytes
        has nothing to stream, and inventing chunk boundaries would be
        behaviour no caller asked it to simulate."""
        self._enter("fetch")
        yield self._adapter_bytes

    def stream(self, machine: Machine, script: bytes) -> Iterator[str]:
        self._enter("stream")
        self.script = script
        # `stop_after` and `silent_after` truncate identically and differ only
        # in what happens at the end -- one ends the stream, the other does
        # not, and that difference is the whole point of having both.
        cut = (
            self._stop_after
            if self._stop_after is not None
            else self._silent_after
        )
        emitted = self._lines[:cut] if cut is not None else self._lines
        for n, line in enumerate(emitted, start=1):
            if self._line_delay:
                # Real output arrives spread over minutes. A double that
                # emits everything in one instant cannot show whether the
                # channel is open or merely fast.
                time.sleep(self._line_delay)
            yield line
            if n == self._pause_at_line:
                self._hold()
        if self._silent_after is not None:
            # Silence, not an ending. A generator that returns tells its reader
            # the run is over; a stalled machine tells it nothing, and only one
            # of those two is what the stall detector exists for.
            threading.Event().wait(SILENCE_S)
            return
        if self._stop_after is not None or self._result is None:
            return
        yield "---RESULT---"
        for line in json.dumps(self._result).splitlines():
            yield line

    def destroy(self, machine_id: int) -> None:
        self.calls.append("destroy")
        self.destroy_attempts += 1
        if self.destroy_attempts <= self._destroy_failures:
            raise RuntimeError("provider refused the destroy call")
        self.destroyed = True

    def list_machine_ids(self) -> list[int]:
        self.calls.append("list_machine_ids")
        listed = not self.destroyed or self._stays_listed
        return [MACHINE_ID] if listed else []

    def close(self) -> None:
        self.closed = True

    # -- the test's grip on a running job -----------------------------------

    def wait_until_paused(self, timeout: float = REACH_S) -> bool:
        """Block until the job reaches its pause. False if it never did."""
        return self.paused.wait(timeout)

    # -- internals ----------------------------------------------------------

    def _hold(self) -> None:
        """Stop here until the test lets go, then carry on as if nothing did.

        Bounded rather than indefinite: a test that forgets to release the
        pause should fail on its own assertions, not wedge the suite.
        """
        self.paused.set()
        self.resume.wait(PAUSE_S)

    def _enter(self, stage: str) -> None:
        self.calls.append(stage)
        if stage == self._pause_at_stage:
            self._hold()
        if stage != self._fail_at:
            return
        if self._fail_unexpectedly:
            # Not an OrchestratorError: the path nobody anticipated is the one
            # that must still tear the machine down.
            raise RuntimeError(f"fake provider exploded at {stage}")
        raise OrchestratorError(
            self._fail_code, f"fake provider failed at {stage}"
        )
