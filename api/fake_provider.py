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
  afterwards.
"""

from __future__ import annotations

import json
import time
from typing import Iterator, Sequence

from .errors import OrchestratorError
from .provider import GpuChoice, Machine

STAGES = ("select_gpu", "create", "await_ready", "push", "fetch", "stream")

# Fixed rather than configurable: tests assert against these, and a knob no
# test turns is a knob that only makes the double harder to read.
MACHINE_ID = 4242
GPU = GpuChoice("L4", 41.31, "INR")


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
        line_delay: float = 0.0,
        destroy_failures: int = 0,
        stays_listed: bool = False,
        adapter_bytes: bytes = b"weights",
    ) -> None:
        if fail_at is not None and fail_at not in STAGES:
            raise ValueError(f"unknown stage {fail_at!r}")
        self._lines = list(lines)
        self._result = result
        self._fail_at = fail_at
        self._fail_code = fail_code
        self._fail_unexpectedly = fail_unexpectedly
        self._stop_after = stop_after
        self._line_delay = line_delay
        self._destroy_failures = destroy_failures
        self._stays_listed = stays_listed
        self._adapter_bytes = adapter_bytes

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

    def fetch(self, machine: Machine, path: str) -> bytes:
        self._enter("fetch")
        return self._adapter_bytes

    def stream(self, machine: Machine, script: bytes) -> Iterator[str]:
        self._enter("stream")
        self.script = script
        emitted = self._lines
        if self._stop_after is not None:
            emitted = emitted[:self._stop_after]
        for line in emitted:
            if self._line_delay:
                # Real output arrives spread over minutes. A double that
                # emits everything in one instant cannot show whether the
                # channel is open or merely fast.
                time.sleep(self._line_delay)
            yield line
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

    # -- internals ----------------------------------------------------------

    def _enter(self, stage: str) -> None:
        self.calls.append(stage)
        if stage != self._fail_at:
            return
        if self._fail_unexpectedly:
            # Not an OrchestratorError: the path nobody anticipated is the one
            # that must still tear the machine down.
            raise RuntimeError(f"fake provider exploded at {stage}")
        raise OrchestratorError(self._fail_code,
                                f"fake provider failed at {stage}")
