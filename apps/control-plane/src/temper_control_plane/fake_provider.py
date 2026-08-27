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

import hashlib
import json
import threading
import time
from collections.abc import Iterable, Iterator, Sequence

from temper_core.errors import OrchestratorError
from temper_core.selection import GpuAvailability

from .limits import RunLimits
from .provider import Machine

# Stage names, not method names: `push` and `fetch` are where a transfer --
# streamed or otherwise -- can fail or pause.
STAGES = (
    "gpu_availability",
    "create",
    "await_ready",
    "push",
    "fetch",
    # The machine's write of its own artifact to the scoped grant (ADR-0009):
    # where the artifact path can fail or pause on the machine's side.
    "artifact_upload",
    # Issue #37: where the machine's own checkpoint writes can fail or pause.
    "checkpoint_upload",
    "stream",
)

# Fixed rather than configurable: tests assert against these, and a knob no
# test turns is a knob that only makes the double harder to read.
MACHINE_ID = 4242
# 8 free devices, matching what spike 6 measured every VM-capable type
# showing -- enough headroom that a test asking for more than one device
# never has to invent a second row.
DEFAULT_AVAILABILITY = (GpuAvailability("L4", 41.31, 8),)
DEFAULT_CURRENCY = "INR"

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
        # Issue #37: the checkpoints the machine writes and reports. Each
        # entry is {"step", "loss"?, "held_out_loss"?, "bytes", "sha256"?} --
        # the bytes the machine PUTs to its slot, and an optional checksum to
        # report (default: the true one, so verification passes). `bytes=None`
        # writes nothing, which is how a test simulates an upload that never
        # landed. An entry whose reported sha256 differs from its bytes is a
        # corrupt upload.
        checkpoints: Sequence[dict] = (),
        pause_at_stage: str | None = None,
        pause_at_line: int | None = None,
        availability: Sequence[GpuAvailability] = DEFAULT_AVAILABILITY,
        currency: str = DEFAULT_CURRENCY,
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
        self._checkpoints = list(checkpoints)
        self._pause_at_stage = pause_at_stage
        self._pause_at_line = pause_at_line
        self._availability = availability
        self._currency = currency

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
        # What `create` was actually asked for -- (gpu_type, num_gpus,
        # storage_gb, name), one per call. `created` alone shows what came
        # back; this is how a test tells disk was passed as a computed
        # parameter rather than a constant (issue #64).
        self.create_calls: list[tuple[str, int, int, str]] = []
        self.destroy_attempts = 0
        self.destroyed = False
        self.pushed: list[tuple[str, bytes]] = []
        self.script: bytes | None = None
        self.closed = False

    # -- protocol -----------------------------------------------------------

    def gpu_availability(self) -> Sequence[GpuAvailability]:
        self._enter("gpu_availability")
        return self._availability

    def currency(self) -> str:
        return self._currency

    def create(
        self, gpu_type: str, num_gpus: int, storage_gb: int, name: str
    ) -> Machine:
        self._enter("create")
        self.create_calls.append((gpu_type, num_gpus, storage_gb, name))
        machine = Machine(MACHINE_ID, handle=f"fake://{name}")
        self.created.append(machine)
        return machine

    def await_ready(self, machine: Machine) -> str:
        self._enter("await_ready")
        return "SSH ready after 0s"

    def push_stream(
        self, machine: Machine, chunks: Iterable[bytes], dest: str
    ) -> None:
        """The streaming push. Enters `push` and records into `pushed` the
        chunks joined whole, so fail_at and the recorded calls read the same
        as any transfer and a caller's test asserts payload, not plumbing."""
        self._enter("push")
        self.pushed.append((dest, b"".join(chunks)))

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
        # ADR-0009: the machine writes its own artifact to the scoped grant
        # before it reports the result, so what the control plane verifies
        # exists by the time the result names it.
        self._write_artifact(_job_spec_from_script(script))
        # Issue #37: the machine writes its checkpoints to their slots before
        # reporting, and reports each one's step, loss and checksum -- the
        # same shape the trainer's background uploader produces.
        self._write_checkpoints(_job_spec_from_script(script))
        yield "---RESULT---"
        for line in json.dumps(self._result).splitlines():
            yield line

    def _write_checkpoints(self, spec) -> None:
        """The machine's half of issue #37, simulated: put each reported
        checkpoint to its slot, then report the set in result.json.

        Mirrors `_write_artifact`: on the filesystem backend the machine
        presents the signed token back to the process that minted it
        (`redeem`); the object-store backend's half is proven by the storage
        tier separately. The reported checksum is the true one unless the
        entry overrides it, so a test can make a corrupt upload on purpose.
        """
        grants = ((spec or {}).get("checkpoint_grants")) or []
        records = []
        for i in range(len(self._checkpoints)):
            if not grants:
                break
            ckpt = self._checkpoints[i]
            # The ring: the i-th checkpoint goes to slot i mod N, overwriting
            # whatever the slot held -- the same shape the trainer's uploader
            # uses, so retention is exercised rather than sidestepped.
            slot = i % len(grants)
            record = {"step": ckpt["step"], "slot": slot}
            for loss_key in ("loss", "held_out_loss"):
                if ckpt.get(loss_key) is not None:
                    record[loss_key] = ckpt[loss_key]
            if ckpt.get("ok") is False:
                # The machine tried and failed, and reports the failure rather
                # than a checksum -- the trainer records a failed upload this
                # way, and the control plane records it as not complete.
                record.update(
                    {"ok": False, "error": "simulated upload failure"}
                )
                records.append(record)
                continue
            payload = ckpt.get("bytes")
            if payload is None:
                # The machine PUT to its slot and got a success it believed,
                # but nothing is at the key -- the report still names a
                # checksum so the control plane has something to verify and
                # fail on (the same shape as a vanished artifact object).
                record.update(
                    {
                        "sha256": ckpt.get("sha256")
                        or hashlib.sha256(b"").hexdigest(),
                        "ok": True,
                    }
                )
                records.append(record)
                continue
            from . import storage
            from .storage import WriteGrant

            grant = WriteGrant(
                url=grants[slot]["url"],
                key=grants[slot]["key"],
                expires_at=grants[slot]["expires_at"],
            )
            if isinstance(storage.STORE, storage.FilesystemStorage):
                storage.STORE.redeem(grant, payload)
            else:
                raise AssertionError(
                    "the fake machine redeems filesystem grants only; tests "
                    "run against the filesystem backend"
                )
            record.update(
                {
                    "sha256": ckpt.get("sha256")
                    or hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "ok": True,
                }
            )
            records.append(record)
        if records and self._result is not None:
            self._result = {**self._result, "checkpoints": records}
        self._enter("checkpoint_upload")

    def _write_artifact(self, spec) -> None:
        """The machine's half of ADR-0009, simulated: put the artifact to the
        scoped grant before reporting the result.

        On the filesystem backend -- the only one a test can use -- the machine
        presents the signed token back to the process that minted it (`redeem`),
        the honest local analogue of a machine PUTting to a pre-signed URL. On
        the object-store backend the machine would PUT over HTTP, which the
        storage tier proves separately; this double has no way to be that HTTP
        client, so it refuses loudly rather than pretending.

        The write lands **before** the stage's pause, so a cancellation that
        arrives while the artifact is being handled is answered the way a real
        one would be -- what already landed is discarded -- rather than racing
        the delete against a write that has not happened yet.
        """
        upload = (spec or {}).get("artifact_upload")
        if not upload:
            return
        if not (self._result or {}).get("adapter_path"):
            # No artifact was produced; there is nothing to write.
            return
        from . import storage
        from .storage import WriteGrant

        grant = WriteGrant(
            url=upload["url"],
            key=upload["key"],
            expires_at=upload["expires_at"],
        )
        if isinstance(storage.STORE, storage.FilesystemStorage):
            storage.STORE.redeem(grant, self._adapter_bytes)
        else:
            raise AssertionError(
                "the fake machine redeems filesystem grants only; tests run "
                "against the filesystem backend"
            )
        self._enter("artifact_upload")

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


# --- a completed job, canned -------------------------------------------------
# For surfaces that need a whole job to reach `complete` without hardware --
# today the browser journeys, which boot the control plane with
# TEMPER_FAKE_PROVIDER and drive a launch to its adapter. One definition, so
# every surface that watches a finished job watches the same one.

DEMO_ADAPTER_BYTES = b"demo adapter weights"

# The published-image reference tests inject so the pull-by-digest path is
# exercised (issue #44). The checked-in contract starts unpublished -- the
# refusal test pins that state -- so the suites that drive a real `run_job`
# to completion hand it a reference the orchestrator will embed in the
# machine's script. It is a fixture value, not a real image, and the simulated
# machine never executes it.
PUBLISHED_IMAGE_REFERENCE = (
    "ghcr.io/thp728/temper/trainer@sha256:"
    "0000000000000000000000000000000000000000000000000000000000000000"
)

DEMO_LINES = (
    "[00:00:00] building trainer image",
    "[00:00:02] image built in 2s",
    "[00:00:02] running training",
    "{'loss': 0.6931, 'step': 10, 'epoch': 0.5}",
)

DEMO_RESULT = {
    "ok": True,
    "stage": "train",
    "adapter_path": "run/adapter_model.safetensors",
    "adapter_sha256": hashlib.sha256(DEMO_ADAPTER_BYTES).hexdigest(),
    "adapter_config": {"r": 16, "lora_alpha": 32},
    # The measured peak the real trainer records via its nvidia-smi sampler
    # (issue #77); the simulated machine carries it so a journey's finished
    # job has an actual to compare against its prediction.
    "peak_memory_gb": 5.31,
    # The export-time template probe (Spec 009 / issue #59) records its result
    # with the artifact on every export; the simulated machine's canned success
    # carries the same shape so every surface that reads a finished job reads
    # one that includes the probe.
    "template_probe": {
        "ok": True,
        "training_ids": 27,
        "serialised_ids": 27,
        "serialised_template": "{{ messages }}",
        "serialised_kwargs": {"enable_thinking": False},
    },
}

# The reserved hyperparameter through which a journey asks the simulated
# machine to end with a named code -- the gap between "a job that succeeds"
# (above) and Spec 007's failed-job journey, until #24 grows fault injection
# into product surface. It is a platform-internal key, not a trainer field, so
# it is carried in `temper_core.surface.PLATFORM_INTERNAL_KEYS` (#33), which
# is what lets creation's hyperparameter validation pass it through, but it is
# honoured only here: the real trainer receives it and ignores it, the
# simulated machine reads it and ends early on the ordinary failure path.
SIMULATED_FAILURE_KEY = "simulated_failure_code"

# Where `_remote_script` writes the jobspec into every script it ships, and
# how that block ends. Parsing this is the same kind of accepted coupling as
# tests/transport_endpoint.py parsing the wire commands: a machine that could
# not read its own job spec would be simulating the wrong thing.
_JOBSPEC_OPEN = b"<<'JOBSPEC'\n"
_JOBSPEC_CLOSE = b"\nJOBSPEC"


def _job_spec_from_script(script: bytes) -> dict | None:
    """The jobspec embedded in a remote script, or None if there is none."""
    start = script.find(_JOBSPEC_OPEN)
    if start < 0:
        return None
    start += len(_JOBSPEC_OPEN)
    end = script.find(_JOBSPEC_CLOSE, start)
    if end < 0:
        return None
    try:
        parsed = json.loads(script[start:end].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


class SimulatedMachine(FakeProvider):
    """`completed_run`'s machine, plus the one thing a canned success cannot
    do: end early because the spec asked it to.

    The failure travels the ordinary path -- result document naming its code,
    OrchestratorError, coded terminal state -- exactly as a trainer that died
    mid-run would, so nothing downstream can tell it apart by shape."""

    def stream(self, machine, script):
        spec = _job_spec_from_script(script)
        requested = ((spec or {}).get("hyperparameters") or {}).get(
            SIMULATED_FAILURE_KEY
        )
        if isinstance(requested, str) and requested.strip():
            code = requested.strip()
            self._result = {
                "ok": False,
                "stage": "train",
                "error_code": code,
                # This sentence is what the user reads under the stable code
                # on the finished-job page, so it says what happened rather
                # than what was simulated.
                "error": (
                    "The training process ended before completing; no "
                    "artifact was produced."
                ),
            }
            # Output stops where the failure begins: the history keeps what
            # ran, not what never got the chance to.
            self._lines = DEMO_LINES[:1]
        yield from super().stream(machine, script)


def completed_run() -> FakeProvider:
    """The fake configured as a small successful job, end to end."""
    from . import config

    return SimulatedMachine(
        lines=DEMO_LINES,
        result=DEMO_RESULT,
        adapter_bytes=DEMO_ADAPTER_BYTES,
        line_delay=config.FAKE_LINE_DELAY_S,
    )
