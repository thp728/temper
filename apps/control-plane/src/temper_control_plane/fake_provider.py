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
* suffer a fault from the surface (issue #24): when a job spec's
  `simulated_failure_code` is a dict naming one of the six faults, the
  machine makes it happen — exhausting memory, diverging, dying mid-run,
  going silent, leaving an orphan, or refusing a destroy — so a recovery can
  be watched rather than argued about. Off by default: no dict, no fault.

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

from temper_core import faults as fault_surface
from temper_core.errors import OrchestratorError
from temper_core.selection import GpuAvailability

from .limits import RunLimits
from .provider import Machine, normalize_status

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
# The machine the `orphan` fault leaves in the provider's listing with no job
# that owns it (issue #24 / #61). Deliberately a different id from the job's
# own machine, so a test can prove the job's teardown did not touch it.
ORPHAN_MACHINE_ID = 7777
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
        # Issue #34: the provider's listing is eventually consistent. After a
        # destroy the machine can read absent, then reappear as `destroying`,
        # then go absent for good. `list_sequence` lets a test script the
        # exact ids (or Machines) returned per `list_machines` call, and
        # `destroying_for` makes the fake report `destroying` for N calls
        # after a successful destroy before becoming absent — the two
        # mechanisms that let the confirmation rule be exercised without
        # hardware.
        list_sequence: Sequence[Sequence[object]] | None = None,
        destroying_for: int = 0,
        # Issue #46: whether the machine answers the spend ceiling's emergency
        # checkpoint request. A wedged machine cannot be asked to save, and
        # `request_checkpoint` must return None rather than a manifest it does
        # not have; this knob makes that state explicit so the unresponsive
        # case is a decision a test makes, not a race it hopes to catch.
        checkpoint_unresponsive: bool = False,
    ) -> None:
        if fail_at is not None and fail_at not in STAGES:
            raise ValueError(f"unknown stage {fail_at!r}")
        if pause_at_stage is not None and pause_at_stage not in STAGES:
            raise ValueError(f"unknown stage {pause_at_stage!r}")
        self._lines = list(lines)
        self._result = result
        # The constructor's own behaviour, kept for the memory recovery
        # (issue #35): the `oom` fault replaces it once, then the retried
        # machine is restored to it, so the recovered run completes normally.
        self._base_lines = list(lines)
        self._base_result = result
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

        # Issue #24: the fault surface. Defaults here are the surface being
        # off; the dict form of `simulated_failure_code` in a job spec sets
        # them at stream time, so a fake constructed for one job never leaks
        # fault behaviour into the next.
        self._orphan_ids: list[int] = []
        self._stop_without_result = False
        # Issue #35: whether the `oom` fault has already fired for this job.
        # The fault exhausts the *first* machine's memory; the memory
        # recovery's retried attempt runs a smaller batch on a fresh machine
        # and fits, so the fault fires once per job rather than once per
        # attempt (firing on every retried machine would turn the
        # demonstration of the recovery into a retry-until-surface loop).
        self._oom_fired = False
        # Observable afterwards: the fault this machine was asked to suffer,
        # or None when the surface was off.
        self.fault_applied: str | None = None

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

        # Issue #34: eventual-consistency scaffolding.
        self._list_sequence: list[list[object]] | None = (
            [list(seq) for seq in list_sequence]
            if list_sequence is not None
            else None
        )
        self._destroying_for = int(destroying_for)
        self._list_calls = 0
        self._destroyed_at_call: int | None = None
        self._checkpoint_unresponsive = bool(checkpoint_unresponsive)

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
        # Issue #74: the machine writes each requested delivery format to its
        # own scoped grant before reporting, the same way it writes the
        # artifact.
        self._write_delivery(_job_spec_from_script(script))
        # Issue #37: the machine writes its checkpoints to their slots before
        # reporting, and reports each one's step, loss and checksum -- the
        # same shape the trainer's background uploader produces.
        self._write_checkpoints(_job_spec_from_script(script))
        # Issue #24: a `worker_kill` fault dies here -- after the checkpoints
        # that a resumption would need have left the machine, and before any
        # result document exists. The orchestrator sees a stream that ended
        # without a result, which is exactly what a killed worker looks like.
        if self._stop_without_result:
            return
        yield "---RESULT---"
        for line in json.dumps(self._result).splitlines():
            yield line

    def _write_checkpoints(self, spec) -> list[dict]:
        """The machine's half of issue #37, simulated: put each reported
        checkpoint to its slot, then report the set in result.json.

        Mirrors `_write_artifact`: on the filesystem backend the machine
        presents the signed token back to the process that minted it
        (`redeem`); the object-store backend's half is proven by the storage
        tier separately. The reported checksum is the true one unless the
        entry overrides it, so a test can make a corrupt upload on purpose.

        Returns the records it wrote, so `request_checkpoint` (issue #46) can
        hand the control plane the manifest of an emergency save.
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
        return records

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
        if not (self._result or {}).get("artifact_path"):
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

    def _write_delivery(self, spec) -> None:
        """The machine's half of issue #74, simulated: write each requested
        delivery format to its own scoped grant before reporting the result.

        Mirrors `_write_artifact` exactly: the spec's `delivery_grants` are
        one scoped write URL per format, the fake redeems each on the
        filesystem backend, and the reported `delivery` records carry each
        format's checksum so the orchestrator verifies what landed. The bytes
        are opaque to the fake, as the adapter's are -- it PUTs them and
        reports their checksum, exactly as the trainer PUTs its produced
        formats.
        """
        grants = (spec or {}).get("delivery_grants") or []
        if not grants:
            return
        from . import storage
        from .storage import WriteGrant

        records: list[dict] = []
        for block in grants:
            format_id = block.get("format")
            payload = _delivery_bytes(format_id)
            grant = WriteGrant(
                url=block["url"],
                key=block["key"],
                expires_at=block["expires_at"],
            )
            if isinstance(storage.STORE, storage.FilesystemStorage):
                storage.STORE.redeem(grant, payload)
            else:
                raise AssertionError(
                    "the fake machine redeems filesystem grants only; tests "
                    "run against the filesystem backend"
                )
            records.append(
                {
                    "format": format_id,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "upload": {"ok": True, "bytes": len(payload)},
                }
            )
            self._enter(f"delivery_upload:{format_id}")
        if records and self._result is not None:
            self._result = {**self._result, "delivery": records}

    def destroy(self, machine_id: int) -> None:
        self.calls.append("destroy")
        self.destroy_attempts += 1
        if self.destroy_attempts <= self._destroy_failures:
            raise RuntimeError("provider refused the destroy call")
        self.destroyed = True
        if self._destroyed_at_call is None:
            self._destroyed_at_call = self._list_calls

    def request_checkpoint(
        self, machine: Machine, job_id: str
    ) -> list[dict] | None:
        """The spend ceiling's emergency checkpoint (issue #46), simulated.

        A responsive machine saves the checkpoints it has produced -- writing
        them to their slots exactly as it would at a normal end, so the
        control plane's verification streams real bytes back -- and reports
        the manifest. A silent machine, or one a test explicitly made
        unresponsive, reports None: the ceiling's checkpoint is best-effort by
        construction, and a wedged machine cannot be asked to save.
        """
        self.calls.append("request_checkpoint")
        if self._silent_after is not None or self._checkpoint_unresponsive:
            return None
        if self.script is None:
            return None
        spec = _job_spec_from_script(self.script)
        records = self._write_checkpoints(spec)
        return records if records else None

    def _list_machines_inner(self) -> list[Machine]:
        """The actual machines the provider bills, with lifecycle status.

        A destroyed machine can still be reported as `destroying` for a
        configurable number of listings before becoming absent — the
        eventual-consistency window the confirmation rule must survive
        (spec 010, spike/teardown.py C17). A scripted `list_sequence`
        overrides everything, so a test can make the provider read absent
        then reappear as destroying.
        """
        # Scripted sequence wins, so a test can exercise the exact
        # absent -> destroying -> absent shape without timing.
        if self._list_sequence is not None:
            idx = (
                self._list_calls - 1
            )  # _list_calls is 1-based after increment
            if 0 <= idx < len(self._list_sequence):
                raw = self._list_sequence[idx]
            elif self._list_sequence:
                raw = self._list_sequence[-1]
            else:
                raw = []
            out: list[Machine] = []
            for item in raw:
                if isinstance(item, Machine):
                    out.append(
                        Machine(
                            item.machine_id,
                            handle=item.handle,
                            status=normalize_status(item.status),
                        )
                    )
                elif isinstance(item, int):
                    out.append(Machine(item, status="running"))
                elif isinstance(item, tuple) and len(item) == 2:
                    mid, st = item
                    out.append(
                        Machine(int(mid), status=normalize_status(str(st)))
                    )
                else:
                    raise ValueError(f"bad list_sequence entry {item!r}")
            # Orphans are still appended unless the sequence already names them
            # — a sequence that wants to hide orphans can just include them
            # explicitly, and one that wants to show them does not need to.
            return out

        ids: list[int] = []
        statuses: dict[int, str] = {}

        # Primary job machine
        listed = not self.destroyed or self._stays_listed
        if listed:
            # After a successful destroy, report destroying for a window
            # before going absent.
            if (
                self.destroyed
                and not self._stays_listed
                and self._destroyed_at_call is not None
                and self._destroying_for > 0
            ):
                since = self._list_calls - self._destroyed_at_call
                if 0 < since <= self._destroying_for:
                    ids.append(MACHINE_ID)
                    statuses[MACHINE_ID] = "destroying"
                elif since > self._destroying_for:
                    pass  # absent for good
                else:
                    ids.append(MACHINE_ID)
                    statuses[MACHINE_ID] = "running"
            else:
                ids.append(MACHINE_ID)
                statuses[MACHINE_ID] = "running"

        # Orphans are always running and never destroying — they are the
        # reconciler's (#61) fixture, not the teardown's.
        for oid in self._orphan_ids:
            if oid not in ids:
                ids.append(oid)
                statuses[oid] = "running"

        return [Machine(mid, status=statuses[mid]) for mid in ids]

    def list_machines(self) -> list[Machine]:
        self.calls.append("list_machines")
        self.calls.append("list_machine_ids")
        self._list_calls += 1
        return self._list_machines_inner()

    def list_machine_ids(self) -> list[int]:
        return [m.machine_id for m in self.list_machines()]

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

# The bytes of a delivery format as the simulated machine uploads them (issue
# #74): opaque, exactly as the adapter's are. The fake PUTs them to the
# format's scoped grant and reports their checksum; a test can override the
# bytes or the reported checksum to make a corrupt upload on purpose.
DEMO_MERGED_BYTES = b"merged model: base + trained change at full precision"
DEMO_QUANTISED_BYTES = b"gguf: a quantised local-inference format"


def _delivery_bytes(format_id: str | None) -> bytes:
    """The canned bytes of a delivery format, keyed by the format id.

    An unknown format has no bytes -- a machine asked to produce something the
    vocabulary does not define cannot fake it, and refusing loudly beats
    uploading garbage the orchestrator would then have to explain.
    """
    if format_id == "merged":
        return DEMO_MERGED_BYTES
    if format_id == "quantised":
        return DEMO_QUANTISED_BYTES
    raise AssertionError(
        f"no simulated bytes for delivery format {format_id!r}; the fake "
        "knows merged and quantised only"
    )


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
    # Issue #49: the machine's pull and download output is the progress data.
    # The simulated machine emits docker's pull shape (layers pulled in
    # parallel, each its own done/total) and huggingface_hub's download bars,
    # and the control plane promotes them into per-phase progress -- so the
    # journeys watch image pull and model download advance with a measured
    # rate, exactly the phases a large real job is dominated by.
    "9b829b73a52f: Pulling fs layer",
    "9b829b73a52f: Downloading [===============> ] 15.19MB/42.42MB",
    "1fe172e4850f: Downloading [======>            ]  8.5MB/25.54MB",
    "9b829b73a52f: Downloading [==================> ] 28.1MB/42.42MB",
    "9b829b73a52f: Extracting [========================> ] 35.2MB/42.42MB",
    "9b829b73a52f: Pull complete",
    "1fe172e4850f: Pull complete",
    "model.safetensors:  10%|█         | 400M/4.00G [00:05<00:45]",
    "model.safetensors:  60%|██████    | 2.4G/4.00G [00:30<00:20]",
    "model.safetensors: 100%|██████████| 4.00G/4.00G [00:40<00:00]",
    # Issue #53: the trainer holds out a portion of the dataset at startup and
    # says so; the simulated machine narrates the same way so the running view
    # shows the split alongside the output that follows it.
    "[00:00:02] held-out split: 1 of 12 rows held out for evaluation "
    "(0 duplicate(s) removed)",
    "[00:00:02] running training",
    "{'loss': 0.6931, 'step': 10, 'epoch': 0.5}",
    # The held-out measurement, as the trainer's eval pass prints it: the
    # control plane promotes eval_loss into a metric event carrying
    # held_out_loss, so the loss chart has a second series to draw.
    "{'eval_loss': 0.52, 'eval_runtime': 2.0, 'epoch': 0.5}",
)

DEMO_RESULT = {
    "ok": True,
    "stage": "train",
    "artifact_path": "run/adapter_model.safetensors",
    "artifact_sha256": hashlib.sha256(DEMO_ADAPTER_BYTES).hexdigest(),
    "adapter_config": {"r": 16, "lora_alpha": 32},
    # Issue #53: the recorded split, shaped exactly as the trainer records it
    # for a 12-row dataset under the default 5% hold-out -- the simulated
    # machine carries the same record so every surface that reads a finished
    # job reads one that says how it was split.
    "held_out_split": {
        "rows_in": 12,
        "rows_removed_duplicates": 0,
        "train_rows": 11,
        "held_out_rows": 1,
        "fraction": 0.05,
        "seed": 42,
    },
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
    # The side-by-side comparison (issue #69): the same held-out prompt run
    # through the base model and the chosen checkpoint (step 20 -- the fake's
    # checkpoints make the last one NOT the best), with the fixed decoding
    # settings recorded so a reader can tell whether two outputs are
    # comparable. The simulated machine carries the same shape the real
    # trainer records, so a journey's finished job shows the comparison. The
    # text is canned -- this is the renderer's fixture, not the generation
    # proof; real generation runs in the trainer and is exercised by the
    # hardware-marked trainer test.
    "comparison": {
        "ok": True,
        "decoding": {
            "temperature": 0.7,
            "max_new_tokens": 128,
            "do_sample": True,
        },
        "selection": {
            "step": 20,
            "basis": "best_held_out_loss",
            "held_out_loss": 0.39,
            "reason": (
                "Step 20 has the lowest held-out loss (0.39) of 3 retained "
                "checkpoint(s)."
            ),
        },
        "rows": [
            {
                "prompt": [{"role": "user", "content": "q0"}],
                "base": "The base model's answer to the held-out question.",
                "tuned": "The tuned model's answer to the held-out question.",
            }
        ],
    },
}

# The hyperparameter through which a journey or an operator asks the simulated
# machine to fail (issue #24, grown from ADR-0026). A string value ends the
# machine early with that named code; a dict value is a fault spec naming one
# of the six faults in the surface. It is a platform-internal key, not a
# trainer field, so it is carried in `temper_core.surface.PLATFORM_INTERNAL_KEYS`
# (#33), which is what lets creation's hyperparameter validation pass it
# through, but it is honoured only here: the real trainer receives it and
# ignores it (the trainer-side faults reach the real trainer through its
# environment instead), and the simulated machine reads it and makes the
# failure happen.
SIMULATED_FAILURE_KEY = fault_surface.HYPERPARAMETER_KEY

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


def _fault_int(spec: dict, key: str, default: int) -> int:
    """A fault parameter as a whole number, refusing a value that is not one.

    A fault spec a caller believes is in effect but is not is worse than a
    refusal, so a malformed parameter refuses loudly rather than silently
    becoming the default.
    """
    raw = spec.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise OrchestratorError(
            "fault_invalid",
            f"fault parameter '{key}' must be a whole number, got {raw!r}",
        ) from None


class SimulatedMachine(FakeProvider):
    """`completed_run`'s machine, plus the two things a canned success cannot
    do: end early because the spec asked it to, and suffer a fault from the
    surface (issue #24).

    The failure travels the ordinary path -- result document naming its code,
    OrchestratorError, coded terminal state -- exactly as a trainer that died
    mid-run would, so nothing downstream can tell it apart by shape. A run
    broken on purpose is still named as such (the `simulated_` codes, the
    fault's own line in the history, the orchestrator's launch event), which
    is what keeps a deliberately broken run from being mistaken for a real
    one.
    """

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
        elif isinstance(requested, dict):
            self._configure_fault(requested)
            # Issue #35: the oom fault exhausts the first machine's memory
            # and the memory recovery's retried attempt -- a smaller batch on
            # a fresh machine -- fits, so the fault fires once per job. The
            # first machine's run fails with the fault's own `simulated_oom`
            # code (a deliberately broken run is never mistaken for a real
            # one); the recovery that follows is the thing the fault exists
            # to prove.
            if requested.get("name") == "oom" and not self._oom_fired:
                self._oom_fired = True
                self._lines = [self._narration("oom"), *self._base_lines]
                self._result = {
                    "ok": False,
                    "stage": "train",
                    "error_code": fault_surface.code_for("oom"),
                    "error": (
                        "The training process ran out of device memory "
                        "(simulated fault); no artifact was produced."
                    ),
                }
            elif requested.get("name") == "oom":
                # The recovery's retried machine -- a smaller batch on a
                # fresh machine -- fits, so it behaves like the normal
                # completed machine it was constructed as.
                self._lines = list(self._base_lines)
                self._result = self._base_result
        yield from super().stream(machine, script)

    # -- the fault surface (issue #24) ---------------------------------------

    def _configure_fault(self, spec: dict) -> None:
        """Apply a fault spec -- the dict form of `simulated_failure_code` --
        to this machine before the run streams.

        The vocabulary, the codes and the parameters each fault accepts live
        in `packages/contracts/fault-surface.json` (read through
        `temper_core.faults`), so the fake and the trainer cannot drift about
        what a fault is called or what code a broken run carries. A spec the
        validator rejects -- an unknown fault, an unknown parameter, a
        malformed value -- is refused loudly rather than run under a fault
        nobody can explain.
        """
        problem = fault_surface.spec_error(spec)
        if problem is not None:
            raise OrchestratorError("fault_invalid", problem)
        name = spec["name"]
        self.fault_applied = name
        if name == "machine_silent":
            # The narration line precedes the silence, so the history says
            # why the silence came before the stall detector has to.
            self._lines = [self._narration(name), *self._lines]
            self._silent_after = _fault_int(spec, "after_line", 3) + 1
        elif name == "destroy_refused":
            self._lines = [self._narration(name), *self._lines]
            self._destroy_failures = _fault_int(spec, "times", 99)
            self._stays_listed = True
        elif name == "orphan":
            self._lines = [self._narration(name), *self._lines]
            self._orphan_ids = [
                _fault_int(spec, "machine_id", ORPHAN_MACHINE_ID)
            ]
        elif name == "oom":
            # The failure itself is applied in `stream`, once per job (issue
            # #35): the first machine exhausts memory and the recovery's
            # retried machine -- a smaller batch -- fits. `fault_applied` is
            # set above; there is nothing more to configure here.
            pass
        elif name == "divergence":
            self._lines = [
                self._narration(name),
                *DEMO_LINES,
                # The loss becomes meaningless, and says so in the stream.
                "{'loss': nan, 'step': 20, 'epoch': 1.0}",
            ]
            # The loss is the fault; what happens next is the recovery's
            # job. A run whose loss has gone meaningless still runs out its
            # duration until the divergence recovery (#36) stops it, so this
            # run completes with a worthless result -- the honest shape of a
            # diverged run, exactly what the real trainer produces when its
            # learning rate is sabotaged.
        elif name == "worker_kill":
            self._lines = [self._narration(name), *DEMO_LINES]
            # The worker dies before any result document exists; `{}` (not
            # None) keeps the stream on the path that writes whatever
            # checkpoints were already produced, then stops it before the
            # result marker -- exactly what a killed worker looks like: no
            # artifact, no result, only what had already left the machine.
            self._result = {}
            self._stop_without_result = True
        else:  # pragma: no cover - guarded by spec_error above
            raise AssertionError(f"unhandled fault {name!r}")

    def _narration(self, name: str) -> str:
        return (
            f"[simulated fault] {name}: {fault_surface.describe(name)}. "
            "This run is deliberately broken."
        )


def completed_run() -> FakeProvider:
    """The fake configured as a small successful job, end to end.

    The reported checkpoints deliberately make the last one NOT the best:
    held-out loss falls to step 20 and rises again by step 30, the overfitting
    shape issue #62 exists to catch, so the journeys exercise the recorded
    best-checkpoint choice (and a non-chosen checkpoint download) on a real
    finished record.
    """
    from . import config

    return SimulatedMachine(
        lines=DEMO_LINES,
        result=DEMO_RESULT,
        adapter_bytes=DEMO_ADAPTER_BYTES,
        line_delay=config.FAKE_LINE_DELAY_S,
        checkpoints=[
            {
                "step": 10,
                "loss": 0.41,
                "held_out_loss": 0.44,
                "bytes": b"ckpt-10",
            },
            {
                "step": 20,
                "loss": 0.35,
                "held_out_loss": 0.39,
                "bytes": b"ckpt-20",
            },
            {
                "step": 30,
                "loss": 0.31,
                "held_out_loss": 0.52,
                "bytes": b"ckpt-30",
            },
        ],
    )


# The bytes of a full fine-tune's artifact as the simulated machine uploads
# them (issue #66): one whole-model archive, larger than any adapter. The
# bytes are opaque to the fake -- it PUTs them to the scoped grant and reports
# their checksum, exactly as the trainer PUTs its model.tar.gz.
DEMO_FULL_MODEL_BYTES = b"full model archive: config + weight shards" * 4


def completed_full_run() -> FakeProvider:
    """The fake configured as a small successful full fine-tuning job, end to
    end (issue #66).

    The predictor only picks full fine-tuning when it is the cheapest
    configuration that fits, so the availability here is an 80 GB H100 -- a
    full fine-tune of the catalog 4B model fits one (predicted ~67 GB), while
    the default single L4 cannot hold it. The canned result is a full model's
    shape: one archive, no adapter config, its own peak.
    """
    from . import config

    return SimulatedMachine(
        lines=DEMO_LINES,
        result={
            "ok": True,
            "stage": "train",
            "artifact_path": "model.tar.gz",
            "artifact_sha256": hashlib.sha256(
                DEMO_FULL_MODEL_BYTES
            ).hexdigest(),
            "artifact_format": "tar.gz",
            "artifact_members": ["config.json", "model.safetensors"],
            "held_out_split": DEMO_RESULT["held_out_split"],
            # Mirrors what `temper_core.memory.predict_peak` expects for a
            # full fine-tune of the catalog 4B model on one H100 (~67 GB) --
            # the measured figure a real full run would record (issue #77).
            "peak_memory_gb": 67.0,
            "template_probe": DEMO_RESULT["template_probe"],
        },
        adapter_bytes=DEMO_FULL_MODEL_BYTES,
        availability=[GpuAvailability("H100", 250.0, 2)],
        line_delay=config.FAKE_LINE_DELAY_S,
    )
