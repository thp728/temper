"""The compute provider seam.

Every interaction with the platform that supplies machines goes through one
protocol. Before this existed the orchestrator built its own client and shelled
out directly, which meant the only way to exercise the money-spending path was
to spend money — so it was never exercised at all.

Two properties of the default implementation were learned expensively and must
not regress:

* **Readiness distinguishes *unreachable* from *authentication failed*.** They
  have opposite remedies, and collapsing both into "no answer" cost an evening
  and produced a wrongly-filed platform bug.
* **Teardown is confirmed by listing machines**, never by the destroy call's
  return value. `destroy` returning cleanly is a claim, not evidence.

`stream` yields lines as they arrive rather than returning output at the end.
That shape is deliberate: it holds whether the transport is SSH stdout today or
an outbound HTTPS push from a deployed control plane later, so changing
transport becomes a new implementation of this protocol rather than a change to
the orchestration logic.

Three methods beyond the six the spec enumerates, each with a reason:

* `await_ready` is separate from `create` so the caller can record the machine's
  id *before* anything can fail. Folding readiness into `create` means a machine
  that exists but never answers is never recorded, and a machine nobody recorded
  is a machine nobody destroys.
* `fetch_stream` is the counterpart to `push_stream`. The artifact no longer
  travels through the control plane -- the machine writes it directly to a
  scoped grant (ADR-0009) -- but fetch remains a transport primitive, driven
  against a real endpoint by the transport tier (ADR-0027) and available to
  any future path that needs bytes off a machine.
* `close` releases whatever the implementation holds open. Plumbing, not domain.

Spec 006 in full: transfers take and return streams of chunks. The expand half
(#30) added `push_stream`/`fetch_stream` beside a buffered pair; every caller
has since moved, so the contract half (#41) deleted the pair -- one way to
move bytes rather than two, and no transfer that ever needs the whole payload
in memory.
"""

from __future__ import annotations

import io
import shlex
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from temper_core.errors import OrchestratorError
from temper_core.selection import GpuAvailability

SSH_READY_TIMEOUT_S = 300
PUSH_TIMEOUT_S = 180
FETCH_TIMEOUT_S = 600

# How much one streamed fetch hands the caller per step. Bounded, not tuned:
# the tests assert memory stays flat as payloads grow, not that a particular
# chunk size was used.
FETCH_CHUNK_BYTES = 256 * 1024

# The three wire commands the transport speaks. Defined once, read twice:
# this module writes them and the transport tier's emulated endpoint
# (tests/transport_endpoint.py) parses them, so the two sides cannot drift.
# The shapes are the real machine's: a push lands via `cat`, a fetch reads
# with `sudo` because artifacts are root-owned on the machine.
STREAM_COMMAND = "bash -s"
PUSH_PREFIX = "mkdir -p "
PUSH_SEPARATOR = " && cat > "
FETCH_PREFIX = "sudo cat "


def push_command(dest: str) -> str:
    """The wire command that writes streamed bytes to `dest`."""
    parent = dest.rsplit("/", 1)[0] or "/"
    return f"{PUSH_PREFIX}{parent}{PUSH_SEPARATOR}{dest}"


def fetch_command(path: str) -> str:
    """The wire command that streams `path` back."""
    return f"{FETCH_PREFIX}{path}"


# The backstop for a stream that never ends, and deliberately *above* the
# orchestrator's duration ceiling rather than below it. It used to be 90
# minutes, which was fine while it was the only bound on a run and wrong the
# moment there was a real one: a 24-hour ceiling that a transport detail kills
# at 90 minutes is not a 24-hour ceiling, and the job would have failed with an
# uncoded `TimeoutExpired` instead of the named outcome the user is owed. This
# now only fires if the guard somehow does not, and the margin is what keeps
# the coded outcome the one that wins the race.
STREAM_BACKSTOP_MARGIN_S = 600


def _stream_timeout() -> float:
    from . import config

    return config.MAX_JOB_DURATION_S + STREAM_BACKSTOP_MARGIN_S


@dataclass(frozen=True)
class Machine:
    """A GPU host provisioned for one job.

    `handle` is how the implementation reaches the machine and is opaque to
    everything outside it — the SSH implementation stores a command, a push
    transport would store a URL. Nothing in the orchestrator reads it.
    """

    machine_id: int
    handle: str = ""


class Provider(Protocol):
    """Everything the orchestrator is allowed to know about compute."""

    def gpu_availability(self) -> Sequence[GpuAvailability]:
        """What the provider has free right now, filtered to machine-capable
        types -- `workload_type == "vm"`, since container-only capacity is a
        separate pool a VM job can never draw from."""

    def currency(self) -> str:
        """The account's billing currency, read live rather than assumed:
        this account bills in INR, and a hard-coded `$` would misreport
        every price by roughly 85x."""

    def create(
        self, gpu_type: str, num_gpus: int, storage_gb: int, name: str
    ) -> Machine: ...

    def await_ready(self, machine: Machine) -> str:
        """Block until the machine is usable. Returns a line worth logging."""

    def push_stream(
        self, machine: Machine, chunks: Iterable[bytes], dest: str
    ) -> None:
        """Stream `chunks` to `dest`, holding at most one chunk whole.

        Refuses unless every chunk reached the wire: an early-ending
        connection is a failure even when the remote exits zero.
        """

    def fetch_stream(self, machine: Machine, path: str) -> Iterator[bytes]:
        """Yield `path`'s bytes in bounded chunks.

        Yielding nothing means the file could not be read. A read that dies
        part-way delivers what crossed before it died -- truncation is caught
        downstream, against the checksum the machine computed.
        """

    def stream(self, machine: Machine, script: bytes) -> Iterator[str]:
        """Run a script and yield its output lines as they are produced."""

    def destroy(self, machine_id: int) -> None: ...

    def list_machine_ids(self) -> list[int]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# the default implementation: JarvisLabs machines, reached over SSH
# ---------------------------------------------------------------------------


def _ssh(handle: str) -> list[str]:
    """Turn a machine's opaque handle into an `ssh` argument vector.

    Split with `shlex`, not `str.split`: the transport tier drives this
    against an endpoint whose handle carries a quoted identity path, and a
    path with spaces in it must arrive at `ssh` as one argument. A provider
    handle is whatever `create` returned; quoting is its grammar.
    """
    base = handle.strip()
    if base.startswith("ssh "):
        base = base[4:]
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "BatchMode=yes",
        *shlex.split(base),
    ]


class JarvisLabsProvider:
    """The real thing. Constructing one requires credentials."""

    def __init__(self) -> None:
        from . import config

        if not config.provider_credentials_present():
            raise OrchestratorError(
                "provider_unauthenticated",
                "JL_API_KEY is not set and no jl config file exists, so no "
                "machine could be created. Put the key in spike/.env or export "
                "JL_API_KEY.",
            )
        self._client = self._connect()

    @staticmethod
    def _connect():
        """The one place a real client is built.

        A test that gets here has escaped its fake and is about to talk to a
        billing account, so the suite replaces this method with a loud failure.
        """
        from jarvislabs import Client

        return Client()

    # -- provisioning -------------------------------------------------------

    def gpu_availability(self) -> Sequence[GpuAvailability]:
        # Rows are kept one-per-node, never merged by gpu_type: two devices
        # of the same type on different nodes cannot be attached to one
        # machine (spike 6), so collapsing them would let the selection
        # search believe capacity exists that no single node actually offers.
        return [
            GpuAvailability(r.gpu_type, r.price_per_hour, r.num_free_devices)
            for r in self._client.account.gpu_availability()
            if r.workload_type == "vm" and r.num_free_devices > 0
        ]

    def currency(self) -> str:
        return self._client.account.currency()

    def create(
        self, gpu_type: str, num_gpus: int, storage_gb: int, name: str
    ) -> Machine:
        # The SDK calls it an instance; on this side of the seam it is a
        # machine, and that translation is the seam's job.
        created = self._client.instances.create(
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            template="vm",
            storage=storage_gb,
            name=name,
        )
        return Machine(created.machine_id, created.ssh_command or "")

    def await_ready(self, machine: Machine) -> str:
        """Poll until sshd answers.

        `Running` from the provider is a claim about the machine, not about
        reachability: measured, SSH refuses for ~40s after the status flips.
        Authentication failures are reported separately from unreachability
        because the fixes are unrelated — one means wait or reprovision, the
        other means the agent is not holding the key.
        """
        t0 = time.time()
        last_auth_error = None
        while time.time() - t0 < SSH_READY_TIMEOUT_S:
            try:
                r = subprocess.run(
                    _ssh(machine.handle) + ["true"],
                    capture_output=True,
                    text=True,
                    timeout=25,
                )
                if r.returncode == 0:
                    return f"SSH ready after {time.time() - t0:.0f}s"
                err = (r.stderr or "").lower()
                if "permission denied" in err or "publickey" in err:
                    last_auth_error = r.stderr.strip()
            except subprocess.TimeoutExpired:
                pass
            time.sleep(5)

        if last_auth_error:
            raise OrchestratorError(
                "ssh_auth_failed",
                "The machine was reachable but rejected the SSH key. The key "
                "is registered, so this is almost always a local agent "
                "problem: check `ssh-add -l` lists the JarvisLabs key. "
                f"Server said: {last_auth_error}",
            )
        raise OrchestratorError(
            "ssh_unreachable",
            f"Machine never accepted SSH within {SSH_READY_TIMEOUT_S}s. It "
            f"reached Running but is not usable; destroying and giving up.",
        )

    # -- moving bytes -------------------------------------------------------

    def push_stream(
        self, machine: Machine, chunks: Iterable[bytes], dest: str
    ) -> None:
        """Stream `chunks` to `dest` over the wire `mkdir + cat` speaks.

        stderr goes to a scratch file rather than a pipe: a pipe nobody
        drains while this loop writes is a remote that can fill it and block,
        which would deadlock the very transfer this method exists to make
        unbounded.

        The push is refused unless every chunk was handed to the wire: the
        remote's `cat` cannot tell EOF-because-done from EOF-because-the
        -connection-died, so its exit status alone would let a truncated
        transfer be recorded as delivered.

        The ceiling is PUSH_TIMEOUT_S of wall clock. At the payload sizes
        pushed today, a total wall-clock bound is the wrong shape -- an
        idle-aware bound is what a growing catalog will need -- and it is
        recorded as known rather than silently inherited.
        """
        with tempfile.TemporaryFile() as errors:
            proc = subprocess.Popen(
                _ssh(machine.handle) + [push_command(dest)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=errors,
            )
            # stdin=PIPE above guarantees the pipe; Popen's type is Optional
            # for callers who passed something else.
            stdin = proc.stdin
            assert stdin is not None
            timed_out = threading.Event()

            def kill() -> None:
                timed_out.set()
                proc.kill()

            watchdog = threading.Timer(PUSH_TIMEOUT_S, kill)
            watchdog.start()
            reaped = False
            drained = False
            try:
                try:
                    for chunk in chunks:
                        stdin.write(chunk)
                    # Only a close that did not raise counts: a failed flush
                    # means the tail of the payload never left this process.
                    stdin.close()
                    drained = True
                except OSError:
                    # The pipe died under us -- most often the watchdog
                    # killing the client, sometimes the remote exiting
                    # early. Which of the two is decided below, after the
                    # wait; raising here would report a symptom instead of
                    # the cause.
                    try:
                        stdin.close()
                    except OSError:
                        pass
                code = proc.wait()
                reaped = True
            finally:
                watchdog.cancel()
                if not reaped and proc.poll() is None:
                    # The chunk source failed mid-push. There is no
                    # legitimate transfer left to finish, so nothing keeps
                    # the child alive.
                    proc.kill()
                    proc.wait()
            if timed_out.is_set():
                raise subprocess.TimeoutExpired(
                    cmd=push_command(dest), timeout=PUSH_TIMEOUT_S
                )
            if not drained:
                raise OrchestratorError(
                    "source_upload_failed",
                    "The connection closed before every byte was written; "
                    f"{dest} must not be treated as delivered.",
                )
            if code != 0:
                errors.seek(0)
                raise OrchestratorError(
                    "source_upload_failed",
                    errors.read().decode("utf-8", "replace")[:300]
                    or f"ssh exited {code}",
                )

    def fetch_stream(self, machine: Machine, path: str) -> Iterator[bytes]:
        """Yield `path`'s bytes in bounded chunks as they cross the wire.

        Yielding nothing means the file could not be read. A watchdog bounds
        a wedged read exactly as a blocking timeout would, and cleanup also
        runs when the consumer abandons the iterator, which is how
        cancellation will stop a download.
        """
        proc = subprocess.Popen(
            _ssh(machine.handle) + [fetch_command(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # stdout=PIPE above guarantees the pipe; Popen's type is Optional
        # for callers who passed something else.
        stdout = proc.stdout
        assert stdout is not None
        timed_out = threading.Event()

        def kill() -> None:
            timed_out.set()
            proc.kill()

        watchdog = threading.Timer(FETCH_TIMEOUT_S, kill)
        watchdog.start()
        try:
            while chunk := stdout.read(FETCH_CHUNK_BYTES):
                yield chunk
        finally:
            # Also reached when the consumer abandons the iterator, which is
            # how cancellation will stop a download; and cleanup must not be
            # skipped, because a surviving child holds the connection open.
            watchdog.cancel()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        # Reached only on normal completion; an abandoned iterator leaves via
        # the finally above before this line, so abandonment never raises a
        # timeout that nobody was reading.
        if timed_out.is_set():
            raise subprocess.TimeoutExpired(
                cmd=fetch_command(path), timeout=FETCH_TIMEOUT_S
            )

    def stream(self, machine: Machine, script: bytes) -> Iterator[str]:
        """Run `script` under bash and yield its output as it is produced.

        stderr is folded into stdout because the remote script's progress
        narration goes to stderr and its result to stdout, and a single ordered
        channel is what a push transport would also deliver. The script is fed
        from a thread: writing standard input inside the read loop deadlocks
        whenever the payload exceeds the pipe buffer, and nothing guarantees
        the script stays small forever.
        """
        proc = subprocess.Popen(
            _ssh(machine.handle) + [STREAM_COMMAND],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        # Both pipes are guaranteed by the PIPE arguments above; Popen's
        # type is Optional for callers who passed something else.
        # text=True additionally makes stdin a TextIOWrapper, which
        # typeshed's coarse IO[Any] cannot say -- and the feed thread below
        # writes bytes through its .buffer.
        stdin = proc.stdin
        assert isinstance(stdin, io.TextIOWrapper)
        stdout = proc.stdout
        assert stdout is not None

        def feed() -> None:
            # Written as bytes, through the text wrapper rather than to it.
            # `text=True` puts a TextIOWrapper on stdin, and on Windows that
            # wrapper translates every "\n" handed to it into "\r\n" -- so a
            # script written as text reaches the remote `bash -s` with a
            # carriage return on every line, and bash answers
            # `$'\r': command not found` before running a thing. The script is
            # already bytes; decoding it only to have it re-encoded with
            # different newlines was the whole defect. stdout stays in text
            # mode, which is the direction the translation is wanted in.
            try:
                stdin.buffer.write(script)
                stdin.buffer.flush()
                stdin.close()
            except Exception:  # noqa: S110
                # The process died. The read loop is what reports that,
                # with the exit status and the output; raising here would
                # replace a diagnosis with a stack trace from a daemon
                # thread nobody is watching.
                pass

        threading.Thread(target=feed, daemon=True, name="stream-stdin").start()

        timed_out = threading.Event()

        def kill() -> None:
            timed_out.set()
            proc.kill()

        timeout = _stream_timeout()
        watchdog = threading.Timer(timeout, kill)
        watchdog.start()
        try:
            for raw in stdout:
                yield raw.rstrip("\r\n")
        finally:
            # Also reached when the consumer abandons the iterator, which is
            # how cancellation will stop a run. Leaving the process alive would
            # leave the training container running on a machine we are about to
            # destroy anyway, but noisily.
            watchdog.cancel()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        if timed_out.is_set():
            raise subprocess.TimeoutExpired(cmd="bash -s", timeout=timeout)

    # -- teardown -----------------------------------------------------------

    def destroy(self, machine_id: int) -> None:
        self._client.instances.destroy(machine_id)

    def list_machine_ids(self) -> list[int]:
        return [i.machine_id for i in self._client.instances.list()]

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


def new_provider() -> Provider:
    """The default provider. Callers that do not care get this one.

    TEMPER_FAKE_PROVIDER (config.FAKE_PROVIDER) substitutes the in-package
    fake, for the browser journeys and a hardware-free demo: a launch driven
    from a test must never be able to reach the billing account, and /health
    advertises which implementation is in force so the journeys can refuse to
    run against a process that could. Imported here rather than at module
    level because fake_provider imports this module's dataclasses."""
    from . import config

    if config.FAKE_PROVIDER:
        from .fake_provider import completed_run

        return completed_run()

    return JarvisLabsProvider()
