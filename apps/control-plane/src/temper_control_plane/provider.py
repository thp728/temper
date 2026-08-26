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
* `fetch` is the counterpart to `push`. Without it the adapter download would be
  the one provider interaction still shelling out behind the seam's back.
* `close` releases whatever the implementation holds open. Plumbing, not domain.
"""

from __future__ import annotations

import shlex
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from temper_core.errors import OrchestratorError

SSH_READY_TIMEOUT_S = 300
PUSH_TIMEOUT_S = 180
FETCH_TIMEOUT_S = 600

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
    """The wire command that writes `payload` bytes to `dest`."""
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
class GpuChoice:
    """A GPU type the provider has free, and what it will cost per hour.

    The currency is read from the account rather than assumed: this account
    bills in INR, and a hard-coded `$` would misreport every price by ~85x.
    """

    gpu_type: str
    price_per_hour: float
    currency: str


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

    def select_gpu(self, preference: Sequence[str]) -> GpuChoice: ...

    def create(self, gpu_type: str, storage_gb: int, name: str) -> Machine: ...

    def await_ready(self, machine: Machine) -> str:
        """Block until the machine is usable. Returns a line worth logging."""

    def push(self, machine: Machine, payload: bytes, dest: str) -> None: ...

    def fetch(self, machine: Machine, path: str) -> bytes:
        """Read a file off the machine. Empty bytes mean it could not be read."""

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

    def select_gpu(self, preference: Sequence[str]) -> GpuChoice:
        avail = {
            r.gpu_type: r
            for r in self._client.account.gpu_availability()
            if r.workload_type == "vm" and r.num_free_devices > 0
        }
        gpu = next((g for g in preference if g in avail), None)
        if not gpu:
            raise OrchestratorError(
                "provider_capacity_unavailable",
                "No VM-capable GPU free. Note that availability is "
                "per-workload-type: some GPUs exist only for containers.",
            )
        return GpuChoice(
            gpu, avail[gpu].price_per_hour, self._client.account.currency()
        )

    def create(self, gpu_type: str, storage_gb: int, name: str) -> Machine:
        # The SDK calls it an instance; on this side of the seam it is a
        # machine, and that translation is the seam's job.
        created = self._client.instances.create(
            gpu_type=gpu_type,
            num_gpus=1,
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

    def push(self, machine: Machine, payload: bytes, dest: str) -> None:
        r = subprocess.run(
            _ssh(machine.handle) + [push_command(dest)],
            input=payload,
            capture_output=True,
            timeout=PUSH_TIMEOUT_S,
        )
        if r.returncode != 0:
            raise OrchestratorError(
                "source_upload_failed",
                r.stderr.decode("utf-8", "replace")[:300],
            )

    def fetch(self, machine: Machine, path: str) -> bytes:
        r = subprocess.run(
            _ssh(machine.handle) + [fetch_command(path)],
            capture_output=True,
            timeout=FETCH_TIMEOUT_S,
        )
        return r.stdout if r.returncode == 0 else b""

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
                proc.stdin.buffer.write(script)
                proc.stdin.buffer.flush()
                proc.stdin.close()
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
            for raw in proc.stdout:
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
    """The default provider. Callers that do not care get this one."""
    return JarvisLabsProvider()
