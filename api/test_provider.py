"""The default provider's transport, exercised without a machine.

The fake covers the orchestration path; nothing covers the code that actually
moves bytes. These tests run the real `stream` against a local process instead
of an SSH one, which is enough to catch its two ways of going wrong: feeding a
large script from the read loop deadlocks, and a command that never ends has to
be killed by something.

The provider is built with `object.__new__` deliberately — constructing one
properly reaches the real client, which the suite refuses.
"""

import subprocess
import sys

import pytest

from api import provider as provider_mod
from api.provider import JarvisLabsProvider, Machine

# Echoes stdin back, unbuffered, so the parent sees lines as they are written.
ECHO = "import sys\nfor line in sys.stdin: sys.stdout.write(line)"
SLEEP = "import time\ntime.sleep(30)"


def transport(monkeypatch, program: str) -> JarvisLabsProvider:
    """A provider whose 'SSH' is a local Python process running `program`."""
    monkeypatch.setattr(provider_mod, "_ssh",
                        lambda handle: [sys.executable, "-u", "-c", program])
    return object.__new__(JarvisLabsProvider)


def test_ssh_command_from_the_provider_is_used_verbatim():
    """The provider hands back `ssh -p 1234 user@host`; the `ssh` is ours."""
    argv = provider_mod._ssh("ssh -p 1234 root@10.0.0.1")
    assert argv[0] == "ssh"
    assert argv[-3:] == ["-p", "1234", "root@10.0.0.1"]
    assert "BatchMode=yes" in argv


def test_stream_yields_the_lines_the_command_produced(monkeypatch):
    p = transport(monkeypatch, ECHO)
    script = b"first\nsecond\nthird\n"
    assert list(p.stream(Machine(1), script)) == ["first", "second", "third"]


def test_stream_does_not_deadlock_on_a_script_larger_than_the_pipe_buffer(
        monkeypatch):
    """The script carries the dataset inline; 64KB is not a hypothetical."""
    p = transport(monkeypatch, ECHO)
    script = ("x" * 200 + "\n").encode() * 1000       # ~200KB
    assert len(list(p.stream(Machine(1), script))) == 1000


def test_stream_kills_a_command_that_outlives_the_backstop(monkeypatch):
    """The backstop, not the limit anyone is meant to meet.

    It is derived from the duration ceiling and sits above it, so that a long
    job fails with the named outcome the orchestrator produces rather than with
    an uncoded transport error. Turned down here to prove it still fires.
    """
    p = transport(monkeypatch, SLEEP)
    monkeypatch.setattr(provider_mod, "_stream_timeout", lambda: 0.5)
    with pytest.raises(subprocess.TimeoutExpired):
        list(p.stream(Machine(1), b"ignored\n"))


def test_abandoning_the_stream_kills_the_command(monkeypatch):
    """Cancellation stops reading; the remote command must not survive it."""
    p = transport(monkeypatch, ECHO)
    lines = p.stream(Machine(1), b"one\ntwo\nthree\n")
    assert next(lines) == "one"
    lines.close()          # what abandoning the iterator does
    # No assertion beyond returning: a surviving child would hang the suite at
    # interpreter exit, which is exactly the failure being guarded against.


# --- readiness: two failures with opposite remedies -------------------------

class Answer:
    """One canned result from an SSH probe."""

    def __init__(self, returncode, stderr=""):
        self.returncode, self.stderr = returncode, stderr
        self.stdout = ""


def probe(monkeypatch, answer):
    """A provider whose readiness probe always gets `answer`, quickly."""
    monkeypatch.setattr(provider_mod, "SSH_READY_TIMEOUT_S", 0.2)
    monkeypatch.setattr(provider_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(provider_mod.subprocess, "run",
                        lambda *a, **k: answer)
    return object.__new__(JarvisLabsProvider)


def test_a_machine_that_answers_is_ready(monkeypatch):
    p = probe(monkeypatch, Answer(0))
    assert "SSH ready" in p.await_ready(Machine(1))


def test_a_rejected_key_is_reported_as_authentication_failure(monkeypatch):
    """Not 'no answer'. The remedy is `ssh-add`, and it is nothing like waiting.

    Collapsing these two into one message cost an evening and produced a
    wrongly-filed platform bug.
    """
    p = probe(monkeypatch, Answer(255, "root@1.2.3.4: Permission denied (publickey)."))
    with pytest.raises(provider_mod.OrchestratorError) as e:
        p.await_ready(Machine(1))
    assert e.value.code == "ssh_auth_failed"
    assert "ssh-add" in str(e.value)


def test_a_machine_that_never_answers_is_reported_as_unreachable(monkeypatch):
    p = probe(monkeypatch, Answer(255, "ssh: connect to host port 22: Connection refused"))
    with pytest.raises(provider_mod.OrchestratorError) as e:
        p.await_ready(Machine(1))
    assert e.value.code == "ssh_unreachable"


# Reports the exact bytes it was fed, so the test can see newline translation.
REPORT_STDIN = (
    "import sys\n"
    "data = sys.stdin.buffer.read()\n"
    "print('CR' if bytes([13]) in data else 'CLEAN')\n"
    "print(repr(data))\n"
)


def test_the_script_arrives_with_the_newlines_it_was_given(monkeypatch):
    """No carriage returns, on any platform.

    `text=True` wraps stdin in a TextIOWrapper, and on Windows that wrapper
    turns every "\n" into "\r\n". The remote shell then reads a script whose
    every line ends in a carriage return and refuses the first one with
    `$'\r': command not found` -- before running any of it.

    The fake provider cannot catch this. It never crosses a pipe, so the
    translation that breaks a real run does not happen to it, and the whole
    suite stayed green while no job could train. This test exists because that
    is exactly what happened.
    """
    p = transport(monkeypatch, REPORT_STDIN)
    script = b"set -euo pipefail\necho one\necho two\n"

    out = list(p.stream(Machine(1), script))

    assert out[0] == "CLEAN", f"newlines were translated: {out[1]}"
    assert repr(script) in out[1]
