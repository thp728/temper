"""The default provider's transport, exercised without a machine.

The fake covers the orchestration path; nothing covers the code that actually
moves bytes. These tests run the real `stream` against a local process instead
of an SSH one, which is enough to catch its two ways of going wrong: feeding a
large script from the read loop deadlocks, and a command that never ends has
to be killed by something.

A local process crosses pipes but not a connection, so what a real connection
does to bytes is covered one tier up, in `test_transport_endpoint.py`, where
the same provider methods are driven against a live SSH endpoint.

The provider is built with `object.__new__` deliberately — constructing one
properly reaches the real client, which the suite refuses.
"""

import hashlib
import itertools
import subprocess
import sys
import tracemalloc

import pytest

from temper_control_plane import provider as provider_mod
from temper_control_plane.provider import JarvisLabsProvider, Machine

# Echoes stdin back, unbuffered, so the parent sees lines as they are written.
ECHO = "import sys\nfor line in sys.stdin: sys.stdout.write(line)"
SLEEP = "import time\ntime.sleep(30)"


def transport(monkeypatch, program: str) -> JarvisLabsProvider:
    """A provider whose 'SSH' is a local Python process running `program`."""
    monkeypatch.setattr(
        provider_mod,
        "_ssh",
        lambda handle: [sys.executable, "-u", "-c", program],
    )
    return object.__new__(JarvisLabsProvider)


def test_ssh_command_from_the_provider_is_used_verbatim():
    """The provider hands back `ssh -p 1234 user@host`; the `ssh` is ours."""
    argv = provider_mod._ssh("ssh -p 1234 root@10.0.0.1")
    assert argv[0] == "ssh"
    assert argv[-3:] == ["-p", "1234", "root@10.0.0.1"]
    assert "BatchMode=yes" in argv


def test_a_quoted_identity_path_stays_one_argument():
    """Handles are parsed, not split on whitespace.

    The transport endpoint's handle quotes an identity path; on this
    developer's machine that path contains a space (`C:/Users/Tejas
    Page/...`). Splitting the handle naively would hand `ssh` two broken
    arguments and fail auth with no hint why.
    """
    argv = provider_mod._ssh('-i "C:/Users/Tejas Page/k/id" -p 2222 u@h')
    assert "-i" in argv
    i = argv.index("-i")
    assert argv[i + 1] == "C:/Users/Tejas Page/k/id"
    assert argv[-1] == "u@h"


def test_stream_yields_the_lines_the_command_produced(monkeypatch):
    p = transport(monkeypatch, ECHO)
    script = b"first\nsecond\nthird\n"
    assert list(p.stream(Machine(1), script)) == ["first", "second", "third"]


def test_stream_does_not_deadlock_on_a_script_larger_than_the_pipe_buffer(
    monkeypatch,
):
    """The script carries the dataset inline; 64KB is not a hypothetical."""
    p = transport(monkeypatch, ECHO)
    script = ("x" * 200 + "\n").encode() * 1000  # ~200KB
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
    lines.close()  # what abandoning the iterator does
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
    monkeypatch.setattr(provider_mod.subprocess, "run", lambda *a, **k: answer)
    return object.__new__(JarvisLabsProvider)


def test_a_machine_that_answers_is_ready(monkeypatch):
    p = probe(monkeypatch, Answer(0))
    assert "SSH ready" in p.await_ready(Machine(1))


def test_a_rejected_key_is_reported_as_authentication_failure(monkeypatch):
    """Not 'no answer'. The remedy is `ssh-add`, and it is nothing like waiting.

    Collapsing these two into one message cost an evening and produced a
    wrongly-filed platform bug.
    """
    p = probe(
        monkeypatch,
        Answer(255, "root@1.2.3.4: Permission denied (publickey)."),
    )
    with pytest.raises(provider_mod.OrchestratorError) as e:
        p.await_ready(Machine(1))
    assert e.value.code == "ssh_auth_failed"
    assert "ssh-add" in str(e.value)


def test_a_machine_that_never_answers_is_reported_as_unreachable(monkeypatch):
    p = probe(
        monkeypatch,
        Answer(255, "ssh: connect to host port 22: Connection refused"),
    )
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


# --- streaming transfer ------------------------------------------------------
#
# Spec 006's expand half: `push_stream` and `fetch_stream` beside the buffered
# pair. The sink and feeder below are the local tier of the same three tiers
# the buffered methods have: what is asserted here is chunk order, byte
# identity, the backstop and abandonment. What a *connection* does to the
# chunks is covered one tier up, in test_transport_endpoint.py; that memory
# stays flat as payloads grow is asserted further down.


# Hashes stdin incrementally -- never holding it whole -- and exits nonzero on
# a digest mismatch, so "the transfer succeeded" means "the bytes that crossed
# the pipe are exactly these". The expected digest arrives via argv.
HASH_SINK = (
    "import hashlib, sys\n"
    "h = hashlib.sha256()\n"
    "while chunk := sys.stdin.buffer.read(65536):\n"
    "    h.update(chunk)\n"
    "sys.exit(0 if h.hexdigest() == sys.argv[1] else 3)\n"
)

# Writes argv[1] bytes in 256 KiB pieces of a repeating 0..255 pattern, so the
# test can recompute the same stream independently and compare digests.
FEEDER = (
    "import sys\n"
    "left = int(sys.argv[1])\n"
    "block = bytes(range(256)) * 1024\n"
    "while left > 0:\n"
    "    n = min(left, len(block))\n"
    "    sys.stdout.buffer.write(block[:n])\n"
    "    left -= n\n"
)


def streaming_transport(monkeypatch, program: str, *args: str):
    """A provider whose 'SSH' is a local process running `program`."""
    monkeypatch.setattr(
        provider_mod,
        "_ssh",
        lambda handle: [sys.executable, "-u", "-c", program, *args],
    )
    return object.__new__(JarvisLabsProvider)


def pattern_chunks(total: int, piece: int = 256 * 1024):
    """The FEEDER's byte sequence, produced as an iterator of pieces.

    One block object reused per yield: the caller's own footprint must not
    grow with the payload either, or the flat-memory assertions would pass or
    fail on the test's bookkeeping rather than the implementation's.
    """
    block = (bytes(range(256)) * 1024)[:piece]
    made = 0
    while made < total:
        yield block
        made += len(block)


def pattern_digest(total: int) -> str:
    """Digest of `pattern_chunks(total)`'s concatenation, held one piece big."""
    h = hashlib.sha256()
    for chunk in pattern_chunks(total):
        h.update(chunk)
    return h.hexdigest()


def peak_traced_during(run) -> int:
    """Peak Python-heap allocation, in bytes, while `run` executes.

    tracemalloc measures this process's own allocations -- which is the thing
    under test: the control plane side of the transfer. The child at the far
    end of the pipe is a separate address space and holds O(one chunk) by its
    own construction. RSS would bury the signal under the interpreter baseline;
    the traced heap catches exactly the failure mode spec 006 names, an
    implementation quietly accumulating chunks. Measured, not assumed.
    """
    tracemalloc.start()
    try:
        run()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_push_stream_delivers_the_chunks_byte_identical(monkeypatch):
    """Chunks arrive concatenated in order, whatever their boundaries.

    The payload is split so a CRLF pair straddles two chunks; a translation
    or a reordering shows up as a digest mismatch. The negative half proves
    the harness has teeth: a deliberately wrong expectation fails the push,
    so a passing one is evidence and not tautology.
    """
    payload = b'{"a": 1}\r\n{"b": 2}\n{"c": 3}\rlast line'
    # 7-byte pieces cut mid-line and mid-CRLF on purpose.
    chunks = [payload[i : i + 7] for i in range(0, len(payload), 7)]

    good = streaming_transport(
        monkeypatch, HASH_SINK, hashlib.sha256(payload).hexdigest()
    )
    good.push_stream(Machine(1), iter(chunks), "/tmp/temper/ds.jsonl")

    wrong = streaming_transport(monkeypatch, HASH_SINK, "0" * 64)
    with pytest.raises(provider_mod.OrchestratorError):
        wrong.push_stream(Machine(1), iter(chunks), "/tmp/temper/ds.jsonl")


def test_fetch_stream_yields_whole_file_across_chunks(monkeypatch):
    """Every byte arrives exactly once, in order, across several chunks."""
    total = 3 * 256 * 1024  # three reader-chunks plus remainder-free tail
    p = streaming_transport(monkeypatch, FEEDER, str(total))

    seen = bytearray()
    count = 0
    for chunk in p.fetch_stream(Machine(1), "/tmp/temper/out.bin"):
        seen += chunk
        count += 1

    assert len(seen) == total
    assert count > 1, "arrived as one blob, not a stream"
    assert hashlib.sha256(bytes(seen)).hexdigest() == pattern_digest(total), (
        "bytes did not survive the trip"
    )


def test_push_stream_kills_a_push_that_outlives_the_backstop(monkeypatch):
    """A wedged remote cannot hold the control plane forever.

    The child reads nothing and never exits, so the write side blocks once
    the pipe fills; the watchdog is what gets the caller back. Turned down
    from its production value to prove it still fires.
    """
    p = streaming_transport(monkeypatch, SLEEP)
    monkeypatch.setattr(provider_mod, "PUSH_TIMEOUT_S", 0.5)
    endless = (b"x" * (1 << 20) for _ in itertools.count())

    with pytest.raises(subprocess.TimeoutExpired):
        p.push_stream(Machine(1), endless, "/tmp/temper/ds.jsonl")


def test_push_stream_refuses_a_connection_that_ends_early(monkeypatch):
    """A truncated push is a failure even when the remote exits zero.

    `cat > dest` cannot tell EOF-because-done from EOF-because-the
    -connection-died, so exit status alone would record a half-delivered
    file as uploaded. The child here stops reading after ten bytes and
    exits 0 -- exactly what a remote looks like when the connection dies
    part-way through -- and the push must still be refused.
    """
    early_exit_sink = "import sys\nsys.stdin.buffer.read(10)\n"
    p = streaming_transport(monkeypatch, early_exit_sink)
    endless = (b"x" * (1 << 20) for _ in itertools.count())

    with pytest.raises(provider_mod.OrchestratorError) as e:
        p.push_stream(Machine(1), endless, "/tmp/temper/ds.jsonl")
    # The not-drained branch specifically: exit status was zero, so this
    # refusal is the method's own verdict, not the remote's.
    assert "before every byte" in str(e.value)


def test_abandoning_a_fetch_stream_kills_the_command(monkeypatch):
    """Cancellation stops reading; the remote command must not survive it."""
    endless_feeder = (
        "import sys\nblock = bytes(range(256)) * 1024\n"
        "while True:\n    sys.stdout.buffer.write(block)\n"
    )
    p = streaming_transport(monkeypatch, endless_feeder)

    chunks = p.fetch_stream(Machine(1), "/tmp/temper/out.bin")
    next(chunks)
    chunks.close()  # what abandoning the iterator does
    # No assertion beyond returning: a surviving child would hang the suite at
    # interpreter exit, which is exactly the failure being guarded against.


# The flat-memory bounds, named once because both directions assert them:
# the largest payload's peak must sit within FLAT_SPREAD of the smallest's,
# and under ABS_CEIL -- a ceiling far below the largest payload, which any
# accumulate-then-send implementation blows through immediately.
FLAT_SPREAD = 1 << 21  # 2 MiB
ABS_CEIL = 8 << 20  # 8 MiB, against a 32 MiB payload

MEMORY_PAYLOADS = (1 << 20, 8 << 20, 32 << 20)  # 1, 8, 32 MiB


def test_push_stream_peak_memory_stays_flat_across_payload_sizes(monkeypatch):
    """Peak held memory does not scale with the payload pushed.

    Asserted both relatively (the largest payload's peak sits within slack of
    the smallest's) and absolutely (a ceiling far below the largest payload,
    which any accumulate-then-send implementation would blow through).
    """
    peaks = []
    for total in MEMORY_PAYLOADS:
        p = streaming_transport(monkeypatch, HASH_SINK, pattern_digest(total))
        peaks.append(
            peak_traced_during(
                lambda p=p, total=total: p.push_stream(
                    Machine(1), pattern_chunks(total), "/tmp/temper/ds.jsonl"
                )
            )
        )

    assert max(peaks) - min(peaks) < FLAT_SPREAD, (
        f"peaks grew with size: {peaks}"
    )
    assert peaks[-1] < ABS_CEIL, (
        f"peak scaled with a {MEMORY_PAYLOADS[-1] >> 20} MiB payload: {peaks}"
    )


def test_fetch_stream_peak_memory_stays_flat_across_payload_sizes(monkeypatch):
    """Peak held memory does not scale with the payload fetched."""
    peaks = []
    for total in MEMORY_PAYLOADS:
        p = streaming_transport(monkeypatch, FEEDER, str(total))

        def consume(p=p, total=total):
            h = hashlib.sha256()
            got = 0
            for chunk in p.fetch_stream(Machine(1), "/tmp/temper/out.bin"):
                h.update(chunk)
                got += len(chunk)
            assert got == total

        peaks.append(peak_traced_during(consume))

    assert max(peaks) - min(peaks) < FLAT_SPREAD, (
        f"peaks grew with size: {peaks}"
    )
    assert peaks[-1] < ABS_CEIL, (
        f"peak scaled with a {MEMORY_PAYLOADS[-1] >> 20} MiB payload: {peaks}"
    )


def test_the_flat_memory_assertion_catches_an_accumulating_implementation(
    monkeypatch,
):
    """The flat-memory assertions are not vacuous.

    The same feeder drives a deliberately accumulating consumer -- the
    defect spec 006 warns that streaming quietly becomes -- and its traced
    peak must blow past the ceiling the real implementation is held under.
    Against an implementation that accumulates in secret, the two tests
    above would pass while memory scaled with every payload; this test is
    the proof they would notice.
    """
    total = MEMORY_PAYLOADS[-1]
    p = streaming_transport(monkeypatch, FEEDER, str(total))

    def accumulate():
        held = []
        for chunk in p.fetch_stream(Machine(1), "/tmp/temper/out.bin"):
            held.append(chunk)
        return len(held)

    peak = peak_traced_during(accumulate)
    assert peak >= total, (
        f"an accumulating consumer measured only {peak} bytes for a "
        f"{total}-byte payload; the measurement is blind"
    )
