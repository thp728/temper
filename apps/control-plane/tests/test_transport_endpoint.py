"""The transport tier: the real provider driven against a real endpoint.

The defect that cost this project a real run, line endings translated on the
way to a remote shell, lived exactly between the fake provider and the real
one. The fake never crosses a connection and is structurally blind to it; the
local-process tests in `test_provider.py` cross pipes but not a connection.

These tests start a real SSH server on 127.0.0.1 (`transport_endpoint.py`),
build the *real* provider implementation without touching credentials, and
drive its `push_stream`, `fetch_stream` and `stream` through the actual `ssh`
binary against it. No GPU, no money, no API key: the endpoint mints an
ephemeral keypair per test, so the suite stays hermetic while covering the
tier where transport defects actually live.
"""

import time

import pytest
from transport_endpoint import TransportEndpoint

from temper_control_plane.provider import JarvisLabsProvider, Machine


@pytest.fixture()
def machine(tmp_path):
    """A provider whose remote is a local SSH endpoint, plus that endpoint."""
    endpoint = TransportEndpoint(key_dir=tmp_path / "keys")
    endpoint.start()
    try:
        yield JarvisLabsProvider.__new__(JarvisLabsProvider), endpoint
    finally:
        endpoint.stop()


def test_streamed_output_arrives_line_by_line_as_produced(machine):
    """Lines arrive while the script is still being produced.

    The script carries sleep directives between its lines; the endpoint
    honours them as it consumes the script, so if output were buffered until
    EOF every timestamp would collapse together and the gaps would vanish.
    """
    provider, endpoint = machine
    script = b"alpha\n#!sleep 0.5\nomega\n"

    stamps = []
    for line in provider.stream(Machine(1, endpoint.handle), script):
        stamps.append((time.monotonic(), line))

    # `stream` folds stderr into stdout, so the ssh client's own chatter
    # arrives alongside the script's output; the assertion tracks the script's
    # lines by identity, not position. First occurrence wins: a buffered
    # implementation must not be able to pass by echoing a line twice late.
    seen: dict[str, float] = {}
    for stamp, line in stamps:
        seen.setdefault(line, stamp)
    assert {"alpha", "omega"} <= seen.keys()
    assert seen["omega"] - seen["alpha"] >= 0.4, (
        f"output arrived at the end, not as produced: {stamps}"
    )


def test_a_clean_script_runs_without_newline_translation(machine):
    """An LF-only script executes; each line is echoed back verbatim.

    This is the test that carries the regression criterion: against the
    pre-fix implementation the script crossed stdin in text mode, picked up
    a carriage return per line, and the endpoint answered with
    command-not-found errors instead of echoes, failing here, and failing
    in production before any job trained.
    """
    provider, endpoint = machine
    script = b"set -euo pipefail\necho one\necho two\n"

    out = list(provider.stream(Machine(1, endpoint.handle), script))

    echoed = [line for line in out if not line.startswith("Warning:")]
    assert echoed == ["set -euo pipefail", "echo one", "echo two"]
    assert all("$'\\r'" not in line for line in out)


def test_the_original_defect_manifests_here_exactly_as_it_did_on_the_machine(
    machine,
):
    """The pre-fix behaviour, reproduced against the endpoint deliberately.

    The script is translated to CRLF the way the old text-mode stdin did, and
    the endpoint answers with the same `$'\\r': command not found` signature
    the real machine produced. This is what proves the tier can see the
    defect: the double cannot produce it at any configuration.
    """
    provider, endpoint = machine
    clean = "set -euo pipefail\necho one\n"
    mangled = clean.replace("\n", "\r\n").encode()

    out = list(provider.stream(Machine(1, endpoint.handle), mangled))

    assert any("$'\\r': command not found" in line for line in out), (
        f"the endpoint could not see a CRLF-mangled script: {out}"
    )


# --- streaming transfer ------------------------------------------------------
#
# The only transfer tier there is now: spec 006's contract half deleted the
# buffered pair, so these tests are the byte-identity proof for the one way
# bytes move. They need no new endpoint behaviour -- the methods speak the
# same `mkdir + cat` / `sudo cat` wire grammar the endpoint emulates, and
# that reuse is itself under test: if a transfer method ever grew its own
# command shape, it would arrive here unemulated and fail.


def test_stream_push_then_stream_fetch_round_trips_text_including_line_endings(
    machine,
):
    """What went in as chunks comes out byte-identical, endings included.

    The payload is cut so a CRLF pair straddles two chunks -- the boundary
    where a chunk-aware implementation would be tempted to normalise.
    """
    provider, endpoint = machine
    dest = "/tmp/temper/dataset.jsonl"
    payload = b'{"a": 1}\n{"b": 2}\r\n{"c": 3}\rlast line'
    chunks = [payload[i : i + 7] for i in range(0, len(payload), 7)]

    provider.push_stream(Machine(1, endpoint.handle), iter(chunks), dest)

    assert endpoint.files[dest] == payload
    fetched = b"".join(
        provider.fetch_stream(Machine(1, endpoint.handle), dest)
    )
    assert fetched == payload


def test_stream_push_then_stream_fetch_round_trips_binary_content_unmodified(
    machine,
):
    """Every byte value survives, in many small chunks, both directions.

    Sized past any pipe or channel buffer (512 KB) and fed in 1000-byte
    pieces, so the transfer crosses hundreds of chunk boundaries; the fetch
    side must also deliver more than one chunk for the stream to count as
    one.
    """
    provider, endpoint = machine
    dest = "/tmp/temper/adapter.bin"
    payload = bytes(range(256)) * 2048
    chunks = (payload[i : i + 1000] for i in range(0, len(payload), 1000))

    assert len(payload) == 512 * 1024
    provider.push_stream(Machine(1, endpoint.handle), chunks, dest)
    assert endpoint.files[dest] == payload

    received = list(provider.fetch_stream(Machine(1, endpoint.handle), dest))
    assert b"".join(received) == payload
    assert len(received) > 1, "arrived as one blob, not a stream"


def test_stream_fetch_of_a_file_that_is_not_there_yields_nothing(machine):
    """The documented contract: yielding nothing means it could not be read."""
    provider, endpoint = machine
    assert (
        list(
            provider.fetch_stream(
                Machine(1, endpoint.handle), "/tmp/temper/absent"
            )
        )
        == []
    )
