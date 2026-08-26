"""The transport tier: the real provider driven against a real endpoint.

The defect that cost this project a real run — line endings translated on the
way to a remote shell — lived exactly between the fake provider and the real
one. The fake never crosses a connection and is structurally blind to it; the
local-process tests in `test_provider.py` cross pipes but not a connection.

These tests start a real SSH server on 127.0.0.1 (`transport_endpoint.py`),
build the *real* provider implementation without touching credentials, and
drive its `push`, `fetch` and `stream` through the actual `ssh` binary against
it. No GPU, no money, no API key: the endpoint mints an ephemeral keypair per
test, so the suite stays hermetic while covering the tier where transport
defects actually live.
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


def test_push_then_fetch_round_trips_text_including_line_endings(machine):
    """What went in comes out byte-identical, endings included.

    This is the test the pre-fix implementation fails: text-mode stdin on
    Windows translated every "\\n" to "\\r\\n" before the payload reached the
    wire, so the stored bytes were not the pushed ones and this assertion
    broke across the whole dataset path.
    """
    provider, endpoint = machine
    dest = "/tmp/temper/dataset.jsonl"
    payload = b'{"a": 1}\n{"b": 2}\r\n{"c": 3}\rlast line'

    provider.push(Machine(1, endpoint.handle), payload, dest)

    assert endpoint.files[dest] == payload
    fetched = provider.fetch(Machine(1, endpoint.handle), dest)
    assert fetched == payload


def test_push_then_fetch_round_trips_binary_content_unmodified(machine):
    """Every byte value survives, including NUL and high bytes.

    Sized past any pipe or channel buffer (512 KB) so the transfer crosses
    many chunk boundaries rather than moving in one piece.
    """
    provider, endpoint = machine
    dest = "/tmp/temper/adapter.bin"
    payload = bytes(range(256)) * 2048

    assert len(payload) == 512 * 1024
    provider.push(Machine(1, endpoint.handle), payload, dest)
    assert provider.fetch(Machine(1, endpoint.handle), dest) == payload


def test_fetch_of_a_file_that_is_not_there_returns_empty_bytes(machine):
    """The documented contract: empty bytes mean it could not be read."""
    provider, endpoint = machine
    assert provider.fetch(Machine(1, "unused"), "/tmp/temper/absent") == b""


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
    command-not-found errors instead of echoes — failing here, and failing
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
