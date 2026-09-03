"""A local SSH endpoint the real provider transport is driven against.

The transport tier closes the gap between the fake provider and the real one:
the fake never crosses a connection, so it is structurally blind to anything a
connection does to bytes; line-ending translation above all, which is the
defect that cost this project a real run.

This module is the machine side of that tier: an in-process SSH server on
127.0.0.1 that answers the exact three command shapes `JarvisLabsProvider`
sends; `mkdir -p <parent> && cat > <dest>`, `sudo cat <path>`, and `bash -s`.
It holds pushed bytes in memory, returns them on fetch, and for `bash -s` it
echoes each script line back as it consumes it, which is what lets tests
observe output arriving line by line rather than at the end. Two deliberate
emulations, each faithful to what the real machine did:

* A script line ending in a carriage return produces bash's actual failure,
  ``/dev/stdin: line N: $'\r': command not found``, and exit status 127.
  The original defect surfaced exactly this way; the emulation keeps that
  signature assertable instead of anecdotal.
* A line beginning with ``#!sleep <seconds>`` pauses before continuing, so a
  test can prove streamed output arrives while the script is still running.

Authentication is an ephemeral keypair minted per endpoint: nothing here
touches a real credential, an agent, or a billing account. Tests drive it with
the *real* `ssh` client through `JarvisLabsProvider.push`, `.fetch` and
`.stream`, unmodified.

The server runs its own asyncio loop on a daemon thread so the provider's
blocking subprocess calls can be driven from plain sync tests.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from pathlib import Path

import asyncssh

from temper_control_plane.provider import (
    FETCH_PREFIX,
    PUSH_PREFIX,
    PUSH_SEPARATOR,
    STREAM_COMMAND,
)

# The directive prefix and the failure signature, named once because the tests
# assert against them by meaning rather than by re-typed literals.
SLEEP_DIRECTIVE = "#!sleep "
CR_FAILURE = "$'\\r': command not found"

_READ_CHUNK = 65536


def _authorised_key_line(key: asyncssh.SSHKey) -> bytes:
    """Type and base64 body of a public key, comment stripped."""
    return b" ".join(key.export_public_key().split()[:2])


def _restrict_to_owner(key_path: Path) -> None:
    """Make the private key acceptable to the platform's `ssh` client.

    Windows OpenSSH refuses an identity file readable by anyone but its
    owner, and pytest's tmp directories inherit an OWNER RIGHTS entry it
    still complains about. Strip inheritance and grant only the current
    user; on POSIX a mode bit does the same job.
    """
    if os.name == "nt":
        user = os.environ.get("USERNAME", "")
        subprocess.run(
            [
                "icacls",
                str(key_path),
                "/inheritance:r",
                "/grant:r",
                f"{user}:F",
            ],
            check=True,
            capture_output=True,
        )
    else:
        key_path.chmod(0o600)


class _EndpointServer(asyncssh.SSHServer):
    """Accepts exactly one pre-shared client key and nothing else."""

    def __init__(self, authorised: bytes) -> None:
        self._authorised = authorised

    def begin_auth(self, username: str) -> bool:
        # Always challenge. Only validate_public_key's verdict ends auth.
        return True

    @staticmethod
    def public_key_auth_supported() -> bool:
        # Off by default in asyncssh's server base; without it the client's
        # key is never even offered to validate_public_key.
        return True

    def validate_public_key(self, username: str, key: asyncssh.SSHKey) -> bool:
        return _authorised_key_line(key) == self._authorised


class TransportEndpoint:
    """A fake machine: an in-process SSH server speaking real SSH.

    `handle` plugs straight into `Machine(handle=...)`; it is opaque to
    everything outside the provider implementation, which parses it into an
    `ssh` command line. `files` is the machine's memory: what a push stored
    and what a fetch would return.
    """

    def __init__(self, key_dir: Path) -> None:
        self.key_dir = key_dir
        self.files: dict[str, bytes] = {}
        self.port = 0
        self.handle = ""
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncssh.SSHAcceptor | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._serve, name="transport-endpoint", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=15):
            raise RuntimeError("the transport endpoint failed to start")

    def stop(self) -> None:
        loop, server = self._loop, self._server
        if loop is not None and server is not None and loop.is_running():
            loop.call_soon_threadsafe(server.close)
        if self._thread is not None:
            self._thread.join(timeout=10)

    # -- server side (runs on the endpoint thread) --------------------------

    def _serve(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        host_key = asyncssh.generate_private_key("ssh-ed25519")
        client_key = asyncssh.generate_private_key("ssh-ed25519")

        self.key_dir.mkdir(parents=True, exist_ok=True)
        key_path = self.key_dir / "id_ed25519"
        key_path.write_bytes(client_key.export_private_key("openssh"))
        _restrict_to_owner(key_path)

        server = await asyncssh.listen(
            "127.0.0.1",
            0,
            encoding=None,  # byte mode both ways: the tier exists to prove
            # bytes survive, so the endpoint must never decode them
            server_host_keys=[host_key],
            server_factory=lambda: _EndpointServer(
                _authorised_key_line(client_key)
            ),
            process_factory=self._handle_process,
        )
        self.port = server.get_port()
        self._server = server
        self.handle = (
            f'-i "{key_path.as_posix()}" -o IdentitiesOnly=yes '
            f"-p {self.port} temper@127.0.0.1"
        )
        self._ready.set()
        try:
            await server.wait_closed()
        finally:
            await asyncio.to_thread(self._cleanup)

    def _cleanup(self) -> None:
        import shutil

        shutil.rmtree(self.key_dir, ignore_errors=True)

    async def _handle_process(
        self, process: asyncssh.SSHServerProcess
    ) -> None:
        command = process.command or ""
        if command == STREAM_COMMAND:
            await self._bash(process)
        elif command.startswith(PUSH_PREFIX) and PUSH_SEPARATOR in command:
            dest = command.rsplit(PUSH_SEPARATOR, 1)[1]
            self.files[dest] = await self._drain(process)
            process.exit(0)
        elif command.startswith(FETCH_PREFIX):
            path = command[len(FETCH_PREFIX) :]
            blob = self.files.get(path)
            if blob is None:
                process.stderr.write(f"cat: {path}: No such file\n".encode())
                process.exit(1)
            else:
                process.stdout.write(blob)
                process.exit(0)
        else:
            process.stderr.write(f"{command}: command not found\n".encode())
            process.exit(127)

    async def _drain(self, process: asyncssh.SSHServerProcess) -> bytes:
        chunks = bytearray()
        while chunk := await process.stdin.read(_READ_CHUNK):
            chunks += chunk
        return bytes(chunks)

    async def _bash(self, process: asyncssh.SSHServerProcess) -> None:
        status = 0
        lineno = 0
        buf = b""

        def answer(raw: bytes) -> float | None:
            """Answer one script line like bash would over the wire.

            Returns the pause the line asked for (a directive), or None.
            """
            nonlocal status, lineno
            lineno += 1
            if raw.endswith(b"\r"):
                process.stdout.write(
                    f"/dev/stdin: line {lineno}: {CR_FAILURE}\n".encode()
                )
                status = 127
                return None
            text = raw.decode("utf-8", "replace")
            if text.startswith(SLEEP_DIRECTIVE):
                return float(text[len(SLEEP_DIRECTIVE) :])
            process.stdout.write(raw + b"\n")
            return None

        while True:
            chunk = await process.stdin.read(_READ_CHUNK)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                pause = answer(raw)
                if pause is not None:
                    await asyncio.sleep(pause)
        if buf:
            answer(buf)
        process.exit(status)
