"""The trainer's half of the artifact write path (ADR-0009).

The machine writes its own artifact to a scoped grant rather than handing it
back through the control plane. These tests pin the two things that makes
true here, on the trainer's side of the seam:

* the artifact is fingerprinted in bounded blocks, so the machine never holds
  a full fine-tune's artifact whole to report a checksum;
* the upload PUTs the file's bytes to the grant URL over a real HTTP
  connection and reports an outcome, so a failed upload is recorded in
  result.json rather than crashing the job before it is reported.

The grant URL is minted by the control plane; this test's server stands in
for it and records what a machine-style PUT actually sent.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import entrypoint


class _RecordingHandler(BaseHTTPRequestHandler):
    def do_PUT(self):
        length = int(self.headers.get("Content-Length", 0))
        self.server.received = {
            "path": self.path,
            "content_type": self.headers.get("Content-Type"),
            "body": self.rfile.read(length),
        }
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # noqa: ARG002 - the server is the test's
        pass


class _GrantServer:
    """A loopback stand-in for the object store a scoped URL points at."""

    def __init__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
        self.httpd.received = None
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}/artifacts/job_1/w"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def test_upload_artifact_puts_the_file_over_http(tmp_path):
    payload = b"weights \x00 binary \r\n" * 1000
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(payload)

    server = _GrantServer()
    try:
        outcome = entrypoint.upload_artifact(server.url, path)
    finally:
        server.close()

    assert outcome == {"ok": True, "bytes": len(payload)}
    received = server.httpd.received
    assert received["path"].endswith("/artifacts/job_1/w")
    assert received["content_type"] == "application/octet-stream"
    assert received["body"] == payload, "the URL did not receive the artifact"


def test_upload_artifact_reports_a_failure_rather_than_raising(tmp_path):
    """A failed upload is a recorded outcome, never an uncaught exception:
    the job's result document must be written no matter what happened."""
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(b"x")

    outcome = entrypoint.upload_artifact(
        "http://127.0.0.1:1/unreachable", path
    )

    assert outcome["ok"] is False
    assert "error" in outcome


def test_collect_artifacts_hashes_without_holding_the_file(
    tmp_path, monkeypatch
):
    """The checksum reported is the whole-file one, computed in bounded
    blocks: a payload spanning several hash blocks must hash identically to
    reading it whole."""
    out = tmp_path / "out"
    run = out / "run"
    run.mkdir(parents=True)
    payload = b"\x00\xff" * (2 << 20)  # 4 MiB: several 1 MiB hash blocks
    (run / "adapter_model.safetensors").write_bytes(payload)
    (run / "adapter_config.json").write_text(json.dumps({"r": 16}))
    monkeypatch.setattr(entrypoint, "OUT_DIR", out)

    info = entrypoint.collect_artifacts("qlora")

    assert info["artifact_sha256"] == hashlib.sha256(payload).hexdigest()
    assert info["artifact_bytes"] == len(payload)
