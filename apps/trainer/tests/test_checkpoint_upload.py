"""The trainer's half of issue #37: checkpoints leave the machine as produced.

The control plane mints one scoped write grant per retention slot and ships
them in the job spec; this module's tests pin what the trainer does with them:

* only a *completed* checkpoint is uploaded -- the directory's own
  `trainer_state.json` must say its step, which is how a save cut off mid-way
  is kept out of storage;
* uploads run in a background thread, one at a time, so training never waits
  for one, and the newest checkpoint is still shipped by a final sweep at
  shutdown;
* each upload records its step and its loss, reading the checkpoint's own log
  history (the held-out loss where the step was evaluated);
* the ring is bounded: the i-th successful upload overwrites slot i mod N, so
  storage never holds more checkpoint objects per job than there are grants;
* a failed upload is retried a bounded number of times and then recorded as a
  failure -- never presented as a completed checkpoint.

The grant URLs are minted by the control plane; these tests stand a loopback
server in for the object store and record what a machine-style PUT sent, the
same way `test_artifact_upload.py` does for the artifact.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import checkpoints
import entrypoint


def checkpoint_dir(root: Path, step: int, loss: float | None = None) -> Path:
    """A completed checkpoint directory: name, state, and one payload file."""
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(f"weights-{step}".encode())
    history = []
    if loss is not None:
        history.append({"step": step, "loss": loss, "epoch": 0.5})
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": history})
    )
    return path


def grants(server_url: str, n: int, job: str = "job_1") -> list[dict]:
    """The `checkpoint_grants` block the control plane writes into the spec."""
    return [
        {
            "url": f"{server_url}/checkpoints/{job}/slot-{i}",
            "key": f"checkpoints/{job}/slot-{i}",
            "expires_at": time.time() + 3600,
        }
        for i in range(n)
    ]


class _RecordingHandler(BaseHTTPRequestHandler):
    def do_PUT(self):
        length = int(self.headers.get("Content-Length", 0))
        with self.server.lock:
            self.server.received[self.path] = self.rfile.read(length)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # noqa: ARG002 - the server is the test's
        pass


class _GrantServer:
    """A loopback stand-in for the object store a scoped URL points at."""

    def __init__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
        self.httpd.received = {}
        self.httpd.lock = threading.Lock()
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class _RecordingUpload:
    """A stand-in for the machine's PUT: records what would have been sent."""

    def __init__(self):
        self.calls: list[tuple[str, Path]] = []
        self.fail_url: str | None = None

    def __call__(self, url: str, path: Path) -> dict:
        self.calls.append((url, path))
        if url == self.fail_url:
            return {"ok": False, "error": "connection refused"}
        return {"ok": True, "bytes": path.stat().st_size}


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition never became true")


def test_completeness_requires_the_state_file_to_agree_with_the_step(tmp_path):
    complete = checkpoint_dir(tmp_path, 10, loss=1.5)
    assert checkpoints.is_complete(complete)

    no_state = tmp_path / "checkpoint-20"
    no_state.mkdir()
    (no_state / "model.safetensors").write_bytes(b"x")
    assert not checkpoints.is_complete(no_state), (
        "a directory with no trainer_state.json is still being written"
    )

    mismatched = tmp_path / "checkpoint-30"
    mismatched.mkdir()
    (mismatched / "trainer_state.json").write_text(
        json.dumps({"global_step": 29})
    )
    assert not checkpoints.is_complete(mismatched), (
        "a state file that does not agree on the step is a cut-off save"
    )


def test_losses_read_the_checkpoints_own_log_history(tmp_path):
    path = checkpoint_dir(tmp_path, 10, loss=1.5)
    (path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 10,
                "log_history": [
                    {"step": 9, "loss": 2.0},
                    {"step": 10, "loss": 1.5, "eval_loss": 1.4},
                ],
            }
        )
    )
    assert checkpoints.checkpoint_losses(path) == (1.5, 1.4)

    no_eval = checkpoint_dir(tmp_path, 20, loss=0.9)
    (no_eval / "trainer_state.json").write_text(
        json.dumps(
            {"global_step": 20, "log_history": [{"step": 20, "loss": 0.9}]}
        )
    )
    assert checkpoints.checkpoint_losses(no_eval) == (0.9, None)


def test_a_step_with_no_recorded_loss_reports_none(tmp_path):
    path = checkpoint_dir(tmp_path, 10)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": 10, "log_history": []})
    )
    assert checkpoints.checkpoint_losses(path) == (None, None)


def test_uploads_happen_in_the_background_as_checkpoints_appear(tmp_path):
    server = _GrantServer()
    try:
        out = tmp_path / "out"
        out.mkdir()
        upload = _RecordingUpload()
        worker = checkpoints.CheckpointUploader(
            out,
            grants(server.url, 3),
            upload=upload,
            log=lambda m: None,
            poll_interval_s=0.05,
        )
        worker.start()
        try:
            checkpoint_dir(out / "run", 10, loss=1.5)
            wait_until(lambda: len(upload.calls) == 1)
            assert upload.calls[0][0].endswith("/slot-0")

            checkpoint_dir(out / "run", 20, loss=1.2)
            wait_until(lambda: len(upload.calls) == 2)
            assert upload.calls[1][0].endswith("/slot-1")
        finally:
            worker.stop_and_finish()
    finally:
        server.close()

    records = worker.snapshot()
    assert [r["step"] for r in records] == [10, 20]
    assert records[0]["slot"] == 0 and records[1]["slot"] == 1
    assert records[0]["loss"] == 1.5
    assert records[0]["ok"] is True
    assert "held_out_loss" not in records[0]


def test_a_partial_checkpoint_is_never_uploaded(tmp_path):
    out = tmp_path / "out"
    (out / "run").mkdir(parents=True)
    partial = out / "run" / "checkpoint-10"
    partial.mkdir()
    (partial / "model.safetensors").write_bytes(b"half-written")

    upload = _RecordingUpload()
    worker = checkpoints.CheckpointUploader(
        out, grants("http://unused", 2), upload=upload, log=lambda m: None
    )
    worker.sweep()
    worker.stop_and_finish()

    assert upload.calls == []
    assert worker.snapshot() == []


def test_the_ring_is_bounded_to_the_number_of_slots(tmp_path):
    server = _GrantServer()
    try:
        out = tmp_path / "out"
        out.mkdir()
        upload = _RecordingUpload()
        worker = checkpoints.CheckpointUploader(
            out,
            grants(server.url, 2),
            upload=upload,
            log=lambda m: None,
            poll_interval_s=0.05,
        )
        worker.start()
        try:
            for step in (10, 20, 30, 40, 50):
                checkpoint_dir(out / "run", step)
                wait_until(
                    lambda s=step: s in {r["step"] for r in worker.snapshot()}
                )
        finally:
            worker.stop_and_finish()
    finally:
        server.close()

    # The i-th success overwrites slot i mod N: five checkpoints through two
    # slots, so only slot 0 and 1 are ever written.
    slots = {r["slot"] for r in worker.snapshot()}
    assert slots == {0, 1}
    # And the newest retained checkpoint sits in the slot the previous cycle
    # freed: step 50 is the 5th success -> slot 4 % 2 == 0.
    by_step = {r["step"]: r for r in worker.snapshot()}
    assert by_step[50]["slot"] == 0
    assert by_step[40]["slot"] == 1


def test_a_failed_upload_is_recorded_not_crashed(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    upload = _RecordingUpload()
    upload.fail_url = grants("http://127.0.0.1:1", 2)[0]["url"]

    worker = checkpoints.CheckpointUploader(
        out,
        grants("http://127.0.0.1:1", 2),
        upload=upload,
        log=lambda m: None,
        max_attempts=1,
    )
    checkpoint_dir(out / "run", 10, loss=1.5)
    worker.sweep()
    worker.stop_and_finish()

    assert worker.snapshot()[0]["ok"] is False
    assert worker.snapshot()[0]["step"] == 10
    assert worker.snapshot()[0]["loss"] == 1.5


def test_the_final_sweep_ships_a_checkpoint_the_poll_never_saw(tmp_path):
    """The last checkpoint of a run is uploaded even if the background loop
    had already stopped looking -- the one a resumption would most want."""
    server = _GrantServer()
    try:
        out = tmp_path / "out"
        out.mkdir()
        upload = _RecordingUpload()
        worker = checkpoints.CheckpointUploader(
            out, grants(server.url, 2), upload=upload, log=lambda m: None
        )
        worker._stop.set()  # the run "ended" before this checkpoint appeared
        checkpoint_dir(out / "run", 60)
        worker.stop_and_finish()
    finally:
        server.close()

    assert len(upload.calls) == 1
    assert worker.snapshot()[0]["step"] == 60


def test_a_checkpoint_puts_its_bytes_over_http_like_the_artifact(tmp_path):
    """The machine-style PUT is real: a checkpoint tar reaches the URL the
    control plane minted, with a declared Content-Length, exactly as the
    artifact does (ADR-0009's shape)."""
    server = _GrantServer()
    try:
        out = tmp_path / "out"
        out.mkdir()
        worker = checkpoints.CheckpointUploader(
            out,
            grants(server.url, 1),
            upload=entrypoint.upload_artifact,
            log=lambda m: None,
        )
        checkpoint_dir(out / "run", 10, loss=1.5)
        worker.stop_and_finish()
    finally:
        server.close()

    record = worker.snapshot()[0]
    assert record["ok"] is True
    body = server.httpd.received["/checkpoints/job_1/slot-0"]
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as tar:
        names = tar.getnames()
    assert "checkpoint-10/model.safetensors" in names
    assert "checkpoint-10/trainer_state.json" in names
    # The reported checksum is the checksum of the bytes that actually landed.
    assert record["sha256"] == hashlib.sha256(body).hexdigest()
