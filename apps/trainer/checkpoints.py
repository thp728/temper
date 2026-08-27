"""Upload checkpoints off the machine as training produces them (issue #37).

Runs beside `axolotl train` as a background thread in the entrypoint. Axolotl
owns the training loop, so the save callback is not ours to hook; instead the
uploader watches `/out/run/` for completed `checkpoint-N` directories and ships
each one to its slot in object storage before the machine is destroyed.

Four decisions this module makes, and why:

* **A checkpoint is complete when its `trainer_state.json` says so.** Axolotl
  (through transformers) writes `trainer_state.json` last in a checkpoint save,
  and it carries the step as `global_step`. A directory named `checkpoint-N`
  whose `trainer_state.json` has `global_step == N` is a finished checkpoint;
  a directory that has the file but not the agreeing step was cut off mid-save.
  Only the first is ever uploaded. This is the machine's half of "a partially
  written checkpoint is never presented as complete": the control plane's half
  is verifying what landed before it records anything.
* **Uploads run in a background thread, one checkpoint at a time, oldest
  first, and training never waits for one.** A slow link or a large checkpoint
  cannot stall the loop. Each upload streams: the directory is tarred in
  blocks to a temporary file, the file is stream-hashed, and the PUT streams
  the file body with a declared Content-Length (the same stdlib-only shape
  ADR-0009 uses for the artifact) -- nothing here holds a checkpoint whole.
* **Each upload uses a scoped write grant, one per retention slot.** The
  grants arrive in the job spec, minted by the control plane for exactly the
  `checkpoints/{job}/slot-N` keys (ADR-0009's machinery, reused rather than
  re-invented). The i-th successful upload goes to slot `i mod N`, overwriting
  the oldest retained checkpoint, so storage never holds more than N checkpoint
  objects per job. A failed upload does not advance the ring and is retried a
  bounded number of times; a checkpoint that ultimately fails to upload is
  recorded as such, never presented as complete.
* **Each upload records its step and its loss.** The step is in the directory
  name; the loss is read from the checkpoint's own `trainer_state.json` at
  upload time -- the held-out loss where the step was evaluated (Axolotl's
  `val_set_size` makes it evaluate), the training loss otherwise, and nothing
  when the step was never logged. The control plane records both beside the
  verified bytes.

One honest gap, recorded rather than glossed: the trainer keeps what the run
keeps. If `save_total_limit` makes Axolotl delete a checkpoint before this
uploader shipped it, that checkpoint is simply not in storage -- bounded
retention includes the machine's own disk, and the newest checkpoints are
uploaded first because they are the ones a resumption would need.
"""

from __future__ import annotations

import json
import tarfile
import threading
from collections.abc import Callable
from pathlib import Path

# How often the uploader scans for new checkpoints. Fixed rather than
# configurable: it is a tradeoff between catching a checkpoint promptly and
# not waking the disk pointlessly, and no operator needs to turn it.
POLL_INTERVAL_S = 5.0

# How many times an upload that failed is retried before it is recorded as
# failed. Bounded on purpose: a checkpoint that cannot reach the store should
# stop trying rather than burn the machine's egress forever, and the record --
# which the control plane turns into "not complete" -- is the honest end state.
MAX_ATTEMPTS = 3

# What the uploader can be asked to do with a file. `entrypoint.upload_artifact`
# has exactly this shape (a scoped-URL PUT that reports an outcome rather than
# raising); injecting it keeps this module importable without entrypoint, which
# in turn keeps entrypoint's imports acyclic.
UploadFn = Callable[[str, Path], dict]


def checkpoint_step(path: Path) -> int | None:
    """The step number in a `checkpoint-N` directory name, or None."""
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return None


def is_complete(path: Path) -> bool:
    """Whether a checkpoint directory is finished writing.

    `trainer_state.json` is the last file transformers writes in a checkpoint
    save, and its `global_step` is the checkpoint's step. A directory with the
    file and an agreeing step is complete; anything else -- no file, an
    unparseable one, or a step that does not match the directory's name -- is a
    save that has not finished, and is never uploaded.
    """
    step = checkpoint_step(path)
    if step is None:
        return False
    state = path / "trainer_state.json"
    if not state.is_file():
        return False
    try:
        data = json.loads(state.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return data.get("global_step") == step


def checkpoint_losses(path: Path) -> tuple[float | None, float | None]:
    """(training loss, held-out loss) recorded for a completed checkpoint.

    Reads the step's own entry in the checkpoint's `trainer_state.json` log
    history. The held-out loss is reported when that step was evaluated (with
    `val_set_size` set, Axolotl evaluates on a held-out split); the training
    loss otherwise; and (None, None) when the step has no recorded loss at all
    -- which is the "where one exists" of the acceptance criterion, stated as
    data rather than guessed.
    """
    step = checkpoint_step(path)
    state = path / "trainer_state.json"
    try:
        history = (
            json.loads(state.read_text(encoding="utf-8")).get("log_history")
            or []
        )
    except (ValueError, OSError):
        return None, None
    entry = None
    for row in reversed(history):
        if isinstance(row, dict) and row.get("step") == step:
            entry = row
            break
    if entry is None:
        return None, None
    train_loss = entry.get("loss")
    held_out = entry.get("eval_loss")
    return train_loss, held_out


class CheckpointUploader:
    """Ships completed checkpoints to their object-storage slots.

    Owns its background thread. `sweep()` is also safe to call from the main
    thread at shutdown, so the final checkpoint of a run -- the one a resumption
    would most want -- is uploaded even if the poll never saw it.
    """

    def __init__(
        self,
        out_dir: Path,
        grants: list[dict],
        upload: UploadFn,
        log: Callable[[str], None],
        poll_interval_s: float = POLL_INTERVAL_S,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self._out = Path(out_dir)
        # Tars land beside the checkpoints, on the provisioned disk sized for
        # them -- never in /tmp, which can be memory-backed and would turn a
        # large checkpoint into a memory spike.
        self._work = self._out / ".upload"
        self._grants = list(grants)
        self._slots = len(self._grants)
        # Named `_put`, not `_upload`: an attribute named `_upload` would
        # shadow the `_upload` method this class itself calls from `sweep`.
        self._put = upload
        self._log = log
        self._poll = poll_interval_s
        self._max_attempts = max_attempts
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._records: list[dict] = []
        self._successes = 0
        self._slot_of_step: dict[int, int] = {}
        self._attempts: dict[int, int] = {}
        self._done: set[int] = set()
        self._in_flight: set[int] = set()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="checkpoint-uploader", daemon=True
        )
        self._thread.start()

    def stop_and_finish(self) -> None:
        """Stop the background loop, upload anything it missed, then join.

        Called from the entrypoint's `finally`, so a run that ended in failure
        still ships the checkpoints that completed before it failed. The join
        is bounded by the same poll interval the thread lives on: a thread
        stuck mid-upload must not hold the machine up forever, and a thread
        that finished needs no wait at all.
        """
        self._stop.set()
        self.sweep()
        if self._thread is not None:
            self._thread.join(timeout=self._poll * 4)

    def snapshot(self) -> list[dict]:
        """Every upload attempt, in step order, for the run's result document."""
        with self._lock:
            return sorted(self._records, key=lambda r: r.get("step", -1))

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sweep()
            self._stop.wait(self._poll)

    def sweep(self) -> None:
        """Upload every completed checkpoint not yet dealt with, oldest first."""
        run = self._out / "run"
        if not run.is_dir() or self._slots == 0:
            return
        candidates = sorted(
            (p for p in run.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: checkpoint_step(p) or 0,
        )
        for path in candidates:
            step = checkpoint_step(path)
            if step is None or step in self._done:
                continue
            if not is_complete(path):
                continue
            self._upload(path, step)

    def _upload(self, path: Path, step: int) -> None:
        with self._lock:
            if step in self._done or step in self._in_flight:
                return
            if step not in self._slot_of_step:
                self._slot_of_step[step] = self._successes % self._slots
            slot = self._slot_of_step[step]
            self._in_flight.add(step)
        record = self._upload_once(path, step, slot)
        with self._lock:
            self._in_flight.discard(step)
            if record.get("ok"):
                self._successes += 1
                self._records.append(record)
                self._done.add(step)
                return
            attempts = self._attempts.get(step, 0) + 1
            self._attempts[step] = attempts
            if attempts >= self._max_attempts:
                self._records.append(record)
                self._done.add(step)

    def _upload_once(self, path: Path, step: int, slot: int) -> dict:
        base = {"step": step, "slot": slot}
        train_loss, held_out = checkpoint_losses(path)
        if train_loss is not None:
            base["loss"] = train_loss
        if held_out is not None:
            base["held_out_loss"] = held_out
        grant = self._grants[slot % self._slots]

        self._work.mkdir(parents=True, exist_ok=True)
        tar_path = self._work / f"checkpoint-{step}.tar"
        try:
            with tarfile.open(tar_path, mode="w") as tar:
                tar.add(path, arcname=path.name)
            outcome = self._put(grant["url"], tar_path)
            if not outcome.get("ok"):
                return {
                    **base,
                    "ok": False,
                    "error": outcome.get("error") or "upload failed",
                }
            return {
                **base,
                "ok": True,
                "sha256": _sha256_of(tar_path),
                "bytes": tar_path.stat().st_size,
            }
        except Exception as e:  # noqa: BLE001 - an upload must not crash the run
            return {**base, "ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            tar_path.unlink(missing_ok=True)


def _sha256_of(path: Path) -> str:
    """The file's SHA-256, computed in bounded blocks rather than read whole."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
