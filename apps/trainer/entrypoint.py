"""Trainer entrypoint ΓÇö job spec in, adapter + result out.

Runs inside the pinned image. The orchestrator writes a job spec and a dataset
to /job, runs this container, and reads /out.

    /job/job.json       the job spec (see job.example.json)
    /job/dataset.jsonl  training data, one JSON object per line
    /out/               config.yaml, checkpoints, adapter/, result.json, train.log

Design notes worth keeping, because each is a decision:

* **Axolotl owns the training loop, we own the contract.** We render YAML and
  invoke `axolotl train`. We do not call TRL/PEFT directly ΓÇö spike 3 showed
  those APIs move underneath you (TRL 1.x dropped `warmup_ratio` from
  SFTConfig), while Axolotl's config surface stayed stable and still exposes it.
* **Correctness settings are not user-settable.** Chat template resolution,
  EOS handling and loss masking are the highest-frequency silent-failure
  surface: they pass every obvious health check and only show up as garbage
  generations. They are decided here, not exposed.
* **The trainer resolves nothing.** Hyperparameters arrive in the job spec
  already resolved by the control plane (#83); this entrypoint applies what it
  is given, refuses a spec missing required values (`spec_incomplete`), and
  echoes unknown keys back rather than dropping them. Two resolvers that could
  disagree meant the one that ran was the one nobody could see.
* **Everything is reported, including failure.** result.json is always written,
  even on a crash, so the orchestrator never has to parse logs to find out what
  happened.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import capability as capability_step
import checkpoints as checkpoint_upload
import comparison as comparison_step
from checkpoint import select_best_checkpoint
from delivery import DeliveryFailure, run_delivery
from split import held_out_split
from template_probe import (
    PROBE_UNAVAILABLE_CODE,
    ProbeOutcome,
    TemplateProbeFailure,
    resolve_template,
    run_probe,
)
from thinking import MixedThinkingDataset
from thinking import detect as detect_thinking


def _contract_path() -> Path:
    """Find the single definition of the trainer's defaults.

    Sibling first: in the image this file sits beside trainer-defaults.json at
    /opt/trainer (copied by the Dockerfile), and in the repo it sits in
    apps/trainer. Falls back to the workspace tree so an import under the test
    suite resolves without the image. The trainer never installs
    packages/core (ADR-0010), so it reads the data, not the resolver module.
    """
    sibling = Path(__file__).resolve().parent / "trainer-defaults.json"
    if sibling.is_file():
        return sibling
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "packages" / "contracts" / "trainer-defaults.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "packages/contracts/trainer-defaults.json not found beside this file "
        "or anywhere in the workspace tree"
    )


def _schema_path() -> Path:
    """Find the pinned image's configuration schema snapshot, sibling or tree.

    Since #33 the trainer's known-key set is the schema of the pinned image
    (issue #33): a key unknown to the trainer is refused loudly and echoed
    back, and 'unknown' means unknown to the trainer, not absent from a
    hand-written list. The snapshot ships beside this file in the image, the
    same way trainer-defaults.json does, and falls back to the workspace tree
    so the test suite resolves it without the image.
    """
    sibling = Path(__file__).resolve().parent / "axolotl-schema.json"
    if sibling.is_file():
        return sibling
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "packages" / "contracts" / "axolotl-schema.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "packages/contracts/axolotl-schema.json not found beside this file "
        "or anywhere in the workspace tree"
    )


JOB_DIR = Path(os.environ.get("JOB_DIR", "/job"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/out"))
CONFIG = OUT_DIR / "config.yaml"
RESULT = OUT_DIR / "result.json"
LOG = OUT_DIR / "train.log"

# The environment variable this trainer reads its fault from (issue #24).
# Trainer-side faults are an environment switch, by the spec's constraint; the
# control plane writes the job's fault spec into this variable on the machine
# and this process reads it, so the fault reaches the trainer without any new
# seam. It is off by default: an unset or empty value means no fault. The name
# is pinned equal to `temper_core.faults.FAULT_ENV` by a test, because the
# trainer image cannot import that package (ADR-0010).
FAULT_ENV = "TEMPER_FAULT_SPEC"

# The run's seed, used once: it fixes both the model's weight init and the
# held-out split (issue #53), so the same job spec reproduces the same run and
# the same split. One value, read twice -- not two seeds that could disagree.
TRAIN_SEED = 42

# The trainer resolves nothing (#83): the control plane applies its resolver to
# the user's overrides before launch and writes the full resolved set into the
# job spec, so there is one resolver, the visible one. What remains here is the
# guard: the keys this entrypoint enforces are not declared here. They are read
# from the single definition in packages/contracts/trainer-defaults.json (#82),
# copied beside this file at build time and resolved through the same data by the
# control-plane resolver. A default added there without the trainer learning it
# would fail this guard before the GPU does any work -- training on a number
# nobody chose is worse than not training -- and the equality is pinned by
# apps/trainer/tests/test_agreement_with_the_domain.py, so the two cannot
# drift. `lora_use_rslora` is in the resolved set because the resolver infers it
# at rank >= 32 rather than carrying it in the data.
#
# The KNOWN set -- which keys this trainer will accept rather than echo back as
# rejected -- is the schema of the pinned image itself (issue #33): a key
# unknown to the trainer is refused loudly and echoed back, and 'unknown' means
# unknown to the trainer, not absent from a hand-written list. The schema
# snapshot ships beside this file, and `lora_use_rslora` is the one derived
# value the resolver adds that the schema does not carry.
_CONTRACT = json.loads(_contract_path().read_text(encoding="utf-8"))
_SCHEMA = json.loads(_schema_path().read_text(encoding="utf-8"))
REQUIRED_HYPERPARAMETERS = set(_CONTRACT["defaults"]) | {"lora_use_rslora"}
KNOWN_HYPERPARAMETERS = {f["name"] for f in _SCHEMA["fields"]} | {
    "lora_use_rslora"
}


def _fault_contract_path() -> Path:
    """Find the fault surface's vocabulary (issue #24), sibling or tree.

    The same lookup as the other contracts this trainer reads: beside this
    file in the image at /opt/trainer, falling back to the workspace tree so
    the host test suite resolves it without the image. The vocabulary -- the
    six fault names, the `simulated_` code each carries and which side makes
    it real -- is data, not code, so the trainer and the control plane's fake
    provider cannot drift about what a fault is called.
    """
    sibling = Path(__file__).resolve().parent / "fault-surface.json"
    if sibling.is_file():
        return sibling
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "packages" / "contracts" / "fault-surface.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "packages/contracts/fault-surface.json not found beside this file "
        "or anywhere in the workspace tree"
    )


_FAULT_CONTRACT = json.loads(
    _fault_contract_path().read_text(encoding="utf-8")
)
_FAULTS_BY_NAME = {f["name"]: f for f in _FAULT_CONTRACT["faults"]}

# Where a deliberately exhausting allocation is parked so it outlives its
# allocator thread (issue #24); empty on every path but an active `oom` fault.
_HELD_MEMORY: list = []


def _fault_entry(name: str) -> dict:
    entry = _FAULTS_BY_NAME.get(name)
    if entry is None:
        raise ValueError(f"unknown simulated fault {name!r}")
    return entry


def _fault_spec_error(spec: dict) -> str | None:
    """A reason this fault spec is invalid, or None when it is well-formed.

    The trainer cannot import `temper_core.faults` (ADR-0010), so it runs the
    same validation the control plane runs, against the same contract data: an
    unknown fault, an unknown parameter, or a malformed parameter is refused
    loudly rather than half-honoured -- a fault spec the caller believes is in
    effect but is not is worse than a refusal.
    """
    name = spec.get("name")
    entry = _FAULTS_BY_NAME.get(name) if isinstance(name, str) else None
    if entry is None:
        return f"'{name}' is not a fault the surface knows."
    allowed = {*(entry.get("params") or []), "name"}
    unknown = sorted(k for k in spec if k not in allowed)
    if unknown:
        return (
            f"fault '{name}' does not take parameter(s) {unknown}; a "
            "parameter the caller believes is in effect but is not is worse "
            "than a refusal."
        )
    for key in ("after_line", "times", "machine_id"):
        if key in spec:
            try:
                int(spec[key])
            except (TypeError, ValueError):
                return (
                    f"fault parameter '{key}' must be a whole number, got "
                    f"{spec[key]!r}"
                )
    if "delay_s" in spec:
        try:
            delay = float(spec["delay_s"])
        except (TypeError, ValueError):
            return (
                f"fault parameter 'delay_s' must be a number, got "
                f"{spec['delay_s']!r}"
            )
        if delay <= 0:
            return (
                f"fault parameter 'delay_s' must be positive, got "
                f"{spec['delay_s']!r}"
            )
    return None


def read_fault_spec(env_value: str | None) -> dict | None:
    """The fault spec from the environment, validated, or None when off.

    Off by default is a safety property: an unset or empty `TEMPER_FAULT_SPEC`
    is the whole surface being off, and a value that is set but not a valid
    fault spec is refused loudly rather than half-honoured.
    """
    if not env_value or not env_value.strip():
        return None
    try:
        spec = json.loads(env_value)
    except json.JSONDecodeError as e:
        raise ValueError(f"{FAULT_ENV} is not JSON: {e}") from e
    if not isinstance(spec, dict):
        raise ValueError(f"{FAULT_ENV} must be an object with a 'name' string")
    problem = _fault_spec_error(spec)
    if problem is not None:
        raise ValueError(problem)
    return spec


def apply_fault(cfg: dict, spec: dict) -> None:
    """Mutate the Axolotl config so a requested trainer-side fault fires.

    Only the divergence fault touches the config: a loss driven to a
    meaningless value is made by handing the optimiser a learning rate no run
    could survive, so this trainer replaces the resolved rate before the
    config is written. It is the one place a fault overrides a resolved value
    on purpose -- the fault is the point -- and the overridden rate is
    recorded in the config the run writes, so the run's record says what
    actually trained. The other trainer-side faults act at runtime
    (`schedule_fault`), not on the config.
    """
    if spec["name"] != "divergence":
        return
    # The resolved spec always carries `learning_rate` (the trainer refuses a
    # spec missing a required value rather than falling back), so it is read,
    # not defaulted -- the trainer holds no defaults of its own.
    cfg["learning_rate"] = float(cfg["learning_rate"]) * 1e6


def schedule_fault(spec: dict) -> None:
    """Arrange the runtime half of a trainer-side fault (issue #24).

    `oom` holds almost all free device memory a little way into the run so
    the next training step genuinely fails; `worker_kill` kills this process
    as a killed worker would die, leaving no result document -- exactly the
    interruption a resumption exists to recover from. `delay_s` is the chosen
    point at which the fault fires, and it is a knob because the right value
    depends on how long this job's model takes to load.
    """
    name = spec["name"]
    if name == "oom":
        threading.Timer(float(spec["delay_s"]), _exhaust_device_memory).start()
    elif name == "worker_kill":
        threading.Timer(float(spec["delay_s"]), _kill_worker).start()


def _exhaust_device_memory() -> None:
    """Hold (almost) all free device memory, so the next step cannot allocate.

    Runs on the machine, never in the host suite: torch is provided by the
    base image. `torch.cuda.mem_get_info` reports free bytes on the current
    device; 98% leaves a little headroom so the allocation itself succeeds and
    the *training step* is what fails -- the shape a genuine out-of-memory
    has. The tensor is handed to a module-level holder so it stays alive for
    the life of the process, which is the point.
    """
    import torch  # base image only (this trainer ships no dependencies)

    if not torch.cuda.is_available():
        return
    free, _total = torch.cuda.mem_get_info()
    try:
        blob = torch.empty(int(free * 0.98), dtype=torch.uint8, device="cuda")
    except RuntimeError:
        return
    _HELD_MEMORY.append(blob)


def _kill_worker() -> None:
    """Kill this process as a killed worker would die (issue #24).

    No result document survives a SIGKILL -- the orchestrator sees a stream
    that ended without a result, which is exactly what an interruption looks
    like, and what a resumption would recover from.
    """
    import os as _os
    import signal

    _os.kill(_os.getpid(), signal.SIGKILL)


# Issue #35: the stable code the trainer reports for a genuine device
# out-of-memory failure. The trainer image cannot import `temper_core`
# (ADR-0010), so the value is pinned equal to
# `temper_core.memory_retry.REAL_OOM_CODE` by a trainer test -- the same
# FAULT_ENV pattern that keeps the wire name of the fault surface defined
# once. The control plane reads either this code (a real exhaustion) or the
# fault surface's own `simulated_oom` (a deliberately caused one) and retries
# with the effective batch preserved.
TRAINER_OOM_CODE = "training_oom"

# A device-memory exhaustion is named by the lines the framework leaves in
# the failed run's output: torch says "CUDA out of memory" (or the CUDA
# runtime reports "out of memory"). Matched case-insensitively so a framework
# capitalisation change cannot miss it, and only when a CUDA context is
# named -- a host that ran out of RAM is not a memory *retry*: halving the
# per-step batch does not help the host.
_OOM_MARKERS = ("cuda out of memory", "out of memory")


def is_oom_tail(line: str) -> bool:
    """Whether `line` is a device out-of-memory failure line.

    CPU out-of-memory (`MemoryError`, "out of memory" without a CUDA context)
    is not something the memory recovery can fix -- halving the batch does not
    add RAM -- so only a CUDA/device exhaustion is named as one.
    """
    low = line.lower()
    return "cuda" in low and any(marker in low for marker in _OOM_MARKERS)


def oom_error_code(tail: Sequence[str], fault_spec: dict | None) -> str | None:
    """The error code a failed run's tail names, or None when it did not run
    out of device memory.

    A run broken deliberately by the `oom` fault keeps the fault surface's own
    `simulated_oom` code -- a deliberately broken run is never mistaken for a
    real one (ADR-0051), and the vocabulary is the contract data this trainer
    already reads. A genuine exhaustion carries the platform's real code, so
    the control plane can tell the two apart while retrying both.
    """
    if not any(is_oom_tail(line) for line in tail):
        return None
    if fault_spec is not None and fault_spec.get("name") == "oom":
        return _fault_entry("oom")["code"]
    return TRAINER_OOM_CODE


class GracefulStop(Exception):
    """Raised by the SIGTERM handler so an ordered stop finalizes the run.

    The spend ceiling's emergency checkpoint (issue #46) asks this trainer to
    save a final checkpoint and exit. The request arrives as a SIGTERM to the
    container, and Python's default SIGTERM behaviour -- die immediately --
    would throw away exactly the checkpoints the request exists to save. This
    handler converts the signal into the graceful path: the `finally` in
    `main` runs, shipping the final checkpoint and writing result.json, and
    the control plane reads the manifest from there.
    """


def _on_sigterm(signum, frame) -> None:
    """Convert SIGTERM into a graceful, finalizing stop (issue #46).

    The spend ceiling's emergency checkpoint signals the container this way;
    raising from the handler is how a signal becomes the run's own exception
    path rather than a silent death. The narration goes to the stream so the
    run's history says what happened.
    """
    log("SIGTERM received, saving a final checkpoint and exiting")
    raise GracefulStop()


# Top-level keys the job spec may carry. Spike 4 caught a real hole here: an
# unknown key at the TOP level passed silently because only `hyperparameters`
# was being validated. A caller who misspells `max_steps` as `maxSteps` would
# have got a full-length training run and no warning. Keys starting with
# "_comment" are documentation and ignored on purpose.
ALLOWED_JOB_KEYS = {
    "job_id",
    "base_model",
    "base_revision",
    "messages_field",
    # Issue #66: the training method the control plane selected at
    # provisioning. The trainer applies the spec's method (adapter or full
    # fine-tune) and does not choose one; a spec without it is a pre-method
    # row and reads as qlora, the only method that ever ran before the column
    # existed -- the same reading `temper_core.artifacts.kind_for` gives a
    # missing method.
    "method",
    "hyperparameters",
    "max_steps",
    "save_steps",
    "resume_from_checkpoint",
    # ADR-0009: the scoped write URL this machine may put its artifact to,
    # minted by the control plane and expiring with the job. Optional -- a
    # standalone run carries no grant and simply leaves the artifact on /out.
    "artifact_upload",
    # Issue #37: the scoped write URLs this machine may put its checkpoints to,
    # one per retention slot, minted by the control plane and expiring with the
    # job. Optional -- a standalone run carries no grants and simply leaves its
    # checkpoints on /out, exactly as it leaves the artifact.
    "checkpoint_grants",
    # Issue #74: the delivery formats this job asked for (the canonical
    # artifact plus optional merged/quantised forms), and the scoped write URLs
    # for the produced formats, one per format, minted by the control plane and
    # expiring with the job -- the same ADR-0009 machinery the artifact and
    # checkpoint grants use. Optional: a run that asked for no extra formats
    # carries neither key.
    "delivery",
    "delivery_grants",
}

# The hyperparameters that mean something only to an adapter run. They are
# part of the one resolved spec every method carries (issue #66), but a full
# fine-tune's config must not claim a rank or a scale it does not use --
# writing them into YAML names values Axolotl would not act on.
ADAPTER_ONLY_HYPERPARAMETERS = (
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_use_rslora",
)


class IncompleteJobSpec(ValueError):
    """The job spec omitted values the trainer refuses to invent."""


def log(msg: str) -> None:
    print(f"[trainer] {msg}", flush=True)


def write_result(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """One JSON object per line, faithful to the source's characters.

    `ensure_ascii=False` so a conversation with non-ASCII text round-trips to
    the same characters it arrived as: escaping to `\\uXXXX` is semantically
    identical after decoding, but the bytes should not be changed for no
    reason on the machine that trains on them.
    """
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_held_out_split(
    parsed: list[dict],
    job: dict,
    out_dir: Path,
) -> tuple[Path | None, dict]:
    """Dedup and split `parsed` (issue #53), write the train/eval files.

    Returns (eval path, or None when nothing was held out, and the recorded
    split). Deduplication runs before the split, so a duplicated row can never
    land on both sides; the fraction is the effective `val_set_size` (the
    product's chosen eval proportion, resolved by the control plane) and the
    seed is the run's own, so the split is reproducible from the record. The
    held-out rows are written to a separate file and never enter the training
    file, so the two cannot overlap.
    """
    hp = spec_hyperparameters(job)
    train_rows, held_out_rows, record = held_out_split(
        parsed,
        fraction=float(hp["val_set_size"]),
        seed=TRAIN_SEED,
        messages_field=job.get("messages_field", "messages"),
    )
    write_jsonl(out_dir / "train.jsonl", train_rows)
    eval_path = None
    if held_out_rows:
        eval_path = out_dir / "eval.jsonl"
        write_jsonl(eval_path, held_out_rows)
    return eval_path, record.to_dict()


# How much of a failed job's output is carried back inside result.json. The
# lines were already streamed; this is so the orchestrator never has to
# reassemble a failure from the event log.
TAIL_LINES = 40

# A line ends at a newline OR a carriage return; see iter_output_lines.
RB_LINE_END = re.compile(rb"[\r\n]")


def iter_output_lines(
    stream: IO[bytes], chunk_size: int = 4096
) -> Iterator[str]:
    """Yield the framework's output a line at a time, as it is produced.

    Not `for line in stream`, for two reasons:

    * **Progress bars never send a newline.** tqdm ΓÇö which transformers uses
      for every epoch ΓÇö redraws with a carriage return. A reader that waits
      for `\n` sees nothing for the whole length of a bar, which on a training
      phase that *is* one bar is indistinguishable from the silence this
      channel exists to remove. So `\r` ends a line here too.
    * **Iterating a pipe in text mode buffers.** Python reads ahead, so lines
      arrive in blocks rather than as they happen.

    Blank lines are dropped: each one would otherwise become an event, and a
    framework that prints a spacer between epochs would triple the log.
    """
    buf = b""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        buf += chunk
        parts = re.split(RB_LINE_END, buf)
        buf = parts.pop()  # whatever follows the last terminator
        for part in parts:
            text = part.decode("utf-8", "replace").strip()
            if text:
                yield text
    tail = buf.decode("utf-8", "replace").strip()
    if tail:
        yield tail


def run_streaming(cmd: list[str]) -> tuple[int, list[str]]:
    """Run `cmd`, relaying its output to stdout as it arrives.

    Returns its exit code and the tail of what it said.

    Three things here are load-bearing, and dropping any one of them
    reintroduces the silence:

    * `bufsize=0` so the pipe is read raw ΓÇö a buffered reader waits to fill a
      block before handing us anything.
    * `PYTHONUNBUFFERED` in the child's environment, because the child is
      itself Python and will otherwise buffer its own stdout when it sees a
      pipe rather than a terminal. This is the innermost of the redirections;
      the two outside it are worthless if this one holds output back.
    * `flush=True` on every relayed line, for the same reason one layer up.

    train.log is still written. It costs nothing, and it is the only copy that
    survives on the machine if the stream itself breaks.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8", errors="replace") as lf:
        for line in iter_output_lines(proc.stdout):
            print(line, flush=True)
            lf.write(line + "\n")
            lf.flush()
            tail.append(line)
    return proc.wait(), list(tail)


def prefetch_model(job: dict) -> None:
    """Download the base model before training, reporting progress (issue #49).

    The model download is the phase that dominates a large job, and it must be
    *invoked* in a mode that reports progress: axolotl downloads the model
    itself during training, but whether that download's bars reach the control
    plane is left to tqdm's own TTY detection, and piped output is exactly
    where bars disappear. Downloading here first -- with progress bars forced
    on, so a non-TTY cannot silence them -- makes the phase visible, and
    axolotl then resolves the model from the HF cache instead of downloading
    again.

    The import is inside the function because this trainer ships no
    dependencies of its own (ADR-0010): `huggingface_hub` and `tqdm` are
    provided by the base image (both are transformers dependencies), and the
    host test suite may not have them. A prefetch that cannot run is not a
    reason to fail the job: training proceeds as it always has, and axolotl
    downloads the model itself -- the failure loses a progress signal, never a
    run.
    """
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from tqdm.std import tqdm as _tqdm  # noqa: PLC0415

        def visible(*args, **kwargs):
            # Never auto-disable on a non-TTY: the whole point is that piped
            # output reports progress, and tqdm's default `disable=None`
            # suppresses bars exactly there.
            kwargs["disable"] = False
            return _tqdm(*args, **kwargs)

        snapshot_download(
            repo_id=job["base_model"],
            revision=job.get("base_revision"),
            tqdm_class=visible,
        )
        log("base model downloaded")
    except Exception as e:  # noqa: BLE001 - see the docstring
        log(
            f"base model prefetch failed; training will download as usual: {e}"
        )


# --- peak VRAM, measured during training (issue #77) --------------------------
# The prediction (temper_core.memory) counts in decimal GB (BYTES_PER_GB =
# 1e9), so the measurement is converted the same way -- one convention for GB,
# never two. nvidia-smi reports used memory per device in MiB; the maximum
# across samples and devices is the figure issue #77 records against the
# prediction, converted with the same 1e9 byte count the predictor used.
_MIB_PER_GB = 1024 * 1024 / 1e9
_SAMPLE_INTERVAL_S = 2.0
_NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,memory.used",
    "--format=csv,noheader,nounits",
]


def max_vram_gb_from_smi(output_lines: Iterable[str]) -> float | None:
    """The peak used VRAM, in decimal GB, across nvidia-smi's per-GPU lines.

    Pure over the lines so it is testable without a GPU: the same parse the
    sampler runs against real output runs against a test's canned lines. A
    line that is not the "index, MiB" shape is skipped rather than failing
    the measurement -- the query is ours, but the machine is someone else's.
    """
    peak: float | None = None
    for line in output_lines:
        fields = line.strip().replace(" MiB", "").split(",")
        if len(fields) < 2:
            continue
        try:
            used_mib = int(fields[1].strip())
        except ValueError:
            continue
        used_gb = used_mib * _MIB_PER_GB
        if peak is None or used_gb > peak:
            peak = used_gb
    return peak


class VramSampler:
    """Sample the machine's per-GPU used VRAM during training, keep the peak.

    A daemon thread so the training stream is never blocked on the sampling;
    the peak is read once after training ends and recorded in result.json.

    Why nvidia-smi and not torch: axolotl runs as a subprocess, so this
    process never allocates on the GPU, and `torch.cuda.max_memory_allocated`
    here would always be zero. nvidia-smi reports *used* memory per device,
    which includes the training process -- the same method the spikes
    measured the 5.31 GB anchor with (spike 6, `bootstrap6.sh`). A machine
    without nvidia-smi (or a container that cannot see it) measures nothing,
    honestly: the result simply carries no peak.
    """

    def __init__(self, interval_s: float = _SAMPLE_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._peak_gb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vram-sampler"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(self._interval_s * 2, 5.0))

    def peak_gb(self) -> float | None:
        return self._peak_gb

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    _NVIDIA_SMI_QUERY,
                    capture_output=True,
                    text=True,
                    timeout=self._interval_s,
                )
            except (OSError, subprocess.SubprocessError):
                return
            if result.returncode != 0:
                return
            peak = max_vram_gb_from_smi(result.stdout.splitlines())
            if peak is not None and (
                self._peak_gb is None or peak > self._peak_gb
            ):
                self._peak_gb = peak
            self._stop.wait(self._interval_s)


def spec_hyperparameters(job: dict) -> dict:
    """The hyperparameters the job spec carries, or a loud refusal.

    The control plane resolves every value before launch, so anything missing
    here is a hole in the spec, not an occasion to pick a number. Naming the
    missing keys is the difference between a fixable refusal and a guess.
    """
    hp = job.get("hyperparameters")
    if not isinstance(hp, dict) or not hp:
        raise IncompleteJobSpec(
            "job specification carries no hyperparameters; the trainer "
            "resolves nothing and was given nothing"
        )
    missing = sorted(k for k in REQUIRED_HYPERPARAMETERS if k not in hp)
    if missing:
        raise IncompleteJobSpec(
            "job specification is missing required hyperparameters "
            f"{missing}; the trainer resolves nothing and will not fall "
            "back to defaults it no longer has"
        )
    return hp


def build_config(
    job: dict, enable_thinking: bool = False, eval_path: Path | None = None
) -> tuple[dict, dict]:
    """Job spec -> Axolotl config. Returns (config, rejected).

    Every hyperparameter is read from the spec exactly as given. Derivations
    that used to live here -- alpha tracking rank, rsLoRA above rank 32 --
    happen in the control plane's resolver before launch; redoing them would
    be the second resolver again.

    `eval_path`, when supplied, is the platform's own held-out split (issue
    #53): the training file is configured with no further internal split and
    the held-out file is given to Axolotl as its test dataset, so the rows
    never overlap -- they are different files.
    """
    hp = spec_hyperparameters(job)
    # The method the control plane selected at provisioning (issue #66). A
    # spec written before the method key existed reads as qlora -- the only
    # method that ever ran before it -- the same reading
    # `temper_core.artifacts.kind_for` gives a missing method. But unlike
    # `kind_for`, which refuses an unknown method, an unknown string here
    # would silently run qlora -- a run that lies about what it did -- so it
    # is refused loudly instead, exactly as an unknown key is.
    method = job.get("method") or "qlora"
    if method not in ("qlora", "full"):
        raise IncompleteJobSpec(
            f"job specification carries unknown method {method!r}; the "
            "trainer executes 'qlora' and 'full' only. A spec without a "
            "method reads as qlora; anything else is refused rather than "
            "silently run as qlora."
        )
    rejected: dict = {}

    # Validate the top level before anything else.
    unknown_top = sorted(
        k
        for k in job
        if k not in ALLOWED_JOB_KEYS and not k.startswith("_comment")
    )
    for k in unknown_top:
        rejected[k] = job[k]
    unknown_hp = sorted(k for k in hp if k not in KNOWN_HYPERPARAMETERS)
    for k in unknown_hp:
        rejected[k] = hp[k]

    cfg = {k: v for k, v in hp.items() if k in KNOWN_HYPERPARAMETERS}
    if method == "full":
        # A full fine-tune configures no adapter: the rank, scale, dropout and
        # rsLoRA flags are still in the resolved spec, but they name nothing
        # Axolotl would act on without an `adapter`, so they are dropped from
        # the config rather than written as claims that do nothing.
        for k in ADAPTER_ONLY_HYPERPARAMETERS:
            cfg.pop(k, None)

    cfg.update(
        {
            "base_model": job["base_model"],
            "output_dir": str(OUT_DIR / "run"),
            # --- precision: bf16, same exponent range as fp32, no loss scaler ----
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": True,
            "flash_attention": True,
            "seed": TRAIN_SEED,
            # --- data ------------------------------------------------------------
            "datasets": [
                {
                    "path": str(OUT_DIR / "train.jsonl"),
                    "type": "chat_template",
                    "field_messages": job.get("messages_field", "messages"),
                }
            ],
            # Let the model's own template decide. Hand-writing role delimiters is
            # the modal production bug in this category.
            "chat_template": "tokenizer_default",
            # Detected from the dataset, never guessed and never exposed. Qwen3
            # emits  thinking blocks through its template by default; training data
            # without them under that template is a silent train/infer mismatch.
            # The SAME value must be applied at serving.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            # Train on the assistant turn only. Full-sequence loss on instruction
            # data teaches the model to recite prompts back.
            "train_on_inputs": False,
            # Packing is a throughput win but needs varlen attention to avoid
            # cross-example contamination. Off until measured per model.
            "sample_packing": False,
            "logging_steps": 1,
            "save_safetensors": True,  # never torch .bin ΓÇö see spike 3 / C14
            # save_total_limit is no longer set here: it now arrives resolved
            # in `hp`, from the same contract `temper_core.disk` reads to
            # size the machine's disk (issue #64) -- a value both must agree
            # on is defined once, in packages/contracts/trainer-defaults.json.
        }
    )

    if method == "full":
        # A full fine-tune trains every weight in bf16: no PEFT adapter and
        # no 4-bit quantisation. Explicit rather than omitted, so the config
        # cannot silently drift into an adapter shape the method never asked
        # for.
        cfg["load_in_4bit"] = False
    else:
        # --- method: QLoRA. NF4 + double-quant, bf16 compute -------------------
        # This said "adapters in bf16" and that was wrong: measured from the
        # first real run's safetensors header, all 504 adapter tensors are F32,
        # which is why the artifact is 132 MB rather than ~66 MB. bf16 is the
        # COMPUTE dtype; trainable parameters are kept in fp32 under 4-bit
        # quantisation, which is standard and correct. The claim was the bug,
        # not the behaviour.
        cfg.update(
            {
                "adapter": "qlora",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                # --- targets: ALL linear, not attention-only ---------------------
                # The MLP is ~78% of every transformer block's parameters, so
                # attention-only leaves most of the model untouched at any rank.
                "lora_target_linear": True,
            }
        )

    if job.get("max_steps"):
        cfg["max_steps"] = int(job["max_steps"])
    if job.get("save_steps"):
        cfg["save_steps"] = int(job["save_steps"])
    if job.get("resume_from_checkpoint"):
        cfg["resume_from_checkpoint"] = job["resume_from_checkpoint"]

    # The split is the platform's (issue #53): the held-out file arrives as
    # `eval_path` and is configured as Axolotl's test dataset, so Axolotl
    # must not carve a second split out of the training file. `val_set_size`
    # is set to zero rather than left to the value in the spec (the fraction
    # our own split already used) because Axolotl accepts either
    # `test_datasets` or a `val_set_size` split, not both.
    cfg["val_set_size"] = 0.0
    # The eval cadence is pinned rather than left to a default: criterion 4
    # and the plateau (issue #53) both need held-out loss measured *during*
    # the run, and relying on Axolotl's unpinned default would make that a
    # hope. Evaluating at each epoch end gives the chart and the plateau
    # their points.
    cfg["eval_strategy"] = "epoch"
    if eval_path is not None:
        cfg["test_datasets"] = [
            {
                "path": str(eval_path),
                "type": "chat_template",
                "field_messages": job.get("messages_field", "messages"),
            }
        ]

    return cfg, rejected


def _checkpoint_step(path: Path) -> int:
    """Step number from a `checkpoint-N` directory name, -1 if unparseable."""
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


# The top-level name of the archive a full fine-tune's artifact is tarred
# into. The archive is what the machine uploads through the one scoped grant
# (ADR-0009): a whole model is a set of files -- config, weight shards,
# tokenizer -- that cannot be addressed by a single pre-minted key the way an
# adapter's two fixed files can, so they travel as one streamed archive and
# the control plane verifies it against this one checksum, exactly as it does
# an adapter.
FULL_MODEL_ARCHIVE = "model.tar.gz"

# Files written beside a full model's weights that are training state, not
# part of a loadable model. Shipped into the artifact they would bloat the
# download with bytes `from_pretrained` ignores; the exact member set a real
# full fine-tune writes is one of the things the hardware verification
# (issue #66) confirms, so this denylist is an assumption, not a measured list.
FULL_MODEL_STATE_NAMES = {"optimizer.pt", "scheduler.pt", "trainer_state.json"}
FULL_MODEL_STATE_SUFFIXES = (".pt",)


def _holds_model(d: Path) -> bool:
    """Whether `d` is a trained model directory: a config beside a weights
    file, single (`model.safetensors` / `pytorch_model.bin`) or sharded
    (`model-00001-of-N.safetensors`)."""
    if not (d / "config.json").is_file():
        return False
    if any(
        (d / cand).is_file()
        for cand in ("model.safetensors", "pytorch_model.bin")
    ):
        return True
    return any(d.glob("model-*.safetensors"))


def collect_artifacts(method: str) -> dict:
    """Find what training produced, and fingerprint it.

    Which artifact gets shipped is not a detail: it is the entire deliverable,
    and "which weights did I actually download?" is a question the product has
    to answer exactly. Two ways to get it wrong were both present in the
    original adapter search:

    * `sorted(rglob(...))[-1]` sorts lexicographically, so once a run produces
      ten checkpoints it picks `checkpoint-9` over `checkpoint-10`.
    * It also preferred a checkpoint over the final artifact Axolotl writes at
      the top of the output directory at the end of training, because
      `run/checkpoint-N/...` sorts after `run/adapter_model.safetensors`.

    The rule is explicit instead: the end-of-training artifact wins; failing
    that, the numerically highest checkpoint. `artifact_source` records which,
    so the answer is in result.json rather than inferred. The method picks the
    shape (issue #66): an adapter is the weights file plus its config; a full
    fine-tune is a whole model directory, shipped as one streamed archive
    because the machine writes one object through one grant.
    """
    run = OUT_DIR / "run"
    ckpt_dirs = sorted(
        (p for p in run.glob("checkpoint-*") if p.is_dir()),
        key=_checkpoint_step,
    )
    info: dict = {"checkpoints": [p.name for p in ckpt_dirs]}
    if method == "full":
        return _collect_full_model(run, ckpt_dirs, info)
    return _collect_adapter(run, ckpt_dirs, info)


def _collect_adapter(run: Path, ckpt_dirs: list[Path], info: dict) -> dict:
    """The adapter-shaped artifact: the weights file and the config beside it.

    Search order, most authoritative first: the final adapter, then
    checkpoints from the highest step down.
    """
    search_dirs = [run, *reversed(ckpt_dirs)]
    adapter = None
    for d in search_dirs:
        for cand in ("adapter_model.safetensors", "adapter_model.bin"):
            if (d / cand).is_file():
                adapter = d / cand
                break
        if adapter:
            break

    if adapter:
        # Hashed in bounded blocks, not read whole: an artifact can be
        # arbitrarily large, and nothing here needs to hold it.
        digest = hashlib.sha256()
        with adapter.open("rb") as f:
            while chunk := f.read(1 << 20):
                digest.update(chunk)
        info.update(
            {
                "artifact_path": str(adapter.relative_to(OUT_DIR)),
                "artifact_bytes": adapter.stat().st_size,
                "artifact_sha256": digest.hexdigest(),
                "artifact_format": adapter.suffix.lstrip("."),
                "artifact_source": (
                    "final" if adapter.parent == run else adapter.parent.name
                ),
            }
        )
        # The config that sits WITH the chosen adapter, not whichever one
        # happened to sort last. A LoRA is not loadable without it, so the
        # orchestrator ships it alongside the weights.
        cfg_path = adapter.parent / "adapter_config.json"
        if cfg_path.is_file():
            info["adapter_config"] = json.loads(cfg_path.read_text())
    return info


def _collect_full_model(run: Path, ckpt_dirs: list[Path], info: dict) -> dict:
    """The full-model artifact: the trained model's directory, tarred.

    Search order matches the adapter's: the final model at the top of the
    output directory wins; failing that, the highest checkpoint. Only the
    chosen directory's top-level files are shipped -- checkpoint
    sub-directories would double the download with the run's whole history,
    and only the one model is the deliverable.

    The archive is written streaming (tarfile streams each member in blocks)
    and hashed after in bounded blocks, so a whole model that is larger than
    any adapter is still never held whole here (the flat-memory rule that
    already guards the artifact path).
    """
    search_dirs = [run, *reversed(ckpt_dirs)]
    model_dir = next((d for d in search_dirs if _holds_model(d)), None)
    if model_dir is None:
        return info

    members = [
        p
        for p in sorted(model_dir.iterdir())
        if p.is_file()
        and p.name not in FULL_MODEL_STATE_NAMES
        and not p.name.endswith(FULL_MODEL_STATE_SUFFIXES)
    ]
    tar_path = OUT_DIR / FULL_MODEL_ARCHIVE
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in members:
            # One top-level `model/` folder in the archive, so extraction
            # produces a loadable directory rather than dumping the files
            # into whatever folder the user extracted into.
            tar.add(path, arcname=f"model/{path.name}")
    digest = hashlib.sha256()
    with tar_path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    info.update(
        {
            "artifact_path": FULL_MODEL_ARCHIVE,
            "artifact_bytes": tar_path.stat().st_size,
            "artifact_sha256": digest.hexdigest(),
            "artifact_format": "tar.gz",
            "artifact_source": (
                "final" if model_dir == run else model_dir.name
            ),
            "artifact_members": [p.name for p in members],
        }
    )
    return info


def upload_artifact(url: str, path: Path) -> dict:
    """PUT the artifact to the scoped write URL (ADR-0009).

    The URL is already an authorisation -- the machine holds no credential that
    outlives the job -- so a plain HTTPS PUT is the whole protocol. The body is
    a file object with a declared Content-Length: urllib streams a file-like
    body in blocks, so an arbitrarily large artifact is never held whole here,
    and the explicit length keeps the request a plain PUT rather than a chunked
    one (a pre-signed S3 URL is not signed for aws-chunked). No timeout is set
    on purpose: a very large artifact may take minutes, and a wedged upload is
    already caught by the job's stall detector rather than by a guess here.

    Returns {"ok": True, "bytes": n} or {"ok": False, "error": ...} so the
    caller records the outcome in result.json without the upload crashing the
    job before it is reported. The control plane verifies what landed against
    the checksum this trainer reports (ADR-0009 point 5); it does not take the
    machine's word that the bytes arrived.
    """
    import urllib.request

    size = path.stat().st_size
    try:
        with path.open("rb") as body:
            # The URL is minted by the control plane for exactly one object;
            # this is the machine's only egress on this path (spike 2).
            request = urllib.request.Request(  # noqa: S310
                url,
                data=body,
                method="PUT",
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(size),
                },
            )
            with urllib.request.urlopen(request):  # noqa: S310
                pass
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "bytes": size}


def load_probe_tokenizer(job: dict):
    """The tokenizer the probe compares both templates through.

    transformers is provided by the base image, so the import is deliberately
    inside this function: this trainer ships no dependencies of its own
    (ADR-0010), and the host test suite never has transformers installed. The
    model was downloaded during training, so from_pretrained resolves from the
    HF cache rather than the network.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415 - base image only

    return AutoTokenizer.from_pretrained(
        job["base_model"], revision=job.get("base_revision")
    )


def probe_export(job: dict, cfg: dict, tokenizer) -> ProbeOutcome:
    """Run the export-time template probe (Spec 009 / issue #59).

    The training side is what Axolotl applied: the tokenizer's own template
    when `chat_template` is `tokenizer_default`, otherwise the directive as
    given. The artifact's serialised side is what the job records as chosen
    -- where #80's override surface writes its result -- falling back to the
    config when nothing was recorded. Two sources, deliberately: a recorded
    override that training never applied (or the reverse) is a real divergence
    the probe can catch, not a value compared with itself. Identical token ids
    are required; the probe runs on every export, whether or not anything was
    overridden.

    The probe's power to catch a divergence is proven by the deliberately
    mismatched template in tests -- Spec 009's testing decision is that a probe
    with no failing test has no evidence of working -- and it is load-bearing
    once #80 lands, when a recorded override can genuinely disagree with what
    trained.
    """
    training_tpl = cfg.get("chat_template", "tokenizer_default")
    training_kwargs = cfg.get("chat_template_kwargs") or {}
    training_template, training_kwargs = resolve_template(
        tokenizer, chat_template=training_tpl, kwargs=training_kwargs
    )

    recorded = job.get("hyperparameters") or {}
    serialised_tpl = recorded.get("chat_template", training_tpl)
    serialised_kwargs = recorded.get("chat_template_kwargs") or training_kwargs
    serialised_template, serialised_kwargs = resolve_template(
        tokenizer, chat_template=serialised_tpl, kwargs=serialised_kwargs
    )
    return run_probe(
        tokenizer,
        training_template=training_template,
        training_kwargs=training_kwargs,
        serialised_template=serialised_template,
        serialised_kwargs=serialised_kwargs,
    )


def checkpoint_records(out_dir: Path) -> list[dict]:
    """The machine's completed checkpoints as selection records.

    Each carries its step and, where the step recorded one, its training and
    held-out losses -- the same shape the control plane's
    `select_best_checkpoint` reads (issue #62), so the comparison compares the
    model selection actually chose, not the last one written. The two
    components use the same pure rule because this module ships the same file
    the control plane imports (ADR-0010).
    """
    run = out_dir / "run"
    if not run.is_dir():
        return []
    records = []
    for path in sorted(
        (p for p in run.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: checkpoint_upload.checkpoint_step(p) or 0,
    ):
        if not checkpoint_upload.is_complete(path):
            continue
        step = checkpoint_upload.checkpoint_step(path)
        train_loss, held_out = checkpoint_upload.checkpoint_losses(path)
        record: dict = {"step": step}
        if train_loss is not None:
            record["loss"] = train_loss
        if held_out is not None:
            record["held_out_loss"] = held_out
        records.append(record)
    return records


class _ModelGenerator:
    """Render a conversation and generate with one loaded model.

    The decoding settings are the fixed constant from `comparison.py`,
    captured at construction: the comparison is made under exactly the
    settings that are recorded on the result, so a reader can tell whether
    two outputs are comparable. The tokenizer renders through the same
    template the run trained with (`tokenizer_default` plus the detected
    thinking mode), so what the models answer is the same question the
    training rows asked.
    """

    def __init__(
        self,
        model,
        tokenizer,
        chat_template,
        chat_template_kwargs,
        decoding,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        # The same resolution the export-time probe applies (issue #59): the
        # `tokenizer_default` directive becomes the tokenizer's own concrete
        # template, so rendering and the artifact's record cannot disagree.
        self._template, self._kwargs = resolve_template(
            tokenizer,
            chat_template=chat_template,
            kwargs=chat_template_kwargs,
        )
        self._decoding = dict(decoding)

    def generate(self, conversation) -> str:
        prompt = self._tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            chat_template=self._template,
            chat_template_kwargs=self._kwargs,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            self._model.device
        )
        output = self._model.generate(**inputs, **self._decoding)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)


def load_comparison_generators(
    job: dict,
    cfg: dict,
    chosen_dir: Path,
    method: str,
    decoding: dict = comparison_step.COMPARISON_DECODING,
):
    """Load the base model and the chosen checkpoint, ready to generate.

    Runs on the machine, never in the host suite: torch, transformers and
    peft are provided by the base image, so the imports are deliberately
    inside the function (ADR-0010 -- this trainer ships no dependencies of
    its own). The base model resolves from the HF cache the run already
    warmed (issue #49's prefetch), and the tuned side is the base plus the
    chosen checkpoint's adapter (a QLoRA run) or the checkpoint directory
    itself (a full fine-tune). The adapter was trained on a 4-bit base;
    applying it to a bf16 base is the standard inference setup and keeps the
    whole comparison on one dtype.
    """
    import torch  # base image only
    from peft import PeftModel  # base image only
    from transformers import (  # base image only  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    base_repo = job["base_model"]
    revision = job.get("base_revision")
    chat_template = cfg.get("chat_template", "tokenizer_default")
    chat_template_kwargs = cfg.get("chat_template_kwargs") or {}

    tokenizer = AutoTokenizer.from_pretrained(base_repo, revision=revision)
    base = AutoModelForCausalLM.from_pretrained(
        base_repo,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if method == "full":
        tuned = AutoModelForCausalLM.from_pretrained(
            str(chosen_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        tuned = PeftModel.from_pretrained(base, str(chosen_dir))
    return (
        _ModelGenerator(
            base, tokenizer, chat_template, chat_template_kwargs, decoding
        ),
        _ModelGenerator(
            tuned, tokenizer, chat_template, chat_template_kwargs, decoding
        ),
    )


@dataclass(frozen=True)
class WarmModels:
    """The two loaded generators and the selection they evaluate.

    ``base`` and ``tuned`` are the loaded generators (real transformers/peft
    models on the machine, doubles in the host suite); ``selection`` is the
    run's own selection rule's record (issue #62) and ``step`` the checkpoint
    the tuned side was loaded from. One object, handed to both eval steps, so
    neither the field list nor the meaning can drift between them.
    """

    base: Any
    tuned: Any
    selection: dict[str, Any]
    step: int


def _load_warm_models(
    job: dict,
    cfg: dict,
    out_dir: Path,
    *,
    make_generators=load_comparison_generators,
) -> tuple[WarmModels | None, str | None, dict[str, Any]]:
    """Select the checkpoint and load the two generators, or explain why not.

    Shared by the comparison and the general-capability slice so the base and
    tuned models are loaded exactly once between them on the warm machine
    (Spec 011's eval runs where the weights already are -- a second load is
    time on a paid machine for nothing). Returns ``(WarmModels, None,
    selection_record)`` on success, or ``(None, reason, selection_record)``
    where ``reason`` is a reader-facing sentence each caller shapes into its
    own failure record. Loading is guarded here, so a model that will not
    load becomes a recorded reason rather than a failed run.
    """
    selection = select_best_checkpoint(checkpoint_records(out_dir))
    selection_record = selection.to_dict()
    if selection.step is None:
        return None, selection.reason, selection_record
    chosen_dir = out_dir / "run" / f"checkpoint-{selection.step}"
    if not chosen_dir.is_dir():
        return (
            None,
            f"the chosen checkpoint (step {selection.step}) was not on the "
            "machine to evaluate",
            selection_record,
        )
    try:
        base_gen, tuned_gen = make_generators(
            job, cfg, chosen_dir, job.get("method") or "qlora"
        )
    except Exception as e:  # noqa: BLE001 - the run must survive the extra step
        return None, f"{type(e).__name__}: {e}", selection_record
    return (
        WarmModels(
            base=base_gen,
            tuned=tuned_gen,
            selection=selection_record,
            step=selection.step,
        ),
        None,
        selection_record,
    )


def run_machine_comparison(
    job: dict,
    cfg: dict,
    out_dir: Path,
    *,
    make_generators=load_comparison_generators,
    loaded=None,
) -> dict:
    """The side-by-side comparison, run on the warm machine after training.

    Reads the trainer's own held-out file (issue #53 -- the same rows the
    eval loss was measured on, never a second sampling of the dataset), picks
    a fixed slice of it, selects the checkpoint the run's own rule (issue
    #62) would choose, generates from the base model and from that
    checkpoint, and returns the record. **Never fails the run**: the whole
    phase -- reading the held-out file, selecting the checkpoint, loading
    the models, and generating -- is guarded, so any failure becomes
    `ok: false` with the reason recorded and the artifact is still delivered
    (Spec 011). `make_generators` is a seam so the host suite can exercise
    the whole flow without torch; `loaded` is the entrypoint's already-loaded
    generator pair plus its selection, so the comparison and the capability
    slice share one model load between them on the warm machine.
    """
    decoding = dict(comparison_step.COMPARISON_DECODING)
    try:
        eval_path = out_dir / "eval.jsonl"
        if not eval_path.is_file():
            return comparison_step.ComparisonOutcome(
                ok=False,
                decoding=decoding,
                reason="nothing was held out, so there was nothing to compare",
            ).to_dict()

        held = [json.loads(line) for line in eval_path.open() if line.strip()]
        messages_field = job.get("messages_field", "messages")
        conversations = [
            row[messages_field]
            for row in comparison_step.select_prompts(held)
            if isinstance(row.get(messages_field), list)
        ]
        if not conversations:
            return comparison_step.ComparisonOutcome(
                ok=False,
                decoding=decoding,
                reason="the held-out rows carried no conversations to compare",
            ).to_dict()

        if loaded is not None:
            base_gen = loaded.base
            tuned_gen = loaded.tuned
            selection_record = loaded.selection
            chosen_step = loaded.step
        else:
            models, reason, selection_record = _load_warm_models(
                job, cfg, out_dir, make_generators=make_generators
            )
            if models is None:
                return comparison_step.ComparisonOutcome(
                    ok=False,
                    decoding=decoding,
                    reason=reason,
                    selection=selection_record,
                ).to_dict()
            base_gen = models.base
            tuned_gen = models.tuned
            chosen_step = models.step

        log(
            f"comparison: {len(conversations)} held-out prompt(s) through the "
            f"base model and the chosen checkpoint (step {chosen_step})"
        )
        outcome = comparison_step.run_comparison(
            conversations=conversations,
            base=base_gen,
            tuned=tuned_gen,
            decoding=decoding,
            selection=selection_record,
        )
        if not outcome.ok:
            # The reason is recorded with the result; the run continues. This
            # is the "evaluation failure never fails the run" guarantee,
            # stated where the failure is swallowed.
            log(
                "comparison failed; recorded and the run continues: "
                f"{outcome.reason}"
            )
        return outcome.to_dict()
    except Exception as e:  # noqa: BLE001 - the run must survive the extra step
        log(
            "comparison failed; recorded and the run continues: "
            f"{type(e).__name__}: {e}"
        )
        return comparison_step.ComparisonOutcome(
            ok=False,
            decoding=decoding,
            reason=f"{type(e).__name__}: {e}",
        ).to_dict()


def run_machine_capability(
    job: dict,
    cfg: dict,
    out_dir: Path,
    *,
    make_generators=load_comparison_generators,
    loaded=None,
) -> dict:
    """The general-capability slice (Spec 011 / issue #73), run on the warm
    machine after training.

    A fixed, versioned slice of general-knowledge questions is answered by
    the base model and by the checkpoint the run's own rule (issue #62) would
    choose, and the difference is reported as a delta with its sample size and
    standard error recorded. It is a smoke test for catastrophic forgetting,
    not a benchmark, and the interface says so from the record. **Never fails
    the run**: the whole phase -- selecting the checkpoint, loading the
    models, and answering -- is guarded, so any failure becomes `ok: false`
    with the reason recorded and the artifact is still delivered (Spec 011).
    `loaded`, when given, is the entrypoint's already-loaded generator pair
    plus its selection, so both eval steps load the two models exactly once.
    """
    decoding = dict(comparison_step.COMPARISON_DECODING)
    try:
        if loaded is not None:
            base_gen = loaded.base
            tuned_gen = loaded.tuned
            selection_record = loaded.selection
            chosen_step = loaded.step
        else:
            models, reason, selection_record = _load_warm_models(
                job, cfg, out_dir, make_generators=make_generators
            )
            if models is None:
                return capability_step.CapabilityOutcome(
                    ok=False,
                    decoding=decoding,
                    reason=reason,
                    selection=selection_record,
                ).to_dict()
            base_gen = models.base
            tuned_gen = models.tuned
            chosen_step = models.step

        log(
            f"capability: {len(capability_step.CAPABILITY_QUESTIONS)} general "
            f"question(s) through the base model and the chosen checkpoint "
            f"(step {chosen_step})"
        )
        outcome = capability_step.run_capability_slice(
            base=base_gen,
            tuned=tuned_gen,
            decoding=decoding,
            selection=selection_record,
        )
        if not outcome.ok:
            # The reason is recorded with the result; the run continues. This
            # is the "evaluation failure never fails the run" guarantee,
            # stated where the failure is swallowed.
            log(
                "capability slice failed; recorded and the run continues: "
                f"{outcome.reason}"
            )
        return outcome.to_dict()
    except Exception as e:  # noqa: BLE001 - the run must survive the extra step
        log(
            "capability slice failed; recorded and the run continues: "
            f"{type(e).__name__}: {e}"
        )
        return capability_step.CapabilityOutcome(
            ok=False,
            decoding=decoding,
            reason=f"{type(e).__name__}: {e}",
        ).to_dict()


def run_machine_evals(
    job: dict,
    cfg: dict,
    out_dir: Path,
    *,
    make_generators=load_comparison_generators,
) -> tuple[dict, dict]:
    """Both on-warm-machine eval steps on one shared model load.

    Returns ``(comparison_record, capability_record)``. The base and tuned
    models are loaded exactly once and handed to both the side-by-side
    comparison (issue #69) and the general-capability slice (issue #73): the
    machine is already warm and a second load is time on a paid machine for
    nothing. **Neither step can fail the run** (Spec 011): a failure to load
    is recorded under both records, a failure in one step under that step --
    the artifact is still delivered either way. `make_generators` is a seam so
    the host suite can exercise the exact wiring without torch.
    """
    try:
        models, load_reason, selection_record = _load_warm_models(
            job, cfg, out_dir, make_generators=make_generators
        )
    except Exception as e:  # noqa: BLE001 - the run must survive the extra step
        models, load_reason, selection_record = (
            None,
            f"{type(e).__name__}: {e}",
            None,
        )
    decoding = dict(comparison_step.COMPARISON_DECODING)
    if models is None:
        log(
            "evaluation could not load the models; recorded under both "
            f"comparison and capability and the run continues: {load_reason}"
        )
        return (
            comparison_step.ComparisonOutcome(
                ok=False,
                decoding=decoding,
                reason=load_reason,
                selection=selection_record,
            ).to_dict(),
            capability_step.CapabilityOutcome(
                ok=False,
                decoding=decoding,
                reason=load_reason,
                selection=selection_record,
            ).to_dict(),
        )
    return (
        run_machine_comparison(job, cfg, out_dir, loaded=models),
        run_machine_capability(job, cfg, out_dir, loaded=models),
    )


def main() -> int:
    started = time.time()
    result: dict = {"ok": False, "started_at": time.time()}
    uploader: checkpoint_upload.CheckpointUploader | None = None
    # Issue #46: the spend ceiling's emergency checkpoint signals this
    # process with SIGTERM; the handler turns that into a graceful, finalizing
    # stop rather than Python's default immediate death.
    import signal

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        job = json.loads((JOB_DIR / "job.json").read_text())
        result["job_id"] = job.get("job_id")
        result["base_model"] = job.get("base_model")
        result["base_revision"] = job.get("base_revision")
        # The method the control plane selected, read once (issue #66): it
        # shapes the config and the artifact, and a missing value reads as
        # qlora, the only method that ever ran before the key existed.
        method = job.get("method") or "qlora"
        result["method"] = method
        log(
            f"job {job.get('job_id')} ΓÇö base_model={job.get('base_model')}"
            f"@{job.get('base_revision') or 'unpinned'} method={method}"
        )

        # Issue #24: the fault surface, read from the environment and off by
        # default. A set-but-malformed spec refuses loudly rather than running
        # under a fault nobody can explain; a valid spec is recorded in the
        # result document and named in the output, so a run this deliberately
        # broke can never be mistaken for one that broke on its own.
        fault_spec = None
        raw_fault = os.environ.get(FAULT_ENV)
        if raw_fault:
            try:
                fault_spec = read_fault_spec(raw_fault)
            except ValueError:
                result["error_code"] = "fault_surface_invalid"
                raise
            result["simulated_fault"] = fault_spec
            log(
                f"[trainer] simulated fault {fault_spec['name']}: "
                f"{_fault_entry(fault_spec['name'])['description']}. "
                "This run is deliberately broken."
            )

        ds = JOB_DIR / "dataset.jsonl"
        if not ds.exists():
            raise FileNotFoundError(f"dataset not found at {ds}")
        parsed = [json.loads(line) for line in ds.open() if line.strip()]
        rows = len(parsed)
        result["dataset_rows"] = rows
        # 10 is the hard floor adopted from OpenAI's enforced minimum; below it
        # a run cannot produce a meaningful adapter, so block rather than waste
        # the GPU.
        if rows < 10:
            raise ValueError(f"dataset has {rows} rows; minimum is 10")
        log(f"dataset: {rows} rows")

        # Decide thinking mode from the data before building the config.
        # A mixed dataset is ambiguous by construction, so it blocks here
        # rather than training half the rows against the wrong template.
        try:
            think = detect_thinking(
                parsed, job.get("messages_field", "messages")
            )
        except MixedThinkingDataset:
            result["error_code"] = "dataset_mixed_thinking"
            raise
        result["thinking"] = think.as_dict()
        log(
            f"thinking mode: {think.enable_thinking} "
            f"({think.with_think}/{think.assistant_turns} assistant turns have  thinking)"
        )

        # An incomplete spec must fail as a named refusal here rather than as
        # a training anomaly minutes into a paid machine. Both the held-out
        # split (which reads the effective val_set_size from the spec) and
        # the config build need the spec, so a spec missing a value fails
        # the same named way whichever of the two trips first.
        try:
            eval_path, split_record = prepare_held_out_split(
                parsed, job, OUT_DIR
            )
            cfg, rejected = build_config(
                job,
                enable_thinking=think.enable_thinking,
                eval_path=eval_path,
            )
        except IncompleteJobSpec:
            result["error_code"] = "spec_incomplete"
            raise
        # The held-out split (issue #53): dedup first, then a deterministic
        # hold-out, so a duplicated row can never land on both sides. The
        # sizes are recorded in result.json and the split is written to
        # separate files, so the held-out rows never enter the training file.
        result["held_out_split"] = split_record
        log(
            f"held-out split: {split_record['held_out_rows']} of "
            f"{split_record['rows_in']} rows held out for evaluation "
            f"({split_record['rows_removed_duplicates']} duplicate(s) removed)"
        )
        # Recorded filtered to keys this trainer knows: the record should name
        # only values that were trained with, and anything else is already
        # echoed verbatim under rejected_overrides.
        result["hyperparameters"] = {
            k: v
            for k, v in (job.get("hyperparameters") or {}).items()
            if k in KNOWN_HYPERPARAMETERS
        }
        if rejected:
            # Not silently dropped: a key the caller sent is something the
            # caller believes is in effect.
            result["rejected_overrides"] = rejected
            log(f"REJECTED unknown keys: {list(rejected)}")

        import yaml  # provided by the base image

        # Issue #24: a divergence fault rewrites the resolved learning rate so
        # the loss genuinely becomes meaningless; the sabotaged value is
        # recorded in the config this run writes, because the run's record
        # says what actually trained.
        if fault_spec is not None:
            apply_fault(cfg, fault_spec)
        CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=True))
        result["config"] = cfg
        log(f"config written to {CONFIG}")

        # Issue #37: when the control plane supplied scoped write grants for
        # this job's checkpoint slots, start the background uploader before
        # training so each checkpoint leaves the machine as it is produced.
        # A standalone run carries no grants and keeps its checkpoints on /out.
        grant_block = job.get("checkpoint_grants")
        if (
            isinstance(grant_block, list)
            and grant_block
            and all(isinstance(g, dict) and g.get("url") for g in grant_block)
        ):
            uploader = checkpoint_upload.CheckpointUploader(
                OUT_DIR, grant_block, upload=upload_artifact, log=log
            )
            uploader.start()
            log(
                f"checkpoint uploader started: {len(grant_block)} "
                f"scoped grant(s)"
            )
        else:
            log(
                "no checkpoint grants in the job spec; checkpoints will be "
                "left on the machine"
            )

        # Issue #49: the base model is downloaded here, first, in a mode that
        # reports progress -- the phase that dominates a large job should not
        # be a blank screen. Axolotl then resolves from the HF cache, so the
        # prefetch is not a second download, and a prefetch that cannot run
        # costs a progress signal, never the job.
        prefetch_model(job)

        cmd = ["axolotl", "train", str(CONFIG)]
        log(f"running: {' '.join(cmd)}")
        # Issue #24: the runtime half of a trainer-side fault is scheduled
        # just before training starts, so oom and worker_kill fire at their
        # chosen point a little way into the run.
        if fault_spec is not None:
            schedule_fault(fault_spec)
        t0 = time.time()
        # Sampled while the trainer runs, not after: the machine is destroyed
        # the moment the job ends, so the peak must be captured during the
        # only window it exists (issue #77's "no run is wasted").
        sampler = VramSampler()
        sampler.start()
        try:
            code, tail = run_streaming(cmd)
        finally:
            sampler.stop()
        result["train_seconds"] = round(time.time() - t0, 1)
        result["exit_code"] = code
        peak = sampler.peak_gb()
        if peak is not None:
            result["peak_memory_gb"] = round(peak, 2)

        if code != 0:
            result["error"] = "axolotl train failed"
            # The tail is kept in result.json even though every one of these
            # lines was already streamed: the result document is what the
            # orchestrator reads, and it should never have to go back and
            # reassemble a failure out of the event log.
            result["log_tail"] = tail
            # Issue #35: name a device-memory exhaustion as the specific
            # failure it is, so the control plane can retry with the effective
            # batch preserved instead of reading a bare training failure. A
            # deliberately caused oom (the fault surface) keeps the fault's
            # own simulated_ code; a genuine one carries the platform's
            # training_oom.
            oom = oom_error_code(tail, fault_spec)
            if oom is not None:
                result["error_code"] = oom
                result["error"] = (
                    "The training process ran out of device memory "
                    f"({oom}); no artifact was produced."
                )
            log(f"training FAILED (exit {code})")
            return code

        result.update(collect_artifacts(method))

        # The export-time template probe (Spec 009 / issue #59): a fixed probe
        # conversation is tokenised through the template used in training and
        # through the template serialised into the artifact, and the ids must
        # be identical. It runs on EVERY export -- whether or not anything was
        # overridden -- because a wrong thinking-mode detection produces a
        # wrong template with no override involved. A mismatch fails the
        # export with the stable code template_probe_mismatch; the message
        # names what differs. If the probe cannot run at all it fails closed
        # with template_probe_unavailable: a guard that silently disappears
        # when it cannot run is no guard, and the templates are load-bearing
        # for every advanced override.
        try:
            probe_outcome = probe_export(job, cfg, load_probe_tokenizer(job))
            if not probe_outcome.ok:
                raise TemplateProbeFailure(probe_outcome)
        except TemplateProbeFailure as e:
            result["error_code"] = e.error_code
            result["template_probe"] = e.outcome.as_dict()
            raise
        except Exception as e:
            # Same record shape as a failed probe, but the probe never ran:
            # fail closed rather than let a guard that could not run read as a
            # pass, and reuse the outcome's serialisation so the record shape
            # stays the one the artifact records.
            unavailable = ProbeOutcome(
                ok=False,
                error_code=PROBE_UNAVAILABLE_CODE,
                message=(
                    "the template probe could not run, so the export cannot "
                    f"prove the templates agree: {type(e).__name__}: {e}"
                ),
            )
            result["error_code"] = unavailable.error_code
            result["template_probe"] = unavailable.as_dict()
            raise
        result["template_probe"] = probe_outcome.as_dict()

        # The two on-warm-machine eval steps (Spec 011): the side-by-side
        # comparison (issue #69) and the general-capability slice (issue #73).
        # The same held-out prompts answered by the base model and by the
        # checkpoint the run's selection rule would choose, and a fixed slice
        # of general questions answered by both -- the smoke test for
        # catastrophic forgetting, never a benchmark. Both load the two models
        # exactly once between them (they run back to back where the weights
        # already are), and neither can fail the run: a failure -- including a
        # failure to load the models -- is recorded under its own key and the
        # artifact is still delivered.
        result["comparison"], result["capability"] = run_machine_evals(
            job, cfg, OUT_DIR
        )

        if result.get("artifact_path"):
            # ADR-0009: when the control plane supplied a scoped write URL, the
            # machine puts its artifact to it directly. The outcome -- not a
            # bare claim -- is recorded, and the control plane verifies what
            # landed against the checksum above before the job may report
            # success. A standalone run has no grant and simply leaves the
            # artifact on /out; that is reported as a fact, not a failure of
            # training.
            grant_block = job.get("artifact_upload")
            if isinstance(grant_block, dict) and grant_block.get("url"):
                result["artifact_upload"] = upload_artifact(
                    grant_block["url"], OUT_DIR / result["artifact_path"]
                )
            else:
                result["artifact_upload"] = {
                    "ok": False,
                    "error": "no write grant in the job spec; the artifact "
                    "was left on the machine",
                }

        # Issue #74: produce the requested delivery formats (merged, quantised)
        # off the correctly merged model, in order -- merge at full precision,
        # then quantise once -- each verified by loading it, each running the
        # template probe, each uploaded through its own scoped grant. A
        # produced format that cannot be loaded fails the export with a stable
        # code (the failure the issue names is silent, so the assertion that
        # the order held and each format loaded is the deliverable). A request
        # for no extra formats records nothing and changes nothing.
        try:
            delivery_records = run_delivery(
                job,
                OUT_DIR,
                upload_artifact,
                tokenizer=load_probe_tokenizer(job),
                probe=probe_export,
                cfg=cfg,
                grants=job.get("delivery_grants"),
            )
            if delivery_records:
                result["delivery"] = delivery_records
        except DeliveryFailure as e:
            result["error_code"] = e.error_code
            result["error"] = str(e)
            raise
        result["ok"] = True
        log(f"training complete in {result['train_seconds']}s")
        return 0

    except GracefulStop:
        # An ordered stop (issue #46, the spend ceiling's emergency
        # checkpoint): not a training error, and the run's terminal reason is
        # the control plane's to decide. The finally below ships the final
        # checkpoint and writes result.json, which is the whole point of the
        # request.
        log("ordered stop, finalizing and writing result.json")
        return 0
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        log(f"ERROR: {result['error']}")
        return 1
    finally:
        # The checkpoints are part of the run's result document on every path,
        # including failure: a run that ended early still reports which
        # checkpoints reached storage, which is exactly the record a resumption
        # would consult. `stop_and_finish` runs here so the final checkpoint --
        # the one a resumption would most want -- is uploaded even if the poll
        # never saw it.
        if uploader is not None:
            uploader.stop_and_finish()
            result["checkpoints"] = uploader.snapshot()
        result["total_seconds"] = round(time.time() - started, 1)
        write_result(result)
        log(f"result written to {RESULT}")


if __name__ == "__main__":
    sys.exit(main())
