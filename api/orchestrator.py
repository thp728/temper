"""Job orchestration: provision, bootstrap, train, collect, destroy.

This is `spike/spike4.py` made durable. The sequence is unchanged because it is
proven; what is added is a job row, state transitions, and events.

Everything that touches the compute provider goes through the injected
`Provider` protocol — one seam, so the whole money-spending path can be
exercised with a fake and no GPU. The default is the real one, so callers that
do not care about testing pass nothing.

Three properties carried over from the spikes, each of which was learned the
expensive way:

* **Teardown runs in `finally`, then is independently confirmed** by listing
  machines. Trusting a destroy call's return value is exactly the assumption
  that leaves a GPU billing overnight. It now runs **before** the terminal
  state transition, so a client that stops polling once the job says it is
  finished still sees the confirmation.
* **Readiness distinguishes *unreachable* from *authentication failed*.** They
  have opposite remedies, and collapsing both into "no answer" cost an evening
  and produced a wrongly-filed platform bug. That logic lives in the default
  provider now, with the two codes intact.
* **The trainer publishes no ports.** `ufw` does not filter Docker-published
  ports and a `DOCKER-USER` rule on the published port never matches, because
  the packet is already DNAT'd. Not publishing is the mitigation that works.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import time
from pathlib import Path

from . import catalog, db
from .errors import OrchestratorError
from .provider import Provider, new_provider

REPO_ROOT = Path(__file__).parent.parent
TRAINER_DIR = REPO_ROOT / "trainer"
ARTIFACTS = REPO_ROOT / "data" / "artifacts"

GPU_PREFERENCE = ["L4", "RTX-PRO6000", "H100"]
STORAGE_GB = 100          # platform minimum for VM instances
MAX_GPU_MINUTES = 90      # safety control, not billing

TRAINER_TARBALL = "/tmp/trainer.tar.gz"
# The machine emits this when it fails before the trainer ever runs, so that a
# pre-training failure still arrives as a result document naming its own code
# rather than as "the trainer produced nothing".
SOURCE_UNPACK_FAILED = json.dumps({
    "stage": "source", "ok": False, "error_code": "source_upload_failed",
    "error": "The trainer sources reached the machine but did not unpack."})
RESULT_MARKER = "---RESULT---"
DESTROY_ATTEMPTS = 3
DESTROY_RETRY_DELAY_S = 5


def _trainer_tarball() -> bytes:
    """Ship trainer/ as one tar. One round trip, no scp dependency."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(TRAINER_DIR.iterdir()):
            if not p.is_file() or p.suffix == ".pyc":
                continue
            # Normalise line endings: a CRLF Dockerfile fails inside the
            # container in ways that read as anything but a line-ending bug.
            data = p.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
            info = tarfile.TarInfo(name=p.name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _remote_script(job: dict, dataset_path: Path, model: catalog.BaseModel,
                   enable_thinking: bool) -> bytes:
    """The on-machine script: build the image, run the job, print the result."""
    job_spec = {
        "job_id": job["id"],
        "base_model": model.repo,
        "hyperparameters": job["hyperparameters"] or {},
    }
    dataset_b64 = dataset_path.read_bytes().hex()
    script = f"""
set -u
say() {{ echo "[$(date +%H:%M:%S)] $*" >&2; }}

mkdir -p /tmp/trainer
tar xzf {TRAINER_TARBALL} -C /tmp/trainer || {{
  say "SOURCE UNPACK FAILED"
  echo "{RESULT_MARKER}"
  echo '{SOURCE_UNPACK_FAILED}'
  exit 0
}}

# The trainer needs no inbound port, so it publishes none. That is the only
# firewall mitigation that actually holds for Docker on this platform.
sudo ufw allow 22/tcp >/dev/null 2>&1
sudo ufw default deny incoming >/dev/null 2>&1
sudo ufw --force enable >/dev/null 2>&1

mkdir -p /tmp/job /tmp/out
python3 -c "
import binascii, pathlib
pathlib.Path('/tmp/job/dataset.jsonl').write_bytes(
    binascii.unhexlify('{dataset_b64}'))
"
cat > /tmp/job/job.json <<'JOBSPEC'
{json.dumps(job_spec, indent=2)}
JOBSPEC

say "building trainer image"
t0=$(date +%s)
sudo docker build -t temper-trainer:job /tmp/trainer >/tmp/build.log 2>&1 || {{
  say "BUILD FAILED"; tail -n 20 /tmp/build.log >&2
  echo '{{"stage":"build","ok":false}}'; exit 0
}}
say "image built in $(( $(date +%s) - t0 ))s"

say "running training"
sudo docker run --rm --gpus all \\
  -v /tmp/job:/job:ro -v /tmp/out:/out -e HF_HOME=/out/hf \\
  temper-trainer:job >/tmp/run.log 2>&1 || say "TRAINER EXITED NONZERO"
tail -n 30 /tmp/run.log >&2

if [ -f /tmp/out/result.json ]; then
  echo "{RESULT_MARKER}"
  sudo cat /tmp/out/result.json
else
  echo '{{"stage":"train","ok":false,"error":"no result.json produced"}}'
fi
"""
    return script.encode("utf-8")


def _consume(job_id: str, lines) -> dict:
    """Turn the machine's output into events, and return the trainer's result.

    Everything before the marker is the job's output and becomes a log event.
    Everything after it is the trainer's result document, which is machinery
    rather than output and is not logged as such.
    """
    result_lines: list[str] = []
    seen_marker = False
    for line in lines:
        text = line.strip()
        if not seen_marker:
            if text == RESULT_MARKER:
                seen_marker = True
            elif text:
                db.add_event(job_id, "log", text[:500])
        else:
            result_lines.append(line)

    if not seen_marker:
        raise OrchestratorError(
            "training_failed",
            "Trainer produced no result.json. See job events.")
    try:
        return json.loads("\n".join(result_lines))
    except json.JSONDecodeError as e:
        raise OrchestratorError(
            "training_failed",
            f"Trainer's result document did not parse: {e}")


def _fetch_adapter(provider: Provider, machine, job_id: str,
                   result: dict) -> str | None:
    rel = result.get("adapter_path")
    if not rel:
        return None
    payload = provider.fetch(machine, f"/tmp/out/{rel}")
    if not payload:
        db.add_event(job_id, "error",
                     "Adapter fetch failed; artifact left on the machine")
        return None

    dest = ARTIFACTS / job_id
    dest.mkdir(parents=True, exist_ok=True)
    local = dest / "adapter_model.safetensors"
    local.write_bytes(payload)

    # Verify against the hash the container computed. A silently truncated
    # transfer produces a file that looks fine and is not.
    got = hashlib.sha256(payload).hexdigest()
    want = result.get("adapter_sha256")
    if want and got != want:
        raise OrchestratorError(
            "artifact_corrupt",
            f"Adapter SHA mismatch: container reported {want[:16]}…, "
            f"downloaded file is {got[:16]}…")
    db.add_event(job_id, "log",
                 f"Adapter verified, {local.stat().st_size / 1e6:.1f} MB")

    # A bare .safetensors is not a loadable adapter: PEFT needs
    # adapter_config.json beside it to know the rank, alpha and target modules.
    # Shipping only the weights would have handed the user a file that looks
    # like the deliverable and cannot be used. The config is already inside
    # result.json, so this costs no extra transfer.
    adapter_config = result.get("adapter_config")
    if adapter_config:
        (dest / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=2), encoding="utf-8")
    else:
        db.add_event(job_id, "error",
                     "No adapter_config.json in the run result; the downloaded "
                     "adapter will not load without one.")
    return str(local)


def _teardown(provider: Provider, job_id: str, machine) -> None:
    """Destroy the machine, then confirm it independently.

    A destroy call that returns cleanly is a claim. The evidence is the machine
    no longer being listed, and a machine that is still listed is billing right
    now — so it is reported as an error an operator cannot miss.
    """
    for attempt in range(DESTROY_ATTEMPTS):
        try:
            provider.destroy(machine.machine_id)
            db.add_event(job_id, "log",
                         f"Machine {machine.machine_id} destroyed")
            break
        except Exception as e:
            db.add_event(job_id, "error", f"Destroy attempt failed: {e}")
            if attempt < DESTROY_ATTEMPTS - 1:
                time.sleep(DESTROY_RETRY_DELAY_S)
    try:
        if machine.machine_id in provider.list_machine_ids():
            db.add_event(job_id, "error",
                         f"STRAY MACHINE {machine.machine_id} still listed — "
                         f"destroy it manually, it is billing")
    except Exception as e:
        db.add_event(job_id, "error",
                     f"Could not confirm teardown of machine "
                     f"{machine.machine_id}: {e}")


def _attempt(provider: Provider, job_id: str, machines: list) -> tuple[str, str, dict]:
    """Do the work. Returns the terminal state to record, but never records it.

    Recording the outcome is the caller's job precisely so that teardown can
    happen in between: the confirmation that the machine is gone must reach the
    event log before the job reports that it is finished.

    `machines` is the caller's handle on anything created, appended to the
    moment it exists — a machine that exists but was never recorded is a
    machine nobody destroys.
    """
    job = db.get_job(job_id)
    dataset = db.get_dataset(job["dataset_id"])
    model = catalog.get(job["base_model"]) or catalog.get(catalog.DEFAULT_MODEL)
    enable_thinking = bool(dataset.get("enable_thinking"))

    db.set_state(job_id, "provisioning", "Selecting a GPU")
    gpu = provider.select_gpu(GPU_PREFERENCE)
    db.set_state(job_id, "provisioning",
                 f"Provisioning {gpu.gpu_type} at "
                 f"{gpu.price_per_hour}{gpu.currency}/hr",
                 gpu_type=gpu.gpu_type, price_per_hour=gpu.price_per_hour,
                 currency=gpu.currency)

    machine = provider.create(gpu.gpu_type, STORAGE_GB, f"temper-{job_id[:12]}")
    machines.append(machine)
    db.set_state(job_id, "preparing",
                 f"Machine {machine.machine_id} running; waiting for SSH",
                 machine_id=machine.machine_id)

    db.add_event(job_id, "log", provider.await_ready(machine))
    provider.push(machine, _trainer_tarball(), TRAINER_TARBALL)

    db.set_state(job_id, "training", "Building image and training")
    ds_path = Path(dataset["path"])
    script = _remote_script(job, ds_path, model, enable_thinking)
    result = _consume(job_id, provider.stream(machine, script))
    if not result.get("ok"):
        # The result document names its own failure where it can. A stage that
        # failed before training started is not a training failure, and telling
        # a user otherwise sends them to read the wrong logs.
        raise OrchestratorError(
            result.get("error_code") or "training_failed",
            result.get("error") or "Training did not complete.")

    db.set_state(job_id, "packaging", "Retrieving adapter")
    adapter = _fetch_adapter(provider, machine, job_id, result)
    return ("complete", "Training complete",
            {"result_json": result, "adapter_path": adapter})


def run_job(job_id: str, provider: Provider | None = None) -> None:
    """Drive one job to a terminal state. Always tears down.

    The provider is injected so the whole path is testable; omit it and the
    real one is built, which is where a missing credential surfaces.
    """
    owns_provider = provider is None
    if owns_provider:
        try:
            provider = new_provider()
        except OrchestratorError as e:
            db.set_state(job_id, "failed", str(e),
                         error_code=e.code, error_message=str(e))
            return
        except Exception as e:
            # Nothing was provisioned, so there is nothing to tear down -- but
            # the job still has to reach a terminal state rather than sit in
            # `queued` forever because the SDK failed to import.
            db.set_state(job_id, "failed", f"{type(e).__name__}: {e}",
                         error_code="internal_error", error_message=str(e))
            return

    machines: list = []
    started = time.time()
    try:
        try:
            outcome = _attempt(provider, job_id, machines)
        except OrchestratorError as e:
            outcome = ("failed", str(e),
                       {"error_code": e.code, "error_message": str(e)})
        except Exception as e:
            outcome = ("failed", f"{type(e).__name__}: {e}",
                       {"error_code": "internal_error",
                        "error_message": str(e)})
        finally:
            # Before the terminal transition, and on every path including one
            # nobody anticipated.
            for machine in machines:
                _teardown(provider, job_id, machine)
            db.add_event(job_id, "log",
                         f"Job finished in {time.time() - started:.0f}s")

        state, message, fields = outcome
        db.set_state(job_id, state, message, **fields)
    finally:
        if owns_provider:
            provider.close()


def launch(job_id: str, provider: Provider | None = None) -> None:
    """Start a job on a background thread.

    A thread rather than Celery: one process is the whole deployment, and a
    queue with one worker and no retries would be ceremony. The cost is honest
    -- a process restart orphans in-flight jobs, which `db.active_jobs()`
    surfaces at startup rather than hiding.
    """
    threading.Thread(target=run_job, args=(job_id, provider), daemon=True,
                     name=f"job-{job_id[:8]}").start()
