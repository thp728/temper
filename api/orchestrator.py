"""Job orchestration: provision, bootstrap, train, collect, destroy.

This is `spike/spike4.py` made durable. The sequence is unchanged because it is
proven; what is added is a job row, state transitions, and events.

Three properties carried over from the spikes, each of which was learned the
expensive way:

* **Teardown runs in `finally`, then is independently confirmed** by listing
  instances. Trusting a destroy call's return value is exactly the assumption
  that leaves a GPU billing overnight.
* **Readiness distinguishes *unreachable* from *authentication failed*.** They
  have opposite remedies, and collapsing both into "no answer" cost an evening
  and produced a wrongly-filed platform bug.
* **The trainer publishes no ports.** `ufw` does not filter Docker-published
  ports and a `DOCKER-USER` rule on the published port never matches, because
  the packet is already DNAT'd. Not publishing is the mitigation that works.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import threading
import time
from pathlib import Path

from . import catalog, config, db

REPO_ROOT = Path(__file__).parent.parent
TRAINER_DIR = REPO_ROOT / "trainer"
ARTIFACTS = REPO_ROOT / "data" / "artifacts"

GPU_PREFERENCE = ["L4", "RTX-PRO6000", "H100"]
STORAGE_GB = 100          # platform minimum for VM instances
SSH_READY_TIMEOUT_S = 300
BOOTSTRAP_TIMEOUT_S = 5400
MAX_GPU_MINUTES = 90      # safety control, not billing


class OrchestratorError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _ssh(ssh_command: str) -> list[str]:
    base = ssh_command.strip()
    if base.startswith("ssh "):
        base = base[4:]
    return ["ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
            "-o", "BatchMode=yes", *base.split()]


def _wait_for_ssh(job_id: str, ssh_command: str) -> None:
    """Poll until sshd answers.

    `Running` from the provider is a claim about the VM, not about
    reachability: measured, SSH refuses for ~40s after the status flips.
    Authentication failures are reported separately from unreachability because
    the fixes are unrelated -- one means wait or reprovision, the other means
    the agent is not holding the key.
    """
    t0 = time.time()
    last_auth_error = None
    while time.time() - t0 < SSH_READY_TIMEOUT_S:
        try:
            r = subprocess.run(_ssh(ssh_command) + ["true"],
                               capture_output=True, text=True, timeout=25)
            if r.returncode == 0:
                db.add_event(job_id, "log",
                             f"SSH ready after {time.time() - t0:.0f}s")
                return
            err = (r.stderr or "").lower()
            if "permission denied" in err or "publickey" in err:
                last_auth_error = r.stderr.strip()
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)

    if last_auth_error:
        raise OrchestratorError(
            "ssh_auth_failed",
            "The VM was reachable but rejected the SSH key. The key is "
            "registered, so this is almost always a local agent problem: "
            "check `ssh-add -l` lists the JarvisLabs key. "
            f"Server said: {last_auth_error}")
    raise OrchestratorError(
        "ssh_unreachable",
        f"VM never accepted SSH within {SSH_READY_TIMEOUT_S}s. It reached "
        f"Running but is not usable; destroying and giving up.")


def _push_sources(ssh_command: str) -> None:
    """Ship trainer/ as a tar on stdin. One round trip, no scp dependency."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in TRAINER_DIR.iterdir():
            if not p.is_file() or p.suffix == ".pyc":
                continue
            # Normalise line endings: a CRLF Dockerfile fails inside the
            # container in ways that read as anything but a line-ending bug.
            data = p.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
            info = tarfile.TarInfo(name=p.name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    r = subprocess.run(
        _ssh(ssh_command) + ["mkdir -p /tmp/trainer && tar xzf - -C /tmp/trainer"],
        input=buf.getvalue(), capture_output=True, timeout=180)
    if r.returncode != 0:
        raise OrchestratorError("source_upload_failed",
                                r.stderr.decode("utf-8", "replace")[:300])


def _remote_script(job: dict, dataset_path: Path, model: catalog.BaseModel,
                   enable_thinking: bool) -> bytes:
    """The on-VM script: build the image, run the job, print the result."""
    job_spec = {
        "job_id": job["id"],
        "base_model": model.repo,
        "hyperparameters": job["hyperparameters"] or {},
    }
    dataset_b64 = dataset_path.read_bytes().hex()
    script = f"""
set -u
say() {{ echo "[$(date +%H:%M:%S)] $*" >&2; }}

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
  echo "---RESULT---"
  sudo cat /tmp/out/result.json
else
  echo '{{"stage":"train","ok":false,"error":"no result.json produced"}}'
fi
"""
    return script.encode("utf-8")


def _fetch_adapter(job_id: str, ssh_command: str, result: dict) -> str | None:
    rel = result.get("adapter_path")
    if not rel:
        return None
    dest = ARTIFACTS / job_id
    dest.mkdir(parents=True, exist_ok=True)
    local = dest / "adapter_model.safetensors"
    r = subprocess.run(
        _ssh(ssh_command) + [f"sudo cat /tmp/out/{rel}"],
        capture_output=True, timeout=600)
    if r.returncode != 0 or not r.stdout:
        db.add_event(job_id, "error", "Adapter fetch failed; artifact left on VM")
        return None
    local.write_bytes(r.stdout)

    # Verify against the hash the container computed. A silently truncated
    # transfer produces a file that looks fine and is not.
    import hashlib
    got = hashlib.sha256(local.read_bytes()).hexdigest()
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


def run_job(job_id: str) -> None:
    """Drive one job to a terminal state. Always tears down."""
    from jarvislabs import Client

    if not config.provider_credentials_present():
        db.set_state(job_id, "failed",
                     "No provider credentials; nothing was provisioned",
                     error_code="provider_unauthenticated",
                     error_message="JL_API_KEY is not set and no jl config file "
                                   "exists, so no VM could be created. Put the "
                                   "key in spike/.env or export JL_API_KEY.")
        return

    job = db.get_job(job_id)
    dataset = db.get_dataset(job["dataset_id"])
    model = catalog.get(job["base_model"]) or catalog.get(catalog.DEFAULT_MODEL)
    enable_thinking = bool(dataset.get("enable_thinking"))
    instance = None
    started = time.time()

    with Client() as client:
        try:
            db.set_state(job_id, "provisioning", "Selecting a GPU")
            avail = {r.gpu_type: r for r in client.account.gpu_availability()
                     if r.workload_type == "vm" and r.num_free_devices > 0}
            gpu = next((g for g in GPU_PREFERENCE if g in avail), None)
            if not gpu:
                raise OrchestratorError(
                    "provider_capacity_unavailable",
                    f"No VM-capable GPU free. Note that availability is "
                    f"per-workload-type: some GPUs exist only for containers.")
            row = avail[gpu]
            currency = client.account.currency()
            db.set_state(job_id, "provisioning",
                         f"Provisioning {gpu} at {row.price_per_hour}{currency}/hr",
                         gpu_type=gpu, price_per_hour=row.price_per_hour,
                         currency=currency)

            instance = client.instances.create(
                gpu_type=gpu, num_gpus=1, template="vm",
                storage=STORAGE_GB, name=f"temper-{job_id[:12]}")
            db.set_state(job_id, "preparing",
                         f"VM {instance.machine_id} running; waiting for SSH",
                         machine_id=instance.machine_id)

            _wait_for_ssh(job_id, instance.ssh_command or "")
            _push_sources(instance.ssh_command or "")

            db.set_state(job_id, "training", "Building image and training")
            ds_path = Path(dataset["path"])
            proc = subprocess.run(
                _ssh(instance.ssh_command) + ["bash -s"],
                input=_remote_script(job, ds_path, model, enable_thinking),
                capture_output=True, timeout=BOOTSTRAP_TIMEOUT_S)

            for line in proc.stderr.decode("utf-8", "replace").splitlines():
                if line.strip():
                    db.add_event(job_id, "log", line.strip()[:500])

            out = proc.stdout.decode("utf-8", "replace")
            marker = "---RESULT---"
            if marker not in out:
                raise OrchestratorError("training_failed",
                                        "Trainer produced no result.json. See job events.")
            result = json.loads(out.split(marker, 1)[1])
            if not result.get("ok"):
                raise OrchestratorError(
                    "training_failed",
                    result.get("error") or "Training did not complete.")

            db.set_state(job_id, "packaging", "Retrieving adapter")
            adapter = _fetch_adapter(job_id, instance.ssh_command, result)
            db.set_state(job_id, "complete", "Training complete",
                         result_json=result, adapter_path=adapter)

        except OrchestratorError as e:
            db.set_state(job_id, "failed", str(e),
                         error_code=e.code, error_message=str(e))
        except Exception as e:
            db.set_state(job_id, "failed", f"{type(e).__name__}: {e}",
                         error_code="internal_error", error_message=str(e))
        finally:
            # Teardown, then independent confirmation. The destroy call's
            # return value is not evidence.
            if instance is not None:
                for attempt in range(3):
                    try:
                        client.instances.destroy(instance.machine_id)
                        db.add_event(job_id, "log",
                                     f"VM {instance.machine_id} destroyed")
                        break
                    except Exception as e:
                        db.add_event(job_id, "error", f"Destroy attempt failed: {e}")
                        time.sleep(5)
                try:
                    live = [i for i in client.instances.list()
                            if i.machine_id == instance.machine_id]
                    if live:
                        db.add_event(job_id, "error",
                                     f"STRAY INSTANCE {instance.machine_id} still "
                                     f"listed — destroy manually, it is billing")
                except Exception:
                    pass
            db.add_event(job_id, "log",
                         f"Job finished in {time.time() - started:.0f}s")


def launch(job_id: str) -> None:
    """Start a job on a background thread.

    A thread rather than Celery: one process is the whole deployment, and a
    queue with one worker and no retries would be ceremony. The cost is honest
    -- a process restart orphans in-flight jobs, which `db.active_jobs()`
    surfaces at startup rather than hiding.
    """
    threading.Thread(target=run_job, args=(job_id,), daemon=True,
                     name=f"job-{job_id[:8]}").start()
