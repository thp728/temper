"""Spike 4 - build the pinned trainer image and prove a real job runs through it.

Spike 3 got QLoRA running but could not resume, because an unpinned stack put
transformers 5.x, trl 1.x and torch 2.5.1 in one environment. The trainer image
exists to fix that by delegating the version matrix to Axolotl. This spike is
the proof that the fix works.

What it answers:

  1. Does the image build from the pinned digest?
  2. Does a real job run end to end through the documented /job -> /out contract?
  3. Does `axolotl train` accept the config the entrypoint renders - especially
     warmup_ratio, the argument TRL 1.x removed?
  4. Does checkpoint/resume work on this stack? (it did not on the last one)
  5. Are unknown job keys refused rather than silently dropped?

Usage:
    python -u spike4.py            # full run, destroys the VM
    python -u spike4.py --keep     # leave it up (COSTS MONEY)
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spike import Findings, header, load_dotenv  # noqa: E402
from spike2 import ssh_base, wait_for_ssh  # noqa: E402

try:
    from jarvislabs import Client
except ImportError:
    sys.exit("jarvislabs SDK not installed. Run: pip install jarvislabs")

GPU_PREFERENCE = ["L4", "RTX-PRO6000", "H100"]
INSTANCE_NAME = "spike4-trainer"
STORAGE_GB = 100
TRAINER_DIR = Path(__file__).parent.parent / "trainer"
BOOTSTRAP_TIMEOUT_S = 3600


def push_trainer_sources(ssh_command: str, f: Findings) -> bool:
    """Ship trainer/ to the VM as a tar on stdin.

    tar rather than scp: one round trip, no extra tooling, and it preserves the
    directory shape docker build expects as its context.
    """
    header("SHIPPING TRAINER SOURCES")
    if not TRAINER_DIR.exists():
        f.record("trainer/ present", False, str(TRAINER_DIR))
        return False

    files = [p for p in TRAINER_DIR.iterdir() if p.is_file()]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in files:
            # Normalise line endings. A CRLF Dockerfile or entrypoint written on
            # Windows breaks inside the container in ways that read as anything
            # except a line-ending problem.
            data = p.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
            info = tarfile.TarInfo(name=p.name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    payload = buf.getvalue()
    f.record("trainer/ packed", True,
             f"{len(files)} files, {len(payload)} bytes gz: "
             f"{', '.join(p.name for p in files)}")

    r = subprocess.run(
        ssh_base(ssh_command) + ["mkdir -p /tmp/trainer && tar xzf - -C /tmp/trainer"],
        input=payload, capture_output=True, timeout=120)
    ok = r.returncode == 0
    f.record("trainer/ unpacked on VM", ok,
             r.stderr.decode("utf-8", "replace")[:200] if not ok else "/tmp/trainer")
    return ok


def run_bootstrap(ssh_command: str, model: str, f: Findings) -> dict | None:
    header("BUILD + RUN")
    script = Path(__file__).parent / "bootstrap4.sh"
    if not script.exists():
        f.record("bootstrap4.sh present", False, str(script))
        return None
    payload = script.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    f.record("bootstrap4.sh present", True, f"{len(payload)} bytes")

    print("  building the image (8.5 GB base pull), then two training runs.")
    t0 = time.time()
    try:
        r = subprocess.run(
            ssh_base(ssh_command) + [f"BASE_MODEL={model} bash -s"],
            input=payload, capture_output=True, timeout=BOOTSTRAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        f.record("bootstrap executed", False, f"timed out after {BOOTSTRAP_TIMEOUT_S}s")
        return None

    for line in r.stderr.decode("utf-8", "replace").splitlines():
        if line.strip():
            print(f"    {line.strip()}")

    out = r.stdout.decode("utf-8", "replace")
    try:
        rep = json.loads(out[out.index("{"):])
    except (ValueError, json.JSONDecodeError):
        f.record("bootstrap executed", False, "stdout was not JSON")
        print(out[:1200])
        return None
    f.record("bootstrap executed", True, f"{time.time()-t0:.0f}s wall clock")
    return rep


def interpret(rep: dict, f: Findings) -> None:
    header("WHAT THIS MEANS")
    r1 = rep.get("result_run1") or {}

    f.record("image built", bool(rep.get("build_ok")),
             f"{rep.get('build_seconds')}s, "
             f"{rep.get('image_bytes', 0) / 1e9:.1f} GB, "
             f"base={rep.get('base_digest_label', '')[:23]}...")

    f.record("job ran through /job -> /out contract", bool(r1.get("ok")),
             f"{r1.get('train_seconds')}s train, "
             f"{r1.get('dataset_rows')} rows, exit={r1.get('exit_code')}")

    rejected = r1.get("rejected_overrides")
    f.record("unknown job keys refused", bool(rejected),
             f"rejected {list(rejected)}" if rejected
             else "NOT refused - a caller could believe an override applied")

    f.record("checkpoint resume", bool(rep.get("resume_ok")),
             f"{rep.get('checkpoint_count')} checkpoints, "
             f"run2 {rep.get('run2_seconds')}s")

    if r1.get("adapter_sha256"):
        print(f"\n  adapter: {r1.get('adapter_bytes', 0) / 1e6:.1f} MB "
              f"({r1.get('adapter_format')}), sha {r1['adapter_sha256'][:16]}...")
        ac = r1.get("adapter_config") or {}
        if ac:
            print(f"  adapter_config: r={ac.get('r')} alpha={ac.get('lora_alpha')} "
                  f"rslora={ac.get('use_rslora')} targets={ac.get('target_modules')}")

    if rep.get("resume_ok"):
        print("\n  RESUME WORKS. This is the thing spike 3 could not do - the")
        print("  pinned Axolotl base removed the transformers/trl/torch conflict")
        print("  that made checkpoints unloadable. C14 is closed.")
    elif r1.get("ok"):
        print("\n  ! Training works but resume does not. Section 31's")
        print("    'survives a forced interruption' criterion still fails.")
    if not r1.get("ok") and r1.get("log_tail"):
        print("\n  --- entrypoint log tail ---")
        for line in r1["log_tail"][-15:]:
            print(f"    {line}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--gpu")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "findings-spike4.json")
    args = ap.parse_args()

    if load_dotenv(Path(__file__).parent / ".env"):
        print("Loaded credentials from .env")

    f, instance, started = Findings(), None, time.time()
    try:
        client = Client()
    except Exception as e:
        print(f"Cannot authenticate: {e}")
        return 2

    with client:
        try:
            header("PREFLIGHT")
            avail = {r.gpu_type: r for r in client.account.gpu_availability()
                     if r.workload_type == "vm" and r.num_free_devices > 0}
            gpu = args.gpu or next((g for g in GPU_PREFERENCE if g in avail), None)
            if not gpu:
                f.record("vm gpu available", False, f"none of {GPU_PREFERENCE} free")
                return 1
            f.record("vm gpu available", True,
                     f"{gpu} @ {avail[gpu].price_per_hour}"
                     f"{client.account.currency()}/hr")

            header(f"PROVISIONING - {gpu}, vm, {STORAGE_GB} GB")
            t0 = time.time()
            instance = client.instances.create(
                gpu_type=gpu, num_gpus=1, template="vm",
                storage=STORAGE_GB, name=INSTANCE_NAME)
            f.record("VM created", True, f"{time.time() - t0:.0f}s to Running",
                     machine_id=instance.machine_id)
            print(f"  machine_id={instance.machine_id}  ip={instance.public_ip}")

            if wait_for_ssh(instance.ssh_command or "", f) is None:
                return 1
            if not push_trainer_sources(instance.ssh_command or "", f):
                return 1

            rep = run_bootstrap(instance.ssh_command or "", args.model, f)
            f.probe_report = rep
            if rep:
                interpret(rep, f)

        except KeyboardInterrupt:
            f.record("run", False, "interrupted")
        except Exception as e:
            f.record("run", False, f"{type(e).__name__}: {e}")
        finally:
            header("TEARDOWN")
            if instance is None:
                print("  Nothing provisioned.")
            elif args.keep:
                print(f"  --keep: {instance.machine_id} LEFT RUNNING and billing.")
                f.record("teardown", False, "skipped via --keep")
            else:
                for attempt in range(1, 4):
                    try:
                        client.instances.destroy(instance.machine_id)
                        f.record("instance destroyed", True, f"attempt {attempt}")
                        break
                    except Exception as e:
                        f.note(f"destroy attempt {attempt}", str(e))
                        time.sleep(5)
                else:
                    f.record("instance destroyed", False,
                             f"DESTROY MANUALLY: jl destroy {instance.machine_id}")
            try:
                stray = [i for i in client.instances.list()
                         if getattr(i, "name", "") == INSTANCE_NAME]
                f.record("no stray instances", not stray,
                         "confirmed" if not stray else f"{len(stray)} left")
            except Exception as e:
                f.note("stray check", str(e))

    print(f"\nElapsed: {time.time() - started:.0f}s")
    f.save(args.out)
    failed = [s for s in f.steps if s["ok"] is False]
    if failed:
        print(f"\n{len(failed)} check(s) failed:")
        for s in failed:
            print(f"  - {s['step']}: {s['detail']}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
