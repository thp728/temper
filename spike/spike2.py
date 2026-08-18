"""Spike 2 — prove the SSH-driven bootstrap, the replacement for C11.

Spike 1 established that a JarvisLabs VM has Docker, a GPU, and egress, but
that startup scripts are silently ignored on --vm instances. So the bootstrap
in the architecture's section 14 has no mechanism to run. This spike proves the
only replacement that keeps custom images: the orchestrator SSHes in and drives
the bootstrap itself.

What it answers:

  1. Can the orchestrator drive a full bootstrap over SSH, unattended?
  2. How long does a realistic training image take to pull? (goes in the ETA --
     it happens on every cold run and is currently a guess)
  3. Does digest pinning work end to end -- resolve, then re-pull by digest?
  4. Does torch inside the container see the GPU, and is bf16 available?
  5. Does an artifact survive out of the container onto the host, and back to
     the orchestrator?
  6. C7: is a port on the VM's public IP reachable from outside?

Usage:
    python -u spike2.py            # full run, destroys the VM afterwards
    python -u spike2.py --keep     # leave it up (COSTS MONEY)

Reuses .env, GPU selection and teardown discipline from spike.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spike import Findings, header, load_dotenv  # noqa: E402

try:
    from jarvislabs import Client
except ImportError:
    sys.exit("jarvislabs SDK not installed. Run: pip install jarvislabs")

GPU_PREFERENCE = ["L4", "RTX-PRO6000", "H100"]
INSTANCE_NAME = "spike2-bootstrap"
STORAGE_GB = 100          # VM minimum, enforced by the platform
TEST_PORT = 8000
SSH_READY_TIMEOUT_S = 240
BOOTSTRAP_TIMEOUT_S = 1800   # image pull dominates; be generous


def ssh_base(ssh_command: str) -> list[str]:
    base = ssh_command.strip()
    if base.startswith("ssh "):
        base = base[4:]
    return ["ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30", *base.split()]


def wait_for_ssh(ssh_command: str, f: Findings) -> float | None:
    """Poll until sshd answers. 'Running' is not 'reachable' (correction C9)."""
    header("WAITING FOR SSH")
    t0 = time.time()
    while time.time() - t0 < SSH_READY_TIMEOUT_S:
        try:
            r = subprocess.run(ssh_base(ssh_command) + ["true"],
                               capture_output=True, text=True, timeout=25)
            if r.returncode == 0:
                dt = time.time() - t0
                f.record("ssh ready", True, f"{dt:.0f}s after Running",
                         ssh_ready_seconds=round(dt))
                return dt
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
    f.record("ssh ready", False, f"no answer within {SSH_READY_TIMEOUT_S}s")
    return None


def run_bootstrap(ssh_command: str, f: Findings) -> dict | None:
    """Pipe bootstrap.sh in over SSH. This IS the C11 replacement mechanism."""
    header("SSH-DRIVEN BOOTSTRAP")
    script = Path(__file__).parent / "bootstrap.sh"
    if not script.exists():
        f.record("bootstrap.sh present", False, str(script))
        return None
    f.record("bootstrap.sh present", True, f"{script.stat().st_size} bytes")

    # Bytes, NOT text=True. On Windows, Python's text mode translates '\n' to
    # os.linesep when writing to stdin, so a clean LF file arrives at bash as
    # CRLF and every line ends in a stray \r -- which surfaces as nonsense like
    # "ambiguous redirect" rather than anything pointing at line endings.
    # Normalise explicitly and hand over raw bytes.
    payload = script.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")

    print("  piping bootstrap.sh over SSH; image pull dominates, be patient...")
    t0 = time.time()
    try:
        r = subprocess.run(
            ssh_base(ssh_command) + ["bash -s"],
            input=payload, capture_output=True, timeout=BOOTSTRAP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        f.record("bootstrap executed", False,
                 f"timed out after {BOOTSTRAP_TIMEOUT_S}s")
        return None

    elapsed = time.time() - t0
    stdout = r.stdout.decode("utf-8", errors="replace")
    stderr = r.stderr.decode("utf-8", errors="replace")
    for line in stderr.splitlines():
        if line.strip():
            print(f"    {line.strip()}")

    if r.returncode != 0:
        f.record("bootstrap executed", False, f"exit {r.returncode}")
        return None

    try:
        report = json.loads(stdout)
    except json.JSONDecodeError:
        f.record("bootstrap executed", False, "stdout was not JSON")
        print(stdout[:800])
        return None

    f.record("bootstrap executed", True, f"{elapsed:.0f}s wall clock",
             bootstrap_seconds=round(elapsed))
    return report


def test_public_port(public_ip: str, f: Findings) -> None:
    """Correction C7. `http_ports` is rejected for VMs, but a VM has a real
    public IP -- so the question is whether a port on it is reachable without
    the platform's proxy. Deliberately tested with UFW untouched."""
    header(f"C7 -- PUBLIC PORT {TEST_PORT}")
    url = f"http://{public_ip}:{TEST_PORT}/"
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                body = resp.read(64).decode(errors="replace").strip()
            f.record("public port reachable", True,
                     f"{url} -> HTTP {resp.status} {body!r}. "
                     "C7 RETRACTED: VM serving works without the port proxy")
            return
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            if attempt == 4:
                f.record("public port reachable", False,
                         f"{url} unreachable ({reason}). C7 STANDS: serving "
                         "needs the container port-proxy or an SSH tunnel")
            time.sleep(4)


def interpret(report: dict, f: Findings) -> None:
    header("WHAT THIS MEANS")
    if report.get("pull_ok"):
        gb = report.get("image_bytes", 0) / 1e9
        s = report.get("pull_seconds", 0)
        print(f"  Image pull: {gb:.1f} GB in {s}s"
              f"{f' ({gb*1000/s:.0f} MB/s)' if s else ''}.")
        print("  This is cold-start latency on EVERY run and belongs in the ETA.")
        if s > 120:
            print("  >2min: worth caching the image on a JarvisLabs filesystem,")
            print("  which the architecture allows for public weights only --")
            print("  re-read that constraint, an image is not tenant data.")
    if report.get("digest_repull_ok"):
        print("  Digest pinning works. Principle 4 (immutable runs) is")
        print("  achievable on this platform.")
    else:
        print("  ! Digest re-pull FAILED -- immutability claim is not supported.")
    if report.get("gpu_in_container_ok"):
        print(f"  torch sees {report.get('gpu_name')}, "
              f"bf16={report.get('bf16_supported')}, "
              f"torch {report.get('torch_version')}.")
    else:
        print("  ! torch cannot see the GPU inside the container -- the whole")
        print("    containerised training path is blocked.")
    if report.get("artifact_ok"):
        print(f"  Artifact survived the container: "
              f"{report.get('artifact_bytes',0)/1e6:.1f} MB via bind mount.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--gpu")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "findings-spike2.json")
    args = ap.parse_args()

    env_file = Path(__file__).parent / ".env"
    if load_dotenv(env_file):
        print(f"Loaded credentials from {env_file.name}")

    f = Findings()
    instance = None
    started = time.time()

    try:
        client = Client()
    except Exception as e:
        print(f"Cannot authenticate: {e}")
        return 2

    with client:
        try:
            header("PREFLIGHT")
            rows = [r for r in client.account.gpu_availability()
                    if r.workload_type == "vm" and r.num_free_devices > 0]
            avail = {r.gpu_type: r for r in rows}
            gpu = args.gpu or next((g for g in GPU_PREFERENCE if g in avail), None)
            if not gpu:
                f.record("vm gpu available", False,
                         f"none of {GPU_PREFERENCE} free")
                return 1
            r = avail[gpu]
            f.record("vm gpu available", True,
                     f"{gpu} @ {r.price_per_hour}{client.account.currency()}/hr, "
                     f"{r.num_free_devices} free")

            header(f"PROVISIONING -- {gpu}, vm, {STORAGE_GB} GB")
            t0 = time.time()
            instance = client.instances.create(
                gpu_type=gpu, num_gpus=1, template="vm",
                storage=STORAGE_GB, name=INSTANCE_NAME,
                http_ports="",           # rejected for VMs; left explicit
            )
            f.record("VM created", True, f"{time.time()-t0:.0f}s to Running",
                     machine_id=instance.machine_id)
            print(f"  machine_id={instance.machine_id}  ip={instance.public_ip}")
            print(f"  ssh: {instance.ssh_command}")

            if wait_for_ssh(instance.ssh_command or "", f) is None:
                return 1

            report = run_bootstrap(instance.ssh_command or "", f)
            f.probe_report = report
            if report:
                interpret(report, f)
                if report.get("port_listening_locally"):
                    test_public_port(instance.public_ip, f)
                else:
                    f.record("public port reachable", False,
                             "listener never started on the VM; C7 untested")

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
                print(f"  Destroy: jl destroy {instance.machine_id}")
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
                         "confirmed via list()" if not stray
                         else f"{len(stray)} still listed")
            except Exception as e:
                f.note("stray check", str(e))

    print(f"\nElapsed: {time.time()-started:.0f}s")
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
