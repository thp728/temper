"""Spike 3: firewall, a real QLoRA run, checkpoint/resume, adapter integrity.

Spike 1 proved a VM has Docker and a GPU. Spike 2 proved the orchestrator can
bootstrap it over SSH and that ports on a VM are reachable -- because nothing
filters them (correction C12). Spike 3 closes C12 and then does the thing the
whole platform exists to do: fine-tune a model.

What it answers:

  1. C12 -- does default-deny UFW actually block the port, without locking us
     out? (the orchestrator now expects the port test to FAIL)
  2. Does the real stack -- transformers/peft/trl/bitsandbytes -- install and
     run QLoRA on an L4, or does it only look plausible on paper?
  3. How long does installing that stack take? That is the evidence for whether
     a purpose-built image is worth a build pipeline.
  4. Does checkpoint/resume restore the step counter, as section 31 requires?
  5. Does the adapter come back intact -- SHA matching between container and
     host, and a trainable-parameter count matching the wiki's arithmetic?
  6. tokens/sec and peak VRAM, to calibrate the estimator's softest numbers.

Usage:
    python -u spike3.py                  # full run, destroys the VM
    python -u spike3.py --keep           # leave it up (COSTS MONEY)
    python -u spike3.py --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spike import Findings, header, load_dotenv  # noqa: E402
from spike2 import ssh_base, wait_for_ssh  # noqa: E402

try:
    from jarvislabs import Client
except ImportError:
    sys.exit("jarvislabs SDK not installed. Run: pip install jarvislabs")

GPU_PREFERENCE = ["L4", "RTX-PRO6000", "H100"]
INSTANCE_NAME = "spike3-qlora"
STORAGE_GB = 100
TEST_PORT = 8000
BOOTSTRAP_TIMEOUT_S = 3600  # model download + install + two training runs

# From wiki/foundations.md, computed off Qwen3-8B's config. The 4B differs, so
# this is a sanity band rather than an equality check -- LoRA r=16 all-linear
# should land well under 1% of parameters on any model in this family.
TRAINABLE_PCT_MAX = 1.5


def run_bootstrap(ssh_command: str, model: str, f: Findings) -> dict | None:
    header("SSH-DRIVEN BOOTSTRAP: firewall, install, QLoRA")
    script = Path(__file__).parent / "bootstrap3.sh"
    if not script.exists():
        f.record("bootstrap3.sh present", False, str(script))
        return None
    f.record("bootstrap3.sh present", True, f"{script.stat().st_size} bytes")

    # Bytes, not text=True -- Windows would translate \n to \r\n on the pipe.
    payload = script.read_text(encoding="utf-8").replace("\r\n", "\n").encode()

    print(
        f"  model={model}. Weights download + pip install + 2 training runs."
    )
    print("  This is the long one; expect several minutes.")
    t0 = time.time()
    try:
        r = subprocess.run(
            ssh_base(ssh_command) + [f"BASE_MODEL={model} bash -s"],
            input=payload,
            capture_output=True,
            timeout=BOOTSTRAP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        f.record(
            "bootstrap executed",
            False,
            f"timed out after {BOOTSTRAP_TIMEOUT_S}s",
        )
        return None

    for line in r.stderr.decode("utf-8", "replace").splitlines():
        if line.strip():
            print(f"    {line.strip()}")

    stdout = r.stdout.decode("utf-8", "replace")
    try:
        report = json.loads(stdout[stdout.index("{") :])
    except (ValueError, json.JSONDecodeError):
        f.record("bootstrap executed", False, "stdout was not JSON")
        print(stdout[:1000])
        return None
    f.record("bootstrap executed", True, f"{time.time() - t0:.0f}s wall clock")
    return report


def test_port_blocked(public_ip: str, rep: dict, f: Findings) -> None:
    """C12. Inverted from spike 2: success here means the port is UNREACHABLE.

    Attempt 1 of spike 3 showed ufw reporting `Status: active` with default-deny
    while the published port stayed wide open, because Docker publishes via
    NAT/FORWARD and ufw only filters INPUT. So this now tests the DOCKER-USER
    chain rule, and separately checks the loopback-bound port as the mitigation
    that needs no firewall at all.
    """
    header(f"C12: PORT {TEST_PORT} SHOULD NOW BE BLOCKED")
    for label, port, expect_blocked in (
        ("published port (DOCKER-USER rule)", TEST_PORT, True),
        ("loopback-bound port", TEST_PORT + 1, True),
    ):
        url = f"http://{public_ip}:{port}/"
        try:
            with urllib.request.urlopen(url, timeout=12) as resp:
                body = resp.read(32).decode(errors="replace").strip()
            reachable, detail = True, f"HTTP {resp.status} {body!r}"
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            reachable, detail = False, str(getattr(e, "reason", e))
        f.record(
            f"blocked from outside: {label}",
            expect_blocked and not reachable,
            f"{url} {'REACHABLE ' + detail if reachable else 'unreachable (' + detail + ')'}",
        )

    if rep.get("docker_user_rule_installed"):
        print(
            "\n  DOCKER-USER rule was installed. If the published port is now"
        )
        print(
            "  blocked but was open under ufw alone, that is the whole finding:"
        )
        print("  ufw is not a firewall for containers.")
    if rep.get("loopback_bound_port_ok"):
        print(
            "  The loopback-bound container answered on 127.0.0.1, so binding"
        )
        print("  to loopback keeps a service working locally while never")
        print(
            "  publishing it. Prefer this wherever public reach is not needed."
        )


def interpret(rep: dict, f: Findings) -> None:
    header("WHAT THIS MEANS")
    tr = rep.get("train_result") or {}

    if rep.get("pip_install_ok"):
        s = rep.get("pip_install_seconds", 0)
        print(f"  Stack install: {s}s. {rep.get('versions', '')}")
        if s > 90:
            print("  >90s of pure setup on EVERY cold run. That is the")
            print("  argument for baking a purpose-built image rather than")
            print("  installing at boot. Build it once, pull it by digest.")

    dropped = tr.get("sftconfig_dropped")
    if dropped:
        print(
            f"\n  ⚠ SFTConfig REJECTED {len(dropped)} of our intended args: "
            f"{', '.join(dropped)}"
        )
        print("  TRL 1.x no longer inherits TrainingArguments. Those defaults")
        print("  cannot be expressed on this version, which is exactly the")
        print("  dependency drift a pinned image exists to prevent.")

    if rep.get("train_ok") and tr:
        print(
            f"\n  QLoRA RAN. steps={tr.get('steps_completed')}, "
            f"loss={tr.get('train_loss')}, "
            f"peak VRAM={tr.get('peak_vram_gb')} GB on a 24 GB L4"
        )
        print(
            f"  model load: {tr.get('load_seconds')}s, "
            f"train: {tr.get('train_seconds')}s"
        )
        if tr.get("tokens_per_second"):
            print(
                f"  {tr['tokens_per_second']} tokens/sec, feeds the MFU "
                "constant, the softest number in the cost model"
            )
        tp = tr.get("trainable_pct")
        if tp is not None:
            ok = tp < TRAINABLE_PCT_MAX
            print(
                f"  trainable: {tr.get('trainable_params'):,} ({tp}%) "
                f"{'✓ matches the LoRA arithmetic' if ok else '✗ UNEXPECTED'}"
            )
        print(f"  checkpoints written: {tr.get('checkpoints')}")
    else:
        print(
            "\n  ! QLoRA did NOT run. The stack does not work as configured;"
        )
        print("    see the log tail above. This blocks the entire build.")

    if tr.get("resume_ok"):
        print(
            f"\n  Resume works: restarted from {tr.get('resume_from')} and "
            f"reached step {tr.get('resume_final_step')}. Section 31's "
            "'survives a forced interruption' criterion is achievable."
        )
    elif "resume_error" in tr:
        print(f"\n  ! Resume FAILED: {tr['resume_error']}")

    c_sha, h_sha = tr.get("adapter_sha256"), rep.get("host_adapter_sha256")
    if c_sha and h_sha:
        print(
            f"\n  Adapter {tr.get('adapter_bytes', 0) / 1e6:.1f} MB, "
            f"SHA {'MATCHES' if c_sha == h_sha else 'MISMATCH'} "
            "between container and host."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--gpu")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "findings-spike3.json",
    )
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
            avail = {
                r.gpu_type: r
                for r in client.account.gpu_availability()
                if r.workload_type == "vm" and r.num_free_devices > 0
            }
            gpu = args.gpu or next(
                (g for g in GPU_PREFERENCE if g in avail), None
            )
            if not gpu:
                f.record(
                    "vm gpu available", False, f"none of {GPU_PREFERENCE} free"
                )
                return 1
            r = avail[gpu]
            f.record(
                "vm gpu available",
                True,
                f"{gpu} @ {r.price_per_hour}{client.account.currency()}/hr, "
                f"{r.vram}GB",
            )

            header(f"PROVISIONING: {gpu}, vm, {STORAGE_GB} GB")
            t0 = time.time()
            instance = client.instances.create(
                gpu_type=gpu,
                num_gpus=1,
                template="vm",
                storage=STORAGE_GB,
                name=INSTANCE_NAME,
            )
            f.record(
                "VM created",
                True,
                f"{time.time() - t0:.0f}s to Running",
                machine_id=instance.machine_id,
            )
            print(
                f"  machine_id={instance.machine_id}  ip={instance.public_ip}"
            )

            if wait_for_ssh(instance.ssh_command or "", f) is None:
                return 1

            rep = run_bootstrap(instance.ssh_command or "", args.model, f)
            f.probe_report = rep
            if rep:
                tr = rep.get("train_result") or {}
                f.record(
                    "stack installed",
                    bool(rep.get("pip_install_ok")),
                    rep.get("versions", ""),
                )
                f.record(
                    "QLoRA trained",
                    bool(rep.get("train_ok")),
                    f"steps={tr.get('steps_completed')} "
                    f"loss={tr.get('train_loss')} "
                    f"peakVRAM={tr.get('peak_vram_gb')}GB",
                )
                f.record(
                    "checkpoint resume",
                    bool(tr.get("resume_ok")),
                    f"from {tr.get('resume_from')} to step "
                    f"{tr.get('resume_final_step')}"
                    if tr.get("resume_ok")
                    else tr.get("resume_error", "n/a"),
                )
                c, h = tr.get("adapter_sha256"), rep.get("host_adapter_sha256")
                f.record(
                    "adapter integrity",
                    bool(c and c == h),
                    f"{tr.get('adapter_bytes', 0) / 1e6:.1f} MB, sha "
                    f"{'match' if c and c == h else 'MISMATCH/absent'}",
                )
                interpret(rep, f)
                test_port_blocked(instance.public_ip, rep, f)

        except KeyboardInterrupt:
            f.record("run", False, "interrupted")
        except Exception as e:
            f.record("run", False, f"{type(e).__name__}: {e}")
        finally:
            header("TEARDOWN")
            if instance is None:
                print("  Nothing provisioned.")
            elif args.keep:
                print(
                    f"  --keep: {instance.machine_id} LEFT RUNNING and billing."
                )
                print(f"  Destroy: jl destroy {instance.machine_id}")
                f.record("teardown", False, "skipped via --keep")
            else:
                for attempt in range(1, 4):
                    try:
                        client.instances.destroy(instance.machine_id)
                        f.record(
                            "instance destroyed", True, f"attempt {attempt}"
                        )
                        break
                    except Exception as e:
                        f.note(f"destroy attempt {attempt}", str(e))
                        time.sleep(5)
                else:
                    f.record(
                        "instance destroyed",
                        False,
                        f"DESTROY MANUALLY: jl destroy {instance.machine_id}",
                    )
            try:
                stray = [
                    i
                    for i in client.instances.list()
                    if getattr(i, "name", "") == INSTANCE_NAME
                ]
                f.record(
                    "no stray instances",
                    not stray,
                    "confirmed via list()"
                    if not stray
                    else f"{len(stray)} left",
                )
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
