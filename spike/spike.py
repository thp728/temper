"""Phase 0 vertical spike — JarvisLabs provisioning probe.

Proves, or disproves, the provider assumptions that everything else in
reference-technical-architecture.md is built on top of. Its §32 names
JarvisLabs VM automation as the highest technical risk in the build, and the
2026-08-17 corrections (§0) found the Docker assumption unverified.

What it answers, in order, stopping at the first one that fails:

  1. Does auth work and is there balance?
  2. Is an SSH key registered? (hard prerequisite for --vm)
  3. Which GPUs are actually available -- for VMs specifically, not containers?
  4. [RESOLVED offline 2026-08-17] Can the SDK create a VM? YES -- pass
     template="vm". The CLI's --vm flag does nothing else; see
     jarvislabs.cli.instance.resolve_vm_template(). Undocumented in the SDK
     reference, which is why this was an open question.
  5. Does a startup script upload and attach?
  6. Does the instance reach running, and how long does that take?
  7. Is Docker there? Does GHCR pull work? Does --gpus all work?  <-- the point
  8. Does the VM always get destroyed, including when things go wrong?

Deliberately does no training. A spike that also trains cannot tell you which
half broke.

Usage:
    export JL_API_KEY=...
    python spike.py --dry-run     # preflight only, provisions nothing
    python spike.py               # full run, destroys the instance afterwards
    python spike.py --keep        # leave it up for manual poking (COSTS MONEY)

Requires: pip install jarvislabs
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    from jarvislabs import Client
except ImportError:
    sys.exit("jarvislabs SDK not installed. Run: pip install jarvislabs")


def load_dotenv(path: Path) -> list[str]:
    """Minimal .env reader. Returns the names of the variables it set.

    Hand-rolled rather than pulling in python-dotenv: this is a spike, the
    format is four lines of KEY=value, and one fewer dependency is one fewer
    thing to pin. Does NOT overwrite variables already set in the environment,
    so an explicitly exported key still wins over a stale file.
    """
    if not path.exists():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


# Cheapest first, and VM-CAPABLE ONLY.
#
# Learned the hard way 2026-08-17: the first run picked A30 (cheapest overall,
# 2 devices free) and the API rejected it with "A30 is available for containers
# in IN2, but VM support is not available yet."
#
# The discriminator is `workload_type` on each availability row -- 'vm' or
# 'container' -- and the same gpu_type appears under both. A30 and A100 (40GB)
# are container-only. Filtering on it is done in preflight(); this list is only
# a cost preference within what is already VM-capable.
#
# VM-capable as of 2026-08-17 (INR/hr, on-demand):
#   L4 41.31 (24GB) | A100-80GB 140.94 | RTX-PRO6000 179.01 (96GB)
#   H100 255.15 (80GB) | H200 378.27 (141GB)
GPU_PREFERENCE = ["L4", "A100-80GB", "RTX-PRO6000", "H100"]

PROBE_SCRIPT = Path(__file__).parent / "probe.sh"
SCRIPT_NAME = "phase0-probe"
INSTANCE_NAME = "phase0-spike"
# NOT a free choice. Learned 2026-08-17 by asking for 40:
#   "Disk size must be at least 100 GB for V2 VM instances. Requested: 40 GB"
# So 100 GB is the VM floor, which is why the SDK defaults to it. Storage bills
# at $0.10/GB-month, so every VM carries a mandatory ~$10/month-equivalent
# storage footprint while it exists -- a fixed cost input the quote engine has
# to include and cannot optimise away.
STORAGE_GB = 100
BOOT_TIMEOUT_S = 600
SSH_RETRIES = 12
SSH_RETRY_DELAY_S = 10


@dataclass
class Findings:
    """Accumulates results so a late failure still returns everything learned."""

    steps: list[dict] = field(default_factory=list)
    probe_report: dict | None = None

    def record(self, step: str, ok: bool, detail: str = "", **extra) -> None:
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}] {step}" + (f" -- {detail}" if detail else ""))
        self.steps.append({"step": step, "ok": ok, "detail": detail, **extra})

    def note(self, step: str, detail: str, **extra) -> None:
        """Neither pass nor fail -- an observation worth carrying back."""
        print(f"  [ .. ] {step} -- {detail}")
        self.steps.append(
            {"step": step, "ok": None, "detail": detail, **extra}
        )

    def save(self, path: Path) -> None:
        payload = {
            "spike": "jarvislabs-phase0",
            "run_at": datetime.now(timezone.utc).isoformat(),
            "steps": self.steps,
            "probe_report": self.probe_report,
        }
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nFindings written to {path}")


def header(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def preflight(client: Client, f: Findings) -> str | None:
    """Checks that cost nothing. Returns a usable GPU type, or None."""
    header("PREFLIGHT -- no resources provisioned")

    # 1. Auth + balance.
    try:
        bal = client.account.balance()
        f.record(
            "auth + balance",
            True,
            f"balance={bal.balance} grants={bal.grants}",
        )
    except Exception as e:
        f.record("auth + balance", False, f"{type(e).__name__}: {e}")
        return None

    # 2. SSH key. --vm creation fails without one, and the failure message is
    #    not obviously about SSH keys, so check it explicitly.
    try:
        keys = client.ssh_keys.list()
        if keys:
            f.record("ssh key registered", True, f"{len(keys)} key(s)")
        else:
            f.record(
                "ssh key registered",
                False,
                "none found -- VM creation WILL fail. "
                "Fix: jl ssh-key add ~/.ssh/id_ed25519.pub --name my-laptop",
            )
            return None
    except Exception as e:
        f.record("ssh key registered", False, f"{type(e).__name__}: {e}")
        return None

    # 3. Availability. The pricing page lists what exists, not what is free
    #    right now -- and VM capacity is tracked separately from container
    #    capacity. Architecture §32 lists capacity between quote and launch as
    #    a named risk; this is the first look at how real that is.
    chosen = None
    try:
        rows = client.account.gpu_availability()

        # The API returns one row per node, so the same gpu_type appears
        # several times per region with different free counts. Aggregate before
        # deciding, and skip rows with nothing free -- a GPU being listed is
        # not the same as a GPU being available.
        # THE critical filter. Availability rows carry workload_type 'vm' or
        # 'container', and the same gpu_type appears under both with different
        # free counts. Aggregating across them is what made the first run pick
        # a container-only A30 and fail at create.
        vm_rows = [r for r in rows if r.workload_type == "vm"]
        container_only = sorted(
            {r.gpu_type for r in rows} - {r.gpu_type for r in vm_rows}
        )
        if container_only:
            f.note(
                "container-only GPUs (unusable for training)",
                ", ".join(container_only),
            )

        free: dict[str, dict] = {}
        for r in vm_rows:
            if r.num_free_devices < 1:
                continue
            cur = free.setdefault(
                r.gpu_type,
                {
                    "free": 0,
                    "price": r.price_per_hour,
                    "spot": r.spot_price,
                    "vram": r.vram,
                    "regions": set(),
                },
            )
            cur["free"] += r.num_free_devices
            cur["regions"].add(r.region)
            cur["price"] = min(cur["price"], r.price_per_hour)

        currency = client.account.currency()
        summary = ", ".join(
            f"{g}({v['free']}free@{v['price']}{currency})"
            for g, v in sorted(free.items(), key=lambda kv: kv[1]["price"])
        )
        f.note("VM-capable availability", summary or "nothing free")

        # Billing currency is account-scoped and is NOT necessarily USD.
        # The quote engine must read this rather than assume it; the
        # architecture's USD anchors are calibration, not billing.
        f.note(
            "billing currency", f"{currency} -- quote must not hard-code USD"
        )

        for gpu in GPU_PREFERENCE:
            if gpu in free:
                chosen = gpu
                break
        if chosen:
            i = free[chosen]
            f.record(
                "gpu selected",
                True,
                f"{chosen}: {i['free']} free @ {i['price']}{currency}/hr, "
                f"{i['vram']}GB, {sorted(i['regions'])}",
            )
        else:
            f.record(
                "gpu selected",
                False,
                f"none of {GPU_PREFERENCE} free for VMs. "
                f"VM-capable and free: {summary or 'none'}",
            )
            return None
    except Exception as e:
        # Do NOT fall back to a guess. Provisioning a GPU we could not confirm
        # is free is how you get a surprise bill or a hang.
        f.record("gpu availability", False, f"{type(e).__name__}: {e}")
        return None

    # 4. VM mode. RESOLVED 2026-08-17 by reading the SDK source rather than the
    #    docs: create() has no `vm` parameter, because --vm is not a parameter.
    #    jarvislabs.cli.instance.resolve_vm_template() sets template="vm", and
    #    that is all --vm does. So the SDK CAN create VMs; the docs just never
    #    say how. Asserted here so a future SDK release that changes it fails
    #    loudly instead of silently provisioning a container.
    try:
        params = list(inspect.signature(client.instances.create).parameters)
        f.note("SDK create() params", ", ".join(params))
        if "template" in params:
            f.record("VM reachable from SDK", True, 'via template="vm"')
        else:
            f.record(
                "VM reachable from SDK",
                False,
                "no template parameter -- SDK contract changed, re-read the source",
            )
    except Exception as e:
        f.note("SDK create() introspection", f"{type(e).__name__}: {e}")

    return chosen


def upload_probe(client: Client, f: Findings) -> str | None:
    """Register probe.sh as a startup script. Returns its id."""
    header("STARTUP SCRIPT")

    if not PROBE_SCRIPT.exists():
        f.record("probe.sh present", False, f"not found at {PROBE_SCRIPT}")
        return None

    body = PROBE_SCRIPT.read_text()
    f.record("probe.sh present", True, f"{len(body)} bytes")

    # NOTE: the field is `script_name`, not `name`, and add()/update() both
    # return a plain bool -- NOT the created object. The first run recorded
    # "id=True" and would have passed the string "True" as script_id, so the
    # probe would never have run. Always re-list to resolve the real id.
    def find_id() -> str | None:
        for s in client.scripts.list():
            if getattr(s, "script_name", None) == SCRIPT_NAME:
                return str(s.script_id)
        return None

    try:
        sid = find_id()
        if sid:
            client.scripts.update(script_id=int(sid), script=body)
            f.record("startup script updated", True, f"id={sid}")
            return sid

        client.scripts.add(script=body, name=SCRIPT_NAME)
        sid = find_id()
        if sid:
            f.record("startup script created", True, f"id={sid}")
            return sid
        f.record(
            "startup script created",
            False,
            "add() reported success but the script is not in list()",
        )
        return None
    except Exception as e:
        f.record("startup script", False, f"{type(e).__name__}: {e}")
        return None


def fetch_probe_report(ssh_command: str, f: Findings) -> dict | None:
    """SSH in and read the report the startup script left behind.

    Retries: the SDK returns when the instance is *running*, but the startup
    script runs after that, and sshd may not be up yet either.
    """
    header("PROBE REPORT")

    if not ssh_command:
        f.record("ssh command available", False, "instance exposed none")
        return None

    base = ssh_command.strip()
    if base.startswith("ssh "):
        base = base[4:]
    # /tmp because VMs log in as `ubuntu` and /root is 700. The sudo fallback
    # covers a startup script that ran with a stricter umask, and `2>/dev/null`
    # on the first arm keeps the real error visible if both fail.
    remote = (
        "cat /tmp/probe-report.json 2>/dev/null "
        "|| sudo cat /tmp/probe-report.json 2>/dev/null "
        "|| sudo cat /root/probe-report.json"
    )
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
        *base.split(),
        remote,
    ]

    t_start = time.time()
    ssh_reachable_at = None

    for attempt in range(1, SSH_RETRIES + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            # "Running" from the API is not "reachable". The first attempts
            # time out at the TCP level while sshd is still coming up; once we
            # get ANY response (even a failed cat), the port is open. That gap
            # is orchestration-relevant: the worker cannot be handed work at
            # the moment the provider says Running.
            if ssh_reachable_at is None and "Connection timed out" not in (
                r.stderr or ""
            ):
                ssh_reachable_at = time.time() - t_start
                f.record(
                    "ssh port reachable",
                    True,
                    f"{ssh_reachable_at:.0f}s after Running "
                    f"(attempt {attempt}) -- 'Running' != 'ready'",
                    ssh_ready_seconds=round(ssh_reachable_at),
                )
            if r.returncode == 0 and r.stdout.strip():
                try:
                    report = json.loads(r.stdout)
                except json.JSONDecodeError:
                    f.note(
                        f"ssh attempt {attempt}",
                        "report present but unparseable",
                    )
                    print(r.stdout[:500])
                    time.sleep(SSH_RETRY_DELAY_S)
                    continue
                f.record(
                    "probe report retrieved",
                    True,
                    f"after {attempt} attempt(s)",
                )
                return report
            f.note(
                f"ssh attempt {attempt}/{SSH_RETRIES}",
                (r.stderr or "no report yet").strip()[:120],
            )
        except subprocess.TimeoutExpired:
            f.note(f"ssh attempt {attempt}/{SSH_RETRIES}", "timed out")
        except FileNotFoundError:
            f.record(
                "ssh client available",
                False,
                "no `ssh` on PATH -- install OpenSSH or read the report manually",
            )
            return None
        time.sleep(SSH_RETRY_DELAY_S)

    f.record(
        "probe report retrieved",
        False,
        f"gave up after {SSH_RETRIES} attempts; ssh in manually: {ssh_command}",
    )
    return None


def interpret(report: dict | None, f: Findings) -> None:
    """Turn the probe into the decisions it forces."""
    header("WHAT THIS MEANS")

    if not report:
        print(
            "  No probe report -- the C1 Docker question is still unanswered."
        )
        return

    docker = report.get("docker_present")
    daemon = report.get("docker_daemon_running")
    ghcr = report.get("ghcr_pull_ok")
    gpus = report.get("docker_gpu_passthrough_ok")

    if docker and daemon and ghcr and gpus:
        print("  C1 RESOLVED, favourably. Docker present, daemon up, GHCR")
        print(
            "  reachable, GPU passthrough works. The immutable-image approach"
        )
        print("  in architecture §14 holds -- correct §0/C1 to resolved and")
        print("  proceed with the bootstrap as written.")
    elif docker and daemon and ghcr and not gpus:
        print(
            "  C1 PARTIAL. Docker works but the container cannot see the GPU."
        )
        print("  Almost certainly a missing nvidia-container-toolkit -- the")
        print(
            "  startup script must install it. Adds boot latency to every run;"
        )
        print("  measure it before quoting durations.")
    elif docker and not daemon:
        print(
            "  C1 PARTIAL. Docker installed but the daemon is not running by"
        )
        print(
            "  default. The startup script has to start it. Cheap fix, but it"
        )
        print("  must be in the script and not assumed.")
    elif not docker:
        print("  C1 RESOLVED, unfavourably. No Docker on a bare VM.")
        print("  DECISION REQUIRED -- log it in decisions.md:")
        print("    (a) startup script installs Docker  -> keeps immutability,")
        print("        costs boot time on every run, and the install itself")
        print("        becomes an unpinned dependency;")
        print(
            "    (b) uv-provisioned env from a lockfile -> fast, but forfeits"
        )
        print("        the immutable-image guarantee principle 10 rests on;")
        print(
            "    (c) use a container template instead of --vm -> Docker becomes"
        )
        print(
            "        irrelevant, at the cost of controlling the image at all."
        )
        print(
            "  Whichever is chosen, architecture §5, §14 and principle 10 all"
        )
        print("  need rewriting, not patching.")

    for host in ("ghcr_io", "huggingface_co", "pypi_org"):
        key = f"egress_{host}"
        if key in report and not report[key]:
            print(f"  ! EGRESS BLOCKED to {host} -- breaks the §15 callback")
            print("    protocol and R2 signed URLs. Escalate before building.")

    if report.get("nvidia_smi_present"):
        print(
            f"  GPU seen: {report.get('gpu_name')} "
            f"({report.get('vram_total_mib')} MiB), "
            f"driver {report.get('driver_version')}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="preflight only; provisions nothing, costs nothing",
    )
    ap.add_argument(
        "--keep",
        action="store_true",
        help="do not destroy the instance (COSTS MONEY until you do)",
    )
    ap.add_argument("--gpu", help="override GPU type")
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "findings.json"
    )
    args = ap.parse_args()

    f = Findings()
    instance = None
    started = time.time()

    # Construct the client outside the main try: a missing key is a setup
    # problem, not a spike result, and a traceback is the wrong output for a
    # tool whose entire job is legible failure.
    env_file = Path(__file__).parent / ".env"
    loaded = load_dotenv(env_file)
    if loaded:
        print(f"Loaded {', '.join(loaded)} from {env_file.name}")

    try:
        client = Client()
    except Exception as e:
        print(f"\nCannot authenticate: {e}\n")
        if env_file.exists() and not loaded:
            print(
                f"{env_file} exists but set nothing -- is JL_API_KEY still blank?"
            )
        print("Fix any one of:")
        print(f"  1. Put JL_API_KEY=<key> in {env_file}")
        print('  2. $env:JL_API_KEY = "<key>"   # this shell only')
        print("  3. jl setup                     # persists to a config file")
        print("\nKey from: https://jarvislabs.ai/settings/api-keys")
        return 2

    with client:
        try:
            gpu = args.gpu or preflight(client, f)
            if gpu is None:
                f.record("preflight", False, "blocked -- fix the above first")
                return 1

            if args.dry_run:
                header("DRY RUN -- stopping before provisioning")
                return 0

            script_id = upload_probe(client, f)

            header(f"PROVISIONING -- {gpu}, vm mode, {STORAGE_GB} GB")
            print(
                "  (SDK create() blocks until running; boot timeout "
                f"{BOOT_TIMEOUT_S}s)"
            )

            # template="vm" IS --vm. Note the constraints this buys, all read
            # from the CLI source and all load-bearing for the architecture:
            #   * VMs are SSH-only -- http_ports is rejected outright
            #   * VMs cannot be spot -- "--spot is only supported for GPU
            #     container instances"
            #   * storage defaults to 100 GB in the SDK, and bills at
            #     $0.10/GB-month
            create_kwargs = dict(
                gpu_type=gpu,
                num_gpus=1,
                template="vm",
                storage=STORAGE_GB,
                name=INSTANCE_NAME,
            )
            if script_id:
                create_kwargs["script_id"] = str(script_id)

            t0 = time.time()
            instance = client.instances.create(**create_kwargs)
            f.record(
                "VM created",
                True,
                f"{time.time() - t0:.0f}s to Running",
                boot_seconds=round(time.time() - t0),
            )

            mid = instance.machine_id
            print(f"  machine_id={mid}  status={instance.status}")
            print(f"  ssh: {instance.ssh_command}")
            f.note("instance", f"id={mid} gpu={gpu}", machine_id=mid)

            report = fetch_probe_report(instance.ssh_command or "", f)
            f.probe_report = report
            interpret(report, f)

        except KeyboardInterrupt:
            print("\n  Interrupted -- tearing down.")
            f.record("run", False, "interrupted by user")
        except Exception as e:
            f.record("run", False, f"{type(e).__name__}: {e}")
        finally:
            # Architecture §15: the orchestrator destroys the VM in a finally
            # path after terminal status or timeout. This is the rehearsal for
            # that, and the single most expensive thing to get wrong -- an
            # orphaned GPU bills until someone notices.
            header("TEARDOWN")
            if instance is None:
                print("  Nothing provisioned.")
            elif args.keep:
                print(
                    f"  --keep set. Instance {instance.machine_id} LEFT RUNNING."
                )
                print(f"  Destroy it: jl destroy {instance.machine_id}")
                f.record(
                    "teardown", False, "skipped via --keep -- still billing"
                )
            else:
                for attempt in range(1, 4):
                    try:
                        client.instances.destroy(instance.machine_id)
                        f.record(
                            "instance destroyed", True, f"attempt {attempt}"
                        )
                        break
                    except Exception as e:
                        f.note(
                            f"destroy attempt {attempt}",
                            f"{type(e).__name__}: {e}",
                        )
                        time.sleep(5)
                else:
                    f.record(
                        "instance destroyed",
                        False,
                        f"ALL ATTEMPTS FAILED -- destroy manually NOW: "
                        f"jl destroy {instance.machine_id}",
                    )

            # Independent confirmation. Trusting the destroy call's return value
            # is exactly the assumption the §17 reconciler exists to catch.
            try:
                live = client.instances.list()
                stray = [
                    i for i in live if getattr(i, "name", "") == INSTANCE_NAME
                ]
                if stray:
                    f.record(
                        "no stray instances",
                        False,
                        f"{len(stray)} still listed -- destroy manually",
                    )
                else:
                    f.record(
                        "no stray instances", True, "confirmed via list()"
                    )
            except Exception as e:
                f.note("stray check", f"{type(e).__name__}: {e}")

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
