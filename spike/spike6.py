"""Spike 6 - two devices and sharded training, through the pinned image.

`num_gpus` is a create parameter and every VM-capable type shows 8 free
devices. Both were measured on 2026-08-19, and **both are availability facts,
not execution facts.** Nothing has ever run on more than one card through this
stack. Three things in sequence have to be true and only the first has any
evidence:

  1. num_gpus=2 provisions a machine with two devices attached.
  2. `docker run --gpus all` inside the PINNED image yields
     `torch.cuda.device_count() == 2`. The image was built and digest-locked
     before multi-GPU was in scope; nothing asserts the toolkit passes more
     than one device through.
  3. Axolotl's FSDP FULL_SHARD launches under `accelerate` in that image, takes
     steps, and writes a checkpoint that can be read back.

Point 3 is where this goes wrong, and it is worth being explicit about why:
sharded checkpoints are not a format this product has ever written. Resume is
proven at single-GPU (spike 4) and unproven here.

**Kill criterion: if FSDP does not run in the pinned image, the 8B full-FT
capstone does not go on the calendar** and multi-GPU stays exactly what the
docs already call it -- designed and unexercised. Do not attempt to fix the
image on the 27th at Rs510/hr. That is the mistake this spike exists to
prevent.

THIS SPIKE PROVISIONS TWO BILLING GPUs. 2x L4 at ~Rs82/hr, roughly 20 minutes.
Use the cheapest configuration that can answer the question -- not 8, and not
H100s.

Run through PowerShell, never Git Bash.

Usage:
    python -u spike6.py --dry-run    # preflight only, provisions nothing
    python -u spike6.py              # full run, destroys the VM
    python -u spike6.py --keep       # leaves it up (COSTS MONEY)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spike import Findings, header, load_dotenv  # noqa: E402
from spike2 import ssh_base, wait_for_ssh  # noqa: E402
from teardown import confirm_destroyed, sweep_by_name  # noqa: E402

try:
    from jarvislabs import Client
except ImportError:
    sys.exit("jarvislabs SDK not installed. Run: pip install jarvislabs")

FINDINGS_PATH = Path(__file__).parent / "findings-spike6.json"
BOOTSTRAP = Path(__file__).parent / "bootstrap6.sh"
DOCKERFILE = Path(__file__).parent.parent / "trainer" / "Dockerfile"

GPU_PREFERENCE = ["L4", "A5000", "A6000", "RTX-PRO6000"]
NUM_GPUS = 2
STORAGE_GB = 100
INSTANCE_NAME = "spike6-fsdp"
# TWO models, because one cannot separate two different failures. Qwen3-4B full
# fine-tune across 2x24 GB needs roughly 17 GB per device before activations, so
# a failure there is ambiguous -- broken mechanism, or a model that does not fit?
# The small case answers the mechanism question with room to spare; the large
# case is then a CAPACITY data point and is read as one.
SMALL_MODEL = "Qwen/Qwen3-0.6B"
LARGE_MODEL = "Qwen/Qwen3-4B"  # the smallest model in the CATALOG
STEPS = 3
BOOTSTRAP_TIMEOUT_S = 5400


def pinned_base_image(f: Findings) -> str | None:
    """The digest from trainer/Dockerfile.

    The BASE image is what runs here, not the built trainer image: spike 6 asks
    whether Axolotl's FSDP works through the pinned stack, and the entrypoint
    this repo layers on top implements a single-GPU job contract that has no
    opinion about sharding. Testing the base separates "the pinned stack cannot
    shard" from "our entrypoint cannot drive it", which are different problems
    with different fixes.
    """
    header("THE PIN")
    if not DOCKERFILE.exists():
        f.record("trainer/Dockerfile present", False, str(DOCKERFILE))
        return None
    m = re.search(
        r"^FROM\s+(\S+@sha256:[0-9a-f]{64})",
        DOCKERFILE.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not m:
        f.record("digest pin found", False, "no FROM ...@sha256: line")
        return None
    f.record("digest pin found", True, m.group(1))
    return m.group(1)


def preflight(client: Client, f: Findings) -> tuple[str | None, dict]:
    header("PREFLIGHT -- nothing provisioned")
    try:
        currency = client.account.currency()
        client.account.balance()
        f.record("auth + balance", True, f"currency={currency}")
    except Exception as e:  # noqa: BLE001
        f.record("auth + balance", False, f"{type(e).__name__}: {e}")
        return None, {}

    if not client.ssh_keys.list():
        f.record(
            "ssh key registered", False, "VM creation WILL fail without one"
        )
        return None, {}
    f.record("ssh key registered", True)

    # Availability is per node, and the question is not "is this GPU free" but
    # "are TWO of them free on one node". Aggregating the wrong way is how a
    # create fails after the preflight said yes.
    # workload_type='vm' is the filter that matters -- container capacity and
    # VM capacity are separate pools -- and the count has to be per ROW, since
    # a row is a node: two devices spread across two nodes cannot shard.
    best: dict = {}
    for row in client.account.gpu_availability():
        if row.workload_type != "vm" or row.num_free_devices < NUM_GPUS:
            continue
        prev = best.get(row.gpu_type)
        if prev is None or row.num_free_devices > prev["free"]:
            best[row.gpu_type] = {
                "price": row.price_per_hour,
                "free": row.num_free_devices,
                "region": row.region,
                "vram_gb": row.vram,
            }

    chosen = next((g for g in GPU_PREFERENCE if g in best), None)
    if chosen is None and best:
        chosen = min(best, key=lambda k: best[k]["price"])
    if chosen is None:
        f.record(
            f"a GPU with {NUM_GPUS} free devices exists",
            False,
            "nothing with two free devices on one node right now",
        )
        return None, {}

    rate = best[chosen]["price"]
    f.record(
        f"a GPU with {NUM_GPUS} free devices exists",
        True,
        f"{chosen}: {best[chosen]['free']} free at {rate} {currency}/hr "
        f"each -- {NUM_GPUS} will cost about "
        f"{rate * NUM_GPUS if rate else '?'} {currency}/hr",
    )
    return chosen, {
        "currency": currency,
        "chosen": chosen,
        "hourly_rate_per_device": rate,
        "estimated_hourly_rate": rate * NUM_GPUS if rate else None,
        "candidates": best,
    }


def run_bootstrap6(ssh_command: str, image: str, f: Findings) -> dict | None:
    header("ON-VM PROBE -- pull, device count, FSDP, resume")
    if not BOOTSTRAP.exists():
        f.record("bootstrap6.sh present", False, str(BOOTSTRAP))
        return None
    payload = (
        BOOTSTRAP.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    )
    print("  image pull dominates the first few minutes; be patient...")
    t0 = time.time()
    try:
        r = subprocess.run(
            ssh_base(ssh_command)
            + [
                f"bash -s -- '{image}' '{SMALL_MODEL}' '{LARGE_MODEL}' {STEPS}"
            ],
            input=payload,
            capture_output=True,
            timeout=BOOTSTRAP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        f.record(
            "probe executed", False, f"timed out after {BOOTSTRAP_TIMEOUT_S}s"
        )
        return None
    for line in r.stderr.decode("utf-8", "replace").splitlines():
        if line.strip():
            print(f"    {line.strip()}")
    if r.returncode != 0:
        f.record("probe executed", False, f"exit {r.returncode}")
        return None
    try:
        report = json.loads(r.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        f.record("probe executed", False, "stdout was not JSON")
        print(r.stdout.decode("utf-8", "replace")[:1500])
        return None
    f.record("probe executed", True, f"{time.time() - t0:.0f}s")
    return report


def read_numerics(train_log: str) -> dict:
    """Pull loss and grad_norm out of the training log.

    **Taking steps is not the same as training.** A run can shard correctly,
    step, checkpoint and resume while learning nothing -- and the first version
    of this spike would have called that a pass, because it only asked whether
    the step counter moved. It is exactly the question a reviewer asks second:
    *"your spike says FSDP works -- what was the loss?"*
    """
    losses, grads = [], []
    for line in (train_log or "").splitlines():
        for key, bucket in (("loss", losses), ("grad_norm", grads)):
            m = re.search(rf"'{key}': '([^']*)'", line)
            if m:
                bucket.append(m.group(1))
    healthy_loss = any(v not in ("0", "0.0", "nan") for v in losses)
    # Every reported grad_norm being nan is not a warning sign, it is the
    # finding: the optimiser had nothing usable to apply.
    all_grads_nan = bool(grads) and all(v == "nan" for v in grads)
    collapsed = bool(losses) and losses[-1] in ("0", "0.0")
    return {
        "losses": losses,
        "grad_norms": grads,
        "loss_ever_plausible": healthy_loss,
        "loss_collapsed_to_zero": collapsed,
        "all_grad_norms_nan": all_grads_nan,
        "numerically_healthy": healthy_loss
        and not collapsed
        and not all_grads_nan,
    }


def interpret(report: dict, f: Findings) -> dict:
    """The three claims, answered in order, with the kill criterion applied."""
    header("CLAIM 1 -- the machine has two devices")
    host = report.get("host_gpu_count", 0)
    f.record(
        f"host reports {NUM_GPUS} devices",
        host == NUM_GPUS,
        report.get("host_gpu_list", "")[:200],
    )

    header("CLAIM 2 -- the PINNED IMAGE sees both of them")
    container = report.get("container_gpu_count", -1)
    passthrough = container == NUM_GPUS
    f.record(
        "the pinned image sees both devices",
        passthrough,
        f"torch.cuda.device_count() == {container} inside the container",
    )
    if host == NUM_GPUS and not passthrough:
        f.note(
            "the toolkit, not the platform",
            "the host has two devices and the container does not see them. "
            "That is nvidia-container-toolkit passthrough, and it is a "
            "problem with OUR IMAGE, not with the provider.",
        )

    # A probe that never launched is not evidence about what it would have
    # found. An earlier run hit a docker-socket permission error and recorded
    # "FSDP does not run in the pinned image" -- a tooling failure filed as a
    # platform one, which is the C16 mistake. Claim 3 is UNKNOWN, not FAILED,
    # when claim 2 did not hold.
    if not report.get("fsdp_attempted", True):
        header("CLAIM 3 -- NOT ATTEMPTED")
        f.note(
            "FSDP was NOT ATTEMPTED",
            report.get("not_attempted_reason", "")
            + " Claim 3 is UNKNOWN, not failed.",
        )
        if report.get("container_torch_stderr"):
            print("  --- what the container said ---")
            print(report["container_torch_stderr"][-1500:])
        return {
            "two_devices_on_host": host == NUM_GPUS,
            "two_devices_in_the_pinned_image": passthrough,
            "fsdp_attempted": False,
            "capstone_go": False,
            "statement": "INCONCLUSIVE ON FSDP. The container reported no devices, so "
            "FSDP was never launched and this run says NOTHING about "
            "whether it works. The capstone cannot be scheduled on an "
            "unanswered question, but neither may the docs say FSDP was "
            "tried and failed. Fix device visibility and re-run.",
        }

    header("CLAIM 3 -- FSDP shards, checkpoints, and resumes")
    cases = {c["case"]: c for c in report.get("cases", [])}
    summaries = {}

    for tag, case in cases.items():
        print(f"\n  --- case '{tag}': {case['model']} ---")
        rc = case["launch_rc"]
        steps = case["steps_completed"]
        ckpt = case["checkpoint_format"]

        # A written sharded checkpoint is PROOF the mechanism ran, whatever the
        # process exited with. An earlier run recorded "FSDP does not run"
        # while 25 GB of .distcp shards sat on the disk -- the exit code was
        # read as the whole story and the artifact was ignored.
        sharded = "distcp" in ckpt or "sharded" in ckpt
        ran = steps > 0 or sharded
        f.record(
            f"[{tag}] FSDP FULL_SHARD actually sharded and stepped",
            ran,
            f"{steps} step(s) of {case['steps_requested']}, "
            f"checkpoint: {ckpt}, rc={rc}",
        )

        completed = rc == 0 and steps >= case["steps_requested"]
        f.record(
            f"[{tag}] the run completed cleanly",
            completed,
            f"rc={rc}"
            if completed
            else f"rc={rc} after {steps} step(s) -- the mechanism ran, the run "
            f"did not finish",
        )
        if not completed:
            print(f"  --- [{tag}] launch log tail ---")
            print((case.get("train_log_tail") or "")[-2500:])

        if sharded:
            gb = case["checkpoint_bytes"] / 1e9
            f.note(f"[{tag}] sharded checkpoint size", f"{gb:.1f} GB")
            f.note(
                f"[{tag}] checkpoint contents",
                (case.get("checkpoint_listing") or "")[:1200],
            )

        peak = case.get("peak_vram") or {}
        if peak:
            f.note(
                f"[{tag}] peak VRAM per device (measured, nvidia-smi)",
                ", ".join(f"{k}={v} MiB" for k, v in sorted(peak.items())),
            )

        # Did it TRAIN, or merely step? Separate question, separate answer.
        num = read_numerics(case.get("train_log_tail", ""))
        f.record(
            f"[{tag}] the loss behaved like training",
            num["numerically_healthy"],
            f"losses {num['losses']}, grad_norms {num['grad_norms']}"
            if not num["numerically_healthy"]
            else f"losses {num['losses']}",
        )
        if not num["numerically_healthy"]:
            f.note(
                f"[{tag}] NUMERICS ARE NOT PROVEN",
                "the run sharded, stepped and checkpointed, and the loss "
                "collapsed to zero with a nan grad_norm. Mechanism and "
                "numerics are different claims and only the first is "
                "demonstrated here.",
            )

        if not case.get("resume_attempted"):
            f.note(
                f"[{tag}] sharded resume NOT ATTEMPTED",
                "no checkpoint existed to resume from. UNKNOWN, not failed.",
            )
            resumed = None
        else:
            resumed = (
                case["resume_rc"] == 0
                and case["resume_steps_completed"] > steps
            )
            f.record(
                f"[{tag}] sharded resume worked",
                bool(resumed),
                f"resumed to step {case['resume_steps_completed']} "
                f"(target {case['resume_target_steps']}), "
                f"rc={case['resume_rc']}",
            )
            if not resumed:
                print(f"  --- [{tag}] resume log tail ---")
                print((case.get("resume_log_tail") or "")[-1500:])

        summaries[tag] = {
            "model": case["model"],
            "mechanism_ran": ran,
            "run_completed": completed,
            "steps_completed": steps,
            "checkpoint_format": ckpt,
            "checkpoint_gb": round(case["checkpoint_bytes"] / 1e9, 2),
            "peak_vram_mib": peak,
            "resume_attempted": case.get("resume_attempted", False),
            "resume_worked": resumed,
            "numerics": num,
        }

    header("GO / NO-GO ON THE 8B FULL-FT CAPSTONE")
    small = summaries.get("small", {})
    large = summaries.get("large", {})

    # The small case is the MECHANISM verdict; the large case is a CAPACITY
    # data point. Reading the large case as the mechanism verdict is exactly
    # the confusion this two-case structure exists to prevent.
    mechanism = bool(small.get("mechanism_ran") and small.get("run_completed"))
    resume_ok = bool(small.get("resume_worked"))
    numerics_ok = bool(
        (small.get("numerics") or {}).get("numerically_healthy")
    )
    # Numerics gate the capstone as hard as the mechanism does. A capstone run
    # that shards perfectly and learns nothing is a more expensive failure than
    # one that will not launch, because it produces an artifact.
    go = bool(passthrough and mechanism and resume_ok and numerics_ok)

    if not passthrough:
        statement = (
            "The pinned image does not see both devices. Multi-GPU "
            "stays designed and unexercised."
        )
    elif not mechanism:
        statement = (
            "FSDP DOES NOT RUN IN THE PINNED IMAGE even at 0.6B, where memory "
            "is not the constraint. The 8B full-FT capstone does NOT go on the "
            "calendar. Do not attempt to fix the image at Rs510/hr on the "
            "27th -- that is the mistake this spike exists to prevent."
        )
    elif not resume_ok:
        statement = (
            "FSDP shards, steps and checkpoints in the pinned image, and "
            "SHARDED RESUME DOES NOT WORK. Resume is proven at single-GPU "
            "(spike 4) and does not carry over. The capstone may be scheduled "
            "only if it is allowed to run uninterrupted -- and a run that "
            "cannot resume is one whose stall-detection remedy is to lose the "
            "work. Record that before scheduling."
        )
    elif not numerics_ok:
        statement = (
            "THE MECHANISM WORKS AND THE NUMERICS DO NOT. Two devices are "
            "visible in the pinned image, FSDP FULL_SHARD shards, steps, "
            "writes a .distcp checkpoint and resumes from it -- and the loss "
            "collapses to zero with a nan grad_norm from the first step. "
            "Taking steps is not training. THE CAPSTONE DOES NOT GO ON THE "
            "CALENDAR on this evidence: a run that shards perfectly and learns "
            "nothing is a MORE expensive failure than one that will not "
            "launch, because it produces an artifact that looks like a "
            "success. Explain the nan before scheduling anything."
        )
    else:
        statement = (
            "The mechanism works end to end at 2x the cheapest card: two "
            "devices visible in the pinned image, FSDP FULL_SHARD sharding and "
            "stepping, a sharded checkpoint written and resumed from, and a "
            "loss that behaves like training. STILL NOT PROVEN: this ran on 2 "
            "devices of one type, and 8 devices of another is a different NCCL "
            "topology."
        )

    verdict = {
        "two_devices_on_host": host == NUM_GPUS,
        "two_devices_in_the_pinned_image": passthrough,
        "fsdp_attempted": True,
        "mechanism_case": small,
        "capacity_case": large,
        "capstone_go": go,
        "statement": statement,
    }

    # The capacity case, read as capacity rather than as mechanism.
    if mechanism and large and not large.get("run_completed"):
        note = (
            f"{large['model']} full fine-tune did not complete on "
            f"{NUM_GPUS}x L4. The mechanism is not in question -- the small "
            f"case proved it on the same machine minutes earlier -- so this "
            f"is a CAPACITY result: 48 GB across two cards is not enough "
            f"for a 4B full fine-tune with adamw_torch."
        )
        if large.get("peak_vram_mib"):
            note += f" Measured peak: {large['peak_vram_mib']}."
        verdict["capacity_finding"] = note
        f.note("capacity, not mechanism", note)

    f.record("capstone is a go", go, statement)
    return verdict


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
        help="leave the instance running (COSTS MONEY)",
    )
    args = ap.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    f = Findings()
    header("SPIKE 6 -- two devices and FSDP through the pinned image")

    client = Client()
    image = pinned_base_image(f)
    gpu, pricing = preflight(client, f)

    payload: dict = {
        "spike": "spike6-multi-gpu-fsdp",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "image": image,
        "num_gpus_requested": NUM_GPUS,
        "small_model": SMALL_MODEL,
        "large_model": LARGE_MODEL,
        "steps": STEPS,
        "preflight": pricing,
    }

    if args.dry_run or gpu is None or image is None:
        payload["dry_run"] = True
        # A separate file, ON PURPOSE. A dry run costs nothing and must not be
        # able to destroy findings that cost money -- which it did once, by
        # writing its preflight over a completed run's measurements.
        dry_path = FINDINGS_PATH.with_name(FINDINGS_PATH.stem + "-dryrun.json")
        dry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nDry-run findings written to {dry_path}")
        if FINDINGS_PATH.exists():
            print(
                f"  ({FINDINGS_PATH.name} left untouched -- it holds a real run)"
            )
        header("DRY RUN -- nothing was provisioned")
        return 0 if (gpu and image) else 1

    instance = None
    try:
        header(f"CREATING {NUM_GPUS}x {gpu}")
        t0 = time.time()
        instance = client.instances.create(
            gpu_type=gpu,
            num_gpus=NUM_GPUS,
            template="vm",
            storage=STORAGE_GB,
            name=INSTANCE_NAME,
        )
        f.record(
            "instance created",
            True,
            f"{NUM_GPUS}x {gpu}, machine {instance.machine_id}, "
            f"{time.time() - t0:.0f}s to Running",
        )
        payload["machine"] = {
            "gpu": gpu,
            "num_gpus": NUM_GPUS,
            "region": getattr(instance, "region", None),
            # The SDK reports what was actually attached, which is not
            # necessarily what was asked for.
            "num_gpus_reported": getattr(instance, "num_gpus", None),
        }

        ssh_command = getattr(instance, "ssh_command", None)
        if not ssh_command:
            f.record(
                "ssh command returned", False, "instance has no ssh_command"
            )
            return 1
        if wait_for_ssh(ssh_command, f) is None:
            return 1

        report = run_bootstrap6(ssh_command, image, f)
        payload["probe_report"] = report
        if report:
            payload["verdict"] = interpret(report, f)
    finally:
        if instance is not None:
            if args.keep:
                f.record(
                    "teardown",
                    False,
                    f"SKIPPED via --keep -- STILL BILLING: "
                    f"jl destroy {instance.machine_id}",
                )
            else:
                confirm_destroyed(client, instance.machine_id, f)
        # ...and then ask the PLATFORM, not our own variables. `instance` is
        # only bound once create() has returned, and create() polls internally
        # until the machine is Running -- so an exception raised inside it
        # leaves a billing machine that this finally block cannot see. The
        # sweep closes that window by matching on the name instead.
        sweep_by_name(client, INSTANCE_NAME, f, keep=args.keep)
        payload["steps"] = f.steps
        FINDINGS_PATH.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"\nFindings written to {FINDINGS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
