"""Spike 5 - the disk ceiling, and how fast model weights arrive.

Two numbers, both load-bearing, neither ever measured.

**Disk.** `create()` enforces a 100 GB minimum on VMs, measured in spike 1 and
recorded as a floor. Nothing has ever asked for more. Llama-3.3-70B in bf16 is
~141 GB of weights, which does not fit inside the floor -- and QLoRA does not
help, because the bf16 weights land on disk before they are quantised on load.
With checkpoints and a merged output, a 70B job needs somewhere between 250 and
400 GB.

**Throughput.** The measured cold start is 2-4 minutes, of which 87-183s is the
trainer image. The only model this product has ever downloaded is ~8 GB. A 70B
model is ~140 GB from Hugging Face onto a JarvisLabs VM at a rate nobody has
measured. At 200 MB/s that is 12 minutes; at 50 MB/s it is 47. **That
difference is the whole ETA of the `preparing` phase** -- a number the quote
promises to the user before they spend anything.

**Kill criterion: if disk cannot exceed ~200 GB, 70B full fine-tuning is off
the table on this provider and the spec says so in those words.** That is a
finding, not a failure, and it is worth having on a Sunday rather than in the
final week.

THIS SPIKE PROVISIONS A BILLING GPU. Roughly 15 minutes on an L4 at ~Rs41/hr,
plus whatever the extra storage costs -- which is one of the things it measures.

Run through PowerShell, never Git Bash: Git Bash ships its own ssh and cannot
see the Windows ssh-agent, and the failure presents as a dead VM.

Usage:
    python -u spike5.py --dry-run     # preflight only, provisions nothing
    python -u spike5.py               # full run, destroys the VM
    python -u spike5.py --keep        # leaves it up (COSTS MONEY)
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

FINDINGS_PATH = Path(__file__).parent / "findings-spike5.json"
BOOTSTRAP = Path(__file__).parent / "bootstrap5.sh"

GPU_PREFERENCE = ["L4", "RTX-PRO6000", "A5000", "A6000", "H100"]
INSTANCE_NAME = "spike5-disk"

# Descending, because a rejection is free and a success costs money. The first
# value that is accepted is the one we keep and pay for, so the ladder starts
# at the size a 70B job needs and stops at the floor spike 1 measured.
# The first run accepted the top of the ladder (1000 GB) with zero rejections,
# which located a FLOOR on the ceiling rather than the ceiling. The top is
# raised so the ladder can actually find it; rejections above it stay free.
STORAGE_LADDER_GB = [8000, 4000, 2000, 1000, 600, 400, 300, 200, 150, 100]

# The largest repo that is not itself an hour of waiting. 8B in bf16 is ~16 GB,
# which is enough to reach steady state and long enough to see whether the rate
# is steady or bursty -- and it is the largest model in the current catalog, so
# the number is directly usable rather than only extrapolated.
DOWNLOAD_REPO = "Qwen/Qwen3-8B"

BOOTSTRAP_TIMEOUT_S = 3600


def ceiling_from_rejection(message: str) -> int | None:
    """Pull the storage ceiling out of an API rejection, if it names one.

    A refusal that names its bound is a better answer than a bisect: it is the
    platform's own number rather than the largest value we happened to try. It
    is also a different product experience from "invalid request", which is
    worth recording about a provider whose features this product reflects.

    Measured 2026-08-23: requesting 8000 GB returns
    `hdd: ensure this value is less than or equal to 7200`, among several other
    complaints about fields a GPU create does not send -- so the pattern is
    anchored on `hdd:` rather than matching the first number in the string.

    Pure, so it can be tested without provisioning anything. The run that first
    saw this message had already created its machine before this function
    existed, and re-provisioning a 4 TB disk to exercise a regex would be
    spending money on something a test covers.
    """
    m = re.search(r"hdd:.*?less than or equal to (\d+)", message)
    return int(m.group(1)) if m else None


def read_storage_bounds(f: Findings) -> dict:
    """What does the SDK itself say about the storage parameter?

    Answers the first method step -- is the ceiling documented, enforced, or
    discovered by rejection? -- before anything is provisioned, because the
    answer changes how the rest of the spike has to be run.
    """
    header("STORAGE PARAMETER -- what the SDK says before anything is created")
    import inspect

    from jarvislabs import instances as inst_mod

    sig = inspect.signature(inst_mod.Instances.create)
    param = sig.parameters.get("storage")
    default = param.default if param else None
    f.record("storage is a create parameter", param is not None,
             f"storage: {param.annotation if param else '?'} = {default}")

    source = inspect.getsource(inst_mod.Instances.create)
    client_side_bound = "storage" in source and any(
        tok in source for tok in ("_validate_storage", "storage >", "storage <")
    )
    # A note, not a failure: "the SDK does not bound it" is the answer to the
    # question, and marking it FAIL would put a red mark next to a finding.
    f.note("where the storage ceiling lives",
           "no client-side upper bound -- the ceiling is DISCOVERED BY "
           "REJECTION at the API, so this spike walks a ladder downwards from "
           "a size a 70B job needs"
           if not client_side_bound else "the SDK bounds it locally")

    return {
        "default_storage_gb": default,
        "client_side_upper_bound": client_side_bound,
        "ceiling_is": "documented" if client_side_bound else "discovered by rejection",
        "ladder_gb": STORAGE_LADDER_GB,
    }


def pick_gpu(client: Client, f: Findings) -> tuple[str | None, dict]:
    """Cheapest VM-capable GPU with a free device, and its rate."""
    header("PREFLIGHT -- nothing provisioned")
    try:
        bal = client.account.balance()
        currency = client.account.currency()
        f.record("auth + balance", True, f"currency={currency}")
    except Exception as e:  # noqa: BLE001
        f.record("auth + balance", False, f"{type(e).__name__}: {e}")
        return None, {}

    if not client.ssh_keys.list():
        f.record("ssh key registered", False,
                 "none found -- VM creation WILL fail. "
                 "Fix: jl ssh-key add ~/.ssh/id_ed25519.pub --name my-laptop")
        return None, {}
    f.record("ssh key registered", True)

    # THE critical filter, and the same one api/provider.py uses: availability
    # rows carry workload_type 'vm' or 'container', and a GPU with free
    # devices for containers may have none for VMs. Spike 1 found the two
    # pools are tracked separately.
    prices: dict = {}
    for row in client.account.gpu_availability():
        if row.workload_type != "vm" or row.num_free_devices < 1:
            continue
        prev = prices.get(row.gpu_type)
        if prev is None or row.price_per_hour < prev["price"]:
            prices[row.gpu_type] = {"price": row.price_per_hour,
                                    "free": row.num_free_devices,
                                    "region": row.region,
                                    "vram_gb": row.vram}

    chosen = next((g for g in GPU_PREFERENCE if g in prices), None)
    if chosen is None and prices:
        chosen = min(prices, key=lambda k: prices[k]["price"])
    if chosen is None:
        f.record("a VM-capable GPU is free", False,
                 "no row with workload_type='vm' has a free device right now")
        return None, {}

    f.record("a VM-capable GPU is free", True,
             f"{chosen} at {prices[chosen]['price']} {currency}/hr, "
             f"{prices[chosen]['free']} free in {prices[chosen]['region']}")
    return chosen, {
        "currency": currency,
        "chosen": chosen,
        "hourly_rate_at_100gb": prices[chosen]["price"],
        "vm_capable_free": prices,
        "balance_read": bal is not None,
    }


def create_with_largest_disk(client: Client, gpu: str, f: Findings) -> tuple:
    """Walk the ladder down until the API accepts. Rejections are free.

    Every rejection is recorded with its message: *how* the platform refuses is
    itself the finding, because a refusal that names the ceiling is a different
    product experience from one that says 'invalid request'.
    """
    header("STORAGE CEILING -- descending until the API accepts")
    attempts = []
    named_ceiling: int | None = None
    for size in STORAGE_LADDER_GB:
        print(f"  requesting {size} GB...")
        t0 = time.time()
        try:
            instance = client.instances.create(
                gpu_type=gpu, num_gpus=1, template="vm",
                storage=size, name=f"{INSTANCE_NAME}-{size}",
            )
        except Exception as e:  # noqa: BLE001 - a rejection is the measurement
            message = f"{type(e).__name__}: {e}"
            attempts.append({"storage_gb": size, "accepted": False,
                             "error": message})
            f.note(f"{size} GB rejected", message)
            # The refusal may name the ceiling outright, which is a better
            # answer than a bisect: it is the platform's own number rather
            # than the largest value we happened to try. Worth looking for
            # every time -- a refusal that names its bound is a different
            # product experience from one that says "invalid request".
            bound = ceiling_from_rejection(message)
            if bound is not None and named_ceiling is None:
                named_ceiling = bound
                f.record("the API NAMES its storage ceiling", True,
                         f"{named_ceiling} GB, quoted in the rejection of "
                         f"{size} GB. Measured, not bisected.")
                # Fold it into the recorded attempt so the findings carry the
                # number and not only the sentence it came in.
                attempts[-1]["ceiling_named_by_the_api_gb"] = named_ceiling
            continue
        elapsed = time.time() - t0
        attempts.append({"storage_gb": size, "accepted": True,
                         "create_seconds": round(elapsed, 1),
                         "ceiling_named_by_the_api_gb": named_ceiling})
        if named_ceiling is not None:
            f.record("storage ceiling found", True,
                     f"{named_ceiling} GB, stated by the API itself. "
                     f"{size} GB was provisioned to confirm the parameter is "
                     f"honoured, not to locate the bound.")
        elif size == STORAGE_LADDER_GB[0]:
            # Accepting the top of the ladder does not find a ceiling; it finds
            # a floor on one. Saying "the ceiling is 8000 GB" here would be
            # claiming a measurement that was not taken.
            f.record("storage ceiling found", False,
                     f"{size} GB -- the TOP of the ladder -- was accepted with "
                     f"zero rejections. The ceiling is therefore AT LEAST "
                     f"{size} GB and WAS NOT FOUND. Raise the ladder to find "
                     f"it; nothing here justifies a number above {size}.")
        else:
            f.record("storage ceiling found", True,
                     f"{size} GB accepted after {len(attempts) - 1} "
                     f"rejection(s), {elapsed:.0f}s to Running. The ceiling is "
                     f"between {size} GB and "
                     f"{STORAGE_LADDER_GB[len(attempts) - 2]} GB.")
        return instance, size, attempts

    f.record("storage ceiling found", False,
             "every size on the ladder was rejected, including the 100 GB floor")
    return None, None, attempts


def price_the_storage(client: Client, gpu: str, size_gb: int,
                      baseline: dict, f: Findings) -> dict:
    """What does the extra disk cost, and is it on the card's rate or separate?

    The quote has to say. If storage is billed separately from the GPU-hour,
    a quote built only from the hourly rate understates a 400 GB job.
    """
    header("PRICING -- with and without the extra storage")
    currency = baseline.get("currency", "?")
    base_rate = baseline.get("hourly_rate_at_100gb")

    # Re-read availability WITH the machine up, so the comparison is against
    # the same surface rather than against a remembered number.
    after = None
    try:
        for row in client.account.gpu_availability():
            if row.gpu_type == gpu and row.workload_type == "vm":
                after = row.price_per_hour
                break
    except Exception as e:  # noqa: BLE001
        f.note("re-reading the rate failed", f"{type(e).__name__}: {e}")

    # Careful with the direction here. The rate NOT moving means storage is
    # NOT inside the GPU-hour -- the opposite of what "the rate covers it"
    # would suggest, and the first version of this line asserted the inverse.
    unchanged = after == base_rate
    f.record("the GPU-hour rate is independent of disk size", unchanged,
             f"{base_rate} -> {after} {currency}/hr with {size_gb} GB attached")
    f.note("what that means for the quote",
           "storage is billed on a SEPARATE line from the GPU-hour, so a quote "
           "derived only from GPU-hours understates a large-disk job"
           if unchanged else
           "the GPU rate moved with the disk, so storage is inside it and the "
           "quote can derive everything from GPU-hours")

    # The instance row carries a running cost. Reading it is the only way to
    # see whether storage lands on the same meter as the card.
    return {
        "currency": currency,
        "hourly_rate_reported_at_100gb": base_rate,
        "hourly_rate_reported_with_extra_storage": after,
        "gpu_hour_rate_unchanged_by_disk_size": unchanged,
        "storage_billed_separately_from_the_gpu_hour": unchanged,
        "note": "The availability surface reports ONE rate per GPU type, and "
                "it did not move when a 10x larger disk was attached. So "
                "storage is billed on a separate line and THE QUOTE MUST ADD "
                "IT EXPLICITLY rather than deriving everything from GPU-hours. "
                "The documented figure is $0.10/GB-month; the instance row's "
                "`cost` field is where the real number appears over time. "
                "MEASURED: the rate. NOT MEASURED: the storage line itself -- "
                "this spike ran for minutes and a GB-month bill does not "
                "resolve in minutes.",
    }


def run_bootstrap5(ssh_command: str, repo: str, fill_gb: int, f: Findings) -> dict | None:
    header("ON-VM PROBE")
    if not BOOTSTRAP.exists():
        f.record("bootstrap5.sh present", False, str(BOOTSTRAP))
        return None
    # Bytes, and LF-normalised. Text mode on Windows turns \n into \r\n on the
    # way to stdin, and bash then fails in ways that never mention line endings.
    payload = BOOTSTRAP.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    print(f"  downloading {repo}; this is the measurement, be patient...")
    t0 = time.time()
    try:
        r = subprocess.run(
            ssh_base(ssh_command) + [f"bash -s -- {repo} {fill_gb}"],
            input=payload, capture_output=True, timeout=BOOTSTRAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        f.record("probe executed", False, f"timed out after {BOOTSTRAP_TIMEOUT_S}s")
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
        print(r.stdout.decode("utf-8", "replace")[:1000])
        return None
    f.record("probe executed", True, f"{time.time() - t0:.0f}s")
    return report


def interpret(report: dict, requested_gb: int, f: Findings) -> dict:
    """Turn the on-VM numbers into the two answers the spec asked for."""
    header("WHAT THE DISK ACTUALLY IS")
    # df reports 1024-byte blocks. Both units are recorded because the
    # platform's "1000" and the filesystem's "969G" are the same disk in
    # different units, and reporting one of them alone reads as a discrepancy.
    total_bytes = report["df_total_kb"] * 1024
    total_gb = total_bytes / 1e9        # decimal GB, the platform's unit
    total_gib = total_bytes / 2 ** 30   # binary GiB, what df prints
    # A filesystem always reports less than the nominal disk -- metadata,
    # reserved blocks, the image itself. 90% is the threshold for "the API gave
    # us what it said", chosen rather than measured, and stated as such.
    honoured = total_gb >= requested_gb * 0.90
    f.record("the disk the API accepted is the disk that exists", honoured,
             f"requested {requested_gb} GB, df reports {total_gb:.0f} GB "
             f"({total_gib:.0f} GiB) usable")
    f.record("the disk is writable", bool(report["write_test_ok"]),
             report.get("write_test_rate") or report.get("write_test_error", "")[:200])

    header("HOW FAST WEIGHTS ARRIVE")
    dl = {}
    if report.get("download_ok"):
        secs = float(report["download_seconds"])
        by = int(report["download_bytes"])
        mean_rate = by / 1e6 / secs if secs else 0

        # Windowed rates from the samples. Whether the rate is steady or bursty
        # is a UI decision, not a trivia question: a live ETA computed from a
        # 30-second window oscillates badly on a bursty link, and the quote
        # would have to smooth it or show a range.
        samples = report.get("download_samples") or []
        windows = []
        for prev, cur in zip(samples, samples[1:]):
            dt = cur["t"] - prev["t"]
            if dt > 0:
                windows.append((cur["bytes"] - prev["bytes"]) / 1e6 / dt)
        import statistics
        spread = None
        if len(windows) > 2:
            mid = statistics.fmean(windows)
            spread = (statistics.pstdev(windows) / mid) if mid else None

        dl = {
            "kind": "measured",
            "repo": report["download_repo"],
            "bytes": by,
            "seconds": round(secs, 1),
            "mean_mb_per_s": round(mean_rate, 1),
            "seconds_per_gb": round(secs / (by / 1e9), 1) if by else None,
            "window_count": len(windows),
            "window_mb_per_s_min": round(min(windows), 1) if windows else None,
            "window_mb_per_s_max": round(max(windows), 1) if windows else None,
            "coefficient_of_variation": round(spread, 2) if spread else None,
            "bursty": bool(spread and spread > 0.35),
            "hf_transfer_enabled": False,
            "hf_transfer_note":
                "hf_transfer was deliberately NOT enabled. The measured rate is "
                "the plain python client's, which is the path the trainer image "
                "takes today. Treat it as a FLOOR: turning hf_transfer on would "
                "raise it, and the quote should not promise a rate the product "
                "does not actually use.",
        }
        f.record("download measured", True,
                 f"{by / 1e9:.1f} GB in {secs:.0f}s = {mean_rate:.0f} MB/s "
                 f"({dl['seconds_per_gb']}s per GB)")
        if dl["bursty"]:
            f.note("throughput is BURSTY",
                   f"window rates {dl['window_mb_per_s_min']}-"
                   f"{dl['window_mb_per_s_max']} MB/s, CoV {spread:.2f}. A live "
                   f"ETA from a short window will oscillate; smooth it or show "
                   f"a range.")
    else:
        f.record("download measured", False, report.get("download_error", "")[:400])

    header("THE KILL CRITERION")
    # 70B in bf16 is ~141 GB of weights. Checkpoints and a merged output take
    # it to 250-400 GB. 200 GB is the line the spec drew.
    supports_70b = total_gb >= 250
    above_kill_line = total_gb >= 200
    verdict = {
        "usable_disk_gb": round(total_gb),
        "supports_70b_full_finetune": supports_70b,
        "above_the_200gb_kill_line": above_kill_line,
        "statement": (
            "70B full fine-tuning has the disk it needs on this provider."
            if supports_70b else
            "The disk clears the 200 GB kill line but is under the 250-400 GB "
            "a 70B job needs with checkpoints and a merged output. 70B QLoRA "
            "from a pre-quantised repo may still fit; 70B full fine-tuning "
            "does not."
            if above_kill_line else
            "DISK CANNOT EXCEED ~200 GB. 70B FULL FINE-TUNING IS OFF THE TABLE "
            "ON THIS PROVIDER. The catalog stays unbounded by design, the "
            "predictor still computes the configuration honestly, and the "
            "README states the largest configuration actually exercised -- the "
            "same pattern already used for multi-GPU."),
    }
    f.record("70B has the disk it needs", supports_70b, verdict["statement"])

    if dl.get("mean_mb_per_s"):
        # The number the quote actually needs, derived and marked as derived.
        verdict["preparing_phase_eta"] = {
            "kind": "derived from the measured rate",
            "seconds_per_gb": dl["seconds_per_gb"],
            "eta_for_8b_bf16_16gb_s": round(16 * dl["seconds_per_gb"]),
            "eta_for_70b_bf16_141gb_s": round(141 * dl["seconds_per_gb"]),
            "caveat": "one measurement, one region, one repo, one time of day. "
                      "A range, not a promise.",
        }
    return {"disk": {"requested_gb": requested_gb,
                     "usable_gb_decimal": round(total_gb, 1),
                     "usable_gib_binary": round(total_gib, 1),
                     "honoured": honoured,
                     "writable": bool(report["write_test_ok"]),
                     "df_human": report.get("df_human"),
                     "root_device": report.get("root_device")},
            "download": dl,
            "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight only; provisions nothing, costs nothing")
    ap.add_argument("--keep", action="store_true",
                    help="leave the instance running (COSTS MONEY)")
    ap.add_argument("--repo", default=DOWNLOAD_REPO,
                    help="Hugging Face repo to time the download of")
    ap.add_argument("--fill-gb", type=int, default=0,
                    help="also fallocate this many GB, to prove the disk size")
    args = ap.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    f = Findings()
    header("SPIKE 5 -- disk ceiling and model download throughput")

    client = Client()
    bounds = read_storage_bounds(f)
    gpu, pricing = pick_gpu(client, f)

    payload: dict = {
        "spike": "spike5-disk-and-download",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "storage_parameter": bounds,
        "preflight": pricing,
    }

    if args.dry_run or gpu is None:
        payload["dry_run"] = True
        # A separate file, ON PURPOSE. A dry run costs nothing and must not be
        # able to destroy findings that cost money -- which it did once, by
        # writing its preflight over a completed run's measurements.
        dry_path = FINDINGS_PATH.with_name(FINDINGS_PATH.stem + "-dryrun.json")
        dry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nDry-run findings written to {dry_path}")
        if FINDINGS_PATH.exists():
            print(f"  ({FINDINGS_PATH.name} left untouched -- it holds a real run)")
        header("DRY RUN -- nothing was provisioned")
        return 0 if gpu else 1

    instance = None
    storage_gb = None
    try:
        instance, storage_gb, attempts = create_with_largest_disk(client, gpu, f)
        payload["storage_ladder"] = attempts
        if instance is None:
            return 1
        payload["machine"] = {"gpu": gpu, "storage_gb": storage_gb,
                              "region": getattr(instance, "region", None)}
        payload["pricing"] = price_the_storage(client, gpu, storage_gb, pricing, f)

        ssh_command = getattr(instance, "ssh_command", None)
        if not ssh_command:
            f.record("ssh command returned", False, "instance has no ssh_command")
            return 1
        if wait_for_ssh(ssh_command, f) is None:
            return 1

        report = run_bootstrap5(ssh_command, args.repo, args.fill_gb, f)
        payload["probe_report"] = report
        if report:
            payload.update(interpret(report, storage_gb, f))
    finally:
        if instance is not None:
            if args.keep:
                f.record("teardown", False,
                         f"SKIPPED via --keep -- STILL BILLING: "
                         f"jl destroy {instance.machine_id}")
            else:
                confirm_destroyed(client, instance.machine_id, f)
        # ...and then ask the PLATFORM, not our own variables. `instance` is
        # only bound once create() has returned, and create() polls internally
        # until the machine is Running -- so an exception raised inside it
        # leaves a billing machine that this finally block cannot see. The
        # sweep closes that window by matching on the name instead.
        sweep_by_name(client, INSTANCE_NAME, f, keep=args.keep)
        payload["steps"] = f.steps
        FINDINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nFindings written to {FINDINGS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
