"""Cancel a real run while it is training, and prove the machine goes away.

The third of the pre-submission hardware runs. Cancelling is the control
that exists entirely for the money: ADR-0003 says it destroys the machine,
produces no adapter, and is not a failure. None of that had ever been
exercised against real hardware.

This launches, waits until the job is genuinely training rather than merely
accepted, cancels, and then watches two things separately: that the job
reaches `cancelled`, and that the provider lists no machines. The second is
the one that matters, and it is asked of the provider rather than read from
the job record -- a job row saying `cancelled` is a claim about a teardown,
not evidence of one.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8001"
DATASET = "ds_0a5d34e37de4"
MODEL = "qwen3-4b"
POLL_S = 10
REACH_TRAINING_CEILING_S = 1500
SETTLE_CEILING_S = 900


def call(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    # S310: the scheme is not user input -- `BASE` is a localhost constant
    # at the top of this file.
    request = urllib.request.Request(  # noqa: S310
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=120
        ) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def machines() -> list:
    sys.path.insert(0, "apps/control-plane/src")
    sys.path.insert(0, "packages/core/src")
    from temper_control_plane.provider import JarvisLabsProvider

    provider = JarvisLabsProvider()
    try:
        return [
            (m.machine_id, m.status, m.name) for m in provider.list_machines()
        ]
    finally:
        provider.close()


def main() -> int:
    say(f"machines before: {machines()}")
    status, job = call(
        "POST",
        "/v1/jobs",
        {"dataset_id": DATASET, "base_model": MODEL, "hyperparameters": {}},
    )
    if status != 201:
        say(f"launch refused {status}: {json.dumps(job)[:400]}")
        return 1
    job_id = job["id"]
    say(f"launched {job_id}")

    started = time.time()
    try:
        # Cancel while it is training, not while it is queued: a cancel that
        # lands before a machine exists proves nothing about teardown.
        while time.time() - started < REACH_TRAINING_CEILING_S:
            _, job = call("GET", f"/v1/jobs/{job_id}")
            if job.get("status") == "training":
                break
            if job.get("status") in ("complete", "failed", "cancelled"):
                say(f"ended before it could be cancelled: {job.get('status')}")
                return 1
            time.sleep(POLL_S)
        live = machines()
        say(f"training now, {time.time() - started:.0f}s in; machines: {live}")

        say("cancelling")
        status, body = call("POST", f"/v1/jobs/{job_id}/cancel")
        say(f"cancel -> {status} {json.dumps(body)[:300]}")

        settle = time.time()
        while time.time() - settle < SETTLE_CEILING_S:
            _, job = call("GET", f"/v1/jobs/{job_id}")
            live = machines()
            say(f"{job.get('status')} / {job.get('error_code')}  {live}")
            if job.get("status") in ("cancelled", "failed", "complete"):
                if not live:
                    break
            time.sleep(POLL_S)

        say(f"final status: {job.get('status')} ({job.get('error_code')})")
        say(f"artifact_key: {job.get('artifact_key')!r}")
        say(f"actuals: {json.dumps(job.get('actuals'))[:400]}")
    finally:
        for attempt in range(1, 13):
            live = machines()
            say(f"listing {attempt}/12: {live}")
            if not live:
                say("teardown CONFIRMED: the account lists no machines")
                break
            time.sleep(20)
        else:
            say("TEARDOWN NOT CONFIRMED -- a machine is still billing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
