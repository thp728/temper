"""Launch one real run with `merged` delivery, on today's fixes.

Confirms three things landed in this session together, on real hardware:

1. The delivery-format producibility gate (ADR-0077) does not block a
   `merged`-only request -- only `quantised` is refused, and that refusal
   is exercised for free in the component suite, not here.
2. The spread-vs-wrapped `chat_template_kwargs` fix (entrypoint.py,
   template_probe.py) renders the held-out comparison with thinking mode
   correctly applied, on a real trained adapter, in the pinned image.
3. A fresh actual-vs-predicted data point for a `merged`-delivery run,
   since the quote still does not price the packaging phase.

The job is watched, never driven -- the worker owns it. Teardown is
confirmed by listing, independently of the job record.
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
DELIVERY = ["merged"]
POLL_S = 20
CEILING_S = 3600


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
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
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
        {
            "dataset_id": DATASET,
            "base_model": MODEL,
            "hyperparameters": {},
            "delivery": DELIVERY,
        },
    )
    if status != 201:
        say(f"launch refused {status}: {json.dumps(job)[:400]}")
        return 1
    job_id = job["id"]
    say(f"launched {job_id}, delivery_request={job.get('delivery_request')}")

    started = time.time()
    try:
        while time.time() - started < CEILING_S:
            _, job = call("GET", f"/v1/jobs/{job_id}")
            status_now = job.get("status")
            say(f"{status_now} / {job.get('error_code')}")
            if status_now in ("complete", "failed", "cancelled"):
                break
            time.sleep(POLL_S)
        else:
            say("CEILING REACHED before a terminal state")
            return 1

        say(f"final status: {job.get('status')} ({job.get('error_code')})")
        say(
            f"delivery_formats: {json.dumps(job.get('delivery_formats'))[:300]}"
        )
        say(f"actuals: {json.dumps(job.get('actuals'))[:600]}")

        comparison = job.get("comparison") or {}
        rows = comparison.get("rows") or []
        for row in rows[:3]:
            say(f"comparison row: {json.dumps(row)[:500]}")

        quote = job.get("quote") or {}
        actual_cost = (job.get("actuals") or {}).get("cost_minor")
        actual_stages = (job.get("actuals") or {}).get("stages") or []
        packaging_priced = any(
            p.get("name") == "packaging" for p in quote.get("phases") or []
        )
        say(
            f"cost: predicted [{quote.get('cost_low_minor')}, "
            f"{quote.get('cost_high_minor')}], actual {actual_cost}"
        )
        say(
            f"duration: predicted [{quote.get('duration_low_s')}, "
            f"{quote.get('duration_high_s')}]"
        )
        say(f"packaging phase priced by the quote: {packaging_priced}")
        say(f"actual stages: {json.dumps(actual_stages)[:500]}")
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
