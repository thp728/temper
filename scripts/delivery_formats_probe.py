"""Launch one real run with merged and quantised ticked, and score the quote.

The second of the pre-submission hardware runs. The question it settles is
recorded before it runs, so the answer is a measurement and not a story:

`quote.quote_for_launch(dataset_id, base_model, hyperparameters, overrides)`
takes no delivery argument, and `quote.PHASES` ends at `teardown` with no
packaging phase. So the frozen estimate structurally cannot price the merge
and the quantise, while the machine bills for both. The prediction is that
the actual total exceeds the predicted high by roughly the packaging time,
and this prints both so the size of the miss is a number.

The run is watched, never driven: the worker owns the job. What this adds is
a reading of the frozen quote against the job's own actuals, and a teardown
confirmed by listing.
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
DELIVERY = ["merged", "quantised"]
POLL_S = 20
CEILING_S = 5400


def call(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    # S310: the scheme is not user input -- `BASE` is a localhost
    # constant at the top of this file.
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


def report(job: dict) -> None:
    quote = job.get("quote") or {}
    actuals = job.get("actuals") or {}
    say("--- the quote this launch froze ---")
    print(json.dumps(quote, indent=2)[:3000], flush=True)
    say("--- what it actually did ---")
    print(json.dumps(actuals, indent=2)[:3000], flush=True)
    say("--- delivery ---")
    print(json.dumps(job.get("delivery"), indent=2)[:2000], flush=True)
    print(json.dumps(job.get("delivery_request"), indent=2)[:500], flush=True)
    phases = quote.get("phases") or []
    say(f"phases the quote priced: {[p.get('name') for p in phases]}")
    say(
        "packaging priced: "
        f"{any('packag' in str(p.get('name')) for p in phases)}"
    )


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
        say(f"launch refused {status}: {json.dumps(job)[:600]}")
        return 1
    job_id = job["id"]
    say(f"launched {job_id} asking for {DELIVERY}")
    quote = job.get("quote") or {}
    say(
        "predicted total: "
        f"{quote.get('total_duration_low_s')}-"
        f"{quote.get('total_duration_high_s')}s"
    )

    started = time.time()
    last = None
    try:
        while time.time() - started < CEILING_S:
            _, job = call("GET", f"/v1/jobs/{job_id}")
            state = (job.get("status"), job.get("stage"))
            if state != last:
                say(f"{state[0]} / {state[1]}  ({time.time() - started:.0f}s)")
                last = state
            if job.get("status") in ("complete", "failed", "cancelled"):
                break
            time.sleep(POLL_S)
        say(
            f"finished as {job.get('status')} "
            f"({job.get('error_code') or 'no error'}) in "
            f"{time.time() - started:.0f}s"
        )
        report(job)
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
