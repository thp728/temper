"""Drive one real serving endpoint end to end, and never leave a machine up.

The endpoint's first hardware run (ADR-0076). It provisions an L4, ships a
model server, waits for the weights, answers one prompt, stops, and confirms
teardown by listing instances -- a destroy call's return value is not
evidence.

Read-only against the product's own HTTP API rather than importing
`serving`, so what it exercises is what a user would reach.

The `finally` is the whole point. If anything here fails, or the process is
interrupted, the endpoint is stopped and the account is listed, because an
orphaned GPU bills until somebody notices.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8001"
JOB_ID = "job_f6df279e2246"
PROMPT = "What is the capital of France?"


def call(
    method: str, path: str, body: dict | None = None, key: str | None = None
):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    # S310: the scheme is not user input -- `BASE` is a localhost
    # constant at the top of this file.
    request = urllib.request.Request(  # noqa: S310
        BASE + path, data=data, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=2400
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
    started = time.time()
    say("starting endpoint (provisions an L4 and loads the model)")
    status, body = call("POST", f"/v1/jobs/{JOB_ID}/endpoint")
    say(f"start -> {status} after {time.time() - started:.0f}s")
    if status != 201:
        say(f"refused: {json.dumps(body)[:600]}")
        say(f"machines after refusal: {machines()}")
        return 1

    key = body["api_key"]
    say(f"endpoint {body['id']} on machine {body['machine_id']}")
    try:
        say(f"prompt: {PROMPT!r}")
        t0 = time.time()
        status, answer = call(
            "POST",
            f"/v1/jobs/{JOB_ID}/endpoint/infer",
            {"prompt": PROMPT},
            key=key,
        )
        say(f"infer -> {status} in {time.time() - t0:.0f}s")
        print("-" * 60, flush=True)
        print(json.dumps(answer, indent=2)[:4000], flush=True)
        print("-" * 60, flush=True)
        if status == 200:
            completion = answer.get("completion", "")
            template = f"[qwen3-4b] tuned response to: {PROMPT}"
            say(f"is the old template: {completion.strip() == template}")
            say(f"contains the prompt verbatim: {PROMPT in completion}")
    finally:
        say("stopping endpoint")
        status, stopped = call("DELETE", f"/v1/jobs/{JOB_ID}/endpoint")
        say(f"stop -> {status} {json.dumps(stopped)[:200]}")
        # Teardown is confirmed by listing, never by the destroy's return.
        for attempt in range(1, 11):
            live = machines()
            say(f"listing {attempt}/10: {live}")
            if not live:
                say("teardown CONFIRMED: the account lists no machines")
                break
            time.sleep(15)
        else:
            say("TEARDOWN NOT CONFIRMED -- a machine is still billing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
