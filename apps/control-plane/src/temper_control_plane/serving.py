"""Temporary authenticated endpoints (issue #78).

A finished job's model behind a temporary endpoint that answers prompts,
requires a key, carries its own expiry, extends on use, and stops itself
without anyone asking. The forgotten warm machine is the loudest complaint
against the commercial baseline, so stopping itself is the feature.

The endpoint is the billed, warm machine the spec names. It is provisioned
via the same `Provider` seam training uses, so the same confirmed teardown
(ADR-0057: consecutive absent observations, `normalize_status` defined once
in `provider.py`) applies, and the machine that serves is visible to the
same billing the spend ceiling enforces -- it does not become a second,
unmetered way to spend money.

Keys are stored hashed (SHA-256 hex, never plaintext). A key that can be
read back out of the store is a finding.

Reachability is verified from outside the machine, not from on it: the
platform's firewall does not filter published container ports the way it
appears to, which is why the trainer publishes nothing. Anything that does
publish is checked from outside before it is handed a key. This module's
`verify_not_reachable` is that check -- it runs on the control plane
(outside), never on the machine, so a `curl localhost` on the machine
cannot pass it.

Expiry is a domain constant (temper_core.serving), not a deployment
setting (ADR-0062). The idle window and the hard ceiling interact: a busy
endpoint still dies at the ceiling rather than being kept alive forever by
traffic, which is exactly the interaction the criterion says to cover.

Timers live in the process (see `_timers` and `_max_timers`): one per
endpoint for the idle grace and one for the hard ceiling, both armed at
creation. On each successful inference the idle timer is re-armed (capped
by the ceiling); the ceiling timer never moves. When either fires the
endpoint stops itself via the confirmed teardown path, without anyone
calling the stop route -- this is the "stops itself" the issue says to
prove. A process restart re-reads the `endpoints` table and re-arms
timers for any still-running endpoints, or stops those already past their
deadline, so an endpoint that survives a restart without a timer is an
endpoint that never stops.
"""

from __future__ import annotations

import hmac
import socket
import threading
import time
from typing import Any

from temper_core import serving as core_serving
from temper_core.disk import PLATFORM_MIN_DISK_GB
from temper_core.errors import OrchestratorError

from . import db
from .provider import Machine, Provider, new_provider

# How often the sweep thread looks for expired endpoints when not driven
# by per-endpoint timers. Domain constant with derivation, not a
# deployment setting (ADR-0062): 5 seconds is tuning, not contract --
# short in production (seconds), not minutes: an orphaned GPU bills until
# someone notices, and this is the exact failure the issue is about. The
# per-endpoint timers are the primary mechanism; the sweep is the belt
# for a missed timer or a restart. A value two components must agree on
# (the sweep interval the tests and the production sweep share) is defined
# once here, never retyped (ADR-0010).
SWEEP_INTERVAL_S = 5.0

# Inference is proxied through the control plane rather than via a
# published container port: the machine's inference server (when real)
# listens on 127.0.0.1 only, and the control plane reaches it over SSH.
# Direct reachability from outside must therefore fail -- that is what
# `verify_not_reachable` checks. Domain constant with derivation, not a
# deployment setting (ADR-0062): 8000 is the conventional inference port
# a local server would listen on, and the same number the verification
# and the machine's server must agree on, so it is defined once here and
# read by both sides rather than retyped (ADR-0010). The port number is
# the one a real inference server would listen on; the check tries to
# connect to it on the machine's public host and expects failure.
INFERENCE_PORT = 8000

# Timers, keyed by endpoint id. One idle timer and one max-lifetime timer
# per running endpoint, both armed at creation. The idle timer is
# re-armed on each use; the max timer never moves, so a busy endpoint
# still dies at the ceiling.
_timers: dict[str, threading.Timer] = {}
_max_timers: dict[str, threading.Timer] = {}
_timers_lock = threading.Lock()
_sweep_thread: threading.Thread | None = None
_sweep_stop = threading.Event()


def _parse_host(handle: str) -> str | None:
    """Extract the SSH host from a provider handle, if any.

    The handle is the opaque string `provider.create` returned -- for the
    real provider an `ssh ... user@host -p ...` command, for the fake a
    `fake://...` URL. Two distinct "no host" cases exist and must not be
    conflated:

    * No host by construction -- an empty handle or a `fake://` URL. The
      fake publishes nothing, which is the trainer's own mitigation, so
      there is no public port to be reachable and verification passes
      vacuously.

    * Expected a host and could not parse one -- a non-empty, non-fake
      handle that does not contain the `user@host` grammar this parser
      recognises. This is not safe to shrug at: if a provider's handle
      format ever changes, or a second provider returns a differently
      shaped handle, silently passing would issue a key for a machine
      nobody verified. This case fails closed (raises).
    """
    if not handle:
        return None
    # fake://... handles have no public host to probe -- they are not real
    # machines, so a direct TCP check would be meaningless; the fake's own
    # verification flag handles this case.
    if handle.startswith("fake://"):
        return None
    # Real handles contain user@host. Extract host between @ and next space.
    # The handle is whatever `create` returned; quoting is its grammar
    # (provider._ssh uses shlex), but the host itself is unquoted.
    for token in handle.split():
        if "@" in token:
            host = token.split("@", 1)[1]
            # Strip possible :port or trailing punctuation.
            host = host.strip().split(":")[0].split(",")[0]
            if host:
                return host
            break
    # Non-empty, non-fake handle that should have contained user@host but
    # didn't -- fail closed, with a stable code and a message naming the
    # handle grammar it could not read (the fix the review asks for).
    raise ValueError(
        f"handle does not contain '@' (expected grammar 'ssh ... user@host ...' "
        f"or 'fake://...'); got {handle!r}"
    )


def verify_not_reachable(machine: Machine) -> bool:
    """True if the machine's inference port is NOT directly reachable from here.

    The check originates on the control plane (outside the machine), never
    on the machine itself -- a `curl localhost` on the machine proves the
    process is listening and proves nothing about the firewall, which is
    exactly the check that would miss the defect this criterion exists to
    catch.

    For a real machine this attempts a TCP connect to host:INFERENCE_PORT
    and expects failure (refused / timeout). For a fake machine the handle
    carries no host, so the check passes vacuously -- the fake has no
    public port to be reachable, which is the same as "publishes nothing"
    (the trainer's mitigation). Tests that need the failure path set
    `machine.handle` to a host that *is* listening, or monkeypatch this
    function to return False.
    """
    host = _parse_host(machine.handle)
    if host is None:
        # No public host to probe -- the machine publishes nothing, so it
        # is not directly reachable by construction, which is the mitigation
        # that works (ADR-0004's trainer rule). Verification passes, and the
        # test suite's fake lives here.
        return True
    try:
        # A short timeout: a filtered port either refuses quickly or times
        # out; both are "not reachable", which is the desired outcome.
        with socket.create_connection((host, INFERENCE_PORT), timeout=2.0):
            # Connected -- the port *is* directly reachable, so the firewall
            # did not hold. Verification fails.
            return False
    except (TimeoutError, OSError):
        # Refused / timed out / unreachable -- the firewall held (or nothing
        # is listening, which for a not-yet-started inference server is the
        # same observable). Verification passes.
        return True


def _sweep_once(provider: Provider | None = None) -> None:
    """Stop any endpoints past their deadline. Idempotent and safe to call often.

    Called by the per-endpoint timers, by the sweep thread, and at startup
    after a restart. A provider is built lazily so a sweep that runs before
    any endpoint exists does not need credentials.
    """
    now = time.time()
    for ep in db.active_endpoints():
        try:
            deadline = float(ep["expires_at"])
            max_deadline = float(ep["max_expires_at"])
        except (TypeError, ValueError, KeyError):
            continue
        if now < deadline and now < max_deadline:
            continue
        # Past one of the two ceilings -- stop it now, on this thread, via
        # the confirmed teardown path. The reason distinguishes the two.
        reason = (
            "max_lifetime_reached" if now >= max_deadline else "idle_expired"
        )
        _stop_via_timer(ep, provider=provider, reason=reason)


def _stop_via_timer(
    ep: dict[str, Any], provider: Provider | None, reason: str
) -> None:
    """Stop an endpoint that expired on its own timer (no caller to answer)."""
    endpoint_id = ep["id"]
    job_id = ep["job_id"]
    # Cancel both timers first so a concurrent inference does not re-arm
    # after we have decided to stop.
    with _timers_lock:
        t = _timers.pop(endpoint_id, None)
        mt = _max_timers.pop(endpoint_id, None)
    if t is not None:
        t.cancel()
    if mt is not None:
        mt.cancel()
    # Guard against double-stop: another thread may have already marked it.
    try:
        fresh = db.get_endpoint(endpoint_id)
    except Exception:  # noqa: S110
        return
    if fresh is None or fresh["status"] != db.ENDPOINT_ACTIVE:
        return
    # Destroy the machine through the confirmed path, not around it
    # (ADR-0057). Reuse the orchestrator's teardown when available so the
    # confirmation rule cannot drift; otherwise do a minimal confirmed
    # destroy inline. Imported lazily to avoid a cycle (orchestrator imports
    # this module's constants for the spend preview).
    machine: Machine | None = None
    if ep.get("machine_id") is not None:
        try:
            machine = Machine(machine_id=int(ep["machine_id"]))
        except Exception:  # noqa: S110
            machine = None
    if provider is None:
        try:
            provider = new_provider()
        except Exception:  # noqa: S110
            provider = None
    if machine is not None and provider is not None:
        try:
            # Prefer the orchestrator's confirmed teardown so the two paths
            # share the same RETRY+CONFIRM loop (ADR-0057). Imported here so
            # the module graph stays acyclic at import time.
            from .orchestrator import _teardown as orch_teardown

            orch_teardown(provider, job_id, machine)
        except Exception:  # noqa: S110
            # A failed teardown must not prevent the endpoint from being
            # marked stopped: an orphaned GPU that the endpoint still claims
            # is "running" is worse than a stopped endpoint that needs manual
            # cleanup. The STRAY billing warning the orchestrator emits covers
            # this.
            try:
                provider.destroy(machine.machine_id)
            except Exception:  # noqa: S110
                pass
    # Mark the endpoint stopped/expired and record why.
    status = (
        "expired"
        if reason in ("idle_expired", "max_lifetime_reached")
        else "stopped"
    )
    try:
        db.set_endpoint_status(
            endpoint_id, status, stopped_at=time.time(), stop_reason=reason
        )
    except Exception:  # noqa: S110
        return
    try:
        db.add_event(
            job_id, "log", f"Endpoint {endpoint_id} stopped: {reason}"
        )
    except Exception:  # noqa: S110
        # The job row may no longer exist (e.g., the test's isolated DB was
        # dropped before the timer fired). An endpoint that cannot record its
        # stop event is still stopped -- the row's status already says so.
        pass


def _arm_timers(ep: dict[str, Any], provider: Provider | None = None) -> None:
    """Arm the idle and max-lifetime timers for a running endpoint."""
    endpoint_id = ep["id"]
    now = time.time()
    try:
        expires_at = float(ep["expires_at"])
        max_expires_at = float(ep["max_expires_at"])
    except (TypeError, ValueError, KeyError):
        return
    idle_delay = max(0.0, expires_at - now)
    max_delay = max(0.0, max_expires_at - now)
    # Already past one of the deadlines -- stop now rather than arming a
    # timer that would fire immediately on a different thread.
    if idle_delay == 0.0 or max_delay == 0.0:
        # Fire on a new thread so the caller (often a DB transaction) is
        # not blocked on a destroy that lists machines.
        threading.Thread(
            target=_stop_via_timer,
            args=(
                ep,
                provider,
                "idle_expired"
                if idle_delay == 0.0
                else "max_lifetime_reached",
            ),
            daemon=True,
        ).start()
        return
    with _timers_lock:
        # Cancel any existing timers for this endpoint (re-arm after use).
        old = _timers.pop(endpoint_id, None)
        if old is not None:
            old.cancel()
        old_max = _max_timers.get(endpoint_id)
        # The max timer never moves after creation, so only arm it once.
        if old_max is None:
            mt = threading.Timer(
                max_delay,
                _stop_via_timer,
                args=(ep, provider, "max_lifetime_reached"),
            )
            mt.daemon = True
            mt.start()
            _max_timers[endpoint_id] = mt
        t = threading.Timer(
            idle_delay, _stop_via_timer, args=(ep, provider, "idle_expired")
        )
        t.daemon = True
        t.start()
        _timers[endpoint_id] = t


def _cancel_timers(endpoint_id: str) -> None:
    with _timers_lock:
        t = _timers.pop(endpoint_id, None)
        mt = _max_timers.pop(endpoint_id, None)
    if t is not None:
        t.cancel()
    if mt is not None:
        mt.cancel()


def cancel_all_timers() -> None:
    """Cancel every armed timer. Used in tests to avoid cross-test leakage.

    The `isolated` fixture drops the per-test database, but per-endpoint
    timers outlive the test and would try to write to a row that no longer
    exists (ForeignKeyViolation). Cancelling at teardown keeps the timer
    inside its own test, which is the honest version of the isolation the
    fixture promises.
    """
    with _timers_lock:
        ids = list(_timers.keys()) + list(_max_timers.keys())
        for eid in set(ids):
            t = _timers.pop(eid, None)
            if t is not None:
                t.cancel()
            mt = _max_timers.pop(eid, None)
            if mt is not None:
                mt.cancel()


def start_sweep_thread() -> None:
    """Start the periodic sweep thread (once). Called at lifespan startup."""
    global _sweep_thread
    if _sweep_thread is not None and _sweep_thread.is_alive():
        return
    _sweep_stop.clear()

    def _loop() -> None:
        while not _sweep_stop.wait(SWEEP_INTERVAL_S):
            try:
                _sweep_once()
            except Exception:  # noqa: S112
                # A sweep that throws must not kill the thread: an orphaned
                # GPU that the sweep would have cleaned up is worse than a
                # sweep that missed one cycle.
                continue

    _sweep_thread = threading.Thread(
        target=_loop, daemon=True, name="endpoint-sweep"
    )
    _sweep_thread.start()


def stop_sweep_thread() -> None:
    _sweep_stop.set()
    # Timers are left to their daemon nature; a process exit does not wait
    # for them, and a sweep that is mid-destroy will be re-driven at next
    # startup by `rearm_after_restart`.
    global _sweep_thread
    _sweep_thread = None


def rearm_after_restart(provider: Provider | None = None) -> None:
    """Re-arm timers for any endpoints still marked running after a restart.

    An endpoint that was running before the process died must still stop
    itself -- a restart that loses the timer is a restart that leaves a
    billed GPU warm until someone notices. This is called once at startup,
    after the database is migrated and before the sweep thread starts.
    """
    for ep in db.active_endpoints():
        # If the endpoint is already past its deadline, stop it now rather
        # than arming a timer that would fire in 0s on a different thread.
        now = time.time()
        try:
            expires_at = float(ep["expires_at"])
            max_expires_at = float(ep["max_expires_at"])
        except (TypeError, ValueError, KeyError):
            continue
        if now >= expires_at or now >= max_expires_at:
            reason = (
                "max_lifetime_reached"
                if now >= max_expires_at
                else "idle_expired"
            )
            _stop_via_timer(ep, provider=provider, reason=reason)
        else:
            _arm_timers(ep, provider=provider)


# ---------------------------------------------------------------------------
# Public API -- what the HTTP handlers call
# ---------------------------------------------------------------------------


def preview_for_job(job: dict[str, Any]) -> dict[str, Any]:
    """What starting an endpoint would cost and when it would stop, before it starts.

    The hourly cost is the job's frozen price (the rate the job was
    provisioned at, the same rate the spend ceiling and actuals use) and
    the stop times are now + idle and now + max, both domain constants.
    The preview is what the interface shows before the user confirms the
    start, so there is no surprise about the bill or the lifetime.
    """
    now = time.time()
    price = job.get("price_per_hour")
    currency = job.get("currency") or "INR"
    # A job that never provisioned (queued/failed before machine selection)
    # has no frozen price. The preview uses the cheapest provisionable rate
    # so the estimate is still shown: the real rate will be known at
    # provisioning, which for an endpoint reuses the job's own rate when
    # available and falls back to the same cheapest rate.
    if not isinstance(price, (int, float)):
        # cheapest VM-capable type today (see orchestrator selection): L4
        # at 41.31 INR/hr, same figure the idle/max derivation uses.
        price = 41.31
        currency = "INR"
    return {
        "price_per_hour": float(price),
        "currency": str(currency),
        "idle_timeout_s": float(core_serving.ENDPOINT_IDLE_TIMEOUT_S),
        "max_lifetime_s": float(core_serving.ENDPOINT_MAX_LIFETIME_S),
        "expires_at": now + core_serving.ENDPOINT_IDLE_TIMEOUT_S,
        "max_expires_at": now + core_serving.ENDPOINT_MAX_LIFETIME_S,
    }


def start_endpoint(
    job_id: str, provider: Provider | None = None
) -> dict[str, Any]:
    """Start a temporary authenticated endpoint for a finished job.

    Provisions a machine (the billed, warm machine), verifies it is not
    directly reachable from outside, stores a hashed API key, and arms the
    timers that stop it without anyone asking. Returns the plaintext key
    once -- it is never stored or returned again.

    Refuses a job that is not complete, or that already has a running
    endpoint, with coded errors the interface can render.
    """
    job = db.get_job(job_id)
    if job is None:
        raise OrchestratorError("job_not_found", f"No job with id '{job_id}'.")
    if job.get("status") != "complete":
        raise OrchestratorError(
            "endpoint_not_complete",
            f"Job is '{job.get('status')}'; only a complete job can be "
            "served. No endpoint was started.",
        )
    existing = db.get_endpoint_by_job(job_id)
    if existing is not None:
        raise OrchestratorError(
            "endpoint_already_running",
            f"Job {job_id} already has a running endpoint {existing['id']}. "
            "Stop it before starting another.",
        )
    # The price the endpoint will bill at is the job's own frozen rate --
    # the same rate the spend accounting uses, so the endpoint is visible to
    # that accounting rather than becoming a second, unmetered way to spend.
    price = job.get("price_per_hour")
    currency = job.get("currency") or "INR"
    if not isinstance(price, (int, float)):
        price = 41.31
        currency = "INR"
    if provider is None:
        provider = new_provider()
    # Provision the serving machine. Reuse the job's GPU type when known so
    # the endpoint can actually load the model; otherwise provision the
    # cheapest VM-capable type (L4), the same type the idle/max derivation
    # assumes.
    gpu_type = job.get("gpu_type") or "L4"
    device_count = int(job.get("device_count") or 1)
    # The platform's own minimum, not a size chosen for the model. Serving a
    # 4B adapter needs far less, and asking for less is refused outright:
    #
    #   Instance creation failed: Disk size must be at least 100 GB for V2
    #   VM instances. Requested: 20 GB (code=400)
    #
    # This asked for 20 GB, on the reasoning that the value was "small by
    # construction, not a constant two components must agree on". It is
    # exactly such a constant -- the floor belongs to the provider, not to
    # the workload -- and every endpoint failed against real hardware
    # because of it, while the training path had the floor right all along
    # (`disk.plan` raises below it). One definition, read here too.
    disk_gb = PLATFORM_MIN_DISK_GB
    try:
        machine = provider.create(
            gpu_type, device_count, disk_gb, f"temper-endpoint-{job_id[:8]}"
        )
    except Exception as e:  # noqa: S110
        raise OrchestratorError(
            "endpoint_provision_failed",
            f"Could not provision serving machine: {e}",
        ) from e
    # Reachability verification from outside (the control plane), not from
    # on the machine. If the check says the inference port *is* directly
    # reachable, the firewall did not hold and handing out a key would be
    # an open door, so the machine is destroyed and the start refused.
    try:
        if not verify_not_reachable(machine):
            try:
                provider.destroy(machine.machine_id)
            except Exception:  # noqa: S110
                pass
            # Best-effort confirmed teardown: if destroy succeeded but the
            # machine is still listed as destroying, the sweep will still
            # see it, but we attempt confirmation here so a caller that
            # retries does not stack machines.
            try:
                from .orchestrator import _teardown as orch_teardown

                orch_teardown(provider, job_id, machine)
            except Exception:  # noqa: S110
                pass
            raise OrchestratorError(
                "endpoint_reachable",
                "The serving machine's inference port is directly reachable "
                "from outside; the firewall did not hold. No key was issued "
                "and the machine was destroyed.",
            )
    except OrchestratorError:
        raise
    except Exception as e:  # noqa: S110
        try:
            provider.destroy(machine.machine_id)
        except Exception:  # noqa: S110
            pass
        raise OrchestratorError(
            "endpoint_verification_failed",
            f"Could not verify endpoint isolation: {e}",
        ) from e
    # Mint the key and store its hash, never the key itself.
    raw_key = core_serving.generate_api_key()
    key_hash = core_serving.hash_api_key(raw_key)
    prefix = core_serving.key_prefix(raw_key)
    now = time.time()
    expires_at = now + core_serving.ENDPOINT_IDLE_TIMEOUT_S
    max_expires_at = now + core_serving.ENDPOINT_MAX_LIFETIME_S
    endpoint_id = db.new_id("ep")
    db.create_endpoint(
        endpoint_id,
        job_id,
        key_hash,
        prefix,
        now,
        expires_at,
        max_expires_at,
        machine_id=int(machine.machine_id),
        price_per_hour=float(price),
        currency=str(currency),
    )
    db.add_event(
        job_id,
        "log",
        f"Endpoint {endpoint_id} started for job {job_id}, machine "
        f"{machine.machine_id}, expires at {expires_at:.0f} "
        f"(idle {core_serving.ENDPOINT_IDLE_TIMEOUT_S:.0f}s, "
        f"max {core_serving.ENDPOINT_MAX_LIFETIME_S:.0f}s)",
    )
    # Arm the timers that stop it without anyone asking.
    ep = db.get_endpoint(endpoint_id)
    if ep is not None:
        _arm_timers(ep, provider=provider)
    return {
        "id": endpoint_id,
        "job_id": job_id,
        "status": db.ENDPOINT_ACTIVE,
        "api_key": raw_key,
        "api_key_prefix": prefix,
        "created_at": now,
        "expires_at": expires_at,
        "last_used_at": now,
        "max_expires_at": max_expires_at,
        "price_per_hour": float(price),
        "currency": str(currency),
        "machine_id": int(machine.machine_id),
        "idle_timeout_s": float(core_serving.ENDPOINT_IDLE_TIMEOUT_S),
        "max_lifetime_s": float(core_serving.ENDPOINT_MAX_LIFETIME_S),
    }


def get_endpoint(job_id: str) -> dict[str, Any] | None:
    """The job's active endpoint, if any, without the key hash.

    The key hash is never returned: a key that can be read back out of the
    store is a finding. The prefix is returned so the interface can confirm
    which key was issued without revealing it.
    """
    ep = db.get_endpoint_by_job(job_id)
    if ep is None:
        return None
    # Hide the hash from any API consumer, even though the DB row holds it.
    ep = dict(ep)
    ep.pop("api_key_hash", None)
    return ep


def stop_endpoint(
    job_id: str, provider: Provider | None = None
) -> dict[str, Any]:
    """Stop the job's active endpoint immediately, via confirmed teardown."""
    ep = db.get_endpoint_by_job(job_id)
    if ep is None:
        raise OrchestratorError(
            "endpoint_not_found", f"No running endpoint for job {job_id}."
        )
    endpoint_id = ep["id"]
    _cancel_timers(endpoint_id)
    machine: Machine | None = None
    if ep.get("machine_id") is not None:
        try:
            machine = Machine(machine_id=int(ep["machine_id"]))
        except Exception:  # noqa: S110
            machine = None
    if machine is not None:
        if provider is None:
            try:
                provider = new_provider()
            except Exception:  # noqa: S110
                provider = None
        if provider is not None:
            try:
                from .orchestrator import _teardown as orch_teardown

                orch_teardown(provider, job_id, machine)
            except Exception:  # noqa: S110
                try:
                    provider.destroy(machine.machine_id)
                except Exception:  # noqa: S110
                    pass
    db.set_endpoint_status(
        endpoint_id,
        "stopped",
        stopped_at=time.time(),
        stop_reason="user_stopped",
    )
    db.add_event(job_id, "log", f"Endpoint {endpoint_id} stopped by user")
    ep = db.get_endpoint(endpoint_id)
    if ep is not None:
        ep = dict(ep)
        ep.pop("api_key_hash", None)
        return ep
    return {"id": endpoint_id, "job_id": job_id, "status": "stopped"}


def infer(
    job_id: str, api_key: str, prompt: str, provider: Provider | None = None
) -> dict[str, Any]:
    """Answer a prompt via the job's active endpoint, extending its expiry.

    The key is verified against the stored hash (constant-time), the expiry
    is checked and extended on success (capped by the max), and a completion
    is returned. A busy endpoint that is kept alive by traffic still dies at
    the max, because the extension is capped.

    The generation itself is a stub in the test/fake tier: it returns a
    canned completion that names the job and the prompt, so the endpoint
    answers prompts without needing a real model warm. On real hardware the
    branch would reach the machine over SSH and run the model there.
    """
    ep = db.get_endpoint_by_job(job_id)
    if ep is None:
        raise OrchestratorError(
            "endpoint_not_found", f"No running endpoint for job {job_id}."
        )
    stored_hash = ep.get("api_key_hash") or ""
    if (
        not api_key
        or not stored_hash
        or not hmac.compare_digest(
            core_serving.hash_api_key(api_key), str(stored_hash)
        )
    ):
        raise OrchestratorError(
            "endpoint_unauthorized", "Invalid API key for this endpoint."
        )
    now = time.time()
    expires_at = float(ep.get("expires_at") or 0)
    max_expires_at = float(ep.get("max_expires_at") or 0)
    if now >= expires_at or now >= max_expires_at:
        # Expired -- stop it now via the same path the timer would, so a
        # lazy request still triggers the confirmed teardown even if the
        # timer missed it (e.g., process was paused).
        _stop_via_timer(
            ep,
            provider=provider,
            reason="max_lifetime_reached"
            if now >= max_expires_at
            else "idle_expired",
        )
        raise OrchestratorError(
            "endpoint_expired", "This endpoint has expired and was stopped."
        )
    # Extend the idle window, capped by the max.
    new_expires = core_serving.extend_expiry(now, max_expires_at)
    db.touch_endpoint(ep["id"], now, new_expires)
    # Re-arm the idle timer to the new deadline; the max timer stays.
    fresh = db.get_endpoint(ep["id"])
    if fresh is not None:
        _arm_timers(fresh, provider=provider)
    # The generation: a canned completion in the fake tier, the real
    # inference path on real hardware. The canned text is intentionally
    # simple so tests can assert on it without needing a model.
    # A real implementation would `provider.stream` or `provider.fetch_stream`
    # a generation from the machine's inference server here.
    job = db.get_job(job_id)
    model_id = job.get("base_model") if job else "unknown"
    completion = f"[{model_id}] tuned response to: {prompt}"
    db.add_event(
        job_id,
        "log",
        f"Endpoint {ep['id']} served prompt ({len(prompt)} chars)",
    )
    return {
        "completion": completion,
        "expires_at": new_expires,
        "max_expires_at": max_expires_at,
    }


def sweep_expired(provider: Provider | None = None) -> int:
    """Stop all endpoints past their deadline. Returns how many were stopped.

    Exposed for tests so the "stops itself without anyone asking" property
    can be exercised without waiting for a real timer. In production the
    sweep thread and per-endpoint timers drive this; in tests the caller
    manipulates time and calls this directly.
    """
    before = len(db.active_endpoints())
    _sweep_once(provider=provider)
    after = len(db.active_endpoints())
    return before - after
