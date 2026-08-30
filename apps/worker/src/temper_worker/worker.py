"""The worker: a separate process that claims queued jobs and drives them.

The orchestration entry point does not change -- it is already a function
over a job record that does not know what called it. What changes is that
a *separate process* calls it, claiming work with a row-level lock that
survives contention.

This is the claim the local file could not express. Two workers, one job,
is a race, and a test that starts them sequentially proves nothing.
``SELECT ... FOR UPDATE SKIP LOCKED`` is the mechanism: ``FOR UPDATE``
takes the row lock, ``SKIP LOCKED`` makes the second worker skip rather
than block. Both halves are proved by the concurrent-claim tests.

The worker is the thing that watches after the move. The control plane
no longer starts threads (``jobs.create`` just inserts the row); the
worker polls ``db.claim_next_job`` and calls ``orchestrator.run_job`` on
what it claimed. ``run_job``'s signature and body are untouched -- if
they needed edits to make this work, the seam leaked, and that is a
finding worth recording rather than patching over (ADR-0066).

Provisioning happens once even if its step is retried: a machine's
identity is recorded before anything else can fail. ``orchestrator._attempt``
already does ``machine = provider.create(...)`` followed immediately by
``db.set_state(..., machine_id=...)`` before ``await_ready``. A machine
provisioned and not recorded is a billing machine nobody owns -- the
failure ADR-0057 and ADR-0063 exist to prevent. The record-first ordering
is what makes a retry safe: a second attempt that finds ``machine_id``
already on the row knows a machine already exists.

Timers that stop a served endpoint from billing forever (ADR-0065) stay
in the control plane. An endpoint is a billed, warm machine whose idle
and max timers are armed in the control-plane process; the worker never
touches them. The control plane's lifespan still re-arms timers after a
restart and sweeps expired endpoints. The worker watches jobs; the
control plane watches endpoints. Both watch spend and teardown through
the same confirmed path (ADR-0057) inside ``run_job``.

The worker also hosts the machine-lifetime reconciler (issue #61): a
scheduled pass on its own thread that destroys machines no live job or
served endpoint owns. It protects the money, not the job, and runs
independently of any workflow.
"""

from __future__ import annotations

import threading

from temper_control_plane import db, orchestrator
from temper_control_plane.correlation import set_correlation_id
from temper_control_plane.logging import configure_logging, get_logger
from temper_control_plane.sentry import init_sentry

from . import reconciler

logger = get_logger(__name__)

# How often a worker with nothing to do checks for queued jobs. Domain
# constant with derivation, not a deployment setting (ADR-0062): 1.0s, up
# from 0.5s after 2026-08-30 CI starvation -- a tight poll is a cost paid
# on every idle tick, multiplied by worker count. At 6 workers × 0.5s that
# is 12 qps idle burning CPU and DB connections that validation's own daemon
# threads also need; at 2 workers × 1.0s (the CI shape after the
# playwright.config.ts cap) it is 2 qps, a 6x reduction, with queue latency
# still <1s which is within the "heartbeat" the comment above promises.
# Short enough that a user who just launched sees the job leave queued
# within a heartbeat, long enough that an idle worker does not hammer the
# database. The value is tuning, not contract, and tests patch it directly
# when they need a faster or slower poll.
WORKER_POLL_INTERVAL_S = 1.0


def run_once() -> bool:
    """Try to claim one queued job and drive it to a terminal state.

    Returns True if a job was claimed (whether it succeeded or failed),
    False if there was nothing queued. The caller decides whether to poll
    again immediately or sleep.
    """

    job = db.claim_next_job()
    if job is None:
        return False
    job_id = job["id"]
    # Re-hydrate the request's correlation identifier so every log line this
    # job emits carries the same identifier the request's error response did
    # (issue #52). A missing correlation is an honest absence for pre-migration
    # rows, not a redaction.
    cid = job.get("correlation_id")
    if isinstance(cid, str) and cid:
        set_correlation_id(cid)
    else:
        # No correlation on the row: honest absence for pre-migration rows or
        # jobs created outside a request context (e.g. direct ``db.create_job``
        # in tests). Clearing prevents a previous job's correlation leaking
        # into this one's logs, and an absent correlation is not a redaction
        # but a true absence -- every log line is still JSON, just without
        # the identifier.
        set_correlation_id(None)
    logger.info(
        "worker claimed job",
        job_id=job_id,
        correlation_id=job.get("correlation_id"),
        machine_id=job.get("machine_id"),
    )
    try:
        orchestrator.run_job(job_id)
    except Exception as e:  # noqa: BLE001 - a failed job must not kill the worker
        logger.exception(
            "worker failed to drive job",
            job_id=job_id,
            correlation_id=job.get("correlation_id"),
            exc_info=e,
        )
        # ``run_job`` already records the failure on the row before raising;
        # a bare exception here means the row may still be non-terminal, so
        # mark it failed rather than leave it stuck provisioning forever.
        try:
            current = db.get_job(job_id)
            if (
                current is not None
                and current["status"] not in db.TERMINAL_STATES
            ):
                db.set_state(
                    job_id,
                    "failed",
                    str(e),
                    error_code="internal_error",
                    error_message=str(e),
                )
        except Exception:  # noqa: S110 - the job is already in an unknown state
            pass
    finally:
        # Clear so the next poll or idle period does not carry the previous
        # job's correlation into an unattributed log line.
        set_correlation_id(None)
    return True


def run_forever(stop: threading.Event | None = None) -> None:
    """Poll forever, claiming and driving jobs until ``stop`` is set."""

    stop = stop or threading.Event()
    while not stop.is_set():
        did_work = False
        try:
            did_work = run_once()
        except Exception as e:  # noqa: BLE001 - the worker must stay up
            logger.exception("worker poll failed", exc_info=e)
        if not did_work:
            # No job was queued; wait a beat before polling again. Using
            # ``Event.wait`` rather than ``time.sleep`` so a stop request
            # wakes the worker immediately instead of after the full interval.
            stop.wait(WORKER_POLL_INTERVAL_S)
        # If work was done, loop immediately to claim the next job without
        # sleeping -- a burst of launches should not wait a poll interval
        # between jobs.


def main() -> None:
    """Entry point for ``python -m temper_worker`` (``__main__.py``) and
    Docker. Also runnable directly via the guard below, since a submodule
    invocation without one previously exited silently doing nothing --
    see the guard's own comment."""

    configure_logging()
    init_sentry()
    logger.info("worker starting", poll_interval_s=WORKER_POLL_INTERVAL_S)
    # Ensure the database is migrated before the first claim. The control
    # plane also migrates at startup; this is idempotent and makes a
    # worker that starts before the control plane still ready.
    try:
        db.init()
    except Exception as e:  # noqa: BLE001 - a failed init must be loud
        logger.exception("worker db init failed", exc_info=e)
        raise
    # Start the machine-lifetime reconciler (issue #61): the scheduled pass
    # that destroys machines no live job or served endpoint owns, on its own
    # thread so a long-running job claim cannot stall it. It runs
    # independently of any workflow -- the pass does not care whether a job
    # is being driven, only whether a listed machine has an owner.
    try:
        reconciler.start_reconciler_thread()
    except Exception as e:  # noqa: BLE001 - a reconciler that fails to start must be loud
        logger.exception("reconciler failed to start", exc_info=e)
        raise
    stop = threading.Event()
    try:
        run_forever(stop)
    except KeyboardInterrupt:
        logger.info("worker stopping on interrupt")
        stop.set()
    finally:
        reconciler.stop_reconciler_thread()


if __name__ == "__main__":
    # `compose.yaml` and `playwright.config.ts` both once invoked this file
    # as `python -m temper_worker.worker` (the submodule) rather than
    # `python -m temper_worker` (the package, which `__main__.py` runs).
    # Without this guard, the submodule form imports the module, defines
    # its functions, and exits 0 having started nothing -- no unit test
    # calls a command line, so only the e2e journeys caught it (every
    # launched job stayed `queued` forever). Both invocations now work.
    main()
