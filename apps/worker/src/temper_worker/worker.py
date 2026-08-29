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
"""

from __future__ import annotations

import logging
import threading

from temper_control_plane import db, orchestrator

logger = logging.getLogger(__name__)

# How often a worker with nothing to do checks for queued jobs. Domain
# constant with derivation, not a deployment setting (ADR-0062): 0.5s is
# tuning, not contract -- short enough that a user who just launched sees
# the job leave queued within a heartbeat, long enough that an idle worker
# does not hammer the database. The value is small by construction and is
# the same one the tests patch when they need a faster or slower poll.
WORKER_POLL_INTERVAL_S = 0.5


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
    logger.info("worker claimed job %s", job_id)
    try:
        orchestrator.run_job(job_id)
    except Exception as e:  # noqa: BLE001 - a failed job must not kill the worker
        logger.exception("worker failed to drive job %s: %s", job_id, e)
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
    return True


def run_forever(stop: threading.Event | None = None) -> None:
    """Poll forever, claiming and driving jobs until ``stop`` is set."""

    stop = stop or threading.Event()
    while not stop.is_set():
        did_work = False
        try:
            did_work = run_once()
        except Exception as e:  # noqa: BLE001 - the worker must stay up
            logger.exception("worker poll failed: %s", e)
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
    Docker. Also runnable directly as ``python -m temper_worker.worker`` via
    the guard below -- both names are correct on purpose after the first one
    got typo'd into ``compose.yaml`` and ``playwright.config.ts``, where a
    silent exit(0) went unnoticed by every unit test and was only caught by
    the e2e journeys leaving every launched job ``queued`` forever."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("worker starting, polling every %ss", WORKER_POLL_INTERVAL_S)
    # Ensure the database is migrated before the first claim. The control
    # plane also migrates at startup; this is idempotent and makes a
    # worker that starts before the control plane still ready.
    try:
        db.init()
    except Exception as e:  # noqa: BLE001 - a failed init must be loud
        logger.exception("worker db init failed: %s", e)
        raise
    stop = threading.Event()
    try:
        run_forever(stop)
    except KeyboardInterrupt:
        logger.info("worker stopping on interrupt")
        stop.set()


if __name__ == "__main__":
    main()
