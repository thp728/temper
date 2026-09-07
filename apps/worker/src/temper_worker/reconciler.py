"""The machine-lifetime reconciler (issue #61, Spec 010).

Nothing looks for a machine that is billing with no job that owns it. The
final-block teardown covers the failures the orchestrator can see; it cannot
cover the process disappearing. This is the scheduled pass that closes that
gap: list machines, match them against jobs that are not in a terminal state
(and against served endpoints, which are not jobs), destroy anything it
cannot account for, record what it did, and mark a job whose machine was
destroyed as unowned failed with a reason.

The distinction that shapes every decision here is the one Spec 010 draws
between durable execution and the reconciler. Durable execution recovers
*the job*: a machine owned by a live job; even a job whose worker crashed
and is waiting for resumption (#60); is *matched* and left alone, because
recovering that job is the resumption path's job, not this pass's. This pass
protects *the money*: it assumes nothing about whether orchestration is
healthy and asks only whether a billing machine has an owner. The two defend
different failures, and this pass would stay even with durable execution
working perfectly.

The one guarantee that shapes the ownership query: ``db.claim_next_job``
only ever looks at ``queued`` rows, so a machine owned by a ``provisioning``
or ``preparing`` job must be matched here. The ownership query reads every
non-terminal state, never a queued-only subset, or this pass would destroy a
machine a worker is actively driving.

Teardown uses the same confirmed destroy the orchestrator's ``_teardown``
does (``confirmed_destroy``, ADR-0057): retry a refused destroy, then
require consecutive absent listings, treating ``destroying`` as not yet
confirmed. A machine in ``destroying`` is therefore never issued a second
destroy by this pass; that is the double-destroy ADR-0057 exists to stop.

The schedule is a daemon thread in the worker process (ADR-0010 hosts the
reconciler beside the claim loop), started by ``worker.main`` and stopping
with the process. It runs independently of any workflow: the pass does not
care whether a job is being driven, only whether a listed machine has an
owner.
"""

from __future__ import annotations

import threading

from temper_control_plane.logging import get_logger

logger = get_logger(__name__)

# How often the scheduled pass runs. Domain constant with derivation, not a
# deployment setting (ADR-0062): it must be short enough that a machine with
# no owner is found long before it bills for anything a human would notice;
# a machine bills per minute, so a thirty-second cadence bounds an orphan's
# unbilled window to well under a minute, and long enough that an idle pass
# does not hammer the provider's listing API on every tick. The serving
# sweep (ADR-0065) uses five seconds because an endpoint is a *warm* machine
# the user is interacting with; this pass guards cold job machines, where
# the same urgency does not apply. The value is tuning, not contract, and
# tests patch it directly when they need a faster loop.
RECONCILE_INTERVAL_S = 30.0

# How long a *starting* endpoint's machine is spared. An endpoint start
# provisions a machine, ships a model server to it and waits for the weights
# to load before it mints a key, and that wait is minutes; a pass landing in
# it would destroy a machine that is doing exactly what it was asked to.
#
# Bounded rather than open, and that bound is the point. If the control
# plane is killed mid-start, the row stays `starting` forever, and an
# ownership claim that never expires would produce an orphan this pass is
# forbidden to collect -- worse than the race it exists to prevent. Past the
# grace the machine is unowned again and gets destroyed, which is right: no
# one is going to serve from it.
#
# Derived from the readiness timeout the start itself honours, plus room for
# provisioning and the image pull before that clock begins.
ENDPOINT_STARTING_GRACE_S = 1800.0

# The stable code and plain reason a job is marked failed with when the
# reconciler destroys the machine it owns as an unowned orphan. A job whose
# current machine has been destroyed cannot continue, and leaving it in a
# non-terminal state forever is exactly the "left running forever" the
# criterion names (issue #61). The code is distinct from any orchestrator
# code so a client can branch on "the platform's reconciler removed my
# machine" separately from a training failure.
ORPHANED_MACHINE_CODE = "orphaned_machine"
ORPHANED_MACHINE_MESSAGE = (
    "The machine this job was running on was found with no live owner and "
    "destroyed by the reconciler. The job cannot continue."
)

# The action vocabulary of the reconciliation log, defined once and read by
# the pass and the tests so an operator reading the table sees a stable set
# of labels. `destroyed` means the destroy was confirmed absent across
# consecutive observations (ADR-0057); `destroy_unconfirmed` means the
# provider refused it and the machine is still billing; the STRAY case;
# `skipped_destroying` means the provider reported it mid-destroy and a
# second destroy would be the double-destroy ADR-0057 exists to stop.
ACTION_DESTROYED = "destroyed"
ACTION_DESTROY_UNCONFIRMED = "destroy_unconfirmed"
ACTION_SKIPPED_DESTROYING = "skipped_destroying"


def _destroy_orphan(
    provider, machine_id: int, status: str, report: dict
) -> None:
    """Destroy one unowned machine and record what happened, then and there.

    The destroy itself goes through the same confirmed path the orchestrator
    uses (`confirmed_destroy`, ADR-0057): retry a refused destroy, escalate
    loudly, then require consecutive absent listings. Each step is recorded on
    the owner job's own history when the machine has an owner, and every
    outcome lands one row in the reconciliation log; a destroyed orphan is
    an event, never a silent removal.

    After a *confirmed* destroy, any job row that still names this machine as
    its current machine and is not terminal is marked failed: a job that
    claims a machine the reconciler has just destroyed as unowned would
    otherwise sit non-terminal forever. This is the branch the "rather than
    left running forever" clause exists for, reachable in the real world by
    the record-first race, where a worker records its machine_id onto a live
    row in the same instant the reconciler is acting on the unowned machine,
    and exercised deterministically in the reconciler tests.

    A destroy that is *not* confirmed (the provider refused it and the machine
    is still billing) does not mark any job failed: the machine is still
    there, so a job that names it has not lost it, and telling the job
    otherwise would be a lie. The reconciliation log records it as
    `destroy_unconfirmed`; the STRAY case an operator must act on, and a
    later pass, or manual removal, takes over.
    """
    from temper_control_plane import db
    from temper_control_plane.orchestrator import confirmed_destroy
    from temper_control_plane.provider import Machine

    machine = Machine(machine_id=machine_id, status=status)
    owners = db.job_rows_for_machine(machine_id)
    owner_job_id = owners[0]["id"] if owners else None

    def record(kind: str, message: str, data: dict | None = None) -> None:
        # A machine with an owner: the destruction belongs on the owner's own
        # history as well as the reconciliation log, so the run's record says
        # what happened to its machine. A machine with no owner (the fault
        # surface's pure orphan) has no history to write to; the reconciliation
        # log is the record.
        if owner_job_id is not None:
            try:
                db.add_event(owner_job_id, kind, message, data)
            except Exception:  # noqa: S110 - the reconciliation log still stands
                pass
        logger.info(
            "reconciler destroy",
            machine_id=machine_id,
            kind=kind,
            message=message,
        )

    confirmed = confirmed_destroy(provider, machine, record)

    if confirmed:
        action = ACTION_DESTROYED
        reason = ORPHANED_MACHINE_MESSAGE
        report["destroyed"].append(machine_id)
    else:
        action = ACTION_DESTROY_UNCONFIRMED
        reason = (
            "destroy refused and machine still listed; manual removal required"
        )
        report["unconfirmed"].append(machine_id)
    db.record_reconciliation(
        machine_id,
        action,
        reason,
        status=status,
        job_id=owner_job_id,
    )

    if not confirmed:
        # The machine is still billing; there is no destroyed machine for a
        # job to have lost. Nothing to fail.
        return

    for owner in owners:
        if owner["status"] in db.TERMINAL_STATES:
            # A terminal owner is a leak from a teardown that failed; its
            # destruction is already on its history. Nothing to fail; the
            # row is already terminal, and marking a complete job failed
            # would be a lie.
            continue
        db.set_state(
            owner["id"],
            "failed",
            ORPHANED_MACHINE_MESSAGE,
            error_code=ORPHANED_MACHINE_CODE,
            error_message=ORPHANED_MACHINE_MESSAGE,
        )
        db.add_event(
            owner["id"],
            "error",
            f"Machine {machine_id} was found with no live owner and destroyed "
            f"by the reconciler; this job cannot continue. "
            f"{ORPHANED_MACHINE_MESSAGE}",
        )
        report["marked_failed"].append(owner["id"])


def reconcile_once(provider=None) -> dict:
    """One reconciliation pass. Returns a report of what it decided.

    The whole pass in one call so a test can drive it deterministically and a
    scheduled loop can call it on an interval. `provider` is injected like
    the orchestrator's: omit it and the real one is built, which is where a
    missing credential surfaces; tests pass the fake.
    """
    from temper_control_plane import db
    from temper_control_plane.provider import (
        endpoint_machine_name,
        machine_name,
        new_provider,
        normalize_status,
    )

    owns_provider = provider is None
    if provider is None:
        try:
            provider = new_provider()
        except Exception as e:  # noqa: BLE001 - a missing credential must not kill the pass
            logger.error(
                "reconciler could not build a provider; nothing was listed",
                error=str(e),
            )
            return {
                "listed": 0,
                "owned": 0,
                "destroying": 0,
                "destroyed": [],
                "unconfirmed": [],
                "marked_failed": [],
                "error": str(e),
            }
    report: dict = {
        "listed": 0,
        "owned": 0,
        "destroying": 0,
        "destroyed": [],
        "unconfirmed": [],
        "marked_failed": [],
    }
    try:
        machines = provider.list_machines()
        report["listed"] = len(machines)
        # The two ownership halves, read once so every machine in this pass
        # is judged against the same snapshot: jobs not in a terminal state,
        # and served endpoints (ADR-0065). A machine in either set is
        # accounted for and left alone.
        owned_by_job = set(db.list_non_terminal_machine_ids())
        owned_by_endpoint = set(
            db.list_active_endpoint_machine_ids(ENDPOINT_STARTING_GRACE_S)
        )
        # The third half, and the one that closes the window. A job cannot
        # record its machine id until `provider.create` returns, but the
        # instance is listed and billing before that -- measured at about
        # twelve seconds against a thirty-second pass, which is how a live
        # machine came to be destroyed as an orphan. The name is set at
        # creation from the job id, so it identifies the owner throughout
        # that window. Derived from live job ids through the one definition
        # both sides read (`machine_name`), never rebuilt from a format
        # string here.
        # An empty name is never ownership: a provider that reports no
        # name must leave id matching as the only test, not protect every
        # unnamed machine at once.
        owned_names = (
            {machine_name(job_id) for job_id in db.list_non_terminal_job_ids()}
            # And the serving half. A served endpoint's machine has its own
            # name, because a job can have a finished training machine and a
            # live serving machine and sparing the wrong one is as bad as
            # destroying the wrong one.
            | {
                endpoint_machine_name(job_id)
                for job_id in db.list_live_endpoint_job_ids(
                    ENDPOINT_STARTING_GRACE_S
                )
            }
        ) - {""}

        for machine in machines:
            status = normalize_status(getattr(machine, "status", None))
            mid = machine.machine_id
            if status == "destroying":
                # Already being torn down by whoever owns it; not ours to
                # double-destroy (ADR-0057). An unowned machine mid-destroy is
                # still recorded, so it is visible rather than silently
                # dropped, but no second destroy is issued.
                report["destroying"] += 1
                if (
                    mid not in owned_by_job
                    and mid not in owned_by_endpoint
                    and getattr(machine, "name", "") not in owned_names
                ):
                    db.record_reconciliation(
                        mid,
                        ACTION_SKIPPED_DESTROYING,
                        "already being destroyed; not yet confirmed (ADR-0057)",
                        status=status,
                    )
                continue
            if (
                mid in owned_by_job
                or mid in owned_by_endpoint
                or (
                    bool(getattr(machine, "name", ""))
                    and getattr(machine, "name", "") in owned_names
                )
            ):
                report["owned"] += 1
                continue
            _destroy_orphan(provider, mid, status, report)

        return report
    finally:
        if owns_provider:
            provider.close()


def reconciler_loop(stop: threading.Event) -> None:
    """Run `reconcile_once` on the interval until `stop` is set.

    A pass that throws must not kill the loop: a machine that the pass would
    have cleaned up is worse than a pass that missed one cycle, so each
    iteration is isolated the way the serving sweep's is.
    """
    while not stop.wait(RECONCILE_INTERVAL_S):
        try:
            report = reconcile_once()
            logger.info(
                "reconciler pass complete",
                listed=report["listed"],
                owned=report["owned"],
                destroying=report["destroying"],
                destroyed=report["destroyed"],
                unconfirmed=report["unconfirmed"],
                marked_failed=report["marked_failed"],
            )
        except Exception as e:  # noqa: BLE001 - a failed pass must not kill the thread
            logger.exception(
                "reconciler pass failed",
                exc_info=e,
            )


# ---------------------------------------------------------------------------
# the scheduled thread (mirrors serving's sweep thread shape)
# ---------------------------------------------------------------------------

_reconciler_thread: threading.Thread | None = None
_reconciler_stop = threading.Event()


def start_reconciler_thread() -> None:
    """Start the periodic reconciler thread (once). Called at worker startup."""
    global _reconciler_thread
    if _reconciler_thread is not None and _reconciler_thread.is_alive():
        return
    _reconciler_stop.clear()

    _reconciler_thread = threading.Thread(
        target=reconciler_loop,
        args=(_reconciler_stop,),
        daemon=True,
        name="machine-reconciler",
    )
    _reconciler_thread.start()


def stop_reconciler_thread() -> None:
    """Ask the reconciler thread to stop. Daemon: the process does not wait."""
    _reconciler_stop.set()
    global _reconciler_thread
    _reconciler_thread = None
