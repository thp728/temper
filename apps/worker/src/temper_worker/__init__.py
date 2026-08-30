"""The worker package: a separate process that claims queued jobs.

See :mod:`temper_worker.worker` for the polling loop. The orchestration
itself (``orchestrator.run_job``) stays in ``temper_control_plane`` and is
called from here unchanged -- the separate process is the change, not the
function it calls (ADR-0066).
"""
