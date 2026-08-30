# Worker

The process that claims queued jobs and drives them (issue #51). It polls
`temper_control_plane.db.claim_next_job` -- `SELECT ... FOR UPDATE SKIP
LOCKED`, so two workers racing for one job never claim the same row and
neither blocks the other -- and calls `orchestrator.run_job` on whatever it
claims. The orchestration entry point is unchanged: it is still a function
over a job record that does not know what called it. See
[ADR-0066](../../docs/adr/0066-orchestration-moves-into-a-worker-process-that-claims-work.md).

Recovering a job left non-terminal by a worker crash mid-run is Spec 010's
reconciler, not this app's job -- see that ADR's "what this does not do"
section.
