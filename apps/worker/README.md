# Worker

The process that claims queued jobs and drives them (issue #51). It polls
`temper_control_plane.db.claim_next_job` -- `SELECT ... FOR UPDATE SKIP
LOCKED`, so two workers racing for one job never claim the same row and
neither blocks the other -- and calls `orchestrator.run_job` on whatever it
claims. The orchestration entry point is unchanged: it is still a function
over a job record that does not know what called it. See
[ADR-0066](../../docs/adr/0066-orchestration-moves-into-a-worker-process-that-claims-work.md).

The worker also hosts the machine-lifetime reconciler (issue #61 / Spec
010): a scheduled pass, on its own thread so a long-running job claim cannot
stall it, that lists machines, matches them against jobs that are not in a
terminal state and against served endpoints, and destroys anything it cannot
account for. It protects the money, not the job. *Recovering* a job left
non-terminal by a worker crash mid-run -- re-driving it so it can finish -- is
the resumption path's job (#60), not this pass's; see that ADR's "what this
does not do" section.
