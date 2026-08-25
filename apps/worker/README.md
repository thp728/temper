# Worker

**Empty on purpose.** ADR-0010 names this directory so that issue #51 has
somewhere agreed to move orchestration into, rather than inventing a name once
twenty other tickets have already assumed one.

Today the orchestrator runs on a thread inside the control plane's own process,
which means a restart kills the job and leaves a machine provisioned, billing,
and unowned. #51 makes this a separate process that claims work with a row-level
lock, and Spec 010 makes an interrupted job resumable.
