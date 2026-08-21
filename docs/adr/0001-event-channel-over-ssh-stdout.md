# ADR-0001 — The event channel is the machine's stdout, pulled over SSH

- **Status:** accepted
- **Date:** 2026-08-21
- **Spec:** `docs/specs/001-orchestrator-core.md`
- **Issue:** [#4](https://github.com/thp728/temper/issues/4)

## Context

A user who launched a job watched nothing happen for about four minutes. The
build-and-train phase was measured at **251 seconds of complete silence** on
2026-08-19, and the thirteen events that did arrive afterwards all carried the
**same timestamp** — so the record could not say how long any phase took, even
after the fact.

Nothing was wrong with the event log. The problem was that no output could
reach it. Between the training framework and the user sat three separate
redirections, each of which alone was enough to hold everything back until the
run was over:

1. **In the container.** The trainer ran `axolotl` with its output going
   straight to a file handle (`/out/train.log`), never to the container's
   stdout.
2. **On the machine.** The remote script redirected the container's output to
   `/tmp/run.log` and printed a 30-line tail of it after the run finished.
3. **In the control plane.** The orchestrator ran the remote command as one
   blocking call and read its output only after it exited.

The third had already been opened by the provider seam (issue #3), which made
`stream` an iterator of lines. That bought nothing on its own, because the two
layers beneath it still held the output.

A fine-tuning product that cannot show a loss value while a job runs is the
hello-world version the brief disqualifies. Opening this channel is also the
prerequisite for three other behaviours in the same spec — cancellation, stall
detection, and the duration ceiling all need a loop over arriving lines to hang
a check on.

## Decision

**Job output is pulled over the SSH connection that already exists**, as the
machine's standard output, and turned into events one line at a time as it
arrives. All three redirections are opened together:

- The trainer relays the framework's output to the container's stdout line by
  line (`run_streaming`), still writing `train.log` as the on-machine copy.
- The remote script writes no log file. The container's output goes to the
  script's stderr, which the provider folds into the same ordered stream —
  stdout is reserved for the result-document protocol, so a training line that
  happened to equal the result marker cannot corrupt what the orchestrator
  parses.
- `PYTHONUNBUFFERED=1` is set on the container. This is the innermost layer and
  the least visible: both the trainer and the framework beneath it are Python,
  and Python buffers stdout whenever it is a pipe rather than a terminal. With
  it unset, the two layers outside it relay nothing and every test above still
  passes.

Each line becomes an event **stamped when it was read**, not when the command
completed.

The `stream` contract — an iterator of lines — is deliberately transport-shaped
rather than SSH-shaped. Moving to a push transport later is a new implementation
of the provider protocol, not a change to the orchestration logic.

## Alternatives considered

**An outbound HTTPS push from the machine to the control plane.** This is what
a deployed product does, and the `stream` contract is shaped so it can be
substituted in. It is rejected *now* because it requires a publicly addressable
control plane, which is out of scope in both phases — everything runs on
localhost behind Docker Compose. Adopting it today would mean building and
securing an ingest endpoint, authenticating the machine to it, and handling
retries when it is unreachable: a second failure domain for a capability the
SSH connection already provides. It also inverts the trust direction, giving
the machine a route into the control plane, when the current design keeps the
machine a pure compute node that nothing but the control plane reaches.

**Polling a log file on the machine.** Keep `/tmp/run.log` and `tail -f` it, or
re-read it over SSH on a timer. Rejected: it adds a poll interval to every
line's latency, needs its own offset bookkeeping to avoid re-reading, and
recovers nothing after the machine is destroyed — the file dies with it. It is
strictly more machinery than reading the stream that is already open, in
exchange for worse latency.

**A reverse SSH tunnel.** Forward a port back to the control plane and have the
container post to it. Rejected: it is the HTTPS push with extra steps, and it
reintroduces a listening port on the path we deliberately removed. `ufw` does
not filter Docker-published ports and a `DOCKER-USER` rule matched on the
published port never fires, because the packet is already DNAT'd and carries the
container port. Not publishing anything is the only mitigation that holds.

**Leaving the trainer's file and shipping it at the end.** Cheapest change, and
it answers "what was the loss?" after the fact. Rejected because it answers
nothing *during* the run, which is the whole complaint, and because it does not
give the orchestrator the loop the rest of the spec needs.

## Consequences

- Output reaches the user while the job runs, and every event carries the time
  its line was actually read. The event log can now say how long each phase
  took.
- The orchestrator has a loop over arriving lines. Cancellation (#6), stall
  detection and the duration ceiling (#7) attach to it rather than needing
  structure of their own.
- Line classification (#5) becomes a pure function over one line: promoting
  step, loss and epoch to structured metric events is a change to that function
  and nothing else.
- Volume is now the control plane's problem. A verbose framework produces one
  event per line, and `--progress=plain` on the image build produces a lot of
  them. Blank lines are dropped at the container; nothing else is filtered yet.
  If this becomes a problem it is a rate or retention decision, recorded then.
- Output is bounded by the life of the SSH connection. A dropped connection
  loses the tail of the stream; the events already written survive, because
  they were written as they arrived rather than at the end. Reconnecting
  mid-run is not implemented and is a Phase B concern alongside durable
  execution.
- `train.log` still exists on the machine. It is redundant while the stream
  holds and is the only copy if it does not, which is worth its zero cost.
- **A failure before the trainer runs now arrives as a result document with its
  own code.** Removing the file redirections forced this: the old script tailed
  `build.log` and echoed a bare JSON object *without* the result marker, so the
  orchestrator never parsed it and reported "no result.json" — a build failure
  reaching the user as `training_failed`. The build failure and the
  missing-result case now echo the marker and carry `image_build_failed` and
  `trainer_no_result`. This is more than the ticket asked for; it is included
  because the alternative was leaving a known mislabelling in code the change
  had to rewrite anyway.
- **One test asserts on the remote script's text**, which the spec's testing
  rules otherwise permit only for the stray-instance path. Recorded here rather
  than granted in a docstring. The justification is specific: the fake provider
  yields lines the script never produced, so a redirection reintroduced in the
  script would close the channel again while every other test still passed. The
  assertion is on redirections in the `docker` commands, not on the shape of the
  script.

## Rollback

Restore the file redirections in `trainer/entrypoint.py` and
`api/orchestrator._remote_script`. The provider seam and the event schema are
unaffected — nothing else depends on the channel being open, and the tests that
assert it fail loudly rather than silently.
