# ADR-0049 — Progress is measured from the output that was going to be thrown away

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/006-streaming-transport-progress-and-artifacts.md`
- **Issue:** [#49](https://github.com/thp728/temper/issues/49)

## Context

One real run wrote several hundred events, the overwhelming majority of them
container-layer pull progress, and the standing plan was to filter those lines
at classification so they never became events at all — a plan that discards
evidence. Meanwhile the phase that dominates a large job, downloading the model
weights, produced no progress signal at all.

Two facts made the standing plan the wrong answer, and both are visible in the
same output. First, the noisy lines carry bytes-so-far and bytes-total: **the
noise is the progress data.** Second, the phase that currently produces no
signal — pulling a hundred gigabytes of weights — is the one a user is most
waiting on, while the phase slated for deletion is the one that is *already*
loud. Filtering the pull lines would solve the flood by throwing away the only
measurement a dominant phase produces, and it would leave the other dominant
phase silent.

There was also a product-shaped reason the plan could not stand: the event
store returns oldest-first with a cap, so a finished job's page stopped partway
through the image pull and never showed that the artifact was verified, the
machine destroyed, or the job finished. Every user landing on a completed job
saw a truncated render of its own history.

A third fact set the surface. The live running-job view (#39) already consumes
one server-pushed event stream over the durable event log. Progress had to ride
that existing stream and render in that existing view — not a second stream,
not a second view.

## Decision

**Progress is promoted, not filtered. Classification gains a progress kind,
and the promoted lines are retained as collapsed detail, so nothing is
discarded.**

- **Classification gains a fourth kind of event.** Layer-pull lines and
  model-download lines stop being log lines and become progress records
  carrying the phase, the bytes done, the bytes expected, and a rate. Docker's
  pull output (a layer id, a status word, a done/total byte pair) becomes
  image-pull progress; huggingface_hub's download bars (a file name, then
  n/total) become model-download progress. A download bar is told apart from a
  tokenization map or a training bar by naming a file — a dot in the
  description — so example counts are never mistaken for bytes.

- **The rate is measured live, not read off a line.** Docker's pull lines carry
  no rate, and the captured download bar omits it, so the rate is measured
  between consecutive readings of the same phase: bytes moved over the time
  between them. The estimate uses that rate, so it corrects itself when actual
  throughput differs from whatever came before — the live figure, never a
  stored prediction (which issue #77's actuals record already owns and which
  this work does not touch).

- **Progress supersedes rather than accumulates.** One record per phase, and a
  new line replaces it. The interface renders the latest per phase. This is
  why progress cannot be modelled as a log line or as a metric: it has no step
  and it replaces rather than appends. Hundreds of pull lines collapse into a
  handful of superseding updates, which is what stops the finished page from
  truncating — the flood no longer exists in the event log.

- **Image pull aggregates across layers.** Docker pulls layers in parallel, so
  no single line is the image's progress. The phase's bytes are the aggregate
  across every layer seen so far, which is what "image pull: 57MB/130MB"
  means. The raw per-layer lines are retained, so the detail is not lost.

- **The raw lines that were promoted are retained but not emitted as events.**
  They are written to the job's own output record and offered as collapsed
  detail per phase. "We keep the whole log" stays true, and the log stops being
  unreadable — solved by promotion rather than by a judgment call about what a
  user is allowed to see. The event cap stays: a job that genuinely produces
  many events should still be paginated rather than silently truncated, and the
  interface should say what it is showing and of how many. Fixing the cause
  does not make the symptom's guardrail unnecessary.

- **The model download is invoked so that it reports progress.** The trainer
  prefetches the base model with `snapshot_download` before training, forcing
  progress bars on so a non-TTY cannot silence them — the machine's output is
  piped, which is exactly where tqdm's default hides bars. Axolotl then
  resolves from the HF cache, so this is not a second download, and a prefetch
  that cannot run costs a progress signal, never the job: training proceeds as
  it always has.

- **Progress rides the existing stream and the existing views.** The durable
  history page carries the per-phase snapshot and the retained output beside
  the events; the live stream pushes the snapshot on the same connection
  whenever it changes. No second channel, no second view — the one server-
  pushed connection the running view already uses now carries progress too.

## Alternatives considered

**Filter layer-pull lines at classification so they never become events.**
Rejected: it discards evidence, and the evidence is the progress data — the
only measurement the pull phase produces. This is the plan the issue exists to
undo, and it would have left the phase that dominates a large job with no
signal at all while deleting the one that already had it.

**Keep the pull lines as log events and rely on the event cap.**
Rejected: the cap is a guardrail against genuine floods, not a licence for the
dominant phase to flood. A user who lands on a completed job would still see a
truncated render of its history, and the noise would still bury the lines that
matter. The cap stays as a guardrail; the flood is removed at the source.

**Promote progress but keep every update as an event, letting the interface
collapse them.**
Rejected: that does not solve the truncation — the events table would still
hold hundreds of rows per pull and the finished page would still stop partway.
The supersede happens at the record, not at the renderer, which is what makes
the durable log small.

**Carry a prediction as the rate, or reuse the quote's figures.**
Rejected: the issue asks for a rate measured from what is actually happening,
so the estimate corrects itself when reality differs from the prediction. A
stored figure would be stale by construction. The prediction-vs-measurement
record (#77) is not reached into; the rate here is a live figure.

**Open a second stream for progress.**
Rejected: #39's stream over the durable log is the channel, and progress is a
new kind of data on it, not a new connection. Extending the existing stream
keeps one replay story and one view.

**Measure image pull as the latest single layer's numbers.**
Rejected: layers pull in parallel, so any single line is one layer's progress,
not the image's. Aggregating across the layers seen so far is what makes
"image pull: 57MB/130MB" mean something, and the per-layer lines are retained
as the detail that explains it.

## Consequences

- The line classifier promotes pull and download output into progress records;
  the orchestrator holds a per-phase tracker that supersedes, measures the live
  rate and retains the raw lines; the events table no longer fills with pull
  chatter, so a finished page renders to its end.
- The events page carries the per-phase snapshot and the retained output; the
  stream pushes progress snapshots on the same connection; the running view and
  the finished record both render a progress region with proportion, measured
  rate, estimate and collapsed detail.
- The trainer prefetches the base model in a progress-reporting mode; the
  journey fake emits pull and download output, so the journeys watch image pull
  and model download advance with a measured rate end to end.
- The one acceptance criterion that cannot be satisfied without hardware — a
  real run watched through the interface showing image-pull and model-download
  progress advancing with a measured rate — is stated as outstanding in the PR
  body rather than quietly reinterpreted as met by the fake run, exactly as
  ADR-0048 treated its hardware-only criterion.
- The layer-pull test fixture follows docker's documented pull output format:
  the repo holds no captured pull transcript (spike 6 redirected the pull's
  output to a file and never kept it), so the format is documented rather than
  captured. Capturing one on a real pull and checking the classifier against it
  line by line is recorded as outstanding. The model-download fixture is the
  genuinely captured bar the suite already carried.

## Rollback

Revert the classifier's progress promotion, the tracker wiring, the trainer's
prefetch and the progress region; pull lines return to log events and the
previous truncating finished page returns. The `job_progress` and `job_output`
tables are additive and can stay without being read, or be dropped with the
schema that created them.
