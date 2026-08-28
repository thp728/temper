"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import BackToUpload from "@/components/BackToUpload";
import FocusHeading from "@/components/FocusHeading";
import LossChart from "@/components/LossChart";
import PlateauNote from "@/components/PlateauNote";
import ProgressRegion from "@/components/ProgressRegion";
import { Button } from "@/components/ui/button";
import {
  TERMINAL_STATUSES,
  formatDuration,
  latestLoss,
  shortRevision,
} from "@/lib/jobs/display";
import { latestHeldOutLoss, lossSeries } from "@/lib/jobs/loss";
import { heldOutPlateau } from "@/lib/jobs/plateau";
import { supersedeProgress } from "@/lib/jobs/progress";
import {
  jobStreamUrl,
  parseJobEvent,
  parseJobOutput,
  parseJobProgress,
} from "@/lib/jobs/stream";
import {
  cancelJobV1JobsJobIdCancelPost,
  getJobV1JobsJobIdGet,
} from "@/lib/api/generated/client";
import type {
  JobEvent,
  JobOutputLine,
  JobProgress,
  JobRecord,
} from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

// The live job view (issue #39): the running half of `/jobs/:id`. The page
// renders this only while the job is non-terminal -- once it ends, the stream
// closes and the page hands back to the server-rendered finished record
// (#40/#77), so the finished view is never duplicated here.
//
// Everything live comes off one server-pushed stream over the durable event
// log: output lines, metric events, per-phase progress and state transitions
// as they are recorded. The page renders the history the server already had,
// opens the stream from its last event id, and appends whatever follows --
// which is what makes leaving and returning show continuous history rather
// than a view that starts where the visitor rejoined. A dropped connection
// reconnects on its own (EventSource replays the last id it received), and
// the stream ends only at a terminal state, which is the page's cue to reload
// into the record.
//
// Progress rides the same connection (issue #49): a `progress` snapshot is
// replaced per phase, never appended, and a `output` line is deduplicated by
// id, so the live view agrees with the durable history about the latest
// figures and the retained raw lines.

function Stat({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <>
      {/* Label and value stay a real dt/dd pair: that adjacency is what a
          screen reader announces, and what the tests read. */}
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd>{children}</dd>
    </>
  );
}

export default function RunningJobView({
  job: initialJob,
  events: initialEvents,
  progress: initialProgress = [],
  output: initialOutput = [],
  datasetFilename,
  streamAfter,
  initialElapsedSeconds,
}: {
  job: JobRecord;
  events: JobEvent[];
  progress?: JobProgress[];
  output?: JobOutputLine[];
  datasetFilename?: string;
  streamAfter: number;
  initialElapsedSeconds: number;
}) {
  const [job, setJob] = useState(initialJob);
  const [events, setEvents] = useState(initialEvents);
  // Seeded from the server so the first paint agrees with the server-rendered
  // HTML, then replaced/updated by the stream's progress and output events.
  const [progress, setProgress] = useState(initialProgress);
  const [output, setOutput] = useState(initialOutput);
  // Output lines are deduplicated by id: the stream re-sends the retained
  // record to a reconnecting client, so an id already held is not appended
  // twice.
  const outputIds = useRef(new Set(initialOutput.map((o) => o.id)));
  // Seeded from the server so the first paint agrees with the server-rendered
  // HTML; the interval below takes over once mounted. Hydration must not see
  // a clock reading the client computed for itself.
  const [elapsed, setElapsed] = useState(initialElapsedSeconds);
  const [cancelRequested, setCancelRequested] = useState(
    initialJob.cancel_requested ?? false,
  );
  const [cancelError, setCancelError] = useState<ApiError | null>(null);

  // Elapsed keeps ticking without a reload: the record gives the origin, the
  // clock is the client's own, and the two never have to agree about how long
  // "now" is.
  const start = initialJob.started_at ?? initialJob.created_at;
  useEffect(() => {
    const timer = setInterval(
      () => setElapsed(Math.max(0, Date.now() / 1000 - start)),
      1000,
    );
    return () => clearInterval(timer);
  }, [start]);

  // The one live channel. `streamAfter` is where the server-rendered history
  // already ended, so the stream picks up exactly where the page stopped.
  useEffect(() => {
    let disposed = false;
    const es = new EventSource(jobStreamUrl(initialJob.id, streamAfter));

    const onEvent = (msg: Event) => {
      let e: JobEvent;
      try {
        e = parseJobEvent((msg as MessageEvent).data);
      } catch {
        return;
      }
      if (disposed) return;
      setEvents((prev) => [...prev, e]);
      if (e.kind === "state") {
        // The status word comes from the record, never from prose: a state
        // event says what is happening, the record says what the job is. The
        // refetch is driven by the stream, not a timer, so the two cannot
        // drift apart about when to look.
        getJobV1JobsJobIdGet(initialJob.id)
          .then((record) => {
            if (disposed) return;
            setJob(record);
            if (TERMINAL_STATUSES.includes(record.status)) {
              // The hand-back, primary path: the terminal transition arrived,
              // the record confirms it, and the page reloads into the
              // server-rendered finished record. A transient refetch failure
              // self-heals, because a dropped connection reconnects and the
              // terminal state event is re-delivered, retrying this refetch.
              window.location.reload();
            }
          })
          .catch(() => {
            // Transient; the stream's reconnect re-delivers the state event
            // and retries this refetch.
          });
      }
    };

    // The hand-back, secondary path: the server also emits an explicit `end`
    // marker after the terminal transition. It is not relied on as the sole
    // signal -- a proxy in the path can fail to propagate the final chunk or
    // the close -- but where it arrives it reloads exactly once.
    const onEnd = () => {
      if (!disposed) window.location.reload();
    };

    // Progress supersedes per phase (issue #49): a snapshot replaces the
    // phase's record, never appends to it. Output lines are retained raw
    // lines, deduplicated by id, so the collapsed detail stays live without
    // duplicating on reconnect.
    const onProgress = (msg: Event) => {
      let p: JobProgress;
      try {
        p = parseJobProgress((msg as MessageEvent).data);
      } catch {
        return;
      }
      if (disposed) return;
      setProgress((prev) => supersedeProgress(prev, [p]));
    };

    const onOutput = (msg: Event) => {
      let line: JobOutputLine;
      try {
        line = parseJobOutput((msg as MessageEvent).data);
      } catch {
        return;
      }
      if (disposed) return;
      if (outputIds.current.has(line.id)) return;
      outputIds.current.add(line.id);
      setOutput((prev) => [...prev, line]);
    };

    es.addEventListener("job", onEvent);
    es.addEventListener("end", onEnd);
    es.addEventListener("progress", onProgress);
    es.addEventListener("output", onOutput);
    return () => {
      disposed = true;
      es.close();
    };
  }, [initialJob.id, streamAfter]);

  const loss = latestLoss(events);
  const heldOutLatest = latestHeldOutLoss(events);

  // The two series the chart draws (issue #53) come from the same stream the
  // view already appends to, so the chart re-renders as each measurement is
  // pushed -- no second request, no timer. A plateau in the held-out series
  // is a signal a non-specialist can read, so it is said in those words.
  const { training, heldOut } = lossSeries(events);
  const plateau = heldOutPlateau(heldOut.map((p) => p.heldOutLoss!));

  const cancel = async () => {
    try {
      await cancelJobV1JobsJobIdCancelPost(initialJob.id);
      setCancelRequested(true);
    } catch (err) {
      // A genuine refusal is shown with its stable code, never swallowed: a
      // user who clicked cancel and got nothing back would not know the
      // request failed. A job that ended between the click and the request
      // answers itself, because the stream's end marker reloads the page into
      // the finished record.
      setCancelError(err instanceof ApiError ? err : NETWORK_ERROR);
    }
  };

  return (
    <section aria-labelledby="job-heading" className="space-y-6">
      <div>
        <FocusHeading id="job-heading">Job {initialJob.id}</FocusHeading>
        <p className="mt-2 text-muted-foreground">
          Dataset{" "}
          <Link
            href={`/datasets/${initialJob.dataset_id}`}
            className="underline hover:no-underline"
          >
            {datasetFilename ?? initialJob.dataset_id}
          </Link>{" "}
          · base model <code>{initialJob.base_model}</code>
          {initialJob.base_revision && (
            <>
              @<code>{shortRevision(initialJob.base_revision)}</code>
            </>
          )}
        </p>
      </div>

      <section aria-labelledby="status-heading">
        <h2 id="status-heading" className="sr-only">
          Status
        </h2>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
          <Stat label="State">
            <strong id="job-state" aria-live="polite" className="text-base">
              {job.status}
            </strong>
          </Stat>
          <Stat label="Elapsed">{formatDuration(elapsed)}</Stat>
          <Stat label="Machine">
            {job.gpu_type
              ? `${job.gpu_type} at ${job.price_per_hour} ${job.currency}/hr`
              : "—"}
          </Stat>
          <Stat label="Latest loss">
            {loss
              ? `${loss.loss}${loss.step !== undefined ? ` at step ${loss.step}` : ""}`
              : "—"}
          </Stat>
          <Stat label="Latest held-out loss">
            {heldOutLatest
              ? `${heldOutLatest.loss}${
                  heldOutLatest.epoch !== undefined
                    ? ` at epoch ${heldOutLatest.epoch}`
                    : ""
                }`
              : "—"}
          </Stat>
        </dl>
      </section>

      {/* Progress is promoted from the output that was going to be thrown away
          (issue #49): image pull and model download advance with a measured
          rate and an estimate, and the raw lines are offered collapsed. It
          sits directly under the status so the longest phases of a job are
          never a blank screen. */}
      <ProgressRegion progress={progress} output={output} />

      <section aria-labelledby="loss-chart-heading" className="space-y-2">
        <h2 id="loss-chart-heading" className="text-lg font-semibold">
          Loss
        </h2>
        <LossChart training={training} heldOut={heldOut} />
        {plateau && (
          // The plateau is announced as it appears: a live region so a
          // non-specialist is told, in plain language, that the held-out loss
          // has stopped improving and why that matters.
          <PlateauNote message={plateau.message} live />
        )}
      </section>

      <section aria-labelledby="output-heading" className="space-y-2">
        <h2 id="output-heading" className="text-lg font-semibold">
          Output
        </h2>
        <div
          role="log"
          aria-label="Output"
          tabIndex={0}
          className="max-h-80 overflow-y-auto rounded-lg border bg-card p-3 font-mono text-xs"
        >
          {events.length === 0 ? (
            <div className="font-sans text-muted-foreground">
              Nothing recorded yet.
            </div>
          ) : (
            events.map((e) => <div key={e.id}>{e.message}</div>)
          )}
        </div>
      </section>

      <section aria-labelledby="cancel-heading" className="space-y-2">
        <h2 id="cancel-heading" className="text-lg font-semibold">
          Cancel this job?
        </h2>
        {/* The consequence -- no artifact will be produced -- is stated beside
            the control, before the request exists (the port of the old watch
            page's warning). The button is styled destructive, and the control
            disappears with the job: a terminal record offers nothing to
            cancel. */}
        <p>
          Cancelling destroys the machine and{" "}
          <strong>no artifact will be produced</strong>. This cannot be undone.
        </p>
        <div className="flex items-center gap-3">
          <Button
            variant="destructive"
            onClick={() => void cancel()}
            disabled={cancelRequested}
          >
            {cancelRequested ? "Cancellation requested" : "Cancel job"}
          </Button>
          {cancelError && (
            // The stable code survives every rendering decision, like every
            // other refusal: it is what a bug report can be pinned to.
            <p className="text-sm text-destructive">
              <code className="rounded bg-muted px-1">{cancelError.code}</code>{" "}
              — {cancelError.message}
            </p>
          )}
        </div>
      </section>

      <div className="flex gap-3">
        <Button variant="outline" asChild>
          <Link href="/jobs">All jobs</Link>
        </Button>
        <BackToUpload />
      </div>
    </section>
  );
}
