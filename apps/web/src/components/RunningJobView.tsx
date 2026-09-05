"use client";

import { useEffect, useRef, useState } from "react";
import { Ban, Loader2 } from "lucide-react";
import CheckpointSection from "@/components/CheckpointSection";
import EndpointSection from "@/components/EndpointSection";
import InstabilityBanner from "@/components/InstabilityBanner";
import JobHeader from "@/components/JobHeader";
import JobStatsGrid from "@/components/JobStatsGrid";
import LogConsole from "@/components/LogConsole";
import LossChart from "@/components/LossChart";
import PlateauNote from "@/components/PlateauNote";
import ProgressRegion from "@/components/ProgressRegion";
import QuoteView from "@/components/QuoteView";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TERMINAL_STATUSES, formatTimestamp, latestLoss } from "@/lib/jobs/display";
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
// Its chrome -- header, status tiles, tabs -- is deliberately the same
// components `JobRecordView` renders (`JobHeader`, `JobStatsGrid`, the same
// four tabs), so the reload from this view into that one changes what a tile
// says, never where it sits or what shape holds it. Only the reload
// (issue: non-terminal/terminal parity) is the seam; everything either side
// of it should look like the same page.
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
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [cancelling, setCancelling] = useState(false);
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

  // Before the first phase reports progress (queued, provisioning, waiting
  // on SSH), `ProgressRegion` renders nothing -- and nothing else in the
  // Overview tab has content yet either, which otherwise leaves it a blank
  // screen the entire time a job spends being provisioned. The narration
  // orchestrator already writes for the Logs tab ("Selecting hardware",
  // "Machine 4242 running; waiting for SSH") says exactly what is happening;
  // this reads the same stream's latest line rather than inventing a second
  // description of the same states. Metric events are skipped -- a raw
  // `{'loss': ...}` line is not a status sentence.
  const latestStatusMessage = [...events]
    .reverse()
    .find((e) => e.kind !== "metric")?.message;

  // The two series the chart draws (issue #53) come from the same stream the
  // view already appends to, so the chart re-renders as each measurement is
  // pushed -- no second request, no timer. A plateau in the held-out series
  // is a signal a non-specialist can read, so it is said in those words.
  const { training, heldOut } = lossSeries(events);
  const plateau = heldOutPlateau(heldOut.map((p) => p.heldOutLoss!));

  const cancel = async () => {
    setCancelling(true);
    setCancelError(null);
    try {
      await cancelJobV1JobsJobIdCancelPost(initialJob.id);
      setCancelRequested(true);
      setCancelDialogOpen(false);
    } catch (err) {
      // A genuine refusal is shown with its stable code, never swallowed, and
      // the dialog stays open so the reader sees why -- closing on failure
      // would read as if nothing had been asked. A job that ended between the
      // click and the request answers itself, because the stream's end
      // marker reloads the page into the finished record.
      setCancelError(err instanceof ApiError ? err : NETWORK_ERROR);
    } finally {
      setCancelling(false);
    }
  };

  return (
    <section aria-labelledby="job-heading" className="space-y-6">
      <JobHeader
        job={job}
        datasetFilename={datasetFilename}
        meta={`started ${formatTimestamp(start)}`}
        actions={
          // The one control a terminal record never offers -- it goes with
          // the header rather than the tabs, so it stays reachable no matter
          // which tab is open, and is simply gone once the job ends.
          <Button
            variant="destructive"
            onClick={() => {
              setCancelError(null);
              setCancelDialogOpen(true);
            }}
            disabled={cancelRequested}
          >
            <Ban aria-hidden className="size-4" />
            {cancelRequested ? "Cancellation requested" : "Cancel job"}
          </Button>
        }
      />

      {/* The consequence -- no artifact will be produced -- is a question
          asked before the request exists, not a caption a click could race
          past: cancelling has no undo, so it earns a confirmation rather
          than firing on the first click. */}
      <Dialog open={cancelDialogOpen} onOpenChange={setCancelDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Cancel this job?</DialogTitle>
            <DialogDescription>
              Cancelling destroys the machine and{" "}
              <strong className="text-foreground">
                no artifact will be produced
              </strong>
              . This cannot be undone.
            </DialogDescription>
          </DialogHeader>

          {cancelError && (
            // The stable code survives every rendering decision, like every
            // other refusal: it is what a bug report can be pinned to.
            <Alert variant="destructive">
              <AlertTitle>
                The cancellation was refused.{" "}
                <code className="rounded bg-muted px-1 text-xs">
                  {cancelError.code}
                </code>
              </AlertTitle>
              <AlertDescription>{cancelError.message}</AlertDescription>
            </Alert>
          )}

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setCancelDialogOpen(false)}
              disabled={cancelling}
            >
              Keep it running
            </Button>
            <Button
              type="button"
              variant="destructive"
              onClick={() => void cancel()}
              disabled={cancelling}
            >
              <Ban aria-hidden className="size-4" />
              {cancelling ? "Cancelling…" : "Yes, cancel job"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <JobStatsGrid job={job} elapsedSeconds={elapsed} loss={loss} />

      {/* Same four tabs as the finished record, same order, same labels: the
          reload that follows a terminal state should feel like the page
          settling, not switching to a different one. */}
      <Tabs defaultValue="overview">
        <TabsList aria-label="Job sections">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="evaluation">Evaluation &amp; loss</TabsTrigger>
          <TabsTrigger value="estimates">Cost &amp; estimates</TabsTrigger>
          <TabsTrigger value="output">Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          {/* Queued/provisioning/preparing have no phase to draw a bar for
              yet -- shown as the narration instead, so the tab says what is
              happening rather than sitting empty until image pull starts. */}
          {progress.length === 0 && (
            <div className="flex items-center gap-3 rounded-[12px] border bg-card p-4 text-sm text-muted-foreground">
              <Loader2 aria-hidden className="size-4 shrink-0 animate-spin" />
              <span>{latestStatusMessage ?? "Waiting to start…"}</span>
            </div>
          )}
          {/* Progress is promoted from the output that was going to be thrown
              away (issue #49): image pull and model download advance with a
              measured rate and an estimate, and the raw lines are offered
              collapsed. It sits directly under the status so the longest
              phases of a job are never a blank screen. */}
          <ProgressRegion progress={progress} output={output} />

          <InstabilityBanner events={events} />

          {/* Checkpoints land during training, not only at the end (#37): a
              run well underway can already have a best-so-far. */}
          <CheckpointSection job={job} />

          {/* Renders nothing before the job completes (its own guard) -- kept
              here anyway so the tab's shape doesn't change the moment the
              record does. */}
          <EndpointSection jobId={job.id} jobStatus={job.status} />
        </TabsContent>

        <TabsContent value="evaluation">
          {/* The loss curve leads this tab (issue #53): it is the one number
              every run gets read against, so it comes before anything else. */}
          <section aria-labelledby="loss-chart-heading" className="space-y-2">
            <h2
              id="loss-chart-heading"
              className="text-xs font-medium tracking-widest uppercase text-foreground"
            >
              Loss
            </h2>
            <p className="text-sm text-muted-foreground">
              Training loss, measured every step; held-out loss, measured
              periodically against data the model never trained on.
            </p>
            <LossChart training={training} heldOut={heldOut} />
            {plateau && <PlateauNote message={plateau.message} live />}
            {heldOutLatest && (
              <p className="text-sm text-muted-foreground">
                Latest held-out loss {heldOutLatest.loss}
                {heldOutLatest.epoch !== undefined
                  ? ` at epoch ${heldOutLatest.epoch}`
                  : ""}
                .
              </p>
            )}
          </section>
        </TabsContent>

        <TabsContent value="estimates">
          {/* The advanced-surface overrides the user froze into the job spec
              (issue #80): what the run is training with, shown while it is
              still going -- the record carries them from launch, so there is
              nothing here that only exists once the job ends. */}
          {job.hyperparameters && Object.keys(job.hyperparameters).length > 0 && (
            <section aria-labelledby="settings-changed-heading" className="space-y-2">
              <h2
                id="settings-changed-heading"
                className="text-xs font-medium tracking-widest uppercase text-foreground"
              >
                Settings you changed
              </h2>
              <p className="text-sm text-muted-foreground">
                These were frozen into this job&apos;s specification at launch
                and cannot be changed afterwards.
              </p>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-1 rounded-[12px] border bg-card p-4 sm:grid-cols-3">
                {Object.entries(job.hyperparameters).map(([key, value]) => (
                  <div key={key}>
                    <dt className="text-sm text-muted-foreground">
                      <code>{key}</code>
                    </dt>
                    <dd className="font-medium">{String(value)}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}

          {job.quote && (
            // The quote the job launched under (issue #72): what it was
            // predicted to cost and how long it was predicted to take, the
            // same figures the finished record still shows once there is an
            // actual to set beside them.
            <QuoteView quote={job.quote} />
          )}
        </TabsContent>

        <TabsContent value="output">
          <section aria-labelledby="output-heading" className="space-y-2">
            <h2
              id="output-heading"
              className="text-xs font-medium tracking-widest uppercase text-foreground"
            >
              Logs
            </h2>
            <LogConsole
              events={events}
              disclosure={
                <span className="inline-flex items-center gap-1.5">
                  <span
                    aria-hidden
                    className="size-1.5 animate-pulse rounded-full bg-success"
                  />
                  Live — {events.length} line{events.length === 1 ? "" : "s"}{" "}
                  so far
                </span>
              }
            />
          </section>
        </TabsContent>
      </Tabs>
    </section>
  );
}
