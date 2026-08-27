"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import BackToUpload from "@/components/BackToUpload";
import FocusHeading from "@/components/FocusHeading";
import { Button } from "@/components/ui/button";
import {
  TERMINAL_STATUSES,
  formatDuration,
  latestLoss,
  shortRevision,
} from "@/lib/jobs/display";
import { jobStreamUrl, parseJobEvent } from "@/lib/jobs/stream";
import {
  cancelJobV1JobsJobIdCancelPost,
  getJobV1JobsJobIdGet,
} from "@/lib/api/generated/client";
import type { JobEvent, JobRecord } from "@/lib/api/generated/client";

// The live job view (issue #39): the running half of `/jobs/:id`. The page
// renders this only while the job is non-terminal -- once it ends, the stream
// closes and the page hands back to the server-rendered finished record
// (#40/#77), so the finished view is never duplicated here.
//
// Everything live comes off one server-pushed stream over the durable event
// log: output lines, metric events and state transitions as they are
// recorded. The page renders the history the server already had, opens the
// stream from its last event id, and appends whatever follows -- which is what
// makes leaving and returning show continuous history rather than a view that
// starts where the visitor rejoined. A dropped connection reconnects on its
// own (EventSource replays the last id it received), and the stream ends only
// at a terminal state, which is the page's cue to reload into the record.

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
  datasetFilename,
  streamAfter,
  initialElapsedSeconds,
}: {
  job: JobRecord;
  events: JobEvent[];
  datasetFilename?: string;
  streamAfter: number;
  initialElapsedSeconds: number;
}) {
  const [job, setJob] = useState(initialJob);
  const [events, setEvents] = useState(initialEvents);
  // Seeded from the server so the first paint agrees with the server-rendered
  // HTML; the interval below takes over once mounted. Hydration must not see
  // a clock reading the client computed for itself.
  const [elapsed, setElapsed] = useState(initialElapsedSeconds);
  const [cancelRequested, setCancelRequested] = useState(
    initialJob.cancel_requested ?? false,
  );

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
              window.location.reload();
            }
          })
          .catch(() => {
            // The next stream event or the stream closing will retry; a
            // transient failure to refresh the record does not end the page.
          });
      }
    };

    const onError = () => {
      // The server closes the stream only when the job is terminal; EventSource
      // reports that as a CLOSED state. A transient drop leaves it CONNECTING,
      // and the browser is already reconnecting on its own -- nothing to do.
      if (es.readyState === EventSource.CLOSED) {
        es.close();
        if (!disposed) window.location.reload();
      }
    };

    es.addEventListener("job", onEvent);
    es.onerror = onError;
    return () => {
      disposed = true;
      es.close();
    };
  }, [initialJob.id, streamAfter]);

  const loss = latestLoss(events);

  const cancel = async () => {
    try {
      await cancelJobV1JobsJobIdCancelPost(initialJob.id);
      setCancelRequested(true);
    } catch {
      // A double-clicked button is not an error, and a job that ends between
      // the click and the request is answered by the record the page then
      // reloads into.
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
        </dl>
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
        {/* The consequence -- no adapter will be produced -- is stated beside
            the control, before the request exists (the port of the old watch
            page's warning). The button is styled destructive, and the control
            disappears with the job: a terminal record offers nothing to
            cancel. */}
        <p>
          Cancelling destroys the machine and{" "}
          <strong>no adapter will be produced</strong>. This cannot be undone.
        </p>
        <Button
          variant="destructive"
          onClick={() => void cancel()}
          disabled={cancelRequested}
        >
          {cancelRequested ? "Cancellation requested" : "Cancel job"}
        </Button>
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
