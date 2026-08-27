import Link from "next/link";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import BackToUpload from "@/components/BackToUpload";
import FocusHeading from "@/components/FocusHeading";
import { Button } from "@/components/ui/button";
import {
  TERMINAL_STATUSES,
  epochNow,
  failureExplanation,
  formatDuration,
  formatTimestamp,
  shortRevision,
} from "@/lib/jobs/display";
import type { JobEvent, JobRecord } from "@/lib/api/generated/client";

// A job's record (#13/#14's screen, ported): the same thing during and after
// the job, so there is no separate finished-job page to drift out of
// agreement with this one. Everything renders from the published record and
// history; once the job is terminal the page carries no scripting at all.

function latestLoss(events: JobEvent[]): { loss: number; step?: number } | null {
  // Scanning backwards: the latest metric wins, whatever order older events
  // arrive in.
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (!e || e.kind !== "metric" || !e.data) continue;
    const loss = e.data["loss"];
    if (typeof loss !== "number") continue;
    const step = e.data["step"];
    return { loss, step: typeof step === "number" ? step : undefined };
  }
  return null;
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      {/* Label and value stay a real dt/dd pair: that adjacency is what a
          screen reader announces, and what the tests read. */}
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd>{children}</dd>
    </>
  );
}

function AdapterSection({ job }: { job: JobRecord }) {
  // `JobRecord` deliberately does not publish the artifact's storage address;
  // whether an artifact exists lives behind the download route (which reads
  // the job row's `artifact_key`), not in anything the page may inspect. A
  // completed job with a result is the only shape that can have produced one,
  // so that is what the offer keys off -- the route answers 409 otherwise.
  const produced = Boolean(job.result);
  return (
    <section aria-labelledby="result-heading" className="space-y-2">
      <h2 id="result-heading" className="text-lg font-semibold">
        Your adapter
      </h2>
      {produced ? (
        <div className="space-y-3">
          <p>
            <Button asChild>
              {/* The artifact travels through the download route; where it
                  is stored is the control plane's business, not the page's. */}
              <a href={`/v1/jobs/${job.id}/adapter`}>Download the adapter</a>
            </Button>
          </p>
          <p className="text-sm text-muted-foreground">
            A zip of the trained weights together with the{" "}
            <code>adapter_config.json</code> that makes them loadable.
          </p>
        </div>
      ) : (
        <p>
          Training finished, but no adapter could be retrieved. The log below
          says what happened to it.
        </p>
      )}
    </section>
  );
}

function FailedSection({ job }: { job: JobRecord }) {
  return (
    <Alert variant="destructive">
      <AlertTitle>Failed</AlertTitle>
      <AlertDescription>
        {/* The stable code survives every rendering decision: it is what a
            search, a bug report or a support question can be pinned to. */}
        <p>
          <code className="rounded bg-muted px-1">{job.error_code}</code>
        </p>
        {failureExplanation(job.error_code) && (
          <p>{failureExplanation(job.error_code)}</p>
        )}
        {job.error_message && <p>{job.error_message}</p>}
      </AlertDescription>
    </Alert>
  );
}

function CancelledSection() {
  return (
    <section aria-labelledby="cancelled-heading" className="space-y-2">
      <h2 id="cancelled-heading" className="text-lg font-semibold">
        Cancelled
      </h2>
      {/* A cancellation is the user's own decision (ADR-0003): it is stated
          as one, with no error code and no destructive framing, because a
          decision presented as a defect teaches users not to cancel. */}
      <p>
        This job was cancelled at your request. No adapter was produced —
        that is what cancelling means here, not a failure of the job.
      </p>
    </section>
  );
}

export default function JobRecordView({
  job,
  events,
  datasetFilename,
}: {
  job: JobRecord;
  events: JobEvent[];
  datasetFilename?: string;
}) {
  const terminal = TERMINAL_STATUSES.includes(job.status);
  const start = job.started_at ?? job.created_at;
  const end = job.finished_at ?? epochNow();
  const loss = latestLoss(events);

  return (
    <section aria-labelledby="job-heading" className="space-y-6">
      {!terminal && (
        // The record re-renders itself until the job ends -- the no-JavaScript
        // equivalent of the old page's poll-and-reload. Live streaming is
        // Spec 008's; this only keeps a returned visitor from reading a stale
        // state.
        <meta httpEquiv="refresh" content="2" />
      )}

      <div>
        <FocusHeading id="job-heading">Job {job.id}</FocusHeading>
        <p className="mt-2 text-muted-foreground">
          Dataset{" "}
          <Link
            href={`/datasets/${job.dataset_id}`}
            className="underline hover:no-underline"
          >
            {datasetFilename ?? job.dataset_id}
          </Link>{" "}
          · base model <code>{job.base_model}</code>
          {job.base_revision && <>@<code>{shortRevision(job.base_revision)}</code></>}
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
          <Stat label="Elapsed">{formatDuration(end - start)}</Stat>
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

      {job.status === "complete" && <AdapterSection job={job} />}
      {job.status === "failed" && <FailedSection job={job} />}
      {job.status === "cancelled" && <CancelledSection />}

      <section aria-labelledby="output-heading" className="space-y-2">
        <h2 id="output-heading" className="text-lg font-semibold">
          Output
        </h2>
        {/* The whole history, oldest first: a finished job is as inspectable
            as a working one, and the lifecycle's early states stay on the
            page even though the job ended elsewhere. */}
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

      <div className="flex gap-3">
        <Button variant="outline" asChild>
          <Link href="/jobs">All jobs</Link>
        </Button>
        <BackToUpload />
      </div>
    </section>
  );
}
