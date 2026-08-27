import Link from "next/link";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import BackToUpload from "@/components/BackToUpload";
import FocusHeading from "@/components/FocusHeading";
import QuoteView from "@/components/QuoteView";
import { Button } from "@/components/ui/button";
import {
  TERMINAL_STATUSES,
  epochNow,
  failureExplanation,
  formatDuration,
  formatMinorCost,
  formatTimestamp,
  shortRevision,
} from "@/lib/jobs/display";
import {
  directionLabel,
  formatGigabytes,
  formatRatio,
  midpoint,
  rangeDirection,
  ratio,
} from "@/lib/jobs/comparison";
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

function ComparisonRow({
  label,
  predicted,
  predictedNote,
  actual,
  actualNote,
  sentence,
}: {
  label: string;
  predicted: string;
  predictedNote: string;
  actual: string;
  actualNote: string;
  sentence: string;
}) {
  // Label/value pairs stay real dt/dd pairs, and each figure states whether
  // it was measured or derived (issue #77): a number without its basis is a
  // number a reader cannot judge.
  return (
    <div className="space-y-2 rounded-lg border bg-card p-4">
      <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
        <div>
          <dt className="text-sm text-muted-foreground">{label}</dt>
          <dd className="font-medium">{predicted}</dd>
          <dd className="text-xs text-muted-foreground">{predictedNote}</dd>
        </div>
        <div>
          <dt className="text-sm text-muted-foreground">Actual</dt>
          <dd className="font-medium">{actual}</dd>
          <dd className="text-xs text-muted-foreground">{actualNote}</dd>
        </div>
      </dl>
      <p className="text-sm text-muted-foreground">{sentence}</p>
    </div>
  );
}

function ComparisonSection({ job }: { job: JobRecord }) {
  // Shown only once a job is terminal and carries both halves of issue #77's
  // comparison: the quote frozen at launch (predicted) and the actuals frozen
  // at the end (measured). A job without a quote has nothing to compare
  // against; a job still working has no actuals yet.
  const quote = job.quote;
  const actuals = job.actuals;
  if (!quote || !actuals) return null;

  const durationMid = midpoint(quote.duration_low_s, quote.duration_high_s);
  const durationRatio = ratio(actuals.duration_s, durationMid);
  const durationDir = rangeDirection(
    actuals.duration_s,
    quote.duration_low_s,
    quote.duration_high_s,
  );
  const peakRatio = ratio(actuals.peak_memory_gb, quote.peak_memory_gb);
  const costMid = midpoint(quote.cost_low_minor, quote.cost_high_minor);
  const costRatio = ratio(actuals.cost_minor, costMid);
  const costDir = rangeDirection(
    actuals.cost_minor,
    quote.cost_low_minor,
    quote.cost_high_minor,
  );

  return (
    <section aria-labelledby="comparison-heading" className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="comparison-heading" className="text-lg font-semibold">
          Prediction vs what happened
        </h2>
        <Link
          href="/calibration"
          className="text-sm text-muted-foreground underline hover:no-underline"
        >
          Predictions vs actuals across runs
        </Link>
      </div>
      <p className="text-sm text-muted-foreground">
        What this job was predicted to take, against what it actually took.
        The predicted figures are estimates; the duration and peak memory are
        measured, and the cost is derived from the measured duration at the
        rate the job froze.
      </p>
      <div className="space-y-2">
        <ComparisonRow
          label="Duration"
          predicted={formatDuration(quote.duration_low_s)}
          predictedNote={`predicted ${formatDuration(quote.duration_high_s)} high (estimate)`}
          actual={
            actuals.duration_s != null
              ? formatDuration(actuals.duration_s)
              : "—"
          }
          actualNote="measured"
          sentence={
            durationRatio != null
              ? `Took ${formatRatio(durationRatio)} the midpoint of its predicted range — ${directionLabel(durationDir)}.`
              : "The run measured no duration to compare."
          }
        />
        <ComparisonRow
          label="Peak memory"
          predicted={formatGigabytes(quote.peak_memory_gb)}
          predictedNote="predicted (arithmetic)"
          actual={formatGigabytes(actuals.peak_memory_gb)}
          actualNote="measured on the machine"
          sentence={
            peakRatio != null
              ? `Measured peak was ${formatRatio(peakRatio)} the prediction.`
              : "No peak memory was measured on this run."
          }
        />
        <ComparisonRow
          label="Cost"
          predicted={`${formatMinorCost(quote.cost_low_minor, quote.currency, quote.minor_unit)} – ${formatMinorCost(quote.cost_high_minor, quote.currency, quote.minor_unit)}`}
          predictedNote="predicted (estimate)"
          actual={
            actuals.cost_minor != null && actuals.currency != null
              ? formatMinorCost(actuals.cost_minor, actuals.currency, 100)
              : "—"
          }
          actualNote="derived from measured duration × frozen rate"
          sentence={
            costRatio != null
              ? `The derived cost was ${formatRatio(costRatio)} the midpoint of its predicted range — ${directionLabel(costDir)}.`
              : "No cost could be derived for this run."
          }
        />
      </div>

      {(actuals.phases ?? []).length > 0 && (
        <div className="rounded-lg border bg-card p-4">
          <dl className="divide-y divide-border">
            {actuals.phases!.map((p) => (
              <div
                key={p.name}
                className="grid grid-cols-[1fr_auto] gap-x-6 gap-y-0.5 py-1"
              >
                <dt className="text-sm text-muted-foreground">{p.name}</dt>
                <dd className="text-sm font-medium text-right">
                  {p.duration_s != null ? formatDuration(p.duration_s) : "—"}
                </dd>
              </div>
            ))}
          </dl>
          <p className="mt-2 text-xs text-muted-foreground">
            What the machine actually did, stage by stage, measured from the
            job&apos;s own state transitions.
          </p>
        </div>
      )}
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

      {job.quote && (
        // The quote the job launched under, frozen into the spec at launch
        // and never updated: a finished run still says what it was predicted
        // to cost and how long it was predicted to take (issue #72).
        <QuoteView quote={job.quote} />
      )}

      {/* The measured half of the record (issue #77): what the prediction
          said against what the run did. Only present on a finished job that
          has both, so the comparison never claims numbers it does not hold. */}
      <ComparisonSection job={job} />

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
