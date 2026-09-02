import Link from "next/link";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import BackToUpload from "@/components/BackToUpload";
import EventLog from "@/components/EventLog";
import FocusHeading from "@/components/FocusHeading";
import LossChart from "@/components/LossChart";
import PlateauNote from "@/components/PlateauNote";
import ProgressRegion from "@/components/ProgressRegion";
import QuoteView from "@/components/QuoteView";
import { Button } from "@/components/ui/button";
import DivergenceRetry from "@/components/DivergenceRetry";
import EndpointSection from "@/components/EndpointSection";
import MemoryRecovery from "@/components/MemoryRecovery";
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
import { decodingSentence, promptText } from "@/lib/jobs/compare";
import {
  changeText,
  scoreText,
  tunedModelLabel,
  uncertaintySentence,
  weightSentence,
} from "@/lib/jobs/capability";
import { lossSeries } from "@/lib/jobs/loss";
import { heldOutPlateau } from "@/lib/jobs/plateau";
import type {
  JobEvent,
  JobOutputLine,
  JobProgress,
  JobRecord,
} from "@/lib/api/generated/client";

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

function ArtifactSection({ job }: { job: JobRecord }) {
  // `JobRecord` deliberately does not publish the artifact's storage address;
  // whether an artifact exists lives behind the download route (which reads
  // the job row's `artifact_key`), not in anything the page may inspect. The
  // published `artifact` record -- its declared kind and what it contains --
  // is what the page may show, so the offer keys off that: the route answers
  // 409 otherwise.
  //
  // Issue #74: the job can also produce delivery formats beyond the canonical
  // artifact (a merged single-file model, a quantised local-inference
  // format). Each is served at the same artifact route with a `format` query
  // parameter, and the published `delivery_formats` record names each with
  // its plain-language purpose, so a user chooses a format by what it is for.
  const artifact = job.artifact;
  const deliveryFormats = job.delivery_formats ?? [];
  const hasDelivery = deliveryFormats.length > 0;
  return (
    <section aria-labelledby="result-heading" className="space-y-2">
      <h2 id="result-heading" className="text-lg font-semibold">
        Your artifact
      </h2>
      {artifact ? (
        <div className="space-y-3">
          <div className="space-y-3 rounded-lg border bg-card p-4">
            <p>
              <Button asChild>
                {/* The artifact travels through the download route; where it
                    is stored is the control plane's business, not the page's. */}
                <a href={`/v1/jobs/${job.id}/artifact`}>
                  Download the artifact
                </a>
              </Button>
            </p>
            {artifact.members.length > 0 && (
              <p className="text-sm text-muted-foreground">
                Contains: {artifact.members.join(", ")}.
              </p>
            )}
            {/* The load path differs by kind -- an adapter is applied to a base
                model, a fully trained model is loaded on its own -- so the
                interface says how to load what it offers (issue #32). */}
            {artifact.loading && (
              <p className="text-sm text-muted-foreground">
                {artifact.loading}
              </p>
            )}
          </div>

          {hasDelivery && (
            <ul className="divide-y divide-border rounded-lg border bg-card">
              {deliveryFormats.map((d) => (
                <li
                  key={d.format}
                  className="flex flex-wrap items-center justify-between gap-3 px-4 py-3"
                >
                  <div className="min-w-0 flex-1 space-y-1">
                    <p className="font-medium">{d.format}</p>
                    {/* What the format is for, in plain language (issue #74):
                        a user chooses by outcome, not by internals. */}
                    <p className="text-sm text-muted-foreground">
                      {d.what_for}
                    </p>
                    {d.members.length > 0 && (
                      <p className="text-xs text-muted-foreground">
                        Contains: {d.members.join(", ")}.
                      </p>
                    )}
                  </div>
                  <Button variant="outline" size="sm" asChild>
                    <a href={`/v1/jobs/${job.id}/artifact?format=${d.format}`}>
                      Download {d.format}
                    </a>
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <p>
          Training finished, but no artifact could be retrieved. The log below
          says what happened to it.
        </p>
      )}
    </section>
  );
}

function CheckpointSection({ job }: { job: JobRecord }) {
  // The result checkpoint is chosen by held-out loss and recorded on the run
  // (issue #62), so the finished record states which one was chosen and why,
  // and offers every retained checkpoint for download -- the user is never
  // locked out of their own run's history. Only retained checkpoints are
  // downloadable; a superseded or failed one is named but not offered.
  const checkpoints = job.checkpoints ?? [];
  if (checkpoints.length === 0) return null;
  const best = job.best_checkpoint;
  return (
    <section aria-labelledby="checkpoints-heading" className="space-y-3">
      <div>
        <h2 id="checkpoints-heading" className="text-lg font-semibold">
          Checkpoints
        </h2>
        <p className="text-sm text-muted-foreground">
          The best checkpoint is chosen by held-out loss and the choice is
          recorded on this run. Every other checkpoint stays downloadable.
        </p>
      </div>
      {best && best.step != null && (
        <div className="space-y-1 rounded-lg border bg-card p-4">
          <p>
            <strong>
              Best checkpoint: step {best.step}
              {best.held_out_loss != null && (
                <> (held-out loss {best.held_out_loss})</>
              )}
            </strong>
          </p>
          <p className="text-sm text-muted-foreground">{best.reason}</p>
        </div>
      )}
      <ul className="divide-y divide-border rounded-lg border bg-card">
        {checkpoints.map((c) => (
          <li
            key={c.step}
            className="flex flex-wrap items-center justify-between gap-3 px-4 py-2"
          >
            <div className="flex items-center gap-2">
              <span className="font-medium">Step {c.step}</span>
              {c.selected && (
                <span className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                  chosen result
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <span>
                {c.held_out_loss != null
                  ? `held-out loss ${c.held_out_loss}`
                  : "no held-out loss recorded"}
              </span>
              {c.verified ? (
                <Button variant="outline" size="sm" asChild>
                  <a href={`/v1/jobs/${job.id}/checkpoints/${c.step}`}>
                    Download
                  </a>
                </Button>
              ) : (
                <span className="text-xs">not retained</span>
              )}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function FailedSection({ job, events }: { job: JobRecord; events: JobEvent[] }) {
  return (
    <Alert variant="destructive">
      <AlertTitle>Failed</AlertTitle>
      <AlertDescription className="space-y-3">
        {/* The stable code survives every rendering decision: it is what a
            search, a bug report or a support question can be pinned to. */}
        <p>
          <code className="rounded bg-muted px-1">{job.error_code}</code>
        </p>
        {failureExplanation(job.error_code) && <p>{failureExplanation(job.error_code)}</p>}
        {job.error_message && <p>{job.error_message}</p>}
        <DivergenceRetry job={job} />
      </AlertDescription>
    </Alert>
  );
}

function InstabilityBanner({ events }: { events: JobEvent[] }) {
  const warnings = events.filter(
    (e) => e.data && (e.data["code"] === "training_instability" || e.data["warning"] === true),
  );
  if (warnings.length === 0) return null;
  return (
    <Alert>
      <AlertTitle>Training instability</AlertTitle>
      <AlertDescription>
        <p>
          Loss is spiking well above its recent average. This is shown as a
          warning rather than an abort — it may be early divergence. Consider
          lowering the learning rate if it continues.
        </p>
        <p className="text-xs text-muted-foreground">
          {warnings[warnings.length - 1]?.message}
        </p>
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
        This job was cancelled at your request. No artifact was produced —
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
      <h2 id="comparison-heading" className="text-lg font-semibold">
        Prediction vs what happened
      </h2>
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
              ? // The actuals' cost is derived in the same currency and minor
                // unit the quote priced (the frozen rate's), so the quote's
                // published minor unit is the unit to show, never a literal.
                formatMinorCost(actuals.cost_minor, actuals.currency, quote.minor_unit)
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

      {(actuals.stages ?? []).length > 0 && (
        <div className="rounded-lg border bg-card p-4">
          <dl className="divide-y divide-border">
            {actuals.stages!.map((p) => (
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

function SideBySideSection({ job }: { job: JobRecord }) {
  // The side-by-side comparison (issue #69): the same held-out prompts
  // answered by the base model and by the checkpoint the run chose, recorded
  // on the machine. Rendered from the published record exactly as recorded --
  // both sides, the decoding settings (so a reader can tell whether two
  // outputs are comparable), and the checkpoint the tuned side compared.
  // A comparison that failed is stated with its reason; it never failed the
  // run, and the page must not dress that as a failure of the job.
  const comparison = job.comparison;
  if (!comparison) return null;
  const rows = comparison.rows ?? [];
  const decoding = decodingSentence(comparison.decoding);
  const selection = comparison.selection;
  const tunedLabel =
    selection?.step != null
      ? `Tuned model (chosen checkpoint, step ${selection.step})`
      : "Tuned model";

  if (rows.length === 0) {
    return (
      <section aria-labelledby="side-by-side-heading" className="space-y-2">
        <h2 id="side-by-side-heading" className="text-lg font-semibold">
          Base model vs your tuned model
        </h2>
        <p className="text-sm text-muted-foreground">
          No side-by-side comparison was produced for this run.
          {comparison.reason && <span> {comparison.reason}</span>}
        </p>
      </section>
    );
  }

  return (
    <section aria-labelledby="side-by-side-heading" className="space-y-3">
      <div>
        <h2 id="side-by-side-heading" className="text-lg font-semibold">
          Base model vs your tuned model
        </h2>
        <p className="text-sm text-muted-foreground">
          The same held-out prompts, answered by the base model and by the
          model this run produced. This is evidence you can read, not a
          benchmark.
        </p>
      </div>
      {rows.map((row, i) => {
        const question = promptText(row.prompt);
        return (
          <div
            key={i}
            className="space-y-2 rounded-lg border bg-card p-4"
          >
            <p className="font-medium">
              {question || `Prompt ${i + 1}`}
            </p>
            <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
              <div>
                <dt className="text-sm text-muted-foreground">Base model</dt>
                <dd className="whitespace-pre-wrap">{row.base}</dd>
              </div>
              <div>
                <dt className="text-sm text-muted-foreground">{tunedLabel}</dt>
                <dd className="whitespace-pre-wrap">{row.tuned}</dd>
              </div>
            </dl>
          </div>
        );
      })}
      {decoding && (
        <p className="text-sm text-muted-foreground">{decoding}</p>
      )}
      {selection?.reason && (
        <p className="text-sm text-muted-foreground">{selection.reason}</p>
      )}
    </section>
  );
}

function CapabilitySection({ job }: { job: JobRecord }) {
  // The general-capability slice (issue #73): a smoke test for catastrophic
  // forgetting, not a benchmark. A fixed, versioned slice of general
  // questions answered by the base model and by the tuned model, reported as
  // a change with the sample size beside the number and the uncertainty
  // stated. Everything renders from the published record -- the threshold
  // behind "large regression" is recorded by the machine and read here, never
  // re-decided (ADR-0010). A slice that failed is stated with its reason; it
  // never failed the run, and the page must not dress that as a job failure.
  const capability = job.capability;
  if (!capability) return null;
  const rows = capability.rows ?? [];
  const decoding = decodingSentence(capability.decoding);
  const selection = capability.selection;
  const tunedLabel = tunedModelLabel(selection);

  if (rows.length === 0) {
    return (
      <section aria-labelledby="capability-heading" className="space-y-2">
        <h2 id="capability-heading" className="text-lg font-semibold">
          General capability
        </h2>
        <p className="text-sm text-muted-foreground">
          No general-capability check was produced for this run.
          {capability.reason && <span> {capability.reason}</span>}
        </p>
      </section>
    );
  }

  return (
    <section aria-labelledby="capability-heading" className="space-y-3">
      <div>
        <h2 id="capability-heading" className="text-lg font-semibold">
          General capability
        </h2>
        <p className="text-sm text-muted-foreground">
          A smoke test for catastrophic forgetting, not a benchmark. The same
          fixed set of general questions, answered by the base model and by
          the model this run produced.
        </p>
      </div>

      {capability.large_regression === true && (
        <Alert variant="destructive">
          <AlertTitle>Large regression</AlertTitle>
          <AlertDescription>
            The tuned model answered at least{" "}
            {capability.regression_threshold ?? 0} fewer general-knowledge
            questions correctly than the base model. This can be a sign that
            fine-tuning degraded general capability.
          </AlertDescription>
        </Alert>
      )}

      <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
        <div>
          <dt className="text-sm text-muted-foreground">Base model</dt>
          <dd className="font-medium">
            {scoreText(capability.base_correct, capability.total)}
          </dd>
        </div>
        <div>
          <dt className="text-sm text-muted-foreground">{tunedLabel}</dt>
          <dd className="font-medium">
            {scoreText(capability.tuned_correct, capability.total)}
          </dd>
        </div>
        <div>
          <dt className="text-sm text-muted-foreground">Change</dt>
          <dd className="font-medium">{changeText(capability)}</dd>
        </div>
        <div>
          <dt className="text-sm text-muted-foreground">Sample</dt>
          <dd className="text-sm">{weightSentence(capability.total)}</dd>
        </div>
      </dl>

      <p className="text-sm text-muted-foreground">
        {uncertaintySentence(capability)} A small sample honestly labelled is
        a smoke test, not a benchmark.
      </p>
      {decoding && (
        <p className="text-sm text-muted-foreground">{decoding}</p>
      )}
      {selection?.reason && (
        <p className="text-sm text-muted-foreground">{selection.reason}</p>
      )}
    </section>
  );
}

export default function JobRecordView({
  job,
  events,
  progress = [],
  output = [],
  datasetFilename,
  total,
}: {
  job: JobRecord;
  events: JobEvent[];
  progress?: JobProgress[];
  output?: JobOutputLine[];
  datasetFilename?: string;
  total?: number;
}) {
  const terminal = TERMINAL_STATUSES.includes(job.status);
  const start = job.started_at ?? job.created_at;
  const end = job.finished_at ?? epochNow();
  const loss = latestLoss(events);

  // The two series the loss chart draws (issue #53), from the same history
  // this record renders. The chart lives here as well as on the running view
  // because the overfitting signal is read after the run, not only while it
  // goes -- and a held-out loss that stopped improving is said in plain
  // language, not left as a number.
  const { training, heldOut } = lossSeries(events);
  const plateau = heldOutPlateau(heldOut.map((p) => p.heldOutLoss!));

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
        {job.is_moe && (
          <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3">
            <p className="text-sm font-medium text-amber-900">
              Mixture-of-experts — untested here
            </p>
            <p className="mt-1 text-sm text-amber-800">
              This model is a mixture-of-experts architecture, which is untested
              here: expert routing changes LoRA target-module selection, memory
              scales with total rather than active parameters, and routing
              interacts poorly with small-batch adapters. It is usable and
              labelled untested — curation is a default, not a boundary.
            </p>
          </div>
        )}
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

      {/* Progress is part of the durable record too (issue #49): what image
          pull and model download reached, measured and estimated, and the raw
          lines that were promoted -- kept whole as collapsed detail. The
          finished page therefore shows how the run got there, not just that
          it did. */}
      <ProgressRegion progress={progress} output={output} />

      {/* The advanced-surface overrides the user froze into the job spec
          (issue #80): what the run actually trained with, shown after it is
          over. The record carries the user's changes -- the resolver's typed
          answer is what reached the trainer -- so this is exactly the
          overrides, nothing more. */}
      {job.hyperparameters && Object.keys(job.hyperparameters).length > 0 && (
        <section aria-labelledby="settings-changed-heading" className="space-y-2">
          <h2 id="settings-changed-heading" className="text-lg font-semibold">
            Settings you changed
          </h2>
          <p className="text-sm text-muted-foreground">
            These were frozen into this job&apos;s specification at launch and
            cannot be changed afterwards.
          </p>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 rounded-lg border bg-card p-4 sm:grid-cols-3">
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

      {job.status === "complete" && <ArtifactSection job={job} />}
      {job.status === "failed" && <FailedSection job={job} events={events} />}
      {/* A memory recovery is an automatic retry with the effective batch
          preserved (issue #35): shown as its own banner whenever the job's
          executions are plural and the last one completed, so the user is
          told a recovery happened and what changed. */}
      <MemoryRecovery job={job} />
      {/* Instability is a warning rather than an abort (issue #36): the
          same exceedance that would become a divergence after 20 steps is
          surfaced at 5 steps as a banner that does not stop the run. */}
      <InstabilityBanner events={events} />
      {job.status === "cancelled" && <CancelledSection />}

      {/* The recorded result checkpoint and every other retained one (issue
          #62): shown whenever the run recorded checkpoints, on whichever
          outcome -- a failed run's checkpoints are still its history. */}
      <CheckpointSection job={job} />

      {/* The side-by-side comparison (issue #69): held-out prompts answered
          by the base model and the chosen checkpoint, with the decoding
          settings recorded. Shown whenever the run recorded one; a failed
          comparison is stated with its reason, never as a failed job. */}
      <SideBySideSection job={job} />

      {/* The general-capability slice (issue #73): a smoke test for
          catastrophic forgetting, not a benchmark -- the same fixed general
          questions answered by both models, reported as a change with the
          sample size and uncertainty stated, and a large regression surfaced
          prominently from the recorded flag. */}
      <CapabilitySection job={job} />

      {/* Temporary authenticated endpoint (issue #78): try the tuned model
          without downloading anything. The endpoint requires a key (stored
          hashed), carries its own expiry from the moment it starts, extends
          on use, and stops itself via a timer -- the forgotten warm machine
          is the loudest complaint against the commercial baseline, so
          stopping itself is the feature. */}
      <EndpointSection jobId={job.id} jobStatus={job.status} />

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

      <section aria-labelledby="loss-chart-heading" className="space-y-2">
        <h2 id="loss-chart-heading" className="text-lg font-semibold">
          Loss
        </h2>
        <LossChart training={training} heldOut={heldOut} />
        {plateau && <PlateauNote message={plateau.message} />}
      </section>

      <EventLog
        jobId={job.id}
        initialEvents={events}
        initialTotal={total ?? events.length}
      />

      <div className="flex gap-3">
        <Button variant="outline" asChild>
          <Link href="/jobs">All jobs</Link>
        </Button>
        <BackToUpload />
      </div>
    </section>
  );
}
