import {
  Download,
  FolderArchive,
  HardDriveDownload,
  Layers,
  Package,
  Paperclip,
  TriangleAlert,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import CheckpointSection from "@/components/CheckpointSection";
import EventLog from "@/components/EventLog";
import InstabilityBanner from "@/components/InstabilityBanner";
import JobHeader from "@/components/JobHeader";
import JobStatsGrid from "@/components/JobStatsGrid";
import LossChart from "@/components/LossChart";
import PlateauNote from "@/components/PlateauNote";
import ProgressRegion from "@/components/ProgressRegion";
import QuoteView from "@/components/QuoteView";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import DivergenceRetry from "@/components/DivergenceRetry";
import EndpointSection from "@/components/EndpointSection";
import MemoryRecovery from "@/components/MemoryRecovery";
import {
  TERMINAL_STATUSES,
  epochNow,
  failureExplanation,
  formatDuration,
  formatMinorCost,
  formatMinorCostRange,
  formatTimestamp,
  latestLoss,
} from "@/lib/jobs/display";
import {
  directionLabel,
  formatGigabytes,
  formatRatio,
  midpoint,
  rangeDirection,
  ratio,
} from "@/lib/jobs/comparison";
import type { Direction } from "@/lib/jobs/comparison";
import { decodingBadge, decodingSentence, promptText } from "@/lib/jobs/compare";
import { formatBytes } from "@/lib/jobs/progress";
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
// history, and the page still emits no literal `<script>` or refresh `<meta>`
// once the job is terminal -- the tabs below are the one interactive surface,
// and they need client JS only to switch which section is visible, not to
// fetch or compute anything.

// Presentational labels and icons keyed by the artifact's real, published
// `kind` (packages/core/src/temper_core/artifacts.py `ARTIFACT_KIND_*`): a
// readable name for a value the record already carries, never a claim the
// record doesn't back. An unrecognised kind falls back to itself rather than
// a made-up label, the same "don't crash on a value this map hasn't learned
// about yet" rule `StatusPill` follows.
const ARTIFACT_LABELS: Record<string, string> = {
  adapter: "LoRA adapter weights",
  full_model: "Full fine-tuned model",
  merged_model: "Merged standalone model",
  quantised_local: "Quantised local model",
};

const ARTIFACT_ICONS: Record<string, React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>> = {
  adapter: FolderArchive,
  full_model: Layers,
  merged_model: Layers,
  quantised_local: HardDriveDownload,
};

function ArtifactCard({
  title,
  kind,
  members,
  bytes,
  description,
  href,
  downloadLabel,
  primary,
}: {
  title: string;
  kind: string;
  members: string[];
  bytes?: number | null;
  description: string;
  href: string;
  downloadLabel: string;
  primary: boolean;
}) {
  const Icon = ARTIFACT_ICONS[kind] ?? Package;
  return (
    <div className="flex flex-col justify-between gap-4 rounded-[12px] border bg-card p-5">
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <Icon
              aria-hidden
              className={
                primary ? "size-5 shrink-0 text-primary" : "size-5 shrink-0 text-muted-foreground"
              }
            />
            <h3 className="font-semibold">{title}</h3>
          </div>
          <span className="shrink-0 rounded border bg-muted px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
            {kind}
          </span>
        </div>
        <p className="text-sm leading-relaxed text-muted-foreground">
          {description}
        </p>
        {members.length > 0 && (
          <div className="flex items-start gap-2 rounded-[8px] border bg-background/60 px-3 py-2 text-xs text-muted-foreground">
            <Paperclip aria-hidden className="mt-0.5 size-3.5 shrink-0" />
            <span>Contains: {members.join(", ")}.</span>
          </div>
        )}
      </div>
      <div className="flex items-center justify-between gap-3 border-t pt-3">
        <span className="text-xs text-muted-foreground">
          {bytes != null ? `Size ~${formatBytes(bytes)}` : ""}
        </span>
        <Button variant={primary ? "default" : "outline"} size="sm" asChild>
          <a href={href}>
            <Download aria-hidden className="size-3.5" />
            {downloadLabel}
          </a>
        </Button>
      </div>
    </div>
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
  const targetCount = (artifact ? 1 : 0) + deliveryFormats.length;
  return (
    <section aria-labelledby="result-heading" className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2
          id="result-heading"
          className="text-xs font-medium tracking-widest uppercase text-foreground"
        >
          Model artifacts &amp; checkpoints
        </h2>
        {targetCount > 0 && (
          <span className="text-xs text-muted-foreground">
            {targetCount} downloadable target{targetCount === 1 ? "" : "s"} ready
          </span>
        )}
      </div>
      {artifact ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <ArtifactCard
            title={ARTIFACT_LABELS[artifact.kind] ?? artifact.kind}
            kind={artifact.kind}
            members={artifact.members}
            bytes={artifact.bytes}
            description={artifact.loading}
            href={`/v1/jobs/${job.id}/artifact`}
            downloadLabel="Download the artifact"
            primary
          />
          {deliveryFormats.map((d) => (
            <ArtifactCard
              key={d.format}
              title={d.format}
              kind={d.kind}
              members={d.members}
              bytes={d.bytes}
              description={d.what_for}
              href={`/v1/jobs/${job.id}/artifact?format=${d.format}`}
              downloadLabel={`Download ${d.format}`}
              primary={false}
            />
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          Training finished, but no artifact could be retrieved. The Output
          tab says what happened to it.
        </p>
      )}
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

function CancelledSection() {
  return (
    <section aria-labelledby="cancelled-heading" className="space-y-2">
      <h2 id="cancelled-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
        Cancelled
      </h2>
      {/* A cancellation is the user's own decision (ADR-0003): it is stated
          as one, with no error code and no destructive framing, because a
          decision presented as a defect teaches users not to cancel. */}
      <p>
        This job was cancelled at your request. No artifact was produced,
        which is what cancelling means here rather than a failure of the job.
      </p>
    </section>
  );
}

// "under" the predicted range is the favourable direction for a duration or
// a cost -- cheaper and faster than the estimate -- so it earns the same
// success/destructive coloring a delta chip uses elsewhere on this page.
// Peak memory has no range to land inside or outside of (the quote predicts
// one arithmetic figure, not a range), so it never takes this color and
// stays a plain, neutral reading.
function directionColor(d: Direction | null): string {
  if (d === "under") return "text-success";
  if (d === "over") return "text-destructive";
  return "";
}

function ComparisonRow({
  label,
  predicted,
  predictedNote,
  actual,
  actualNote,
  actualColor,
  sentence,
}: {
  label: string;
  predicted: string;
  predictedNote: string;
  actual: string;
  actualNote: string;
  actualColor?: string;
  sentence: string;
}) {
  // Label/value pairs stay real dt/dd pairs (issue #77): the adjacency is
  // what a screen reader announces. Each figure states its own basis -- the
  // prediction its estimate or arithmetic note, the actual whether it was
  // measured or derived -- because a number without its basis is one a
  // reader cannot judge. The Actual figure leads through weight and color
  // (bold, green when it came in under): what happened is what the reader
  // came for.
  return (
    <div className="rounded-[12px] border bg-card p-6">
      <h3 className="text-sm text-muted-foreground">{label}</h3>
      <dl className="mt-3 grid grid-cols-2 gap-4">
        <div>
          <dt className="font-mono text-xs text-muted-foreground">Predicted</dt>
          <dd className="mt-1 font-mono text-xl tabular-nums">{predicted}</dd>
          <dd className="mt-1 text-xs text-muted-foreground">{predictedNote}</dd>
        </div>
        <div className="text-right">
          <dt className="font-mono text-xs text-muted-foreground">Actual</dt>
          <dd className={`mt-1 font-mono text-xl font-bold tabular-nums ${actualColor ?? ""}`}>
            {actual}
          </dd>
          <dd className="mt-1 text-xs text-muted-foreground">{actualNote}</dd>
        </div>
      </dl>
      <p className="mt-4 border-t pt-3 font-mono text-xs text-muted-foreground">{sentence}</p>
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
      <h2 id="comparison-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
        Prediction vs what happened
      </h2>
      <p className="text-sm text-muted-foreground">
        What this job was predicted to take, against what it actually took.
        The predicted figures are estimates; duration and peak memory were
        measured directly, and cost is derived from the measured duration at
        the rate the job froze.
      </p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
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
          actualColor={directionColor(durationDir)}
          sentence={
            durationRatio != null
              ? `Took ${formatRatio(durationRatio)} the midpoint of its predicted range, ${directionLabel(durationDir)}.`
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
          predicted={formatMinorCostRange(quote.cost_low_minor, quote.cost_high_minor, quote.currency, quote.minor_unit)}
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
          actualColor={directionColor(costDir)}
          sentence={
            costRatio != null
              ? `The derived cost was ${formatRatio(costRatio)} the midpoint of its predicted range, ${directionLabel(costDir)}.`
              : "No cost could be derived for this run."
          }
        />
      </div>

      {(actuals.stages ?? []).length > 0 && (
        <div className="overflow-hidden rounded-[12px] border bg-card">
          <div className="border-b bg-muted/20 px-6 py-2.5">
            <h3 className="font-mono text-xs font-medium tracking-widest uppercase text-muted-foreground">
              Measured stage durations
            </h3>
          </div>
          <dl className="divide-y divide-border px-6">
            {actuals.stages!.map((p) => (
              <div
                key={p.name}
                className="grid grid-cols-[1fr_auto] items-baseline gap-x-6 gap-y-1 py-3 font-mono text-sm"
              >
                <dt className="text-foreground">{p.name}</dt>
                <dd className="text-right text-muted-foreground">
                  {p.duration_s != null ? formatDuration(p.duration_s) : "—"}
                </dd>
              </div>
            ))}
          </dl>
          <p className="border-t bg-muted/10 px-6 py-3 font-mono text-xs text-muted-foreground">
            What the machine actually did, stage by stage.
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
  const badge = decodingBadge(comparison.decoding);
  const selection = comparison.selection;
  const tunedLabel =
    selection?.step != null
      ? `Tuned model (chosen checkpoint, step ${selection.step})`
      : "Tuned model";
  // The chosen checkpoint's own held-out loss, shown beside the tuned column
  // only when the comparison's selection names the same checkpoint the run's
  // `best_checkpoint` recorded -- cross-checked against the record, not
  // assumed.
  const tunedLoss =
    job.best_checkpoint?.step != null && job.best_checkpoint.step === selection?.step
      ? job.best_checkpoint.held_out_loss
      : null;

  if (rows.length === 0) {
    return (
      <section aria-labelledby="side-by-side-heading" className="space-y-2">
        <h2 id="side-by-side-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
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
        <h2 id="side-by-side-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
          Base model vs your tuned model
        </h2>
        <p className="text-sm text-muted-foreground">
          The same held-out prompts, answered by the base model and by the
          model this run produced. This is evidence you can read, not a
          benchmark.
        </p>
      </div>
      <div className="space-y-3">
        {rows.map((row, i) => {
          const question = promptText(row.prompt);
          return (
            <div key={i} className="overflow-hidden rounded-[12px] border bg-card">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-muted/30 px-4 py-2">
                {/* A plain scan aid, not the question -- kept visually
                    minor (muted, no weight, a symbol rather than a word) so
                    it can never read as a second copy of the prompt text
                    directly below it, even as a placeholder fallback. */}
                <span className="font-mono text-xs text-muted-foreground">
                  #{i + 1}
                </span>
                {badge && (
                  <span className="font-mono text-xs text-muted-foreground">{badge}</span>
                )}
              </div>
              <div className="space-y-3 p-4">
                <p className="font-medium">
                  {question || `Prompt ${i + 1}`}
                </p>
                <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="rounded-[8px] border bg-background/40 p-3">
                    <dt className="mb-1.5 flex items-center gap-1.5 text-sm text-muted-foreground">
                      <span aria-hidden className="size-1.5 rounded-full bg-muted-foreground" />
                      Base model
                    </dt>
                    <dd className="text-sm whitespace-pre-wrap italic">
                      &ldquo;{row.base}&rdquo;
                    </dd>
                  </div>
                  <div className="rounded-[8px] border border-primary/30 bg-primary/5 p-3">
                    <div className="mb-1.5 flex items-center justify-between gap-2">
                      <dt className="flex items-center gap-1.5 text-sm text-primary">
                        <span aria-hidden className="size-1.5 rounded-full bg-primary" />
                        {tunedLabel}
                      </dt>
                      {tunedLoss != null && (
                        <span className="font-mono text-xs text-success">
                          loss {tunedLoss}
                        </span>
                      )}
                    </div>
                    <dd className="text-sm whitespace-pre-wrap italic">
                      &ldquo;{row.tuned}&rdquo;
                    </dd>
                  </div>
                </dl>
              </div>
            </div>
          );
        })}
      </div>
      {(decoding || selection?.reason) && (
        <p className="text-sm text-muted-foreground">
          {decoding}
          {decoding && selection?.reason ? " " : ""}
          {selection?.reason}
        </p>
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
        <h2 id="capability-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
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
        <h2 id="capability-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
          General capability
        </h2>
        <p className="text-sm text-muted-foreground">
          A smoke test for catastrophic forgetting, not a benchmark. The same
          fixed set of general questions, answered by the base model and by
          the model this run produced.
        </p>
      </div>

      {capability.large_regression === true && (
        <Alert variant="destructive" className="border-destructive/30 bg-destructive/10">
          <TriangleAlert aria-hidden />
          <AlertTitle className="text-xs font-semibold tracking-widest uppercase">
            Large regression
          </AlertTitle>
          <AlertDescription>
            The tuned model answered at least{" "}
            {capability.regression_threshold ?? 0} fewer general-knowledge
            questions correctly than the base model. This can be a sign that
            fine-tuning degraded general capability.
          </AlertDescription>
        </Alert>
      )}

      <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-[12px] border bg-card p-4">
          <dt className="text-sm text-muted-foreground">Base model</dt>
          <dd className="mt-1 font-mono text-lg font-semibold tabular-nums">
            {scoreText(capability.base_correct, capability.total)}
          </dd>
        </div>
        <div className="rounded-[12px] border bg-card p-4">
          <dt className="text-sm text-muted-foreground">{tunedLabel}</dt>
          <dd
            className={`mt-1 font-mono text-lg font-semibold tabular-nums ${
              capability.large_regression ? "text-destructive" : ""
            }`}
          >
            {scoreText(capability.tuned_correct, capability.total)}
          </dd>
        </div>
        <div className="rounded-[12px] border bg-card p-4">
          <dt className="text-sm text-muted-foreground">Change</dt>
          <dd
            className={`mt-1 font-mono text-lg font-semibold tabular-nums ${
              capability.large_regression ? "text-destructive" : ""
            }`}
          >
            {changeText(capability)}
          </dd>
        </div>
        <div className="rounded-[12px] border bg-card p-4">
          <dt className="text-sm text-muted-foreground">Sample</dt>
          <dd className="mt-1 font-mono text-sm text-muted-foreground">
            {weightSentence(capability.total)}
          </dd>
        </div>
      </dl>

      <p className="text-sm text-muted-foreground">
        {uncertaintySentence(capability)} A small sample honestly labelled is
        a smoke test, not a benchmark.
        {decoding ? ` ${decoding}` : ""}
        {selection?.reason ? ` ${selection.reason}` : ""}
      </p>
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

      <JobHeader
        job={job}
        datasetFilename={datasetFilename}
        meta={formatTimestamp(end)}
      />

      <JobStatsGrid job={job} elapsedSeconds={end - start} loss={loss} />

      {/* The record used to run as one long page; a finished job's history
          made that a scroll through everything at once regardless of what a
          reader came for. Tabs group it by the question a reader actually
          has -- what did it produce, how good is it, what did it cost, what
          did it say -- so each stays a readable length. This is the one
          place on this page that now depends on client JS to switch between
          sections (Radix mounts only the active tab's content); everything
          each tab holds is still rendered from the same server-fetched
          record, and reaching a specific section is one tab click away. */}
      <Tabs defaultValue="overview">
        <TabsList aria-label="Job record sections">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="evaluation">Evaluation &amp; loss</TabsTrigger>
          <TabsTrigger value="estimates">Cost &amp; estimates</TabsTrigger>
          <TabsTrigger value="output">Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          {/* Progress is part of the durable record too (issue #49): what
              image pull and model download reached, measured and estimated,
              and the raw lines that were promoted -- kept whole as collapsed
              detail. The finished page therefore shows how the run got
              there, not just that it did. */}
          <ProgressRegion progress={progress} output={output} />

          {job.status === "complete" && <ArtifactSection job={job} />}
          {job.status === "failed" && (
            <FailedSection job={job} events={events} />
          )}
          {/* A memory recovery is an automatic retry with the effective batch
              preserved (issue #35): shown as its own banner whenever the
              job's executions are plural and the last one completed, so the
              user is told a recovery happened and what changed. */}
          <MemoryRecovery job={job} />
          {/* Instability is a warning rather than an abort (issue #36): the
              same exceedance that would become a divergence after 20 steps is
              surfaced at 5 steps as a banner that does not stop the run. */}
          <InstabilityBanner events={events} />
          {job.status === "cancelled" && <CancelledSection />}

          {/* The recorded result checkpoint and every other retained one
              (issue #62): shown whenever the run recorded checkpoints, on
              whichever outcome -- a failed run's checkpoints are still its
              history. */}
          <CheckpointSection job={job} />

          {/* Temporary authenticated endpoint (issue #78): try the tuned
              model without downloading anything. The endpoint requires a key
              (stored hashed), carries its own expiry from the moment it
              starts, extends on use, and stops itself via a timer -- the
              forgotten warm machine is the loudest complaint against the
              commercial baseline, so stopping itself is the feature. */}
          <EndpointSection jobId={job.id} jobStatus={job.status} />
        </TabsContent>

        <TabsContent value="evaluation">
          {/* The loss curve leads this tab (issue #53): it is the one number
              every run gets read against, so it comes before the qualitative
              comparisons rather than after them. */}
          <section aria-labelledby="loss-chart-heading" className="space-y-2">
            <h2 id="loss-chart-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
              Loss
            </h2>
            <p className="text-sm text-muted-foreground">
              Training loss, measured every step; held-out loss, measured
              periodically against data the model never trained on.
            </p>
            <LossChart training={training} heldOut={heldOut} best={job.best_checkpoint} />
            {plateau && <PlateauNote message={plateau.message} />}
          </section>

          {/* The side-by-side comparison (issue #69): held-out prompts
              answered by the base model and the chosen checkpoint, with the
              decoding settings recorded. Shown whenever the run recorded one;
              a failed comparison is stated with its reason, never as a
              failed job. */}
          <SideBySideSection job={job} />

          {/* The general-capability slice (issue #73): a smoke test for
              catastrophic forgetting, not a benchmark -- the same fixed
              general questions answered by both models, reported as a change
              with the sample size and uncertainty stated, and a large
              regression surfaced prominently from the recorded flag. */}
          <CapabilitySection job={job} />
        </TabsContent>

        <TabsContent value="estimates">
          {/* The advanced-surface overrides the user froze into the job spec
              (issue #80): what the run actually trained with, shown after it
              is over. The record carries the user's changes -- the
              resolver's typed answer is what reached the trainer -- so this
              is exactly the overrides, nothing more. */}
          {job.hyperparameters && Object.keys(job.hyperparameters).length > 0 && (
            <section aria-labelledby="settings-changed-heading" className="space-y-2">
              <h2 id="settings-changed-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
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
            // The quote the job launched under, frozen into the spec at
            // launch and never updated: a finished run still says what it
            // was predicted to cost and how long it was predicted to take
            // (issue #72).
            <QuoteView quote={job.quote} />
          )}

          {/* The measured half of the record (issue #77): what the
              prediction said against what the run did. Only present on a
              finished job that has both, so the comparison never claims
              numbers it does not hold. */}
          <ComparisonSection job={job} />
        </TabsContent>

        <TabsContent value="output">
          <EventLog
            jobId={job.id}
            initialEvents={events}
            initialTotal={total ?? events.length}
          />
        </TabsContent>
      </Tabs>
    </section>
  );
}
