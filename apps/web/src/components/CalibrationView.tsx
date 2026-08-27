import Link from "next/link";
import { formatDuration, formatTimestamp } from "@/lib/jobs/display";
import { formatGigabytes, formatRatio } from "@/lib/jobs/comparison";
import type {
  Calibration,
  CalibrationMetric,
  CalibrationPhase,
  CalibrationRun,
} from "@/lib/api/generated/client";

// The aggregate view (issue #77): predictions against measurements across
// runs, so a systematically wrong estimate is visible rather than absorbed
// into a better-looking average. Each roll reports its own count, so
// "calibrated against N real runs" is only as honest as N is visible.
//
// The ratio is actual over predicted midpoint: 1.0 means the prediction
// matched on average, above 1 the actuals ran over it, below 1 under it.
// Cost is compared as a ratio only -- the absolute amounts are denominated in
// each run's own currency, and a number without its unit is a number a reader
// cannot judge (the repo rule: currency travels with the amount, never as a
// formatting choice).

function metricSentence(m: CalibrationMetric): string {
  if (m.mean_ratio == null) {
    return "No comparable runs yet.";
  }
  if (m.mean_ratio > 1.05) {
    return `On average the actual ran ${formatRatio(m.mean_ratio)} the prediction — the estimate under-predicted.`;
  }
  if (m.mean_ratio < 0.95) {
    return `On average the actual ran ${formatRatio(m.mean_ratio)} the prediction — the estimate over-predicted.`;
  }
  return "On average the actual matched the prediction.";
}

function MetricCard({
  title,
  metric,
  predicted,
  actual,
}: {
  title: string;
  metric: CalibrationMetric;
  predicted: string | null;
  actual: string | null;
}) {
  return (
    <div className="rounded-lg border bg-card p-4">
      <h3 className="font-medium">{title}</h3>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <dt className="text-muted-foreground">Runs compared</dt>
        <dd>{metric.count}</dd>
        {predicted != null && (
          <>
            <dt className="text-muted-foreground">Mean predicted</dt>
            <dd>{predicted}</dd>
          </>
        )}
        {actual != null && (
          <>
            <dt className="text-muted-foreground">Mean actual</dt>
            <dd>{actual}</dd>
          </>
        )}
        <dt className="text-muted-foreground">Mean ratio</dt>
        <dd>{formatRatio(metric.mean_ratio)}</dd>
        <dt className="text-muted-foreground">Ratio bounds</dt>
        <dd>
          {metric.min_ratio != null && metric.max_ratio != null
            ? `${formatRatio(metric.min_ratio)} – ${formatRatio(metric.max_ratio)}`
            : "—"}
        </dd>
      </dl>
      <p className="mt-2 text-sm text-muted-foreground">
        {metricSentence(metric)}
      </p>
    </div>
  );
}

function PhaseRow({ phase }: { phase: CalibrationPhase }) {
  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-6 gap-y-0.5 py-1 text-sm sm:grid-cols-[1fr_auto_auto_auto]">
      <dt className="text-muted-foreground">
        {phase.name}
        <span className="block text-xs">
          predicted as {phase.quotes_phases?.join(" + ") ?? phase.name}
        </span>
      </dt>
      <dd className="font-medium text-right">
        {formatDuration(phase.mean_actual ?? 0)}
      </dd>
      <dd className="text-right text-muted-foreground">
        {phase.mean_predicted != null
          ? formatDuration(phase.mean_predicted)
          : "—"}
      </dd>
      <dd className="text-right font-medium">{formatRatio(phase.mean_ratio)}</dd>
    </div>
  );
}

function RunRow({ run }: { run: CalibrationRun }) {
  const comparison = (run.comparison ?? {}) as Record<
    string,
    {
      actual?: number | null;
      ratio?: number | null;
    }
  >;
  const duration = comparison["duration"];
  const peak = comparison["peak_memory"];
  const cost = comparison["cost"];
  return (
    <tr className="border-b">
      <td className="py-2 pr-3">
        <Link
          href={`/jobs/${run.job_id}`}
          className="underline hover:no-underline"
        >
          {run.job_id}
        </Link>
      </td>
      <td className="py-2 pr-3">
        <code>{run.base_model}</code>
      </td>
      <td className="py-2 pr-3">{run.status}</td>
      <td className="py-2 pr-3 text-right">
        {duration?.actual != null ? formatDuration(duration.actual) : "—"}
      </td>
      <td className="py-2 pr-3 text-right">{formatRatio(duration?.ratio)}</td>
      <td className="py-2 pr-3 text-right">
        {peak?.actual != null ? formatGigabytes(peak.actual) : "—"}
      </td>
      <td className="py-2 pr-3 text-right">{formatRatio(peak?.ratio)}</td>
      <td className="py-2 text-right">{formatRatio(cost?.ratio)}</td>
    </tr>
  );
}

export default function CalibrationView({ data }: { data: Calibration }) {
  // A metric key may be absent when nothing has been compared yet; the empty
  // shape renders the card's "no comparable runs" state rather than crashing.
  const EMPTY_METRIC: CalibrationMetric = { count: 0 };
  const duration = data.metrics["duration"] ?? EMPTY_METRIC;
  const peak = data.metrics["peak_memory"] ?? EMPTY_METRIC;
  const cost = data.metrics["cost"] ?? EMPTY_METRIC;

  return (
    <section aria-labelledby="calibration-heading" className="space-y-6">
      <div>
        <h1 id="calibration-heading" className="text-2xl font-semibold">
          Predictions vs what happened
        </h1>
        <p className="mt-2 text-muted-foreground">
          {data.count}{" "}
          {data.count === 1
            ? "terminal run has"
            : "terminal runs have"}{" "}
          both a frozen quote and measured actuals. Each figure states its
          basis: predicted figures are estimates, duration and peak memory are
          measured, and cost is derived from measured duration at the frozen
          rate. The ratio is actual over predicted midpoint — a systematically
          wrong estimate shows up as a mean ratio away from 1 rather than
          being absorbed into a better-looking average.
        </p>
      </div>

      {data.count === 0 ? (
        <p>
          No runs to compare yet.{" "}
          <Link href="/" className="underline hover:no-underline">
            Upload a dataset
          </Link>{" "}
          and launch a job — recording starts with the first run, not the
          last.
        </p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-3">
          <MetricCard
            title="Duration"
            metric={duration}
            predicted={
              duration.mean_predicted != null
                ? formatDuration(duration.mean_predicted)
                : null
            }
            actual={
              duration.mean_actual != null
                ? formatDuration(duration.mean_actual)
                : null
            }
          />
          <MetricCard
            title="Peak memory"
            metric={peak}
            predicted={
              peak.mean_predicted != null
                ? formatGigabytes(peak.mean_predicted)
                : null
            }
            actual={
              peak.mean_actual != null
                ? formatGigabytes(peak.mean_actual)
                : null
            }
          />
          <MetricCard
            title="Cost"
            metric={cost}
            predicted={null}
            actual={null}
          />
        </div>
      )}

      {data.phases.length > 0 && (
        <div className="rounded-lg border bg-card p-4">
          <h2 className="text-lg font-semibold">Stage by stage</h2>
          <dl className="mt-2 divide-y divide-border">
            {data.phases.map((phase) => (
              <PhaseRow key={phase.name} phase={phase} />
            ))}
          </dl>
          <p className="mt-2 text-xs text-muted-foreground">
            Each stage&apos;s prediction is the summed quote phases that map to
            it (issue #72), measured against the job&apos;s own stage
            durations. Teardown is not isolable from the terminal transition
            and is left out.
          </p>
        </div>
      )}

      {data.runs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <caption className="pb-2 text-left text-muted-foreground">
              Every compared run, so an outlier can be named rather than
              pointed at.
            </caption>
            <thead>
              <tr className="border-b text-left">
                <th scope="col" className="py-2 pr-3 font-medium">Job</th>
                <th scope="col" className="py-2 pr-3 font-medium">Model</th>
                <th scope="col" className="py-2 pr-3 font-medium">Status</th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">Duration</th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">Dur. ratio</th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">Peak</th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">Peak ratio</th>
                <th scope="col" className="py-2 text-right font-medium">Cost ratio</th>
              </tr>
            </thead>
            <tbody>
              {data.runs.map((run) => (
                <RunRow key={run.job_id} run={run} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
