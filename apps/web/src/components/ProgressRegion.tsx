import type { JobOutputLine, JobProgress } from "@/lib/api/generated/client";
import {
  formatBytes,
  formatEta,
  formatRate,
  outputByPhase,
  proportion,
} from "@/lib/jobs/progress";

// The progress region (issue #49): one row per phase -- image pull, model
// download -- each drawing a proportion from its bytes, the rate measured
// live between readings, the estimate derived from it, and the raw lines that
// were promoted into the phase offered as collapsed detail, so nothing is
// discarded. Shared by the running view (fed by the stream) and the finished
// record (fed by the durable history), so the two cannot drift apart about
// what a phase's progress is.
//
// The label/value pairs stay real dt/dd pairs, and every figure states what
// it is: `done of total` is where the phase is, the rate says it is measured,
// and the estimate says it is derived from that measurement -- a figure
// without its basis is a number a reader cannot judge.

function PhaseRow({
  phase,
  output,
}: {
  phase: JobProgress;
  output: JobOutputLine[];
}) {
  const pct = proportion(phase.done, phase.total);
  const eta = formatEta(phase.eta_s);
  const lines = output.map((o) => o.line);

  return (
    <div className="rounded-lg border bg-card p-3">
      <dl className="space-y-1">
        <div className="flex items-baseline justify-between gap-2">
          <dt className="text-sm font-medium">{phase.phase}</dt>
          <dd className="text-sm text-muted-foreground">
            {phase.done != null && phase.total != null
              ? `${formatBytes(phase.done)} of ${formatBytes(phase.total)}`
              : phase.message ?? "starting…"}
          </dd>
        </div>
      </dl>

      <div
        role="progressbar"
        aria-label={phase.phase}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct != null ? Math.round(pct * 100) : undefined}
        className="mt-2 h-2 w-full overflow-hidden rounded-full bg-muted"
      >
        <div
          className="h-full rounded-full bg-primary transition-all"
          style={{ width: `${pct != null ? pct * 100 : 0}%` }}
        />
      </div>

      <p className="mt-1 text-xs text-muted-foreground">
        {formatRate(phase.rate)}
        {eta != null && <span> · about {eta} left</span>}
      </p>

      {lines.length > 0 && (
        <details className="mt-2 text-xs">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
            {lines.length} line{lines.length === 1 ? "" : "s"}
          </summary>
          <pre className="mt-1 max-h-48 overflow-y-auto whitespace-pre-wrap rounded border bg-muted/40 p-2 font-mono text-[11px]">
            {lines.join("\n")}
          </pre>
        </details>
      )}
    </div>
  );
}

export default function ProgressRegion({
  progress,
  output,
}: {
  progress: JobProgress[];
  output: JobOutputLine[];
}) {
  if (progress.length === 0) return null;
  const linesByPhase = outputByPhase(output);

  return (
    <section aria-labelledby="progress-heading" className="space-y-2">
      <h2 id="progress-heading" className="text-lg font-semibold">
        Progress
      </h2>
      <div className="space-y-3">
        {progress.map((p) => (
          <PhaseRow key={p.phase} phase={p} output={linesByPhase.get(p.phase) ?? []} />
        ))}
      </div>
    </section>
  );
}
