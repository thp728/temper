import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Card,
  CardContent,
} from "@/components/ui/card";
import type {
  DatasetRecord,
  TokenDistribution,
} from "@/lib/api/generated/client";

// The token count (issue #42) is produced by the counting phase that runs
// after validation, so it has its own state on the record: while it runs this
// shows progress; when it lands this shows the total, the distribution across
// rows, and the rows that would be truncated at the trainer's default sequence
// length; when it fails it says so without blocking the report.

function percentOf(record: DatasetRecord): number | null {
  const progress = record.counting_progress;
  const total = progress?.bytes_total;
  if (!total) return null;
  return Math.min(100, Math.round(((progress.bytes_read ?? 0) / total) * 100));
}

function histogramBars(dist: TokenDistribution): {
  label: string;
  count: number;
}[] {
  const edges = dist.histogram_edges;
  const histogram = dist.histogram;
  return edges.map((edge, i) => {
    const next = edges[i + 1];
    return {
      label: next !== undefined ? `${edge}–${next - 1}` : `${edge}+`,
      count: histogram[i] ?? 0,
    };
  });
}

function DistributionBars({ dist }: { dist: TokenDistribution }) {
  const bars = histogramBars(dist);
  const max = Math.max(1, ...bars.map((b) => b.count));
  return (
    <div className="mt-3 space-y-1" role="img" aria-label="Token count distribution across rows">
      {bars.map((b, i) =>
        b.count > 0 ? (
          <div key={i} className="flex items-center gap-2">
            <span className="w-20 shrink-0 text-right text-xs text-muted-foreground">
              {b.label}
            </span>
            <div
              role="progressbar"
              aria-valuenow={b.count}
              aria-valuemin={0}
              aria-valuemax={max}
              aria-label={`${b.count} rows between ${b.label} tokens`}
              className="h-3 min-w-[2px] rounded bg-primary"
              style={{ width: `${Math.max(2, (b.count / max) * 100)}%` }}
            />
            <span className="text-xs text-muted-foreground">{b.count}</span>
          </div>
        ) : null,
      )}
    </div>
  );
}

function truncationNote(dist: TokenDistribution): string {
  return `${dist.truncated_rows.toLocaleString()} row${
    dist.truncated_rows === 1 ? "" : "s"
  } ${dist.truncated_rows === 1 ? "is" : "are"} longer than the ${
    dist.sequence_len
  }-token sequence length and would be truncated during training.`;
}

export default function TokenCountView({
  record,
}: {
  record: DatasetRecord;
}) {
  const status = record.token_count_status;
  const report = record.report;
  const dist = report?.token_distribution ?? null;
  const total = report?.token_count ?? null;

  if (status === "counting") {
    const percent = percentOf(record);
    return (
      <Card>
        <CardContent>
          <Alert role="status" className="mt-0">
            <AlertTitle>Counting tokens…</AlertTitle>
            <AlertDescription>
              The count lands on this report when it finishes — validation is
              already complete, so this does not hold anything up.
            </AlertDescription>
          </Alert>
          {percent !== null ? (
            <div role="status" aria-live="polite" className="mt-3 space-y-2">
              <p className="text-sm text-muted-foreground">
                {percent}% — {record.counting_progress?.rows ?? 0} rows counted
              </p>
              <div
                role="progressbar"
                aria-valuenow={percent}
                aria-valuemin={0}
                aria-valuemax={100}
                className="h-2 w-full overflow-hidden rounded-full bg-muted"
              >
                <div
                  className="h-full bg-primary transition-all"
                  style={{ width: `${percent}%` }}
                />
              </div>
            </div>
          ) : (
            <p role="status" className="mt-3 text-sm text-muted-foreground">
              Reading rows…
            </p>
          )}
        </CardContent>
      </Card>
    );
  }

  if (status === "failed") {
    return (
      <Alert role="status" className="mt-0">
        <AlertTitle>Token count unavailable</AlertTitle>
        <AlertDescription>
          The count could not be produced this time. You can still proceed —
          the quote will show the count as unknown.
        </AlertDescription>
      </Alert>
    );
  }

  if (status === "done" && total !== null && dist !== null) {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Card>
            <CardContent>
              <dl>
                <dt className="text-sm text-muted-foreground">Token count</dt>
                <dd className="mt-1 text-2xl font-semibold">
                  {total.toLocaleString()}
                </dd>
              </dl>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <dl>
                <dt className="text-sm text-muted-foreground">Longest row</dt>
                <dd className="mt-1 text-2xl font-semibold">
                  {dist.max_row_tokens.toLocaleString()}
                </dd>
              </dl>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <dl>
                <dt className="text-sm text-muted-foreground">
                  Would be truncated
                </dt>
                <dd className="mt-1 text-2xl font-semibold">
                  {dist.truncated_rows.toLocaleString()}
                </dd>
              </dl>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardContent>
            <h2 className="text-lg font-semibold">Token count distribution</h2>
            <p className="text-sm text-muted-foreground">
              How the rows are spread across token-count buckets, counted with
              the default model&apos;s tokenizer.
            </p>
            <DistributionBars dist={dist} />
            <p className="mt-3 text-sm text-muted-foreground">
              {truncationNote(dist)}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return null;
}
