import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Card,
  CardContent,
} from "@/components/ui/card";
import { DatasetRecordTokenCountStatus } from "@/lib/api/generated/client";
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

type Bar = { label: string; count: number; edge: number };

function histogramBars(dist: TokenDistribution): Bar[] {
  const edges = dist.histogram_edges;
  const histogram = dist.histogram;
  return edges.map((edge, i) => {
    const next = edges[i + 1];
    return {
      label: next !== undefined ? `${edge}–${next - 1}` : `${edge}+`,
      count: histogram[i] ?? 0,
      edge,
    };
  });
}

// The histogram's edges are a fixed ladder (temper_core's binning) reaching
// 131072 tokens, far past any sequence-length cutoff in practice -- rendered
// one bar per edge, a long tail of rows would push the grid well past a
// single row and dwarf the bins anyone actually reads. Everything at or past
// the cutoff answers one question -- "would this row be truncated?" -- so
// those bins collapse into a single "{cutoff}+" bar instead of one per edge.
// Below the cutoff, every bin stays, since that shape is what "how are my
// rows sized" is asking about. For the trainer's default ladder (edges below
// 2048: 0, 128, 256, 512, 1024, 1536), this is 7 bars total.
function capAtCutoff(bars: Bar[], sequenceLen: number): Bar[] {
  const cutoff = bars.findIndex((b) => b.edge >= sequenceLen);
  if (cutoff === -1) return bars;
  const head = bars.slice(0, cutoff);
  const tail = bars.slice(cutoff);
  const tailCount = tail.reduce((sum, b) => sum + b.count, 0);
  return [
    ...head,
    { label: `${sequenceLen}+`, count: tailCount, edge: sequenceLen },
  ];
}

// Vertical columns, one per bin: height carries the share of rows, colour
// marks the bins at or past the trainer's cutoff (`sequence_len`, already on
// the report) as the ones that would be truncated -- the same fact the note
// below states in words.
function DistributionBars({ dist }: { dist: TokenDistribution }) {
  const bars = capAtCutoff(histogramBars(dist), dist.sequence_len);
  const max = Math.max(1, ...bars.map((b) => b.count));
  const totalRows = dist.rows_counted || bars.reduce((sum, b) => sum + b.count, 0);

  return (
    <div
      role="img"
      aria-label="Token count distribution across rows"
      className="mt-3"
    >
      <div className="flex items-center justify-between font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
        <span>Sequence length bins (tokens)</span>
        <span>Cutoff threshold: {dist.sequence_len.toLocaleString("en-US")}</span>
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2 sm:grid-cols-5 lg:grid-cols-7">
        {bars.map((b, i) => {
          // An empty bin past the cutoff has nothing to warn about -- red is
          // reserved for a bin that actually holds rows that would be
          // truncated, not for the shape of the ladder itself.
          const overCutoff = b.edge >= dist.sequence_len && b.count > 0;
          const percent =
            totalRows > 0 ? Math.round((b.count / totalRows) * 100) : 0;
          return (
            <div key={i} className="flex flex-col gap-1">
              <div className="flex h-20 items-end rounded bg-muted/30 p-1">
                <div
                  className={`w-full rounded-t transition-colors ${
                    overCutoff ? "bg-destructive/70" : "bg-primary/70"
                  }`}
                  style={{
                    height: b.count > 0 ? `${Math.max(4, (b.count / max) * 100)}%` : 0,
                  }}
                />
              </div>
              <div className="text-center">
                <div
                  className={`font-mono text-xs font-medium ${
                    overCutoff ? "text-destructive" : "text-foreground"
                  }`}
                >
                  {b.label}
                </div>
                <div
                  className={`font-mono text-[11px] ${
                    overCutoff ? "text-destructive" : "text-muted-foreground"
                  }`}
                >
                  {b.count.toLocaleString("en-US")} ({percent}%)
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function truncationNote(dist: TokenDistribution): string {
  return `${dist.truncated_rows.toLocaleString("en-US")} row${
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

  if (status === DatasetRecordTokenCountStatus.counting) {
    const percent = percentOf(record);
    return (
      <Card>
        <CardContent>
          <Alert role="status" className="mt-0">
            <AlertTitle>Counting tokens…</AlertTitle>
            <AlertDescription>
              The count lands on this report when it finishes. Validation is
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

  if (status === DatasetRecordTokenCountStatus.failed) {
    return (
      <Alert role="status" className="mt-0">
        <AlertTitle>Token count unavailable</AlertTitle>
        <AlertDescription>
          The count could not be produced this time. You can still proceed, and
          the quote will show the count as unknown.
        </AlertDescription>
      </Alert>
    );
  }

  if (status === DatasetRecordTokenCountStatus.done && total !== null && dist !== null) {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Card>
            <CardContent>
              <dl>
                <dt className="text-sm text-muted-foreground">Token count</dt>
                <dd className="mt-1 text-2xl font-semibold">
                  {total.toLocaleString("en-US")}
                </dd>
              </dl>
              {dist.rows_counted > 0 && (
                <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                  Avg {Math.round(dist.total_tokens / dist.rows_counted)} tok/sample
                </p>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <dl>
                <dt className="text-sm text-muted-foreground">Longest row</dt>
                <dd className="mt-1 text-2xl font-semibold">
                  {dist.max_row_tokens.toLocaleString("en-US")}
                </dd>
              </dl>
              <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                {Math.round((dist.max_row_tokens / dist.sequence_len) * 100)}%
                of the {dist.sequence_len.toLocaleString("en-US")} cutoff
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <dl>
                <dt className="text-sm text-muted-foreground">
                  Would be truncated
                </dt>
                <dd className="mt-1 text-2xl font-semibold">
                  {dist.truncated_rows.toLocaleString("en-US")}
                </dd>
              </dl>
              {dist.rows_counted > 0 && (
                <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                  {Math.round((dist.truncated_rows / dist.rows_counted) * 100)}%
                  of rows
                </p>
              )}
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
