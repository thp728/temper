import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import BackToUpload from "@/components/BackToUpload";
import FocusHeading from "@/components/FocusHeading";
import type { DatasetRecord } from "@/lib/api/generated/client";

// What a user sees while validation runs in the background: a proportion
// complete, not a frozen page (spec 006 / issue #31). The page re-renders
// itself through a meta refresh -- the no-JavaScript equivalent of the old
// page's poll-and-reload, the same mechanism the job record uses while a job
// is running -- until the record's status flips and this component is replaced
// by the report.
function percentOf(record: DatasetRecord): number | null {
  const progress = record.progress;
  const total = progress?.bytes_total;
  if (!total) return null;
  return Math.min(100, Math.round(((progress.bytes_read ?? 0) / total) * 100));
}

export default function ValidationProgressView({
  record,
}: {
  record: DatasetRecord;
}) {
  const percent = percentOf(record);
  const importing = record.status === "importing";

  return (
    <section aria-labelledby="validating-heading" className="space-y-6">
      {/* The record is re-fetched on every reload, so the page advances
          through "importing" (an import's fetch, if that's how this dataset
          arrived) and "validating" until the report takes its place. */}
      <meta httpEquiv="refresh" content="2" />

      <div>
        <FocusHeading id="validating-heading">{record.filename}</FocusHeading>
        <Alert role="status" className="mt-3">
          <AlertTitle>
            {importing
              ? "Importing your dataset…"
              : "Validating your dataset…"}
          </AlertTitle>
          <AlertDescription>
            This page reloads itself while{" "}
            {importing ? "the import" : "validation"} runs, so you can leave
            it open.
          </AlertDescription>
        </Alert>
      </div>

      {percent !== null ? (
        <div role="status" aria-live="polite" className="space-y-2">
          <p className="text-sm text-muted-foreground">
            {percent}% — {record.progress?.rows ?? 0} rows read
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
        <p role="status" className="text-sm text-muted-foreground">
          {importing ? "Fetching rows…" : "Reading rows…"}
        </p>
      )}

      <BackToUpload />
    </section>
  );
}
