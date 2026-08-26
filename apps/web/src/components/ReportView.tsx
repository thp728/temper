import Link from "next/link";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import BackToUpload from "@/components/BackToUpload";
import FocusHeading from "@/components/FocusHeading";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
} from "@/components/ui/card";
import type {
  DatasetRecord,
  PreviewRow,
  ValidationIssue,
} from "@/lib/api/generated/client";

function thinkingText(enableThinking: boolean | null | undefined): string {
  if (enableThinking === true) return "Detected";
  if (enableThinking === false) return "Not detected";
  return "Undetermined";
}

// What the detection means for the run, in the same terms the domain uses:
// the label alone tells a user a fact, the explanation tells them what will
// happen to their training because of it.
function thinkingExplanation(
  enableThinking: boolean | null | undefined,
): string | null {
  if (enableThinking === true) {
    return "Assistant turns contain reasoning traces, so the model will be trained with thinking mode enabled.";
  }
  if (enableThinking === false) {
    return "The model will be trained to answer directly.";
  }
  return "The dataset was not valid far enough to detect it.";
}

function issueLocation(issue: ValidationIssue): string {
  return issue.line === null || issue.line === undefined
    ? "Whole file"
    : `Line ${issue.line}`;
}

function IssueList({ issues }: { issues: ValidationIssue[] }) {
  return (
    <ul className="space-y-2">
      {issues.map((issue, i) => (
        // Issues are a rendered report, not mutable data: position is stable
        // for the lifetime of the page.
        <li key={i} className="rounded-lg border bg-card p-3">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="font-medium">{issueLocation(issue)}</span>
            <code className="rounded bg-muted px-1 text-xs">
              {issue.code}
            </code>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            {issue.message}
          </p>
        </li>
      ))}
    </ul>
  );
}

function Stat({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <Card>
      <CardContent>
        {/* Label and value stay a real dt/dd pair: that adjacency is what a
            screen reader announces, and what the tests read. */}
        <dl>
          <dt className="text-sm text-muted-foreground">{label}</dt>
          <dd className="mt-1 text-2xl font-semibold">{value}</dd>
        </dl>
      </CardContent>
    </Card>
  );
}

function PreviewTurns({ row }: { row: PreviewRow }) {
  const messages = row.messages ?? [];
  return (
    <ul className="mt-2 space-y-1 text-sm">
      {messages.map((turn, j) => (
        <li key={j}>
          <span className="font-medium">{turn.role ?? "(no role)"}:</span>{" "}
          <span className="text-muted-foreground">{turn.content}</span>
        </li>
      ))}
    </ul>
  );
}

// The validation report: what was found, what blocks, what merely warns, and
// how the first rows were understood. Everything on this page comes from the
// published report shape -- the same dict the API returns, typed by the
// generated client. No hand-written casts.
export default function ReportView({ record }: { record: DatasetRecord }) {
  const report = record.report;
  if (!report) {
    return (
      <p role="status">
        This dataset has not been validated yet. Refresh to try again.
      </p>
    );
  }

  const blocked = !report.valid;

  return (
    <section aria-labelledby="report-heading" className="space-y-6">
      <div>
        <FocusHeading>{record.filename}</FocusHeading>
        <Alert
          role="status"
          variant={blocked ? "destructive" : "default"}
          className={`mt-3 ${
            blocked ? "" : "border-green-300 text-green-900"
          }`}
        >
          <AlertTitle>
            {blocked
              ? "This dataset was rejected. Fix the problems below and upload again."
              : "Validation passed. You can proceed to choose a model and launch."}
          </AlertTitle>
        </Alert>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Rows found" value={report.row_count} />
        <Stat label="Usable rows" value={report.usable_rows} />
        <Stat
          label="Schema"
          value={
            <span className="text-lg">{report.schema_type ?? "Not recognised"}</span>
          }
        />
        <Stat
          label="Thinking mode"
          value={
            <span className="text-lg">
              {thinkingText(report.enable_thinking)}
            </span>
          }
        />
      </div>

      <p className="text-sm text-muted-foreground">
        {thinkingExplanation(report.enable_thinking)}
      </p>

      {report.errors.length > 0 && (
        <section aria-labelledby="problems-heading">
          <h2 id="problems-heading" className="text-lg font-semibold">
            Problems ({report.errors.length})
          </h2>
          <p className="text-sm text-muted-foreground">
            This dataset cannot be trained on until every problem below is
            fixed. Fix the named lines and upload again.
          </p>
          <div className="mt-3">
            <IssueList issues={report.errors} />
          </div>
        </section>
      )}

      {report.warnings.length > 0 && (
        <section aria-labelledby="warnings-heading">
          <h2 id="warnings-heading" className="text-lg font-semibold">
            Warnings ({report.warnings.length})
          </h2>
          <p className="text-sm text-muted-foreground">
            These do not block proceeding.
          </p>
          <div className="mt-3">
            <IssueList issues={report.warnings} />
          </div>
        </section>
      )}

      {report.preview.length > 0 && (
        <section aria-labelledby="preview-heading">
          <h2 id="preview-heading" className="text-lg font-semibold">
            Preview — how your first rows were understood
          </h2>
          <div className="mt-3 space-y-3">
            {report.preview.map((row, i) => (
              <Card key={i}>
                <CardContent>
                  <p className="text-sm text-muted-foreground">Row {i + 1}</p>
                  {(row.messages?.length ?? 0) === 0 ? (
                    <p className="mt-1 text-sm">
                      No messages list found in this row.
                    </p>
                  ) : (
                    <PreviewTurns row={row} />
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      )}

      <div className="flex gap-3">
        {blocked ? (
          <BackToUpload />
        ) : (
          <>
            <Button asChild>
              <Link
                href={{
                  pathname: "/jobs/new",
                  query: { dataset_id: record.id },
                }}
              >
                Choose a model and continue
              </Link>
            </Button>
            <BackToUpload />
          </>
        )}
      </div>
    </section>
  );
}
