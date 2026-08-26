import Link from "next/link";
import type {
  DatasetRecord,
  ValidationIssue,
} from "@/lib/api/generated/client";

function thinkingText(enableThinking: boolean | null | undefined): string {
  if (enableThinking === true) return "Detected";
  if (enableThinking === false) return "Not detected";
  return "Unknown";
}

function where(issue: ValidationIssue): string {
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
        <li key={i} className="rounded-md border border-neutral-200 bg-white p-3">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="font-medium">{where(issue)}</span>
            <code className="rounded bg-neutral-100 px-1 text-xs">
              {issue.code}
            </code>
          </div>
          <p className="mt-1 text-sm text-neutral-700">{issue.message}</p>
        </li>
      ))}
    </ul>
  );
}

// The validation report: what was found, what blocks, what merely warns, and
// how the first rows were understood. Everything on this page comes from the
// published report shape -- the same dict the API returns, typed by the
// generated client.
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
        <h1 id="report-heading" className="text-2xl font-semibold break-all">
          {record.filename}
        </h1>
        <p
          role="status"
          className={`mt-3 rounded-md border p-4 ${
            blocked
              ? "border-red-300 bg-red-50 text-red-900"
              : "border-green-300 bg-green-50 text-green-900"
          }`}
        >
          {blocked
            ? "This dataset was rejected. Fix the problems below and upload again."
            : "Validation passed. You can proceed to choose a model and launch."}
        </p>
      </div>

      <dl className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="rounded-md border border-neutral-200 bg-white p-4">
          <dt className="text-sm text-neutral-500">Rows found</dt>
          <dd className="mt-1 text-2xl font-semibold">{report.row_count}</dd>
        </div>
        <div className="rounded-md border border-neutral-200 bg-white p-4">
          <dt className="text-sm text-neutral-500">Usable rows</dt>
          <dd className="mt-1 text-2xl font-semibold">
            {report.usable_rows}
          </dd>
        </div>
        <div className="rounded-md border border-neutral-200 bg-white p-4">
          <dt className="text-sm text-neutral-500">Thinking mode</dt>
          <dd className="mt-1 text-lg font-semibold">
            {thinkingText(report.enable_thinking)}
          </dd>
        </div>
      </dl>

      {report.errors.length > 0 && (
        <section aria-labelledby="problems-heading">
          <h2 id="problems-heading" className="text-lg font-semibold">
            Problems ({report.errors.length})
          </h2>
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
          <p className="text-sm text-neutral-600">
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
            {report.preview.map((row, i) => {
              const messages = Array.isArray(row.messages) ? row.messages : [];
              return (
                <div
                  key={i}
                  className="rounded-md border border-neutral-200 bg-white p-4"
                >
                  <p className="text-sm text-neutral-500">Row {i + 1}</p>
                  {messages.length === 0 ? (
                    <p className="mt-1 text-sm text-neutral-700">
                      No messages list found in this row.
                    </p>
                  ) : (
                    <ul className="mt-2 space-y-1 text-sm">
                      {messages.map((m, j) => {
                        const turn = m as { role?: unknown; content?: unknown };
                        return (
                          <li key={j}>
                            <span className="font-medium">
                              {String(turn.role)}:
                            </span>{" "}
                            <span className="text-neutral-700">
                              {typeof turn.content === "string"
                                ? turn.content
                                : JSON.stringify(turn.content)}
                            </span>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      <div className="flex gap-3">
        {blocked ? (
          <Link
            href="/"
            className="rounded-md border border-neutral-300 bg-white px-4 py-2 text-sm font-medium hover:bg-neutral-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900 focus-visible:ring-offset-2"
          >
            Back to upload
          </Link>
        ) : (
          <>
            {/* Until this journey's next screen is ported (#38), continuing
                hands over to the existing server-rendered create-job page,
                proxied through this origin so the journey stays in one place. */}
            <a
              href={`/jobs/new?dataset_id=${encodeURIComponent(record.id)}`}
              className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900 focus-visible:ring-offset-2"
            >
              Choose a model and continue
            </a>
            <Link
              href="/"
              className="rounded-md border border-neutral-300 bg-white px-4 py-2 text-sm font-medium hover:bg-neutral-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900 focus-visible:ring-offset-2"
            >
              Upload another dataset
            </Link>
          </>
        )}
      </div>
    </section>
  );
}
