"use client";

import Link from "next/link";
import {
  AlertCircle,
  Braces,
  Brain,
  ChevronRight,
  Cpu,
  Hash,
  History,
  ListChecks,
  Rows3,
  ShieldCheck,
} from "lucide-react";
import { Alert, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import DatasetActionsMenu from "@/components/DatasetActionsMenu";
import EmptyStatePanel from "@/components/EmptyStatePanel";
import FocusHeading from "@/components/FocusHeading";
import SampleExplorer from "@/components/SampleExplorer";
import StatusPill from "@/components/StatusPill";
import TokenCountView from "@/components/TokenCountView";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { formatTimestamp } from "@/lib/jobs/display";
import type { ApiError } from "@/lib/api/mutator";
import type {
  DatasetRecord,
  DatasetReport,
  JobRecord,
  ValidationIssue,
} from "@/lib/api/generated/client";

function thinkingText(enableThinking: boolean | null | undefined): string {
  if (enableThinking === true) return "Detected";
  if (enableThinking === false) return "Not detected";
  return "Undetermined";
}

// The unrecognised_schema message is one sentence with an inline JSON
// example wedged into prose ("...JSONL: {"messages": [...]}. Keys found
// instead: [...]."). Reading the example straight out of that string, rather
// than re-declaring the accepted shape here, keeps this single source of
// truth in the validator that owns it (packages/core/src/temper_core/validation.py).
function parseUnrecognisedSchemaMessage(
  message: string,
): { prefix: string; schema: string; keysFound: string } | null {
  const match = /^(.*?:)\s*(\{.*\})\.\s*(Keys found instead:.*)$/s.exec(
    message,
  );
  if (!match) return null;
  const [, prefix, schema, keysFound] = match as unknown as [
    string,
    string,
    string,
    string,
  ];
  return { prefix, schema, keysFound };
}

// Reformats the compact JSON-like example from the validator message into
// readable, indented lines for the code block -- a display transform only,
// not a re-declaration of the schema itself.
function prettyPrintJsonish(raw: string): string {
  let compact = "";
  let inString = false;
  for (const ch of raw) {
    if (ch === '"') inString = !inString;
    if (!inString && /\s/.test(ch)) continue;
    compact += ch;
  }
  let out = "";
  let depth = 0;
  inString = false;
  const indent = (d: number) => "  ".repeat(d);
  for (const ch of compact) {
    if (ch === '"') inString = !inString;
    if (inString) {
      out += ch;
      continue;
    }
    switch (ch) {
      case "{":
      case "[":
        depth += 1;
        out += `${ch}\n${indent(depth)}`;
        break;
      case "}":
      case "]":
        depth -= 1;
        out = out.replace(/\n *$/, "") + `\n${indent(depth)}${ch}`;
        break;
      case ",":
        out += `,\n${indent(depth)}`;
        break;
      case ":":
        out += ": ";
        break;
      default:
        out += ch;
    }
  }
  return out;
}

function IssueMessage({ issue }: { issue: ValidationIssue }) {
  if (issue.code === "unrecognised_schema") {
    const parsed = parseUnrecognisedSchemaMessage(issue.message);
    if (parsed) {
      return (
        <>
          <p className="mt-1 text-sm text-muted-foreground">
            {parsed.prefix}
          </p>
          <pre className="mt-2 overflow-x-auto rounded-md bg-muted p-2.5 text-xs">
            <code className="font-mono">
              {prettyPrintJsonish(parsed.schema)}
            </code>
          </pre>
          <p className="mt-2 text-sm text-muted-foreground">
            {parsed.keysFound}
          </p>
        </>
      );
    }
  }
  return <p className="mt-1 text-sm text-muted-foreground">{issue.message}</p>;
}

function issueLocation(issue: ValidationIssue): string {
  return issue.line === null || issue.line === undefined
    ? "Whole file"
    : `Line ${issue.line}`;
}

// Collapses a sorted run of line numbers into ranges ("6–16, 20, 45–50") so
// a hundred consecutive bad lines read as one span instead of a hundred
// rows.
function compressRanges(lineNumbers: number[]): string {
  const sorted = [...lineNumbers].sort((a, b) => a - b);
  const parts: string[] = [];
  let start = sorted[0]!;
  let prev = sorted[0]!;
  for (const n of sorted.slice(1)) {
    if (n === prev + 1) {
      prev = n;
      continue;
    }
    parts.push(start === prev ? `${start}` : `${start}–${prev}`);
    start = n;
    prev = n;
  }
  parts.push(start === prev ? `${start}` : `${start}–${prev}`);
  return parts.join(", ");
}

// The label a group of same-code issues is filed under: singular "Line 12"
// when there is exactly one line-scoped issue (matching issueLocation's own
// wording), "Lines 6–16" once there is more than one.
function groupLineLabel(issues: ValidationIssue[]): string {
  const lineNumbers = issues
    .map((issue) => issue.line)
    .filter((line): line is number => line !== null && line !== undefined);
  const wholeFileCount = issues.length - lineNumbers.length;
  if (lineNumbers.length === 0) return "Whole file";
  const ranges = compressRanges(lineNumbers);
  const label = lineNumbers.length === 1 ? `Line ${ranges}` : `Lines ${ranges}`;
  return wholeFileCount > 0 ? `${label}, whole file` : label;
}

// A hundred-plus problems are rarely a hundred distinct causes -- they're a
// handful of causes repeated down the file. Grouping by code turns the wall
// of cards into one row per cause, with the affected lines collapsed into
// ranges and the individual lines one click away.
function groupByCode(issues: ValidationIssue[]): ValidationIssue[][] {
  const order: string[] = [];
  const byCode = new Map<string, ValidationIssue[]>();
  for (const issue of issues) {
    if (!byCode.has(issue.code)) {
      order.push(issue.code);
      byCode.set(issue.code, []);
    }
    byCode.get(issue.code)!.push(issue);
  }
  return order.map((code) => byCode.get(code)!);
}

// `codeCounts` is the report's true per-code total, uncapped -- it is what
// lets a group whose issues were partly dropped by the cap say "showing 99
// of 110" against its own cause, instead of a page-level "11 more, not
// shown" that names no cause at all. Absent on a report stored before this
// field existed, in which case a group can only speak to what it holds.
// Errors and warnings share this list but read as two different severities:
// red codes and badges name what blocks training, yellow names what merely
// flags it, so the color alone tells the two sections apart while scanning.
const severityStyle: Record<
  "error" | "warning",
  { codeClass: string; badgeVariant: "destructive" | "warning" }
> = {
  error: {
    codeClass: "rounded bg-destructive/10 px-1 text-xs text-destructive",
    badgeVariant: "destructive",
  },
  warning: {
    codeClass: "rounded bg-warning/10 px-1 text-xs text-warning",
    badgeVariant: "warning",
  },
};

function IssueList({
  issues,
  codeCounts,
  severity,
}: {
  issues: ValidationIssue[];
  codeCounts?: Record<string, number>;
  severity: "error" | "warning";
}) {
  const groups = groupByCode(issues);
  const { codeClass, badgeVariant } = severityStyle[severity];
  return (
    <ul className="space-y-2">
      {groups.map((group) => {
        const code = group[0]!.code;
        const total = codeCounts?.[code] ?? group.length;
        if (total === 1) {
          const issue = group[0]!;
          return (
            <li key={code} className="rounded-lg border bg-card p-3">
              <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                <span className="font-medium">{issueLocation(issue)}</span>
                <code className={codeClass}>{issue.code}</code>
              </div>
              <IssueMessage issue={issue} />
            </li>
          );
        }
        const truncated = total > group.length;
        return (
          <li key={code} className="rounded-lg border bg-card">
            <details className="group">
              <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2.5 [&::-webkit-details-marker]:hidden">
                <ChevronRight
                  className="size-3.5 shrink-0 text-muted-foreground transition-transform group-open:rotate-90"
                  aria-hidden
                />
                <span className="shrink-0 font-medium">
                  {groupLineLabel(group)}
                </span>
                <code className={`shrink-0 ${codeClass}`}>{code}</code>
                <span className="truncate text-sm text-muted-foreground">
                  {group[0]!.message}
                </span>
                <Badge
                  variant={badgeVariant}
                  className="ml-auto shrink-0 font-mono"
                >
                  {total}
                </Badge>
              </summary>
              <ul className="max-h-64 space-y-1.5 overflow-y-auto border-t px-3 py-2.5">
                {group.map((issue, i) => (
                  <li key={i} className="flex gap-2 text-sm">
                    <span className="shrink-0 font-medium">
                      {issueLocation(issue)}
                    </span>
                    <span className="text-muted-foreground">
                      {issue.message}
                    </span>
                  </li>
                ))}
              </ul>
              {truncated && (
                <p className="border-t px-3 py-2 text-xs text-muted-foreground">
                  Showing {group.length} of {total} -- the rest are the same
                  cause.
                </p>
              )}
            </details>
          </li>
        );
      })}
    </ul>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
  hint,
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  label: string;
  value: React.ReactNode;
  /** A second, smaller line under the value -- always derived from fields
      the report already publishes, never a fact the report doesn't have. */
  hint?: React.ReactNode;
}) {
  return (
    <Card className="group relative overflow-hidden">
      <div
        className="absolute inset-x-0 top-0 h-px bg-white/5 opacity-0 transition-opacity group-hover:opacity-100"
        aria-hidden="true"
      />
      <CardContent>
        <Icon className="mb-2 size-4 text-muted-foreground" aria-hidden />
        {/* Label and value stay a real dt/dd pair: that adjacency is what a
            screen reader announces, and what the tests read. */}
        <dl>
          <dt className="text-sm text-muted-foreground">{label}</dt>
          <dd className="mt-1 text-2xl font-semibold">{value}</dd>
        </dl>
        {hint && (
          <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
            {hint}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// Wireframe 2's "Quality" card, without inventing a score the validator
// never computed: the pass rate is usable rows over rows found, and the
// hint names the same error/truncation counts already on the report.
function dataHealth(report: DatasetReport): {
  percent: number | null;
  hint: string;
} {
  const percent =
    report.row_count > 0
      ? Math.round((report.usable_rows / report.row_count) * 1000) / 10
      : null;
  const errorCount = report.error_count ?? report.errors.length;
  const truncated = report.token_distribution?.truncated_rows;
  const hint =
    `${errorCount} error${errorCount === 1 ? "" : "s"}` +
    (truncated ? ` · ${truncated} truncated` : "");
  return { percent, hint };
}

// The right-hand rail on the wireframe's bento layout: the jobs launched
// against this dataset, newest first. There is no "jobs for dataset X"
// endpoint, so the page fetches every job and hands this the ones that
// match -- a failure there is this panel's problem alone, never the
// report's.
// The wireframe's bento layout dropped the old footer's "Choose a model and
// continue" button along with it; a ready dataset still needs exactly one
// clear way to start a job scoped to it, so that link moves into this
// panel's header, where it stays visible whether or not a job exists yet.
// Distinct from the sidebar's own "New job" link (which opens /jobs/new with
// no dataset picked, not this dataset already selected) so the two never
// collide on accessible name -- and so what the button does is legible on its
// own.
function NewJobButton({ datasetId }: { datasetId: string }) {
  return (
    <Button size="sm" asChild>
      <Link href={{ pathname: "/jobs/new", query: { dataset_id: datasetId } }}>
        New job with this dataset
      </Link>
    </Button>
  );
}

function RecentJobsPanel({
  datasetId,
  jobs,
  jobsError,
}: {
  datasetId: string;
  jobs: JobRecord[];
  jobsError: ApiError | null;
}) {
  const shown = jobs.slice(0, 5);
  return (
    <section
      aria-labelledby="recent-jobs-heading"
      className="flex flex-col overflow-hidden rounded-[12px] border bg-card"
    >
      <div className="flex items-center justify-between gap-2 border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <History className="size-4 text-muted-foreground" aria-hidden />
          <h2 id="recent-jobs-heading" className="text-sm font-medium">
            Associated jobs
          </h2>
          {jobs.length > 0 && (
            <span className="font-mono text-xs text-muted-foreground">
              {jobs.length} total
            </span>
          )}
        </div>
        <NewJobButton datasetId={datasetId} />
      </div>
      {jobsError ? (
        <p className="p-4 text-sm text-muted-foreground">
          <code className="rounded bg-muted px-1">{jobsError.code}</code> —{" "}
          {jobsError.message}
        </p>
      ) : shown.length === 0 ? (
        <div className="p-4">
          <EmptyStatePanel icon={History} heading="No jobs yet" headingAs="h3">
            Launch a job against this dataset and it appears here.
          </EmptyStatePanel>
        </div>
      ) : (
        <ul className="divide-y">
          {shown.map((job) => (
            <li key={job.id}>
              <Link
                href={`/jobs/${job.id}`}
                className="block px-4 py-3 transition-colors hover:bg-muted/40"
              >
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-xs text-primary">
                    {job.id}
                  </span>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {formatTimestamp(job.created_at)}
                  </span>
                </div>
                <div className="mb-1.5 flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Cpu className="size-3.5 shrink-0" aria-hidden />
                  <code className="truncate">{job.base_model}</code>
                </div>
                <StatusPill status={job.status} />
              </Link>
            </li>
          ))}
        </ul>
      )}
      {jobs.length > 0 && (
        <div className="border-t p-2 text-center">
          <Button variant="link" size="sm" asChild>
            <Link href="/jobs">View all jobs</Link>
          </Button>
        </div>
      )}
    </section>
  );
}

// The validation report: what was found, what blocks, what merely warns, and
// how the first rows were understood. Everything on this page comes from the
// published report shape -- the same dict the API returns, typed by the
// generated client. No hand-written casts.
//
// A client component so the launch wizard can render the same report inside
// a modal without navigating away: the report page and the dialog read the
// same props, never two implementations. In the dialog the dataset actions
// (rename, delete) stay hidden — deleting mid-wizard would yank the launch
// out from under itself, and that choice belongs on the report page.
export default function ReportView({
  record,
  jobs = [],
  jobsError = null,
  showActions = true,
  stacked = false,
}: {
  record: DatasetRecord;
  jobs?: JobRecord[];
  jobsError?: ApiError | null;
  showActions?: boolean;
  // Stack the preview over the job history instead of beside it: the report
  // page has the full width, but a modal does not, and the side-by-side
  // bento that breathes on the page squeezes in a dialog.
  stacked?: boolean;
}) {
  const report = record.report;
  if (!report) {
    return (
      <p role="status">
        This dataset has not been validated yet. Refresh to try again.
      </p>
    );
  }

  const blocked = !report.valid;
  const health = dataHealth(report);

  return (
    <section aria-labelledby="report-heading" className="space-y-6">
      <div className="space-y-4">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 flex-wrap items-center gap-3">
            <FocusHeading>{record.filename}</FocusHeading>
            <span
              className={`inline-flex shrink-0 items-center rounded-full border px-2 py-0.5 font-mono text-xs font-medium uppercase ${
                blocked
                  ? "border-destructive/30 bg-destructive/10 text-destructive"
                  : "border-success/30 bg-success/10 text-success"
              }`}
            >
              {blocked ? "Needs fixes" : "Ready"}
            </span>
          </div>
          {showActions && (
            <DatasetActionsMenu
              id={record.id}
              filename={record.filename}
              redirectOnDeleteTo="/datasets"
            />
          )}
        </div>
        {/* Passing is what the badge above says permanently; this banner is
            only worth a reader's attention while there is something to act
            on, so it renders for a rejected dataset alone. Full width: it is
            the page's one call to action, not a note beside the filename. */}
        {blocked && (
          <Alert role="status" variant="destructive">
            <AlertCircle className="size-4" aria-hidden />
            <AlertTitle>
              This dataset was rejected. Fix the errors below and upload
              again.
            </AlertTitle>
          </Alert>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <Stat
          icon={Rows3}
          label="Rows found"
          value={report.row_count}
          hint="as uploaded"
        />
        <Stat
          icon={ListChecks}
          label="Usable rows"
          value={report.usable_rows}
          hint={
            report.row_count - report.usable_rows > 0
              ? `${report.row_count - report.usable_rows} discarded`
              : "none discarded"
          }
        />
        <Stat
          icon={Braces}
          label="Schema"
          value={
            <span className="text-lg">{report.schema_type ?? "Not recognised"}</span>
          }
          hint="auto-detected"
        />
        <Stat
          icon={Brain}
          label="Thinking mode"
          value={
            <span className="text-lg">
              {thinkingText(report.enable_thinking)}
            </span>
          }
          hint="affects training format"
        />
        <Stat
          icon={ShieldCheck}
          label="Data health"
          value={health.percent === null ? "—" : `${health.percent}%`}
          hint={health.hint}
        />
      </div>

      {/* The token count (issue #42) is produced by the counting phase that
          runs after validation: while it runs this shows progress, and when
          it lands this shows the total, the distribution and the rows that
          would be truncated. Nothing here holds the report up. */}
      <TokenCountView record={record} />

      {report.errors.length > 0 && (
        <section aria-labelledby="errors-heading">
          <h2
            id="errors-heading"
            className="flex items-center gap-2 text-lg font-semibold"
          >
            Errors{" "}
            <span className="sr-only">
              ({report.error_count ?? report.errors.length})
            </span>
            <Badge variant="destructive" className="font-mono" aria-hidden>
              {report.error_count ?? report.errors.length}
            </Badge>
          </h2>
          <p className="text-sm text-muted-foreground">
            This dataset cannot be trained on until every error below is
            fixed. Fix the named lines and upload again.
          </p>
          <div className="mt-3">
            <IssueList
              issues={report.errors}
              codeCounts={report.error_code_counts}
              severity="error"
            />
          </div>
        </section>
      )}

      {report.warnings.length > 0 && (
        <section aria-labelledby="warnings-heading">
          <h2
            id="warnings-heading"
            className="flex items-center gap-2 text-lg font-semibold"
          >
            Warnings{" "}
            <span className="sr-only">
              ({report.warning_count ?? report.warnings.length})
            </span>
            <Badge variant="warning" className="font-mono" aria-hidden>
              {report.warning_count ?? report.warnings.length}
            </Badge>
          </h2>
          <p className="text-sm text-muted-foreground">
            These do not block proceeding.
          </p>
          <div className="mt-3">
            <IssueList
              issues={report.warnings}
              codeCounts={report.warning_code_counts}
              severity="warning"
            />
          </div>
        </section>
      )}

      {/* Bento layout: the preview takes two thirds on desktop, the dataset's
          own job history sits beside it in the third -- the wireframe's pairing
          of "what's in the file" with "what has been done with it". A
          rejected dataset cannot launch a job, so that pairing has nothing to
          show yet; the preview takes the full width instead of sitting next
          to a panel that can only promise a workflow this dataset can't use. */}
      <div
        className={
          stacked
            ? "grid grid-cols-1 gap-4"
            : "grid grid-cols-1 gap-4 lg:grid-cols-3"
        }
      >
        <div className={stacked ? "" : blocked ? "lg:col-span-3" : "lg:col-span-2"}>
          {report.preview.length > 0 && (
            <section
              aria-labelledby="preview-heading"
              className="h-full overflow-hidden rounded-[12px] border bg-card"
            >
              <div className="flex items-center gap-2 border-b px-4 py-3">
                <Hash className="size-4 text-muted-foreground" aria-hidden />
                <h2 id="preview-heading" className="text-sm font-medium">
                  Preview: how your first rows were understood
                </h2>
              </div>
              <SampleExplorer rows={report.preview} />
            </section>
          )}
        </div>
        {!blocked && (
          <RecentJobsPanel
            datasetId={record.id}
            jobs={jobs}
            jobsError={jobsError}
          />
        )}
      </div>
    </section>
  );
}
