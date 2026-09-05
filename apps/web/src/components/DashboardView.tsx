import Link from "next/link";
import { Clock, Flag, Plus, Wallet } from "lucide-react";
import ActiveJobsSection from "@/components/ActiveJobsSection";
import EmptyStatePanel from "@/components/EmptyStatePanel";
import StatusPill from "@/components/StatusPill";
import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import {
  TERMINAL_STATUSES,
  formatDuration,
  formatMinorCost,
} from "@/lib/jobs/display";
import type { JobRecord } from "@/lib/api/generated/client";

// The home page: what is training now, what finished recently, and spend &
// time across finished jobs. Everything here is read from the real contract --
// the active table shows the job's own status vocabulary and GPU, the finished
// table reads duration and cost from the frozen actuals, and the spend & time
// glimpse aggregates what finished jobs actually cost and how long they took,
// because that is what helps a user budget and plan the next job. A number
// with no data behind it (a live progress percentage, eval scores) is dropped
// rather than faked.
//
// Vocabulary is CONTEXT.md's, not the wireframe's: jobs, never "runs";
// no "cluster" -- a machine is provisioned for one job and destroyed when it
// ends, so there is no standing fleet to monitor.

// "Recently finished" is capped: the list page shows every job, this screen
// shows the tail of the record a visitor is most likely to want back.
const RECENT_FINISHED_LIMIT = 5;

function byCreatedAtDesc(a: JobRecord, b: JobRecord): number {
  return b.created_at - a.created_at;
}

// The empty panel is `EmptyStatePanel`, shared with the jobs and calibration
// screens so one empty state has one look everywhere.
const LINK = "text-primary transition-colors hover:text-primary-hover";

function FinishedJobsSection({ jobs }: { jobs: JobRecord[] }) {
  return (
    <section aria-labelledby="finished-jobs-heading" className="space-y-3">
      <h2 id="finished-jobs-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
        Recently finished jobs
      </h2>
      {jobs.length === 0 ? (
        // No action here: the dashboard's one primary step is the header's
        // "Start a new job" -- three identical buttons down the page would
        // bury it.
        <EmptyStatePanel icon={Flag} heading="No finished jobs yet">
          A job appears here once it completes, fails, or is cancelled.
        </EmptyStatePanel>
      ) : (
        <>
          <p
            id="finished-jobs-subtitle"
            className="text-sm text-muted-foreground"
          >
            Up to {RECENT_FINISHED_LIMIT} most recent terminal jobs, newest
            first.
          </p>
          <div className="overflow-hidden rounded-[12px] border bg-card">
            <div className="overflow-x-auto">
              <table
                aria-describedby="finished-jobs-subtitle"
                className="w-full table-fixed border-collapse text-sm tabular-nums"
              >
                <thead>
                  <tr className="border-b text-left">
                    <th
                      scope="col"
                      className="w-[32%] px-4 py-3 font-medium text-muted-foreground"
                    >
                      Job
                    </th>
                    <th
                      scope="col"
                      className="w-[16%] px-4 py-3 font-medium text-muted-foreground"
                    >
                      Status
                    </th>
                    <th
                      scope="col"
                      className="w-[20%] px-4 py-3 font-medium text-muted-foreground"
                    >
                      Model
                    </th>
                    <th
                      scope="col"
                      className="w-[16%] px-4 py-3 text-right font-medium text-muted-foreground"
                    >
                      Duration
                    </th>
                    <th
                      scope="col"
                      className="w-[16%] px-4 py-3 text-right font-medium text-muted-foreground"
                    >
                      Cost
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job) => {
                    const actuals = job.actuals;
                    return (
                      <tr
                        key={job.id}
                        className="border-b border-border bg-background/50 last:border-b-0"
                      >
                        <td className="truncate px-4 py-4">
                          <Link href={`/jobs/${job.id}`} className={LINK}>
                            {job.id}
                          </Link>
                        </td>
                        <td className="px-4 py-4">
                          <StatusPill status={job.status} />
                        </td>
                        <td className="px-4 py-4">
                          <code>{job.base_model}</code>
                        </td>
                        <td className="px-4 py-4 text-right">
                          {actuals?.duration_s != null
                            ? formatDuration(actuals.duration_s)
                            : "—"}
                        </td>
                        <td className="px-4 py-4 text-right">
                          {actuals?.cost_minor != null &&
                          actuals.currency != null &&
                          job.quote?.minor_unit != null
                            ? // The actuals' cost is derived in the same currency
                              // and minor unit the quote priced, so the quote's
                              // published minor unit is the unit to show (the
                              // same rule the finished record applies).
                              formatMinorCost(
                                actuals.cost_minor,
                                actuals.currency,
                                job.quote.minor_unit,
                              )
                            : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </section>
  );
}

function SpendTimeSection({ jobs }: { jobs: JobRecord[] }) {
  const finished = jobs.filter((j) => TERMINAL_STATUSES.includes(j.status));

  if (finished.length === 0) {
    return (
      <section aria-labelledby="spend-time-heading" className="space-y-3">
        <h2 id="spend-time-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
          Spend & time
        </h2>
        <EmptyStatePanel icon={Wallet} heading="No spend yet">
          Finish a job and its actual cost and duration appear here.
        </EmptyStatePanel>
      </section>
    );
  }

  // Aggregate total spend by currency+minorUnit so a mixed-currency account
  // does not sum incomparable amounts.
  const spendMap = new Map<
    string,
    { total: number; currency: string; minorUnit: number; count: number }
  >();
  let durationSum = 0;
  let durationCount = 0;
  let completeCount = 0;

  for (const job of finished) {
    if (job.status === "complete") completeCount += 1;
    const actuals = job.actuals;
    if (actuals?.duration_s != null && typeof actuals.duration_s === "number") {
      durationSum += actuals.duration_s;
      durationCount += 1;
    }
    if (
      actuals?.cost_minor != null &&
      actuals.currency != null &&
      typeof actuals.cost_minor === "number"
    ) {
      const minorUnit = job.quote?.minor_unit ?? 100;
      const key = `${actuals.currency}:${minorUnit}`;
      const existing = spendMap.get(key);
      if (existing) {
        existing.total += actuals.cost_minor;
        existing.count += 1;
      } else {
        spendMap.set(key, {
          total: actuals.cost_minor,
          currency: actuals.currency,
          minorUnit,
          count: 1,
        });
      }
    }
  }

  const totalSpendText =
    spendMap.size === 0
      ? "—"
      : Array.from(spendMap.values())
          .map((v) => formatMinorCost(v.total, v.currency, v.minorUnit))
          .join(" · ");

  const spendNote =
    spendMap.size === 0
      ? "No cost recorded yet."
      : spendMap.size === 1
        ? `Across ${Array.from(spendMap.values())[0]!.count} finished job${Array.from(spendMap.values())[0]!.count === 1 ? "" : "s"} with cost.`
        : `Across ${Array.from(spendMap.values()).reduce((a, v) => a + v.count, 0)} finished jobs with cost.`;

  const avgSpendText =
    spendMap.size === 0
      ? "—"
      : Array.from(spendMap.values())
          .map((v) =>
            formatMinorCost(
              Math.round(v.total / v.count),
              v.currency,
              v.minorUnit,
            ),
          )
          .join(" · ");

  const avgSpendNote =
    spendMap.size === 0 ? "No cost recorded yet." : "Per job with cost.";

  const avgDurationText =
    durationCount === 0 ? "—" : formatDuration(durationSum / durationCount);

  const successRate = Math.round((completeCount / finished.length) * 100);
  const successText = `${successRate}% · ${completeCount}/${finished.length} complete`;

  return (
    <section aria-labelledby="spend-time-heading" className="space-y-3">
      <h2 id="spend-time-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
        Spend & time
      </h2>
      <p className="text-sm text-muted-foreground">
        Based on {finished.length}{" "}
        {finished.length === 1 ? "finished job" : "finished jobs"}.
      </p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-[12px] border bg-card p-4">
          <div className="flex items-center gap-2">
            <Wallet
              aria-hidden="true"
              className="size-4 text-muted-foreground"
            />
            <h3 className="font-medium">Total spent</h3>
          </div>
          <p className="mt-2 text-2xl font-semibold tracking-tight tabular-nums">
            {totalSpendText}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">{spendNote}</p>
        </div>
        <div className="rounded-[12px] border bg-card p-4">
          <div className="flex items-center gap-2">
            <Wallet
              aria-hidden="true"
              className="size-4 text-muted-foreground"
            />
            <h3 className="font-medium">Avg spend per job</h3>
          </div>
          <p className="mt-2 text-2xl font-semibold tracking-tight tabular-nums">
            {avgSpendText}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">{avgSpendNote}</p>
        </div>
        <div className="rounded-[12px] border bg-card p-4">
          <div className="flex items-center gap-2">
            <Clock
              aria-hidden="true"
              className="size-4 text-muted-foreground"
            />
            <h3 className="font-medium">Average duration</h3>
          </div>
          <p className="mt-2 text-2xl font-semibold tracking-tight tabular-nums">
            {avgDurationText}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {durationCount === 0
              ? "No duration recorded yet."
              : `Across ${durationCount} job${durationCount === 1 ? "" : "s"} with duration.`}
          </p>
        </div>
        <div className="rounded-[12px] border bg-card p-4">
          <div className="flex items-center gap-2">
            <Flag aria-hidden="true" className="size-4 text-muted-foreground" />
            <h3 className="font-medium">Success rate</h3>
          </div>
          <p className="mt-2 text-2xl font-semibold tracking-tight tabular-nums">
            {successText}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {finished.length - completeCount === 0
              ? "No failures or cancellations."
              : `${finished.length - completeCount} failed or cancelled.`}
          </p>
        </div>
      </div>
    </section>
  );
}

export default function DashboardView({
  jobs,
  now,
}: {
  jobs: JobRecord[];
  now: number;
}) {
  const active = jobs
    .filter((job) => !TERMINAL_STATUSES.includes(job.status))
    .sort(byCreatedAtDesc);
  const finished = jobs
    .filter((job) => TERMINAL_STATUSES.includes(job.status))
    .sort(byCreatedAtDesc)
    .slice(0, RECENT_FINISHED_LIMIT);

  return (
    <section aria-labelledby="dashboard-heading" className="space-y-8">
      {/* Breadcrumb for style parity: Dashboard is top-level so it is
          not a link, just a current page marker like Datasets page. */}
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbPage>Dashboard</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-1">
          <h1
            id="dashboard-heading"
            className="text-2xl font-semibold tracking-tight text-balance"
          >
            Training overview
          </h1>
          <p className="text-sm text-muted-foreground">
            What is training now, what finished recently, and what
            you&apos;ve spent.
          </p>
        </div>
        {/* A job starts by picking a dataset: the launch screen carries its own
            picker, so the action lands there directly. Named apart from the
            sidebar's "New job" -- two controls, one route, and an ambiguous
            name would break the journeys' accessible-name queries. It shares
            its name with the active-jobs empty state's own button below: same
            action, same route, so tests reach either by scoping to a
            container or using `getAllByRole`. */}
        <Button asChild>
          <Link href="/jobs/new">
            <Plus aria-hidden="true" className="size-4" />
            Start a new job
          </Link>
        </Button>
      </div>

      <SpendTimeSection jobs={jobs} />
      <ActiveJobsSection jobs={active} now={now} />
      <FinishedJobsSection jobs={finished} />
    </section>
  );
}
