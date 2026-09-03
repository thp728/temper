"use client";

import Link from "next/link";
import { Rocket } from "lucide-react";
import { useState } from "react";
import EmptyStatePanel from "@/components/EmptyStatePanel";
import StatusPill from "@/components/StatusPill";
import { Button } from "@/components/ui/button";
import { formatDuration } from "@/lib/jobs/display";
import type { JobRecord } from "@/lib/api/generated/client";

const LINK = "text-primary transition-colors hover:text-primary-hover";
const PAGE_SIZE = 5;

function elapsed(job: JobRecord, now: number): number {
  return Math.max(0, now - (job.started_at ?? job.created_at));
}

export default function ActiveJobsSection({
  jobs,
  now,
}: {
  jobs: JobRecord[];
  now: number;
}) {
  const [page, setPage] = useState(0);
  const totalPages = Math.max(1, Math.ceil(jobs.length / PAGE_SIZE));
  // Clamp displayed page without mutating state during render storm
  const safePage = Math.min(page, totalPages - 1);
  const start = safePage * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, jobs.length);
  const pageJobs = jobs.slice(start, end);

  if (jobs.length === 0) {
    return (
      <section aria-labelledby="active-jobs-heading" className="space-y-3">
        <h2 id="active-jobs-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
          Active jobs
        </h2>
        <EmptyStatePanel
          icon={Rocket}
          heading="No active jobs"
          action={
            <Button asChild>
              <Link href="/datasets">Start a new job</Link>
            </Button>
          }
        >
          Launch a job and it appears here while it trains, from provisioning
          through packaging.
        </EmptyStatePanel>
      </section>
    );
  }

  return (
    <section aria-labelledby="active-jobs-heading" className="space-y-3">
      <h2 id="active-jobs-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
        Active jobs
      </h2>
      <p id="active-jobs-subtitle" className="text-sm text-muted-foreground">
        Jobs in progress, newest first.
      </p>
      <div className="overflow-hidden rounded-[12px] border bg-card">
        <div className="overflow-x-auto">
          <table
            aria-describedby="active-jobs-subtitle"
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
                  Elapsed
                </th>
                <th
                  scope="col"
                  className="w-[16%] px-4 py-3 text-right font-medium text-muted-foreground"
                >
                  GPU
                </th>
              </tr>
            </thead>
            <tbody>
              {pageJobs.map((job) => (
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
                    {formatDuration(elapsed(job, now))}
                  </td>
                  <td className="px-4 py-4 text-right">
                    {job.gpu_type ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {jobs.length > PAGE_SIZE && (
        <div className="flex items-center justify-between px-1 text-sm">
          <span
            className="text-muted-foreground"
            aria-live="polite"
            data-testid="active-jobs-pagination-info"
          >
            Showing {start + 1}–{end} of {jobs.length} · Page {safePage + 1} of {totalPages}
          </span>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={safePage === 0}
              aria-label="Previous page"
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={safePage >= totalPages - 1}
              aria-label="Next page"
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </section>
  );
}
