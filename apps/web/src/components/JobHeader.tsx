import Link from "next/link";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSlashSeparator,
} from "@/components/ui/breadcrumb";
import FocusHeading from "@/components/FocusHeading";
import { shortRevision } from "@/lib/jobs/display";
import type { JobRecord } from "@/lib/api/generated/client";

// The identity block above a job's status row, shared by the running view
// and the finished record (issue: non-terminal/terminal parity) so the one
// thing that never changes about a job -- what it is, what it trains on --
// sits in an identical spot whether the page is still streaming or has
// reloaded into the durable record. `meta` is the one segment that differs
// (a live job names when it started; a finished one names when it ended),
// passed in rather than decided here.
export default function JobHeader({
  job,
  datasetFilename,
  meta,
  actions,
}: {
  job: JobRecord;
  datasetFilename?: string;
  meta: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <>
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink asChild>
              <Link href="/jobs">Jobs</Link>
            </BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSlashSeparator />
          <BreadcrumbItem>
            <BreadcrumbPage className="font-mono text-xs">
              {job.id}
            </BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      <div>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <FocusHeading id="job-heading">
              <code>{job.base_model}</code>
            </FocusHeading>
            <p className="mt-2 text-muted-foreground">
              Dataset{" "}
              <Link
                href={`/datasets/${job.dataset_id}`}
                className="underline hover:no-underline"
              >
                {datasetFilename ?? job.dataset_id}
              </Link>
              {job.base_revision && (
                <>
                  {" · "}model revision{" "}
                  <code title={job.base_revision}>
                    {shortRevision(job.base_revision)}
                  </code>
                </>
              )}
              {" · "}
              {meta}
            </p>
          </div>
          {actions && <div className="shrink-0">{actions}</div>}
        </div>
        {job.is_moe && (
          <div className="mt-3 rounded-[12px] border border-amber-200 bg-amber-50 p-3">
            <p className="text-sm font-medium text-amber-900">
              Mixture-of-experts, untested here
            </p>
            <p className="mt-1 text-sm text-amber-800">
              This model is a mixture-of-experts architecture, which is
              untested here: expert routing changes LoRA target-module
              selection, memory scales with total rather than active
              parameters, and routing interacts poorly with small-batch
              adapters. It is usable and labelled untested, because curation
              is a default rather than a boundary.
            </p>
          </div>
        )}
      </div>
    </>
  );
}
