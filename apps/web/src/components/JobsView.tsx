import Link from "next/link";
import { Inbox } from "lucide-react";
import EmptyStatePanel from "@/components/EmptyStatePanel";
import StatusPill from "@/components/StatusPill";
import { Button } from "@/components/ui/button";
import { formatTimestamp } from "@/lib/jobs/display";
import type { JobRecord } from "@/lib/api/generated/client";

// The jobs list (#14's screen, ported): every job, newest first, each with
// its outcome beside it and a link to its full record -- the event history,
// and the artifact when there is one. Static HTML; nothing on it needs
// JavaScript, exactly as the page it replaces. The status renders through the
// same pill the dashboard uses, so one status has one look on both screens.

// The way a text link reads: the accent for emphasis, a step lighter on
// hover, never an underline.
const LINK = "text-primary transition-colors hover:text-primary-hover";

function Outcome({ job }: { job: JobRecord }) {
  return (
    <>
      <StatusPill status={job.status} />
      {job.error_code && (
        // A failure names its stable code in the list itself: finding the
        // job and learning what happened are one step, not two.
        <code className="ml-1 rounded bg-muted px-1 text-xs">
          {job.error_code}
        </code>
      )}
    </>
  );
}

export default function JobsView({
  jobs,
  datasetNames,
}: {
  jobs: JobRecord[];
  datasetNames: Record<string, string>;
}) {
  if (jobs.length === 0) {
    return (
      <section aria-labelledby="jobs-heading" className="space-y-4">
        <h1 id="jobs-heading" className="text-2xl font-semibold">
          Jobs
        </h1>
        <EmptyStatePanel
          icon={Inbox}
          heading="No jobs yet"
          headingAs="h2"
          action={
            <Button asChild>
              <Link href="/datasets">Select a dataset</Link>
            </Button>
          }
        >
          Every job you launch appears here, newest first, with its status
          beside it.
        </EmptyStatePanel>
      </section>
    );
  }

  return (
    <section aria-labelledby="jobs-heading" className="space-y-4">
      <h1 id="jobs-heading" className="text-2xl font-semibold">
        Jobs
      </h1>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <caption className="pb-2 text-left text-muted-foreground">
            Every job, newest first. Each links to its full record — the event
            history, and the artifact when there is one.
          </caption>
          <thead>
            <tr className="border-b text-left">
              <th scope="col" className="py-2 pr-3 font-medium">Job</th>
              <th scope="col" className="py-2 pr-3 font-medium">Status</th>
              <th scope="col" className="py-2 pr-3 font-medium">Base model</th>
              <th scope="col" className="py-2 pr-3 font-medium">Dataset</th>
              <th scope="col" className="py-2 font-medium">Created</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id} className="border-b">
                <td className="py-2 pr-3">
                  <Link href={`/jobs/${job.id}`} className={LINK}>
                    {job.id}
                  </Link>
                </td>
                <td className="py-2 pr-3">
                  <Outcome job={job} />
                </td>
                <td className="py-2 pr-3">
                  {/* The model id is what every other surface calls it. */}
                  <code>{job.base_model}</code>
                </td>
                <td className="py-2 pr-3">
                  {datasetNames[job.dataset_id] ?? job.dataset_id}
                </td>
                <td className="py-2">{formatTimestamp(job.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
