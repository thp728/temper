import type { Metadata } from "next";
import BackToUpload from "@/components/BackToUpload";
import FocusHeading from "@/components/FocusHeading";
import JobRecordView from "@/components/JobRecordView";
import RunningJobView from "@/components/RunningJobView";
import {
  getDatasetV1DatasetsDatasetIdGet,
  getEventsV1JobsJobIdEventsGet,
  getJobV1JobsJobIdGet,
} from "@/lib/api/generated/client";
import { load } from "@/lib/api/load";
import { TERMINAL_STATUSES, epochNow } from "@/lib/jobs/display";

export const metadata: Metadata = {
  title: "Job",
};

// A job's state moves while no request is looking, so the record is read at
// request time; a running job's page re-renders itself from a live stream
// (see RunningJobView), and a finished one is the completed record (see
// JobRecordView).
export const dynamic = "force-dynamic";

function NotFound({ id }: { id: string }) {
  return (
    <section aria-labelledby="job-heading" className="space-y-4">
      <FocusHeading id="job-heading">Not found</FocusHeading>
      <p>
        No job with id{" "}
        <code className="rounded bg-neutral-100 px-1">{id}</code>.
      </p>
      <div className="flex gap-3">
        <BackToUpload />
      </div>
    </section>
  );
}

export default async function JobPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const { data: job, error } = await load(() => getJobV1JobsJobIdGet(id));

  if (error?.status === 404) {
    return <NotFound id={id} />;
  }
  if (error || !job) {
    return (
      <section aria-labelledby="job-heading" className="space-y-4">
        <FocusHeading id="job-heading">
          The job record could not be loaded
        </FocusHeading>
        <p>
          <code className="rounded bg-neutral-100 px-1">{error?.code}</code> —{" "}
          {error?.message}
        </p>
        <BackToUpload />
      </section>
    );
  }

  const [{ data: events }, { data: dataset }] = await Promise.all([
    load(() => getEventsV1JobsJobIdEventsGet(id)),
    // For the dataset's name; the row can be gone without the job being
    // unreadable, in which case its id stands in.
    load(() => getDatasetV1DatasetsDatasetIdGet(job.dataset_id)),
  ]);

  if (!TERMINAL_STATUSES.includes(job.status)) {
    // The running half of this route: the history recorded so far is rendered
    // server-side (so a returning visitor finds continuous history from the
    // first paint), and the live view streams everything that follows. Once
    // the job is terminal the stream closes and the page hands back to the
    // finished record below.
    const history = events?.events ?? [];
    const start = job.started_at ?? job.created_at;
    return (
      <RunningJobView
        job={job}
        events={history}
        datasetFilename={dataset?.filename}
        streamAfter={events?.last_id ?? 0}
        initialElapsedSeconds={Math.max(0, epochNow() - start)}
      />
    );
  }

  // The history is what makes the record complete -- how it ended sits
  // beside what it was doing until then. A history that cannot be fetched
  // does not take the record down with it.
  return (
    <JobRecordView
      job={job}
      events={events?.events ?? []}
      datasetFilename={dataset?.filename}
    />
  );
}
