import type { Metadata } from "next";
import BackToUpload from "@/components/BackToUpload";
import FocusHeading from "@/components/FocusHeading";
import JobsView from "@/components/JobsView";
import {
  listDatasetsV1DatasetsGet,
  listJobsV1JobsGet,
} from "@/lib/api/generated/client";
import { load } from "@/lib/api/load";
import { epochNow } from "@/lib/jobs/display";

export const metadata: Metadata = {
  title: "Jobs",
};

// Job state changes outside any request's knowledge, so the list is read at
// request time and never cached into a lie.
export const dynamic = "force-dynamic";

export default async function JobsPage() {
  const { data: listing, error } = await load(() => listJobsV1JobsGet());

  if (error || !listing) {
    return (
      <section aria-labelledby="jobs-heading" className="space-y-4">
        <FocusHeading id="jobs-heading">The job list could not be loaded</FocusHeading>
        <p>
          <code className="rounded bg-neutral-100 px-1">{error?.code}</code> —{" "}
          {error?.message}
        </p>
        <BackToUpload />
      </section>
    );
  }

  // Names make a run identifiable ("support-chats.jsonl", not "ds_q4f…").
  // A dataset row that has gone does not take the list down with it: the id
  // it leaves behind beats a page that never renders.
  const { data: datasets } = await load(() => listDatasetsV1DatasetsGet());
  const datasetNames = Object.fromEntries(
    (datasets?.datasets ?? []).map((d) => [d.id, d.filename]),
  );

  // The clock reading travels with the HTML: the list's relative dates render
  // on the server first, and hydration must see the same strings the server
  // sent rather than a clock the client computed for itself.
  return (
    <JobsView jobs={listing.jobs} datasetNames={datasetNames} now={epochNow()} />
  );
}
