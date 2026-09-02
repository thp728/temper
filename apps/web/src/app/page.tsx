import type { Metadata } from "next";
import DashboardView from "@/components/DashboardView";
import FocusHeading from "@/components/FocusHeading";
import { listJobsV1JobsGet } from "@/lib/api/generated/client";
import { load } from "@/lib/api/load";
import { epochNow } from "@/lib/jobs/display";

export const metadata: Metadata = {
  // The root layout's title.template reaches child segments only, and this
  // page shares the root segment with it -- so the suffix is written here in
  // full rather than relying on the template every other page gets.
  title: "Dashboard - Temper",
};

// The dashboard reads live job state at request time; it is never cached into
// a lie about what is running or recorded. The calibration aggregate (issue
// #77) is no longer fetched here — it was observability, not end-user value,
// and is kept at /calibration for direct access (see CalibrationView.tsx top
// comment). Dashboard now aggregates spend & time from the jobs themselves.
export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const { data: listing, error: jobsError } = await load(() =>
    listJobsV1JobsGet(),
  );

  if (jobsError || !listing) {
    return (
      <section aria-labelledby="dashboard-heading" className="space-y-4">
        <FocusHeading id="dashboard-heading">
          The dashboard could not be loaded
        </FocusHeading>
        <p>
          <code className="rounded bg-muted px-1">{jobsError?.code}</code> —{" "}
          {jobsError?.message}
        </p>
      </section>
    );
  }

  return (
    <DashboardView
      jobs={listing.jobs}
      // The clock read at request time, behind the same seam the running-job
      // page uses: elapsed is a server-side reading of the job's own stamps,
      // not a client clock racing them.
      now={epochNow()}
    />
  );
}
