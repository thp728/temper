import type { Metadata } from "next";
import Link from "next/link";
import BackToUpload from "@/components/BackToUpload";
import ReportView from "@/components/ReportView";
import TokenCountPoll from "@/components/TokenCountPoll";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSlashSeparator,
} from "@/components/ui/breadcrumb";
import ValidationProgressView from "@/components/ValidationProgressView";
import { DatasetRecordTokenCountStatus } from "@/lib/api/generated/client";
import {
  getDatasetV1DatasetsDatasetIdGet,
  listJobsV1JobsGet,
} from "@/lib/api/generated/client";
import { load } from "@/lib/api/load";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

export const metadata: Metadata = {
  title: "Dataset",
};

// The fetch is what the try/catch guards; rendering happens after it, so a
// component error here is an error boundary's job, not this function's.
async function loadDataset(id: string) {
  try {
    return { record: await getDatasetV1DatasetsDatasetIdGet(id), error: null };
  } catch (err) {
    // A report that cannot be fetched is rendered as the refusal it is,
    // with its stable code -- the same contract every page honours.
    const apiError =
      err instanceof ApiError
        ? err
        : NETWORK_ERROR;
    return { record: null, error: apiError };
  }
}

// There is no "list jobs for a dataset" endpoint (spec 007 never asked for
// one), so this fetches every job and keeps the ones that name this dataset
// -- the same trade the jobs list already makes for `datasetNames`. A load
// failure here is not the report's failure: the report renders either way,
// with the jobs panel saying it could not be loaded.
async function loadDatasetJobs(id: string) {
  const { data, error } = await load(() => listJobsV1JobsGet());
  const jobs = (data?.jobs ?? [])
    .filter((job) => job.dataset_id === id)
    .sort((a, b) => b.created_at - a.created_at);
  return { jobs, error };
}

function DatasetBreadcrumb({ label }: { label: string }) {
  return (
    <Breadcrumb className="mb-6">
      <BreadcrumbList>
        <BreadcrumbItem>
          <BreadcrumbLink asChild>
            <Link href="/datasets">Datasets</Link>
          </BreadcrumbLink>
        </BreadcrumbItem>
        <BreadcrumbSlashSeparator />
        <BreadcrumbItem>
          <BreadcrumbPage className="max-w-[28ch] truncate">{label}</BreadcrumbPage>
        </BreadcrumbItem>
      </BreadcrumbList>
    </Breadcrumb>
  );
}

function NotFound({ id }: { id: string }) {
  return (
    <section aria-labelledby="not-found-heading" className="space-y-4">
      <DatasetBreadcrumb label={id} />
      <h1 id="not-found-heading" className="text-2xl font-semibold">
        Not found
      </h1>
      <p>
        No dataset with id{" "}
        <code className="rounded bg-neutral-100 px-1">{id}</code>.
      </p>
      <BackToUpload />
    </section>
  );
}

export default async function DatasetPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const { record, error } = await loadDataset(id);

  if (record?.status === "validating" || record?.status === "importing") {
    // An import passes through "importing" (its bytes are still being
    // fetched from a third party) before "validating"; an upload skips
    // straight to "validating". Both run in the background and both re-fetch
    // on a meta refresh until the report lands.
    return (
      <div className="space-y-0">
        <DatasetBreadcrumb label={record.filename} />
        <ValidationProgressView record={record} />
      </div>
    );
  }
  if (record) {
    // While the token count is being produced (issue #42) the report is
    // already complete, so this renders it with a counting indicator. The
    // count lands on its own: TokenCountPoll re-renders this route while the
    // user is still on it, and never fires once they have navigated on (a
    // meta-refresh cannot promise that -- it is scheduled at parse time and
    // fires even after a client-side navigation has left the page, which
    // yanked a launching user back to the report).
    const { jobs, error: jobsError } = await loadDatasetJobs(id);
    return (
      <div className="space-y-0">
        <DatasetBreadcrumb label={record.filename} />
        {record.token_count_status ===
          DatasetRecordTokenCountStatus.counting && (
          <TokenCountPoll datasetId={record.id} />
        )}
        <ReportView record={record} jobs={jobs} jobsError={jobsError} />
      </div>
    );
  }
  if (error?.status === 404) {
    return <NotFound id={id} />;
  }

  return (
    <section aria-labelledby="report-error-heading" className="space-y-4">
      <DatasetBreadcrumb label={id} />
      <h1 id="report-error-heading" className="text-2xl font-semibold">
        The report could not be loaded
      </h1>
      <p>
        <code className="rounded bg-neutral-100 px-1">{error?.code}</code> —{" "}
        {error?.message}
      </p>
      <BackToUpload />
    </section>
  );
}
