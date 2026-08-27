import type { Metadata } from "next";
import BackToUpload from "@/components/BackToUpload";
import ReportView from "@/components/ReportView";
import ValidationProgressView from "@/components/ValidationProgressView";
import { getDatasetV1DatasetsDatasetIdGet } from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

export const metadata: Metadata = {
  title: "Validation report",
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

function NotFound({ id }: { id: string }) {
  return (
    <section aria-labelledby="not-found-heading" className="space-y-4">
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

  if (record?.status === "validating") {
    // Validation runs in the background; this view shows its progress and
    // re-fetches on a meta refresh until the report lands.
    return <ValidationProgressView record={record} />;
  }
  if (record) {
    return <ReportView record={record} />;
  }
  if (error?.status === 404) {
    return <NotFound id={id} />;
  }

  return (
    <section aria-labelledby="report-error-heading" className="space-y-4">
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
