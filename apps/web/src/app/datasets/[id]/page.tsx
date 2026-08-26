import type { Metadata } from "next";
import Link from "next/link";
import ReportView from "@/components/ReportView";
import { getDatasetV1DatasetsDatasetIdGet } from "@/lib/api/generated/client";
import { ApiError } from "@/lib/api/mutator";

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
        : new ApiError(0, "network_error", "Could not reach Temper.");
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
      <Link
        href="/"
        className="rounded-md border border-neutral-300 bg-white px-4 py-2 text-sm font-medium hover:bg-neutral-100"
      >
        Back to upload
      </Link>
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

  if (record) {
    return <ReportView record={record} />;
  }
  if (error?.status === 404) {
    return <NotFound id={id} />;
  }

  return (
    <section aria-labelledby="error-heading" className="space-y-4">
      <h1 id="error-heading" className="text-2xl font-semibold">
        Something went wrong
      </h1>
      <p>
        <code className="rounded bg-neutral-100 px-1">{error?.code}</code> —{" "}
        {error?.message}
      </p>
      <Link
        href="/"
        className="rounded-md border border-neutral-300 bg-white px-4 py-2 text-sm font-medium hover:bg-neutral-100"
      >
        Back to upload
      </Link>
    </section>
  );
}
