import type { Metadata } from "next";
import Link from "next/link";
import BackToUpload from "@/components/BackToUpload";
import FocusHeading from "@/components/FocusHeading";
import LaunchForm from "@/components/LaunchForm";
import { Button } from "@/components/ui/button";
import {
  getJobSpecPreviewV1JobsSpecGet,
  listModelsV1ModelsGet,
} from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

export const metadata: Metadata = {
  title: "Choose a base model",
};

// The shape both fetches share when they fail: the data is null and the
// refusal is typed, exactly as the mutator raises it.
async function load<T>(fetch: () => Promise<T>): Promise<{
  data: T | null;
  error: ApiError | null;
}> {
  try {
    return { data: await fetch(), error: null };
  } catch (err) {
    return {
      data: null,
      error: err instanceof ApiError ? err : NETWORK_ERROR,
    };
  }
}

// A refusal keeps its stable code on the page -- the same contract every
// screen honours, never laundered into a generic "something went wrong".
function Refusal({
  heading,
  code,
  message,
}: {
  heading: string;
  code?: string;
  message: string;
}) {
  return (
    <section aria-labelledby="launch-heading" className="space-y-4">
      <FocusHeading id="launch-heading">{heading}</FocusHeading>
      <p>
        <code className="rounded bg-neutral-100 px-1">{code}</code> — {message}
      </p>
      <BackToUpload />
    </section>
  );
}

export default async function NewJobPage({
  searchParams,
}: {
  searchParams: Promise<{ dataset_id?: string }>;
}) {
  const { dataset_id: datasetId } = await searchParams;
  if (!datasetId) {
    return (
      <Refusal
        heading="Choose a base model"
        code="no_dataset"
        message="No dataset was given. Upload one first — its validation report decides whether a job can start at all."
      />
    );
  }

  const [{ data: catalog, error: catalogError }, { data: preview, error: previewError }] =
    await Promise.all([
      load(() => listModelsV1ModelsGet()),
      load(() => getJobSpecPreviewV1JobsSpecGet({ dataset_id: datasetId })),
    ]);

  if (previewError || !preview) {
    // A dataset that cannot start a job is refused before anything can be
    // committed, with the same stable code and message the launch itself
    // would raise.
    return (
      <Refusal
        heading="Dataset not usable"
        code={previewError?.code}
        message={previewError?.message ?? "The dataset could not be checked."}
      />
    );
  }

  if (catalogError || !catalog) {
    return (
      <Refusal
        heading="The catalog could not be loaded"
        code={catalogError?.code}
        message={catalogError?.message ?? "The model catalog is unavailable."}
      />
    );
  }

  const usableRows = preview.dataset.report?.usable_rows;

  return (
    <section aria-labelledby="launch-heading" className="space-y-6">
      <div className="space-y-2">
        <FocusHeading id="launch-heading">Choose a base model</FocusHeading>
        <p className="text-muted-foreground">
          For dataset{" "}
          <span className="font-medium text-foreground">
            {preview.dataset.filename}
          </span>
          {usableRows !== null && usableRows !== undefined && (
            <> ({usableRows} usable rows)</>
          )}
          . Everything you are about to commit to is on this page; nothing has
          been spent yet.
        </p>
      </div>

      {preview.warning && (
        // The feasibility estimate reaches the user here rather than after
        // the money starts: this is the last moment they can still act on it.
        <div
          role="alert"
          className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-amber-900"
        >
          <p className="font-semibold">Before you launch</p>
          <p>
            <code className="rounded bg-amber-100 px-1 text-xs">
              {preview.warning.code}
            </code>
          </p>
          <p className="mt-1 text-sm">{preview.warning.message}</p>
        </div>
      )}

      <LaunchForm catalog={catalog} preview={preview} />

      {/* The old screen's way back: the report this launch was reached from. */}
      <Button variant="outline" asChild>
        <Link href={`/datasets/${preview.dataset.id}`}>
          Back to the validation report
        </Link>
      </Button>
    </section>
  );
}
