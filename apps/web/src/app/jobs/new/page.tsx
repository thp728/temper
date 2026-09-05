import type { Metadata } from "next";
import Link from "next/link";
import BackToUpload from "@/components/BackToUpload";
import FocusHeading from "@/components/FocusHeading";
import NewJobWizard from "@/components/NewJobWizard";
import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSlashSeparator,
} from "@/components/ui/breadcrumb";
import {
  getAdvancedSurfaceV1SurfaceGet,
  getJobSpecPreviewV1JobsSpecGet,
  listDatasetsV1DatasetsGet,
  listModelsV1ModelsGet,
} from "@/lib/api/generated/client";
import { load } from "@/lib/api/load";

export const metadata: Metadata = {
  title: "New job",
};

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

  // /jobs/new always shows the wizard: with a dataset id the preview arrives
  // server-rendered, without one the Sources step picks it and re-requests
  // the preview from there. Either way the URL stays the source of truth.
  const previewRequest = datasetId
    ? load(() => getJobSpecPreviewV1JobsSpecGet({ dataset_id: datasetId }))
    : Promise.resolve({ data: null, error: null });

  const [
    { data: catalog, error: catalogError },
    { data: preview, error: previewError },
    { data: surface, error: surfaceError },
    { data: datasetList },
  ] = await Promise.all([
    load(() => listModelsV1ModelsGet()),
    previewRequest,
    load(() => getAdvancedSurfaceV1SurfaceGet()),
    load(() => listDatasetsV1DatasetsGet()),
  ]);

  if (datasetId && (previewError || !preview)) {
    // A dataset that cannot start a job is refused before anything can be
    // committed, with the same stable code and message the launch itself
    // would raise — and a way out that stays in the flow.
    return (
      <section aria-labelledby="launch-heading" className="space-y-4">
        <FocusHeading id="launch-heading">Dataset not usable</FocusHeading>
        <p>
          <code className="rounded bg-neutral-100 px-1">
            {previewError?.code}
          </code>{" "}
          — {previewError?.message ?? "The dataset could not be checked."}
        </p>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" asChild>
            <Link href="/jobs/new">Choose a different dataset</Link>
          </Button>
          <BackToUpload />
        </div>
      </section>
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

  const datasets = [...(datasetList?.datasets ?? [])].sort(
    (a, b) => b.created_at - a.created_at,
  );

  return (
    <section aria-labelledby="launch-heading" className="space-y-6">
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink asChild>
              <Link href="/jobs">Jobs</Link>
            </BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSlashSeparator />
          <BreadcrumbItem>
            <BreadcrumbPage>New job</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      <div className="space-y-2">
        <FocusHeading id="launch-heading">New job</FocusHeading>
        <p className="text-muted-foreground">
          Four steps — sources, hyperparameters, hardware, review.
        </p>
      </div>

      <NewJobWizard
        catalog={catalog}
        preview={preview ?? null}
        // The advanced surface is generated from the trainer's own schema and
        // published through the contract. A load failure does not block the
        // launch -- the job is still offered with the defaults, which is what
        // a first-time user gets anyway (issue #80).
        surface={surfaceError || !surface ? null : surface}
        // The models admitted from outside the catalog, each with its persisted
        // probe result shown beside it (issue #58).
        admitted={catalog.admitted ?? []}
        // The datasets the Sources step offers beside the current one, so
        // switching never leaves the screen.
        datasets={datasets}
      />
    </section>
  );
}
