import type { Metadata } from "next";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import DatasetsView from "@/components/DatasetsView";
import { listDatasetsV1DatasetsGet } from "@/lib/api/generated/client";
import { load } from "@/lib/api/load";
import { epochNow } from "@/lib/jobs/display";

export const metadata: Metadata = {
  title: "Datasets",
};

export const dynamic = "force-dynamic";

// The dataset manager: upload/import JSONL and see previously used datasets.
// Server-fetches the list (jobs/AGENTS.md: never cached into a lie) and hands
// it to the client view for search filtering. The view keeps UploadForm and
// ImportForm visible with their existing accessible names so the e2e journeys
// (upload-journey.spec.ts) still find getByLabelText("Dataset file (.jsonl)")
// and getByLabelText("Public repository").
export default async function UploadPage() {
  const { data, error } = await load(() => listDatasetsV1DatasetsGet());
  const datasets = [...(data?.datasets ?? [])].sort((a, b) => b.created_at - a.created_at);

  return (
    <section aria-labelledby="upload-heading" className="space-y-8">
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbPage>Datasets</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      <DatasetsView datasets={datasets} loadError={error} now={epochNow()} />
    </section>
  );
}
