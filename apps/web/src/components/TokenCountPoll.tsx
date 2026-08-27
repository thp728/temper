"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  DatasetRecordTokenCountStatus,
  getDatasetV1DatasetsDatasetIdGet,
} from "@/lib/api/generated/client";

// The token count (issue #42) is produced by a background phase, so the
// report shows it once it lands. The report must update without a full page
// reload, and without yanking a user who has already navigated on: a
// `<meta http-equiv="refresh">` fires even after a client-side navigation has
// left the page (the browser schedules it when the tag is parsed and does not
// cancel it), which pulled a user mid-launch back to the report. This poll
// lives inside the page's React tree, so when the user navigates away the
// component unmounts, the interval is cleared, and nothing can navigate the
// tab out from under the next page. While the user is still on the report it
// re-renders the current route on the same cadence the meta-refresh used, so
// the landed count (and the counting progress) appears on its own.
export default function TokenCountPoll({ datasetId }: { datasetId: string }) {
  const router = useRouter();

  useEffect(() => {
    const id = window.setInterval(async () => {
      try {
        const record = await getDatasetV1DatasetsDatasetIdGet(datasetId);
        if (
          record.token_count_status !== DatasetRecordTokenCountStatus.counting
        ) {
          window.clearInterval(id);
        }
        // Re-render the current route: the count (or its failure) is now on
        // the record, and the report swaps in whatever it holds.
        router.refresh();
      } catch {
        // A transient fetch failure is not a reason to wedge the report; the
        // next tick retries.
      }
    }, 2000);
    return () => window.clearInterval(id);
  }, [router, datasetId]);

  return null;
}
