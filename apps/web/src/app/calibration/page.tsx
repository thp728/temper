// Route kept but delinked from primary navigation — see note at top of
// `src/components/CalibrationView.tsx` for why the aggregate was removed from
// the nav tree (observability, not end-user value) and what replaced it on the
// dashboard. The page still renders at /calibration for direct/internal access.

import type { Metadata } from "next";
import BackToUpload from "@/components/BackToUpload";
import CalibrationView from "@/components/CalibrationView";
import FocusHeading from "@/components/FocusHeading";
import { calibrationSummaryV1CalibrationGet } from "@/lib/api/generated/client";
import { load } from "@/lib/api/load";

export const metadata: Metadata = {
  title: "Predictions vs what happened",
};

// The aggregate reads every terminal job's frozen actuals at request time;
// it is never cached into a lie about what has been recorded so far.
export const dynamic = "force-dynamic";

export default async function CalibrationPage() {
  const { data, error } = await load(() => calibrationSummaryV1CalibrationGet());

  if (error || !data) {
    return (
      <section aria-labelledby="calibration-heading" className="space-y-4">
        <FocusHeading id="calibration-heading">
          The comparison could not be loaded
        </FocusHeading>
        <p>
          <code className="rounded bg-neutral-100 px-1">{error?.code}</code> —{" "}
          {error?.message}
        </p>
        <BackToUpload />
      </section>
    );
  }

  return <CalibrationView data={data} />;
}
