import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { JobEvent } from "@/lib/api/generated/client";

// Instability is a warning rather than an abort (issue #36): the same
// exceedance that would become a divergence after 20 steps is surfaced at 5
// steps as a banner that does not stop the run. Shared by the running view
// and the finished record so the wording -- and whether it shows at all --
// cannot drift between the two: a run that warned while live still shows it
// was warned about once the record is read after the fact.
export default function InstabilityBanner({ events }: { events: JobEvent[] }) {
  const warnings = events.filter(
    (e) =>
      e.data &&
      (e.data["code"] === "training_instability" || e.data["warning"] === true),
  );
  if (warnings.length === 0) return null;
  return (
    <Alert>
      <AlertTitle>Training instability</AlertTitle>
      <AlertDescription>
        <p>
          Loss is spiking well above its recent average. This is shown as a
          warning rather than an abort, because it may be early divergence.
          Consider lowering the learning rate if it continues.
        </p>
        <p className="text-xs text-muted-foreground">
          {warnings[warnings.length - 1]?.message}
        </p>
      </AlertDescription>
    </Alert>
  );
}
