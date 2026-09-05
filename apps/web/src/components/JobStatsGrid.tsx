import { Clock, Cpu, TrendingDown, Wallet } from "lucide-react";
import BentoStat from "@/components/BentoStat";
import StatusPill, { statusIcon, statusIconSpins } from "@/components/StatusPill";
import { formatDuration, formatMinorCost, machineLabel } from "@/lib/jobs/display";
import type { JobRecord } from "@/lib/api/generated/client";

// The five-tile status row, identical on a job that is still running and one
// that has finished (issue: non-terminal/terminal parity): same tiles, same
// order, same labels. A job's page reloads from the live view into the
// finished record the moment it ends, and this is what keeps that reload
// from reading as a redesign -- Cost simply moves from "-" to a figure, and
// the State icon stops spinning, in place.
//
// `elapsedSeconds` and `loss` are computed by the caller rather than derived
// here: the running view's elapsed keeps ticking off a client interval and
// its loss comes from a live-appended stream, while the finished record's
// are both static reads of the durable history -- this component only knows
// how to lay a number out, not where it comes from.
export default function JobStatsGrid({
  job,
  elapsedSeconds,
  loss,
}: {
  job: JobRecord;
  elapsedSeconds: number;
  loss: { loss: number; step?: number } | null;
}) {
  const Icon = statusIcon(job.status);
  return (
    <section aria-labelledby="status-heading">
      <h2 id="status-heading" className="sr-only">
        Status
      </h2>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <BentoStat
          icon={Icon}
          iconClassName={
            statusIconSpins(job.status)
              ? "mb-2 size-4 animate-spin text-muted-foreground"
              : "mb-2 size-4 text-muted-foreground"
          }
          label="State"
          valueClassName="mt-2"
        >
          <strong id="job-state" aria-live="polite">
            <StatusPill status={job.status} />
          </strong>
        </BentoStat>
        <BentoStat icon={Clock} label="Elapsed">
          {formatDuration(elapsedSeconds)}
        </BentoStat>
        <BentoStat
          icon={Cpu}
          label="Machine"
          valueClassName="mt-1 text-lg font-semibold tracking-tight tabular-nums"
        >
          {machineLabel(job)}
        </BentoStat>
        <BentoStat
          icon={TrendingDown}
          label="Latest loss"
          valueClassName="mt-1 text-lg font-semibold tracking-tight tabular-nums"
        >
          {loss
            ? `${loss.loss}${loss.step !== undefined ? ` at step ${loss.step}` : ""}`
            : "—"}
        </BentoStat>
        <BentoStat icon={Wallet} label="Cost">
          {job.actuals?.cost_minor != null &&
          job.actuals.currency != null &&
          job.quote?.minor_unit != null
            ? formatMinorCost(
                job.actuals.cost_minor,
                job.actuals.currency,
                job.quote.minor_unit,
              )
            : "—"}
        </BentoStat>
      </div>
    </section>
  );
}
