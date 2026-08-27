// The two loss series a running job charts (issue #53), extracted from the
// metric events of the durable log. Both ride the same stream, in the order
// they were recorded, and share one x-axis: the index of the measurement in
// that combined stream. That is deliberate -- the held-out rows do not carry a
// step number (the trainer prints eval rows by epoch), so the honest shared
// axis is "in the order the measurements were recorded", not a step the eval
// side does not have.

import type { JobEvent } from "@/lib/api/generated/client";

export interface MetricPoint {
  /** Index into the combined metric stream; the shared x-axis. */
  x: number;
  /** Training loss, when this point is a training measurement. */
  loss?: number;
  /** Held-out loss, when this point is an evaluation measurement. */
  heldOutLoss?: number;
  step?: number;
  epoch?: number;
}

// One metric stream, in the order the events were recorded. A metric event
// that carries no usable loss (a diverged run prints nan, which the backend
// deliberately does not promote) is not a point on the chart.
export function metricStream(events: JobEvent[]): MetricPoint[] {
  const out: MetricPoint[] = [];
  for (const e of events) {
    if (!e || e.kind !== "metric" || !e.data) continue;
    const loss = e.data["loss"];
    const heldOutLoss = e.data["held_out_loss"];
    if (typeof loss !== "number" && typeof heldOutLoss !== "number") continue;
    const step = e.data["step"];
    const epoch = e.data["epoch"];
    out.push({
      x: out.length,
      loss: typeof loss === "number" ? loss : undefined,
      heldOutLoss:
        typeof heldOutLoss === "number" ? heldOutLoss : undefined,
      step: typeof step === "number" ? step : undefined,
      epoch: typeof epoch === "number" ? epoch : undefined,
    });
  }
  return out;
}

// The latest measured held-out loss in a job's history, mirroring the shape
// `latestLoss` returns for training loss so the two stats read the same way.
export function latestHeldOutLoss(
  events: JobEvent[],
): { loss: number; epoch?: number } | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (!e || e.kind !== "metric" || !e.data) continue;
    const loss = e.data["held_out_loss"];
    if (typeof loss !== "number") continue;
    const epoch = e.data["epoch"];
    return { loss, epoch: typeof epoch === "number" ? epoch : undefined };
  }
  return null;
}
