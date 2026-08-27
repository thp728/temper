import { describe, expect, it } from "vitest";
import { latestHeldOutLoss, metricStream } from "@/lib/jobs/loss";
import type { JobEvent } from "@/lib/api/generated/client";

// The two series a running job charts (issue #53), pulled out of the same
// durable event stream the training loss comes from. These pin the extraction:
// training and held-out measurements share one x-axis, eval rows are not
// mistaken for training rows, and the latest held-out value is the newest
// measurement of it whatever else arrived after.

function metric(
  data: Record<string, unknown>,
  overrides: Partial<JobEvent> = {},
): JobEvent {
  return {
    id: 1,
    job_id: "job_x",
    ts: 0,
    kind: "metric",
    message: "",
    data,
    ...overrides,
  };
}

describe("metricStream", () => {
  it("builds one stream in the order the events were recorded", () => {
    const stream = metricStream([
      metric({ loss: 1.9, step: 1 }),
      metric({ held_out_loss: 0.52, epoch: 0.5 }),
      metric({ loss: 1.2, step: 2 }),
    ]);
    expect(stream).toEqual([
      { x: 0, loss: 1.9, step: 1 },
      { x: 1, heldOutLoss: 0.52, epoch: 0.5 },
      { x: 2, loss: 1.2, step: 2 },
    ]);
  });

  it("does not let a held-out row leak into the training series", () => {
    const stream = metricStream([metric({ held_out_loss: 0.5, epoch: 1 })]);
    expect(stream[0]!.loss).toBeUndefined();
    expect(stream[0]!.heldOutLoss).toBe(0.5);
  });

  it("skips metric events that carry no usable loss", () => {
    const stream = metricStream([
      metric({}),
      metric({ loss: 1.0 }),
      metric({ step: 3 }),
    ]);
    expect(stream).toEqual([{ x: 0, loss: 1.0 }]);
  });

  it("ignores non-metric events entirely", () => {
    const stream = metricStream([
      metric({ loss: 1.0 }),
      { ...metric({ loss: 2.0 }), kind: "log", data: { loss: 2.0 } },
    ]);
    expect(stream).toHaveLength(1);
  });
});

describe("latestHeldOutLoss", () => {
  it("returns the newest held-out measurement", () => {
    const events = [
      metric({ held_out_loss: 0.6, epoch: 0.5 }, { id: 2 }),
      metric({ loss: 1.2, step: 10 }, { id: 3 }),
      metric({ held_out_loss: 0.5, epoch: 1 }, { id: 4 }),
    ];
    expect(latestHeldOutLoss(events)).toEqual({ loss: 0.5, epoch: 1 });
  });

  it("returns null when no held-out loss has been measured", () => {
    expect(latestHeldOutLoss([metric({ loss: 1.2, step: 1 })])).toBeNull();
  });
});
