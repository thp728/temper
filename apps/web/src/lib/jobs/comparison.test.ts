import { describe, expect, it } from "vitest";
import {
  directionLabel,
  formatGigabytes,
  formatRatio,
  midpoint,
  rangeDirection,
  ratio,
} from "@/lib/jobs/comparison";

// The comparison helpers mirror temper_core.calibration (packages/core): the
// ratio is actual over predicted midpoint, the direction is where the actual
// lands on the predicted range. Both halves are tested against the same
// expectation, so a drift fails a test rather than silently changing what a
// reader is told.

describe("comparison", () => {
  it("midpoint is the middle of a predicted range", () => {
    expect(midpoint(10, 20)).toBe(15);
    expect(midpoint(null, 20)).toBeNull();
  });

  it("ratio is actual over predicted", () => {
    expect(ratio(10, 5)).toBe(2); // under-predicted by 2x
    expect(ratio(2.5, 5)).toBe(0.5); // over-predicted by 2x
    expect(ratio(null, 5)).toBeNull();
    expect(ratio(5, 0)).toBeNull();
  });

  it("direction is the landing of the actual on the range", () => {
    expect(rangeDirection(3, 10, 20)).toBe("under");
    expect(rangeDirection(15, 10, 20)).toBe("inside");
    expect(rangeDirection(30, 10, 20)).toBe("over");
    expect(rangeDirection(null, 10, 20)).toBeNull();
  });

  it("labels each direction in plain language", () => {
    expect(directionLabel("under")).toContain("under");
    expect(directionLabel("inside")).toContain("inside");
    expect(directionLabel("over")).toContain("overran");
    expect(directionLabel(null)).toContain("could not");
  });

  it("formats gigabytes and ratios for display", () => {
    expect(formatGigabytes(5.31)).toBe("5.31 GB");
    expect(formatGigabytes(null)).toBe("—");
    expect(formatRatio(1.234)).toBe("1.23×");
    expect(formatRatio(null)).toBe("—");
  });
});
