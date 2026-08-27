import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import CalibrationView from "@/components/CalibrationView";
import type { Calibration } from "@/lib/api/generated/client";

// The aggregate view (issue #77): predictions against measurements across
// runs. The assertions are what an operator reads -- how many runs a figure
// rests on, the mean ratio with its bounds (where a systematically wrong
// estimate shows up), the stage-by-stage reconciliation of the quote's phases
// with the job's stages, and the runs table that lets an outlier be named.

function calibration(): Calibration {
  return {
    count: 2,
    metrics: {
      duration: {
        count: 2,
        mean_predicted: 678.5,
        mean_actual: 300,
        mean_ratio: 0.44,
        min_ratio: 0.44,
        max_ratio: 0.44,
      },
      peak_memory: {
        count: 2,
        mean_predicted: 5.4,
        mean_actual: 5.31,
        mean_ratio: 0.98,
        min_ratio: 0.98,
        max_ratio: 0.98,
      },
      cost: {
        count: 2,
        mean_predicted: 779,
        mean_actual: 344,
        mean_ratio: 0.44,
        min_ratio: 0.44,
        max_ratio: 0.44,
      },
    },
    phases: [
      {
        name: "preparing",
        count: 2,
        mean_predicted: 54,
        mean_actual: 55,
        mean_ratio: 1.02,
        min_ratio: 1.02,
        max_ratio: 1.02,
        quotes_phases: ["readiness"],
      },
      {
        name: "training",
        count: 2,
        mean_predicted: 597,
        mean_actual: 210,
        mean_ratio: 0.35,
        min_ratio: 0.35,
        max_ratio: 0.35,
        quotes_phases: ["image_pull", "model_download", "training"],
      },
    ],
    runs: [
      {
        job_id: "job_a",
        base_model: "qwen3-4b",
        status: "complete",
        created_at: 1000,
        comparison: {
          duration: { actual: 300, ratio: 0.44 },
          peak_memory: { actual: 5.31, ratio: 0.98 },
          cost: { actual: 344, ratio: 0.44 },
        },
      },
    ],
  };
}

describe("CalibrationView", () => {
  it("states how many runs the comparison rests on", () => {
    render(<CalibrationView data={calibration()} />);
    expect(
      screen.getByText(/2 terminal runs have both a frozen quote/),
    ).toBeVisible();
    // Every card reports its own count: "calibrated against N real runs" is
    // only as honest as N is visible.
    expect(screen.getAllByText("Runs compared")).toHaveLength(3);
    expect(screen.getAllByText("2").length).toBeGreaterThan(0);
  });

  it("surfaces a systematically wrong estimate rather than absorbing it", () => {
    render(<CalibrationView data={calibration()} />);
    // Duration ratio 0.44 -> the estimate over-predicted by over 2x, stated
    // out loud rather than folded into a better-looking average.
    expect(
      screen.getAllByText(/the estimate over-predicted/i).length,
    ).toBeGreaterThan(0);
    // The ratio bounds are shown, so an outlier is a visible point in the
    // range, not a number hidden in the mean.
    expect(screen.getAllByText("0.44× – 0.44×").length).toBeGreaterThan(0);
  });

  it("shows the stage-by-stage reconciliation of the two vocabularies", () => {
    render(<CalibrationView data={calibration()} />);
    expect(screen.getByRole("heading", { name: "Stage by stage" })).toBeVisible();
    expect(screen.getAllByText(/preparing/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/readiness/).length).toBeGreaterThan(0);
    expect(
      screen.getByText(/image_pull \+ model_download \+ training/),
    ).toBeVisible();
  });

  it("names each run so an outlier can be pointed at", () => {
    render(<CalibrationView data={calibration()} />);
    const row = screen.getByRole("link", { name: "job_a" });
    expect(row).toHaveAttribute("href", "/jobs/job_a");
    expect(screen.getByText("qwen3-4b")).toBeVisible();
  });

  it("says plainly when there is nothing to compare yet", () => {
    render(
      <CalibrationView
        data={{ count: 0, metrics: {}, phases: [], runs: [] }}
      />,
    );
    expect(screen.getByText(/No runs to compare yet/)).toBeVisible();
    expect(screen.queryByRole("table")).toBeNull();
  });
});
