import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import DashboardView from "@/components/DashboardView";
import type { JobRecord } from "@/lib/api/generated/client";

// The dashboard (the rework's home page): what a user lands on sees active
// jobs with their own status vocabulary and elapsed time, recently finished
// jobs with their outcome and frozen actuals, and spend & time across
// finished jobs. The assertions are what a person reads, never markup structure.

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    id: "job_abc123def456",
    dataset_id: "ds_xyz789",
    base_model: "qwen3-4b",
    status: "complete",
    cancel_requested: false,
    created_at: 1000,
    warnings: [],
    ...overrides,
  };
}

// Four hours and twelve minutes after the fixture's `created_at`, so an
// active job's elapsed time is a known, assertable value.
const NOW = 1000 + 4 * 3600 + 12 * 60;

describe("DashboardView", () => {
  it("shows what is active, what finished, and what the record rests on", () => {
    render(
      <DashboardView
        now={NOW}
        jobs={[
          job({
            id: "job_active01",
            status: "training",
            gpu_type: "l4",
          }),
          job({
            id: "job_done02",
            created_at: 2000,
            status: "complete",
            actuals: {
              duration_s: 600,
              cost_minor: 400,
              currency: "INR",
            },
            quote: { minor_unit: 100 } as JobRecord["quote"],
          }),
        ]}
      />,
    );

    // The active job: its id links to the record, and the status vocabulary
    // is the real one -- the dashboard never invents a "COMPLETED" badge.
    const activeLink = screen.getByRole("link", { name: "job_active01" });
    expect(activeLink).toHaveAttribute("href", "/jobs/job_active01");
    expect(screen.getByText("training")).toBeVisible();
    expect(screen.getByText("l4")).toBeVisible();
    // Elapsed is derived from started_at/created_at, stated in the clock.
    expect(screen.getByText("4h 12m 00s")).toBeVisible();

    // The finished job reads its measured duration and derived cost from the
    // actuals, with the currency beside the amount.
    const doneLink = screen.getByRole("link", { name: "job_done02" });
    expect(doneLink).toHaveAttribute("href", "/jobs/job_done02");
    expect(screen.getByText("complete")).toBeVisible();
    // Cost appears in the finished table and in spend & time (total + avg per
    // job); for a single job all three are INR 4.00.
    expect(screen.getAllByText("INR 4.00").length).toBeGreaterThanOrEqual(3);

    // The spend & time glimpse aggregates finished jobs.
    expect(screen.getByRole("heading", { name: "Spend & time" })).toBeVisible();
    expect(screen.getByText(/Based on 1 finished job/)).toBeVisible();
    expect(screen.getByText("Total spent")).toBeVisible();
    expect(screen.getByText("Avg spend per job")).toBeVisible();
    expect(screen.getByText("Average duration")).toBeVisible();
    expect(screen.getByText("Success rate")).toBeVisible();

    // The primary action starts a job the honest way: from a dataset.
    expect(
      screen.getByRole("link", { name: "Start a new job" }),
    ).toHaveAttribute("href", "/datasets");
  });

  it("shows a failed or cancelled job's outcome without its stable code", () => {
    render(
      <DashboardView
        now={NOW}
        jobs={[
          job({
            id: "job_failed01",
            status: "failed",
            error_code: "gpu_stalled",
            actuals: null,
          }),
          job({
            id: "job_cancelled02",
            status: "cancelled",
            result: null,
            actuals: null,
          }),
        ]}
      />,
    );

    // The outcome shows here; the stable code is the job's own record's
    // detail, not repeated on the dashboard.
    expect(screen.getByText("failed")).toBeVisible();
    expect(screen.queryByText("gpu_stalled")).not.toBeInTheDocument();
    expect(screen.getByText("cancelled")).toBeVisible();
  });

  it("reads sensibly on a fresh install, telling the user what to do next", () => {
    render(<DashboardView now={NOW} jobs={[]} />);

    // The empty states are panels -- one look on every screen: a heading,
    // one line of supporting copy, and, where the section is the page's own
    // next step, the action beneath.
    expect(
      screen.getByRole("heading", { name: "No active jobs" }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "No finished jobs yet" }),
    ).toBeVisible();
    expect(screen.getByText(/A job appears here once it completes/)).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "No spend yet" }),
    ).toBeVisible();
    expect(screen.getByText(/actual cost and duration appear here/)).toBeVisible();

    // Every empty state points at the way forward. The active-jobs panel's
    // own button shares its name with the header's -- same action, same
    // route -- so both are asserted together.
    const startLinks = screen.getAllByRole("link", {
      name: "Start a new job",
    });
    expect(startLinks).toHaveLength(2);
    for (const link of startLinks) {
      expect(link).toHaveAttribute("href", "/datasets");
    }
    // Calibration is no longer linked from the dashboard (kept at /calibration
    // for direct access; see CalibrationView.tsx top comment).
    expect(
      screen.queryByRole("link", { name: "Full comparison" }),
    ).not.toBeInTheDocument();
  });

  it("aggregates avg spend per job across finished jobs", () => {
    render(
      <DashboardView
        now={NOW}
        jobs={[
          job({
            id: "job_done01",
            status: "complete",
            actuals: { duration_s: 600, cost_minor: 400, currency: "INR" },
            quote: { minor_unit: 100 } as JobRecord["quote"],
          }),
          job({
            id: "job_done02",
            status: "complete",
            actuals: { duration_s: 300, cost_minor: 200, currency: "INR" },
            quote: { minor_unit: 100 } as JobRecord["quote"],
          }),
        ]}
      />,
    );

    // Total 600 minor = INR 6.00, avg 300 minor = INR 3.00, avg duration 450s = 7m 30s
    expect(screen.getByText("INR 6.00")).toBeVisible();
    expect(screen.getByText("INR 3.00")).toBeVisible();
    expect(screen.getByText("7m 30s")).toBeVisible();
    expect(screen.getByText(/Based on 2 finished jobs/)).toBeVisible();
  });

  it("paginates active jobs at 5 per page", async () => {
    const user = userEvent.setup();
    // 12 active jobs, newest first = highest created_at.
    // created_at 1000 + i ensures job_active11 is newest.
    const activeJobs = Array.from({ length: 12 }, (_, i) =>
      job({
        id: `job_active${String(i).padStart(2, "0")}`,
        status: "training",
        created_at: 1000 + i * 100,
        gpu_type: "l4",
      }),
    );
    render(<DashboardView now={NOW} jobs={activeJobs} />);

    // Page 1: newest 5 = 11,10,09,08,07
    expect(screen.getByTestId("active-jobs-pagination-info")).toHaveTextContent(
      "Showing 1–5 of 12 · Page 1 of 3",
    );
    expect(screen.getByRole("link", { name: "job_active11" })).toBeVisible();
    expect(screen.getByRole("link", { name: "job_active07" })).toBeVisible();
    expect(screen.queryByRole("link", { name: "job_active06" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Previous page" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled();

    // Page 2
    await user.click(screen.getByRole("button", { name: "Next page" }));
    expect(screen.getByTestId("active-jobs-pagination-info")).toHaveTextContent(
      "Showing 6–10 of 12 · Page 2 of 3",
    );
    expect(screen.getByRole("link", { name: "job_active06" })).toBeVisible();
    expect(screen.getByRole("link", { name: "job_active02" })).toBeVisible();
    expect(screen.queryByRole("link", { name: "job_active11" })).not.toBeInTheDocument();

    // Page 3: last 2
    await user.click(screen.getByRole("button", { name: "Next page" }));
    expect(screen.getByTestId("active-jobs-pagination-info")).toHaveTextContent(
      "Showing 11–12 of 12 · Page 3 of 3",
    );
    expect(screen.getByRole("link", { name: "job_active01" })).toBeVisible();
    expect(screen.getByRole("link", { name: "job_active00" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled();

    // Back to page 2
    await user.click(screen.getByRole("button", { name: "Previous page" }));
    expect(screen.getByTestId("active-jobs-pagination-info")).toHaveTextContent(
      "Showing 6–10 of 12 · Page 2 of 3",
    );
  });

  it("does not show pagination when 5 or fewer active jobs", () => {
    const five = Array.from({ length: 5 }, (_, i) =>
      job({ id: `job_active0${i}`, status: "training", created_at: 1000 + i }),
    );
    render(<DashboardView now={NOW} jobs={five} />);
    expect(screen.queryByTestId("active-jobs-pagination-info")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Next page" })).not.toBeInTheDocument();
  });
});
