import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import JobsView from "@/components/JobsView";
import type { JobRecord } from "@/lib/api/generated/client";

// The assertions are what a user reads: each job with its outcome beside it,
// a failure naming its code, and every entry leading to the full record.
// Never markup structure.

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    id: "job_abc123def456",
    dataset_id: "ds_xyz789",
    base_model: "qwen3-4b",
    status: "complete",
    cancel_requested: false,
    created_at: new Date(2025, 7, 26, 1, 2, 3).getTime() / 1000,
    warnings: [],
    ...overrides,
  };
}

describe("JobsView", () => {
  it("lists every job with its outcome and enough detail to identify it", () => {
    render(
      <JobsView
        jobs={[
          job(),
          job({
            id: "job_failed01",
            dataset_id: "ds_other42",
            base_model: "qwen3-8b",
            status: "failed",
            error_code: "gpu_stalled",
          }),
          job({
            id: "job_cancelled02",
            dataset_id: "ds_other42",
            status: "cancelled",
            result: null,
          }),
        ]}
        datasetNames={{ ds_xyz789: "support-chats.jsonl", ds_other42: "poems.jsonl" }}
      />,
    );

    const link = screen.getByRole("link", { name: "job_abc123def456" });
    expect(link).toHaveAttribute("href", "/jobs/job_abc123def456");
    expect(screen.getByText("complete")).toBeVisible();
    expect(screen.getByText("failed")).toBeVisible();
    // A cancellation appears as its own outcome -- it is in no error code, so
    // its presence can only come from the outcome column.
    expect(screen.getByText("cancelled")).toBeVisible();
    // A failure names its stable code beside its outcome...
    expect(screen.getByText("gpu_stalled")).toBeVisible();
    // ...and each row identifies its job: model and dataset by name.
    expect(screen.getByText("qwen3-8b")).toBeVisible();
    expect(screen.getByText("support-chats.jsonl")).toBeVisible();
  });

  it("falls back to the dataset id when the dataset row is gone", () => {
    render(<JobsView jobs={[job()]} datasetNames={{}} />);
    expect(screen.getByText("ds_xyz789")).toBeVisible();
  });

  it("says so when there are no jobs yet, and offers the way in", () => {
    render(<JobsView jobs={[]} datasetNames={{}} />);
    // The empty state is the shared panel: heading, copy, and the first
    // step as the action.
    expect(screen.getByRole("heading", { name: "No jobs yet" })).toBeVisible();
    expect(screen.getByText(/newest first, with its status beside it/)).toBeVisible();
    expect(
      screen.getByRole("link", { name: /select a dataset/i }),
    ).toHaveAttribute("href", "/datasets");
  });

  it("renders the list as a table with column headers", () => {
    render(<JobsView jobs={[job()]} datasetNames={{}} />);
    expect(screen.getByRole("table")).toBeVisible();
    for (const header of ["Job", "Status", "Base model", "Dataset", "Created"]) {
      expect(screen.getByRole("columnheader", { name: header })).toBeVisible();
    }
  });
});
