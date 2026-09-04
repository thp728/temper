import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

// The page's clock, pinned: every render passes it, so the relative dates
// below read deterministically whatever wall-clock the suite runs under.
const NOW = Date.UTC(2026, 8, 4, 12, 0, 0) / 1000;

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
            actuals: {
              duration_s: 600,
              cost_minor: 400,
              currency: "INR",
            },
          }),
          job({
            id: "job_cancelled02",
            dataset_id: "ds_other42",
            status: "cancelled",
            result: null,
          }),
        ]}
        datasetNames={{ ds_xyz789: "support-chats.jsonl", ds_other42: "poems.jsonl" }} now={NOW}
      />,
    );

    const link = screen.getByRole("link", { name: "job_abc123def456" });
    expect(link).toHaveAttribute("href", "/jobs/job_abc123def456");
    // Status words also label the filter's options, so outcome assertions
    // scope to the table itself.
    const table = screen.getByRole("table");
    expect(within(table).getByText("complete")).toBeVisible();
    expect(within(table).getByText("failed")).toBeVisible();
    // A cancellation appears as its own outcome -- it is in no error code, so
    // its presence can only come from the outcome column.
    expect(within(table).getByText("cancelled")).toBeVisible();
    // A failure shows its outcome only -- the stable code is the job
    // record's detail, not repeated in the list.
    expect(within(table).queryByText("gpu_stalled")).not.toBeInTheDocument();
    // ...and each row identifies its job: model and dataset by name. The
    // model names also label the filter's menu items, so the row assertion
    // scopes to the table itself.
    expect(within(table).getByText("qwen3-8b")).toBeVisible();
    expect(screen.getByText("support-chats.jsonl")).toBeVisible();
    // Duration reads the frozen actuals as a clock; a job with none shows a dash.
    expect(screen.getByText("00:10:00")).toBeVisible();
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(1);
  });

  it("lists newest first even when the API returns another order", () => {
    render(
      <JobsView
        jobs={[
          job({ id: "job_old", created_at: 1000 }),
          job({ id: "job_new", created_at: 3000 }),
          job({ id: "job_mid", created_at: 2000 }),
        ]}
        datasetNames={{}} now={NOW}
      />,
    );

    const table = screen.getByRole("table");
    const links = within(table).getAllByRole("link");
    expect(links.map((l) => l.textContent)).toEqual([
      "job_new",
      "job_mid",
      "job_old",
    ]);
  });

  it("narrows the list through the search box", async () => {
    const user = userEvent.setup();
    render(
      <JobsView
        jobs={[
          job({ id: "job_alpha", base_model: "qwen3-4b" }),
          job({
            id: "job_beta",
            dataset_id: "ds_other42",
            base_model: "qwen3-8b",
          }),
        ]}
        datasetNames={{ ds_xyz789: "support-chats.jsonl", ds_other42: "poems.jsonl" }} now={NOW}
      />,
    );

    await user.type(screen.getByRole("searchbox", { name: "Search jobs" }), "beta");
    expect(
      screen.getByRole("link", { name: "job_beta" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("link", { name: "job_alpha" }),
    ).not.toBeInTheDocument();

    // A query matching nothing says so and offers the way back.
    await user.clear(screen.getByRole("searchbox", { name: "Search jobs" }));
    await user.type(
      screen.getByRole("searchbox", { name: "Search jobs" }),
      "zzz-no-such-job",
    );
    expect(
      screen.queryByRole("link", { name: "job_beta" }),
    ).not.toBeInTheDocument();

    // Clearing the search and filters brings every job back.
    await user.click(
      screen.getByRole("button", { name: "Clear search and filters" }),
    );
    expect(
      screen.getByRole("link", { name: "job_alpha" }),
    ).toBeVisible();
  });

  it("reorders the list through the column headers", async () => {
    const user = userEvent.setup();
    render(
      <JobsView
        jobs={[
          job({ id: "job_old", base_model: "qwen3-8b", created_at: 1000 }),
          job({ id: "job_new", base_model: "qwen3-4b", created_at: 3000 }),
        ]}
        datasetNames={{}} now={NOW}
      />,
    );

    const table = screen.getByRole("table");
    // Newest first out of the gate.
    expect(
      within(table).getAllByRole("link").map((l) => l.textContent),
    ).toEqual(["job_new", "job_old"]);

    // One click on Created flips to oldest first...
    await user.click(screen.getByRole("button", { name: "Created" }));
    expect(
      within(table).getAllByRole("link").map((l) => l.textContent),
    ).toEqual(["job_old", "job_new"]);

    // ...and another column reads A-Z.
    await user.click(screen.getByRole("button", { name: "Base model" }));
    expect(
      within(table).getAllByRole("link").map((l) => l.textContent),
    ).toEqual(["job_new", "job_old"]);
  });

  it("narrows the list to one status through the filter", async () => {
    const user = userEvent.setup();
    render(
      <JobsView
        jobs={[
          job({ id: "job_done", status: "complete" }),
          job({ id: "job_broken", status: "failed" }),
        ]}
        datasetNames={{}} now={NOW}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: "Filter by status" }),
    );
    await user.click(screen.getByRole("menuitem", { name: "failed" }));
    expect(
      screen.getByRole("link", { name: "job_broken" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("link", { name: "job_done" }),
    ).not.toBeInTheDocument();
  });

  it("narrows the list to one model through the filter", async () => {
    const user = userEvent.setup();
    render(
      <JobsView
        jobs={[
          job({ id: "job_small", base_model: "qwen3-4b" }),
          job({ id: "job_big", base_model: "qwen3-8b" }),
        ]}
        datasetNames={{}} now={NOW}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: "Filter by base model" }),
    );
    await user.click(screen.getByRole("menuitem", { name: "qwen3-8b" }));
    expect(
      screen.getByRole("link", { name: "job_big" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("link", { name: "job_small" }),
    ).not.toBeInTheDocument();
  });

  it("pages the list at ten rows per page", async () => {
    const user = userEvent.setup();
    const many = Array.from({ length: 12 }, (_, i) =>
      job({ id: `job_${String(i).padStart(2, "0")}`, created_at: 1000 + i }),
    );
    render(<JobsView jobs={many} datasetNames={{}} now={NOW} />);

    const table = screen.getByRole("table");
    // Newest first, ten to a page.
    expect(within(table).getAllByRole("link")).toHaveLength(10);
    expect(screen.getByTestId("jobs-pagination-info")).toHaveTextContent(
      "Showing 1–10 of 12 · Page 1 of 2",
    );

    await user.click(screen.getByRole("button", { name: "Next page" }));
    expect(within(table).getAllByRole("link")).toHaveLength(2);
    expect(screen.getByTestId("jobs-pagination-info")).toHaveTextContent(
      "Showing 11–12 of 12 · Page 2 of 2",
    );

    await user.click(screen.getByRole("button", { name: "Previous page" }));
    expect(within(table).getAllByRole("link")).toHaveLength(10);
  });

  it("falls back to the dataset id when the dataset row is gone", () => {
    render(<JobsView jobs={[job()]} datasetNames={{}} now={NOW} />);
    expect(screen.getByText("ds_xyz789")).toBeVisible();
  });

  it("says so when there are no jobs yet, and offers the way in", () => {
    render(<JobsView jobs={[]} datasetNames={{}} now={NOW} />);
    // The empty state is the shared panel: heading, copy, and the first
    // step as the action.
    expect(screen.getByRole("heading", { name: "No jobs yet" })).toBeVisible();
    expect(screen.getByText(/newest first, with its status beside it/)).toBeVisible();
    expect(
      screen.getByRole("link", { name: /select a dataset/i }),
    ).toHaveAttribute("href", "/datasets");
  });

  it("renders the list as a table with column headers", () => {
    render(<JobsView jobs={[job()]} datasetNames={{}} now={NOW} />);
    expect(screen.getByRole("table")).toBeVisible();
    for (const header of ["Job", "Status", "Base model", "Dataset", "Created", "Duration"]) {
      expect(screen.getByRole("columnheader", { name: header })).toBeVisible();
    }
  });
});
