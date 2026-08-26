import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import JobRecordView from "@/components/JobRecordView";
import type { JobEvent, JobRecord } from "@/lib/api/generated/client";

// The assertions are what a user reads on a job's record: how it ended, what
// it produced, why it failed, and that a cancellation is a decision rather
// than a defect. The port's behaviour, pinned.

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    id: "job_abc123def456",
    dataset_id: "ds_xyz789",
    base_model: "qwen3-4b",
    base_revision: "a".repeat(40),
    status: "complete",
    cancel_requested: false,
    created_at: new Date(2025, 7, 26, 1, 0, 0).getTime() / 1000,
    started_at: new Date(2025, 7, 26, 1, 0, 5).getTime() / 1000,
    finished_at: new Date(2025, 7, 26, 1, 4, 35).getTime() / 1000,
    gpu_type: "L4",
    price_per_hour: 41.31,
    currency: "INR",
    warnings: [],
    result: { ok: true },
    ...overrides,
  };
}

function event(overrides: Partial<JobEvent> = {}): JobEvent {
  return {
    id: 1,
    job_id: "job_abc123def456",
    ts: new Date(2025, 7, 26, 1, 0, 5).getTime() / 1000,
    kind: "state",
    message: "queued",
    data: null,
    ...overrides,
  };
}

describe("JobRecordView", () => {
  it("shows the state with its spend and latest loss beside it", () => {
    render(
      <JobRecordView
        job={job()}
        datasetFilename="support-chats.jsonl"
        events={[
          event(),
          event({
            id: 2,
            kind: "metric",
            message: "{'loss': 0.6931, 'step': 10}",
            data: { loss: 0.6931, step: 10 },
          }),
        ]}
      />,
    );

    // Label/value pairs are read as pairs; several say "—" so match precisely.
    const value = (label: string) =>
      screen.getByText(label).nextElementSibling?.textContent;
    expect(value("State")).toBe("complete");
    expect(value("Elapsed")).toBe("4m 30s");
    expect(value("Machine")).toBe("L4 at 41.31 INR/hr");
    expect(value("Latest loss")).toContain("0.6931");
    expect(value("Latest loss")).toContain("step 10");

    // The job is identifiable: dataset by name (linking to its report), model
    // and pinned revision in short form.
    expect(
      screen.getByRole("link", { name: "support-chats.jsonl" }),
    ).toHaveAttribute("href", "/datasets/ds_xyz789");
    expect(screen.getByText("qwen3-4b")).toBeVisible();
    expect(screen.getByText("a".repeat(12))).toBeVisible();
  });

  it("offers a completed job's artifact through the download route", () => {
    render(<JobRecordView job={job()} events={[]} />);
    const link = screen.getByRole("link", { name: /Download the adapter/ });
    expect(link).toHaveAttribute("href", "/v1/jobs/job_abc123def456/adapter");
    // The zip promises the file that makes it loadable.
    expect(screen.getByText(/adapter_config\.json/)).toBeVisible();
  });

  it("says when training finished but no adapter could be retrieved", () => {
    render(
      <JobRecordView job={job({ result: null })} events={[]} />,
    );
    expect(screen.getByText(/no adapter could be retrieved/i)).toBeVisible();
    expect(screen.queryByRole("link", { name: /Download the adapter/ })).toBeNull();
  });

  it("shows a failure's stable code with its reason in plain language", () => {
    render(
      <JobRecordView
        job={job({
          status: "failed",
          error_code: "gpu_stalled",
          error_message:
            "No output for 900.0s (stall limit); machine destroyed.",
          result: null,
        })}
        events={[
          event({ message: "[00:00:00] building trainer image", kind: "log" }),
        ]}
      />,
    );

    expect(screen.getByText("gpu_stalled")).toBeVisible();
    expect(screen.getByText(/stopped by a safety limit/)).toBeVisible();
    expect(
      screen.getByText(/No output for 900\.0s \(stall limit\)/),
    ).toBeVisible();
    // What the job was doing before it died stays readable.
    expect(screen.getByText("[00:00:00] building trainer image")).toBeVisible();
    expect(screen.queryByRole("link", { name: /Download the adapter/ })).toBeNull();
  });

  it("presents a cancellation as a decision rather than a defect", () => {
    render(
      <JobRecordView
        job={job({ status: "cancelled", finished_at: 1756162800, result: null })}
        events={[event({ message: "Cancelled at your request." })]}
      />,
    );

    // The decision is stated in the outcome section...
    expect(
      screen.getByText(/that is what cancelling means here/i),
    ).toBeVisible();
    // ...and the history records it in the same words the run recorded.
    expect(screen.getByRole("log")).toHaveTextContent(
      "Cancelled at your request.",
    );
    // No error code, no destructive framing: nothing on the page may read
    // the user's own decision as something that went wrong.
    expect(screen.queryByText(/^failed/i)).toBeNull();
    expect(screen.queryByText("gpu_stalled")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("keeps the full history of a finished job readable", () => {
    render(
      <JobRecordView
        job={job()}
        events={[
          event({ message: "queued" }),
          event({ id: 2, message: "Selecting a GPU" }),
        ]}
      />,
    );
    expect(screen.getByRole("log")).toHaveTextContent("queued");
    expect(screen.getByRole("log")).toHaveTextContent("Selecting a GPU");
  });

  it("renders without scripting once the job is terminal", () => {
    const { container } = render(<JobRecordView job={job()} events={[]} />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector('meta[http-equiv="refresh"]')).toBeNull();
  });

  it("re-renders itself while the job is still working, and shows it as it is", () => {
    render(
      <JobRecordView
        job={job({ status: "training", finished_at: null, result: null })}
        events={[]}
      />,
    );
    // The no-JavaScript equivalent of the old page's poll-and-reload: a
    // returned visitor does not read a stale state. React 19 hoists the tag
    // into the document head, where a browser honours it.
    const refresh = document.querySelector('meta[http-equiv="refresh"]');
    expect(refresh).toHaveAttribute("content", "2");
    expect(screen.getByText("State").nextElementSibling?.textContent).toBe(
      "training",
    );
    // A working job offers no download yet.
    expect(
      screen.queryByRole("link", { name: /Download the adapter/ }),
    ).toBeNull();
  });
});
