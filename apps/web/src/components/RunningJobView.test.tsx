import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import RunningJobView from "@/components/RunningJobView";
import type { JobEvent, JobRecord } from "@/lib/api/generated/client";
import { ApiError } from "@/lib/api/mutator";

// The live view's wiring, pinned: what a user sees as a job runs, and what
// the two server calls it makes are. The generated client is mocked (the
// transport is proven by the journeys), and the stream is a fake EventSource
// so a test can push an event in and assert on what the user then sees.

const getJobMock = vi.hoisted(() => vi.fn());
const cancelJobMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api/generated/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/generated/client")>()),
  getJobV1JobsJobIdGet: getJobMock,
  cancelJobV1JobsJobIdCancelPost: cancelJobMock,
}));

// A fake EventSource standing in for jsdom, which has none. The component
// reads EventSource.CLOSED and builds the stream URL; both are exercised.
class FakeEventSource {
  static CLOSED = 2;
  static instances: FakeEventSource[] = [];
  url: string;
  readyState = 0;
  private listeners: Record<string, ((e: MessageEvent) => void)[]> = {};

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, cb: (e: MessageEvent) => void) {
    (this.listeners[type] ??= []).push(cb);
  }

  emit(type: string, data: unknown) {
    for (const cb of this.listeners[type] ?? []) {
      cb(new MessageEvent(type, { data: JSON.stringify(data) }));
    }
  }

  close() {
    this.readyState = FakeEventSource.CLOSED;
  }
}

function stream(): FakeEventSource {
  const es = FakeEventSource.instances.at(-1);
  if (!es) throw new Error("no EventSource was opened");
  return es;
}

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    id: "job_abc123def456",
    dataset_id: "ds_xyz789",
    base_model: "qwen3-4b",
    base_revision: "a".repeat(40),
    status: "training",
    cancel_requested: false,
    created_at: new Date(2025, 7, 26, 1, 0, 0).getTime() / 1000,
    started_at: new Date(2025, 7, 26, 1, 0, 5).getTime() / 1000,
    finished_at: null,
    gpu_type: null,
    price_per_hour: null,
    currency: "INR",
    warnings: [],
    result: null,
    ...overrides,
  };
}

function event(overrides: Partial<JobEvent> = {}): JobEvent {
  return {
    id: 1,
    job_id: "job_abc123def456",
    ts: new Date(2025, 7, 26, 1, 0, 5).getTime() / 1000,
    kind: "log",
    message: "queued",
    data: null,
    ...overrides,
  };
}

function renderView(
  overrides: { job?: JobRecord; events?: JobEvent[] } = {},
) {
  return render(
    <RunningJobView
      job={overrides.job ?? job()}
      events={overrides.events ?? [event({ id: 1, kind: "state", message: "training" })]}
      datasetFilename="support-chats.jsonl"
      streamAfter={1}
      initialElapsedSeconds={65}
    />,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  getJobMock.mockReset();
  cancelJobMock.mockReset();
  FakeEventSource.instances = [];
});

beforeEach(() => {
  vi.stubGlobal("EventSource", FakeEventSource);
});

describe("RunningJobView", () => {
  it("opens the stream from where the rendered history ended", () => {
    renderView();
    expect(stream().url).toBe("/v1/jobs/job_abc123def456/stream?after=1");
  });

  it("shows a running job's state, elapsed and latest loss beside it", () => {
    renderView({
      events: [
        event({ id: 1, kind: "state", message: "training" }),
        event({
          id: 2,
          kind: "metric",
          message: "{'loss': 0.6931, 'step': 10}",
          data: { loss: 0.6931, step: 10 },
        }),
      ],
    });

    const value = (label: string) =>
      screen.getByText(label).nextElementSibling?.textContent;
    expect(value("State")).toBe("training");
    expect(value("Elapsed")).toBe("1m 05s");
    expect(value("Machine")).toBe("—");
    expect(value("Latest loss")).toContain("0.6931");
    expect(value("Latest loss")).toContain("step 10");
    expect(screen.getByRole("log")).toHaveTextContent("training");
  });

  it("announces state changes to screen readers", () => {
    // The watch page's aria-live region, ported: a state change is announced
    // rather than only painted, so a screen-reader user is not left guessing
    // where the job is (spec 007, stories 18-19).
    renderView();
    const state = screen.getByText("training", { selector: "strong" });
    expect(state).toHaveAttribute("id", "job-state");
    expect(state).toHaveAttribute("aria-live", "polite");
  });

  it("appends streamed output without a refresh", async () => {
    renderView();
    stream().emit("job", event({ id: 2, kind: "log", message: "building image" }));
    await waitFor(() =>
      expect(screen.getByRole("log")).toHaveTextContent("building image"),
    );
  });

  it("shows the latest measured loss as it becomes available", async () => {
    renderView();
    stream().emit("job", event({
      id: 2,
      kind: "metric",
      message: "{'loss': 1.9042, 'step': 10}",
      data: { loss: 1.9042, step: 10 },
    }));
    await waitFor(() =>
      expect(
        screen.getByText("Latest loss").nextElementSibling,
      ).toHaveTextContent("1.9042"),
    );
  });

  it("refetches the record on a state transition and updates the status", async () => {
    getJobMock.mockResolvedValue(
      job({
        status: "preparing",
        gpu_type: "L4",
        price_per_hour: 41.31,
      }),
    );
    renderView();
    stream().emit("job", event({ id: 2, kind: "state", message: "Waiting for SSH" }));

    await waitFor(() =>
      expect(screen.getByText("preparing").closest("dd")).toHaveTextContent(
        "preparing",
      ),
    );
    // The provisioned machine appears as the record reports it.
    expect(
      screen.getByText("Machine").nextElementSibling,
    ).toHaveTextContent("L4 at 41.31 INR/hr");
    expect(getJobMock).toHaveBeenCalledWith("job_abc123def456");
  });

  it("offers a cancel control that is clearly destructive", () => {
    renderView();
    const button = screen.getByRole("button", { name: "Cancel job" });
    expect(button).toBeEnabled();
    expect(
      screen.getByText(/no adapter will be produced/i),
    ).toBeVisible();
    expect(screen.getByText(/cannot be undone/i)).toBeVisible();
  });

  it("records the cancellation request and stops offering it twice", async () => {
    cancelJobMock.mockResolvedValue({});
    renderView();
    await userEvent.click(screen.getByRole("button", { name: "Cancel job" }));

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Cancellation requested" }),
      ).toBeDisabled(),
    );
    expect(cancelJobMock).toHaveBeenCalledWith("job_abc123def456");
  });

  it("shows a refused cancellation with its stable code", async () => {
    cancelJobMock.mockRejectedValue(
      new ApiError(409, "job_already_terminal", "Job is already 'complete'."),
    );
    renderView();
    await userEvent.click(screen.getByRole("button", { name: "Cancel job" }));

    await waitFor(() =>
      expect(screen.getByText("job_already_terminal")).toBeVisible(),
    );
    expect(screen.getByText(/already 'complete'/)).toBeVisible();
  });

  it("hands back to the record when a state event confirms a terminal status", async () => {
    const reload = vi.fn();
    Object.defineProperty(window, "location", {
      value: { reload },
      configurable: true,
    });
    getJobMock.mockResolvedValue(
      job({ status: "complete", finished_at: new Date().getTime() / 1000 }),
    );
    renderView();
    stream().emit("job", event({ id: 2, kind: "state", message: "Training complete" }));
    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
  });

  it("hands back to the record when the stream delivers its end marker", () => {
    const reload = vi.fn();
    Object.defineProperty(window, "location", {
      value: { reload },
      configurable: true,
    });
    renderView();
    stream().emit("end", {});
    expect(reload).toHaveBeenCalledTimes(1);
  });
});
