import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import RunningJobView from "@/components/RunningJobView";
import type {
  JobEvent,
  JobOutputLine,
  JobProgress,
  JobRecord,
} from "@/lib/api/generated/client";
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
  overrides: {
    job?: JobRecord;
    events?: JobEvent[];
    progress?: JobProgress[];
    output?: JobOutputLine[];
  } = {},
) {
  return render(
    <RunningJobView
      job={overrides.job ?? job()}
      events={overrides.events ?? [event({ id: 1, kind: "state", message: "training" })]}
      progress={overrides.progress ?? []}
      output={overrides.output ?? []}
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
    expect(value("Elapsed")).toBe("00:01:05");
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

  it("shows the latest held-out loss as it becomes available", async () => {
    renderView();
    stream().emit("job", event({
      id: 2,
      kind: "metric",
      message: "{'eval_loss': 0.52, 'epoch': 0.5}",
      data: { held_out_loss: 0.52, epoch: 0.5 },
    }));
    await waitFor(() =>
      expect(
        screen.getByText("Latest held-out loss").nextElementSibling,
      ).toHaveTextContent("0.52"),
    );
    expect(
      screen.getByText("Latest held-out loss").nextElementSibling,
    ).toHaveTextContent("epoch 0.5");
  });

  it("charts training loss against step and held-out loss against epoch, each on its own plot", () => {
    const metric = (
      id: number,
      data: Record<string, number>,
    ) => event({
      id,
      kind: "metric",
      message: JSON.stringify(data),
      data,
    });
    renderView({
      events: [
        event({ id: 1, kind: "state", message: "training" }),
        metric(2, { loss: 1.9, step: 1 }),
        metric(3, { held_out_loss: 0.7, epoch: 0.5 }),
        metric(4, { loss: 0.9, step: 2 }),
        metric(5, { held_out_loss: 0.6, epoch: 1.0 }),
      ],
    });
    // Two plots, each with its own axis and its own line drawn -- not one
    // shared chart with two series forced onto the same x.
    const trainingChart = screen.getByRole("img", {
      name: /training loss against step/i,
    });
    const heldOutChart = screen.getByRole("img", {
      name: /held-out loss against epoch/i,
    });
    expect(trainingChart.querySelectorAll("polyline")).toHaveLength(1);
    expect(heldOutChart.querySelectorAll("polyline")).toHaveLength(1);
    expect(screen.getByText("Training loss")).toBeVisible();
    expect(screen.getByText("Held-out loss")).toBeVisible();
  });

  it("surfaces a held-out loss that stops improving, in plain language", () => {
    const heldOut = (id: number, loss: number) =>
      event({
        id,
        kind: "metric",
        message: `{'eval_loss': ${loss}, 'epoch': 1.0}`,
        data: { held_out_loss: loss, epoch: 1.0 },
      });
    renderView({
      events: [
        event({ id: 1, kind: "state", message: "training" }),
        heldOut(2, 0.7),
        heldOut(3, 0.7),
        heldOut(4, 0.7),
      ],
    });
    const note = screen.getByRole("status");
    expect(note).toHaveTextContent(/overfitting/i);
    expect(note).toHaveTextContent(/held-out loss has not improved/i);
  });

  it("says nothing while the held-out loss is still improving", () => {
    const heldOut = (id: number, loss: number) =>
      event({
        id,
        kind: "metric",
        message: `{'eval_loss': ${loss}, 'epoch': 1.0}`,
        data: { held_out_loss: loss, epoch: 1.0 },
      });
    renderView({
      events: [
        event({ id: 1, kind: "state", message: "training" }),
        heldOut(2, 0.9),
        heldOut(3, 0.6),
        heldOut(4, 0.4),
      ],
    });
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("shows a phase's progress with a measured rate from the snapshot", () => {
    renderView({
      progress: [
        {
          phase: "image pull",
          done: 15190000,
          total: 42420000,
          rate: 8_000_000,
          eta_s: 3,
          ts: 1,
          message: "9b829b73a52f: Downloading [==> ] 15.19MB/42.42MB",
        },
      ],
    });

    expect(screen.getByText("image pull")).toBeVisible();
    const bar = screen.getByRole("progressbar", { name: "image pull" });
    expect(bar).toHaveAttribute("aria-valuenow", "36");
    expect(screen.getByText(/15\.2 MB of 42\.4 MB/)).toBeVisible();
    expect(screen.getByText(/8\.0 MB\/s/)).toBeVisible();
    expect(screen.getByText(/about 00:00:03 left/)).toBeVisible();
  });

  it("supersedes a phase's progress rather than accumulating it", async () => {
    renderView();
    stream().emit("progress", {
      phase: "image pull",
      done: 15190000,
      total: 42420000,
      rate: 8_000_000,
      eta_s: 3,
      ts: 1,
    });
    await waitFor(() =>
      expect(
        screen.getByRole("progressbar", { name: "image pull" }),
      ).toHaveAttribute("aria-valuenow", "36"),
    );
    stream().emit("progress", {
      phase: "image pull",
      done: 42420000,
      total: 42420000,
      rate: 10_000_000,
      eta_s: 0,
      ts: 2,
    });
    await waitFor(() =>
      expect(
        screen.getByRole("progressbar", { name: "image pull" }),
      ).toHaveAttribute("aria-valuenow", "100"),
    );
    expect(
      screen.getAllByRole("progressbar", { name: "image pull" }),
    ).toHaveLength(1);
  });

  it("offers the retained raw lines as collapsed detail", async () => {
    renderView({
      progress: [
        { phase: "image pull", done: 42420000, total: 42420000, rate: 0, eta_s: 0, ts: 1 },
      ],
      output: [
        { id: 1, phase: "image pull", line: "9b829b73a52f: Pulling fs layer" },
        { id: 2, phase: "image pull", line: "9b829b73a52f: Pull complete" },
      ],
    });

    const summary = screen.getByText("2 lines");
    expect(summary).toBeVisible();
    // Collapsed by default: the detail is offered, not broadcast.
    const details = summary.closest("details")!;
    expect(details).not.toHaveAttribute("open");
    expect(screen.getByText(/Pull complete/)).not.toBeVisible();

    await userEvent.click(summary);
    expect(details).toHaveAttribute("open");
    expect(screen.getByText(/Pull complete/)).toBeVisible();
  });

  it("appends streamed retained lines without duplicating on replay", async () => {
    renderView({
      progress: [
        { phase: "image pull", done: 15190000, total: 42420000, rate: 8_000_000, eta_s: 3, ts: 1 },
      ],
      output: [{ id: 1, phase: "image pull", line: "a: Pulling fs layer" }],
    });
    stream().emit("output", {
      id: 1,
      phase: "image pull",
      line: "a: Pulling fs layer",
    });
    stream().emit("output", {
      id: 2,
      phase: "image pull",
      line: "a: Pull complete",
    });
    // The replayed id is deduplicated and the new line appended: the summary
    // counts two lines, not three, and the new line is in the DOM (collapsed
    // behind the detail's summary).
    await waitFor(() => expect(screen.getByText("2 lines")).toBeVisible());
    expect(screen.getByText(/Pull complete/)).toBeTruthy();
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
      screen.getByText(/no artifact will be produced/i),
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
