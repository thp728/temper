import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TokenCountPoll from "@/components/TokenCountPoll";
import { DatasetRecordTokenCountStatus } from "@/lib/api/generated/client";
import type { DatasetRecord } from "@/lib/api/generated/client";

// The poll's wiring, pinned: while the count is being produced it re-renders
// the route so the landed count appears, and it stops the moment the phase
// ends. What it does not do -- navigate the tab -- is what makes it the safe
// replacement for the meta-refresh that used to yank a user who had navigated
// on (see the component's docstring).

const getMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api/generated/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/generated/client")>()),
  getDatasetV1DatasetsDatasetIdGet: getMock,
}));

const refresh = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({
  useRouter: () => ({ refresh }),
}));

function record(status: string): DatasetRecord {
  return {
    id: "ds_abc123",
    filename: "d.jsonl",
    created_at: 1756160400,
    status: "valid",
    token_count_status: status as DatasetRecordTokenCountStatus,
    report: null,
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  getMock.mockReset();
  refresh.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("TokenCountPoll", () => {
  it("re-renders the route each tick while the count is being produced", async () => {
    getMock.mockResolvedValue(record("counting"));
    render(<TokenCountPoll datasetId="ds_abc123" />);

    await vi.advanceTimersByTimeAsync(2000);
    expect(getMock).toHaveBeenCalledWith("ds_abc123");
    expect(refresh).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(2000);
    expect(refresh).toHaveBeenCalledTimes(2);
  });

  it("stops polling once the count has landed", async () => {
    getMock
      .mockResolvedValueOnce(record("counting"))
      .mockResolvedValue(record("done"));
    render(<TokenCountPoll datasetId="ds_abc123" />);

    await vi.advanceTimersByTimeAsync(2000);
    await vi.advanceTimersByTimeAsync(2000);
    expect(refresh).toHaveBeenCalledTimes(2);

    // The interval is cleared on the terminal tick; no further polls.
    await vi.advanceTimersByTimeAsync(10_000);
    expect(getMock).toHaveBeenCalledTimes(2);
  });

  it("survives a transient fetch failure and keeps polling", async () => {
    getMock
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValue(record("done"));
    render(<TokenCountPoll datasetId="ds_abc123" />);

    await vi.advanceTimersByTimeAsync(2000);
    await vi.advanceTimersByTimeAsync(2000);
    // The failed tick refreshed nothing and was retried; the landing tick
    // refreshed and stopped.
    expect(refresh).toHaveBeenCalledTimes(1);
  });
});
