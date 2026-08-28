import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import EventLog from "@/components/EventLog";
import type { JobEvent } from "@/lib/api/generated/client";

function ev(id: number, message: string): JobEvent {
  return {
    id,
    job_id: "job_abc",
    ts: Date.now() / 1000,
    kind: "log",
    message,
    data: null,
  };
}

// Mock the generated client for the pagination path. The real client is
// exercised by the control-plane tests; component tests assert on what the
// user sees with the transport mocked out.
vi.mock("@/lib/api/generated/client", async (importOriginal) => {
  const orig = (await importOriginal()) as Record<string, unknown>;
  return {
    ...orig,
    getEventsV1JobsJobIdEventsGet: vi.fn(async () => ({
      events: [],
      last_id: 0,
      total: 0,
      progress: [],
      output: [],
    })),
  };
});

import { getEventsV1JobsJobIdEventsGet } from "@/lib/api/generated/client";

describe("EventLog", () => {
  it("shows the disclosure even when not truncated", () => {
    render(<EventLog jobId="job_abc" initialEvents={[ev(1, "a"), ev(2, "b")]} initialTotal={2} />);
    expect(screen.getByTestId("event-disclosure")).toHaveTextContent("Showing 2 of 2 events");
    expect(screen.queryByRole("button", { name: /Load more/ })).toBeNull();
    expect(screen.getByRole("log")).toHaveTextContent("a");
  });

  it("states the cut and offers the full record when paginated", async () => {
    const many = Array.from({ length: 500 }, (_, i) => ev(i + 1, `log ${i + 1}`));
    render(<EventLog jobId="job_abc" initialEvents={many} initialTotal={734} />);
    expect(screen.getByTestId("event-disclosure")).toHaveTextContent("Showing 500 of 734 events");
    expect(screen.getByRole("button", { name: /Load more/ })).toHaveTextContent(/234 remaining/);
  });

  it("paging appends the tail and updates the disclosure", async () => {
    const user = userEvent.setup();
    const first = Array.from({ length: 2 }, (_, i) => ev(i + 1, `log ${i + 1}`));
    // Second page returns the remaining 1 event and keeps total.
    vi.mocked(getEventsV1JobsJobIdEventsGet).mockResolvedValueOnce({
      events: [ev(3, "log 3")],
      last_id: 3,
      total: 3,
      progress: [],
      output: [],
    } as never);

    render(<EventLog jobId="job_abc" initialEvents={first} initialTotal={3} />);
    expect(screen.getByTestId("event-disclosure")).toHaveTextContent("Showing 2 of 3 events");
    await user.click(screen.getByRole("button", { name: /Load more/ }));
    expect(screen.getByTestId("event-disclosure")).toHaveTextContent("Showing 3 of 3 events");
    expect(screen.getByRole("log")).toHaveTextContent("log 3");
    expect(screen.queryByRole("button", { name: /Load more/ })).toBeNull();
    expect(getEventsV1JobsJobIdEventsGet).toHaveBeenCalledWith("job_abc", { after: 2, limit: 500 });
  });
});
