import { describe, expect, it } from "vitest";
import {
  failureExplanation,
  formatDuration,
  formatDurationRange,
  formatMinorCost,
  formatTimestamp,
  shortRevision,
} from "@/lib/jobs/display";

// Formatting is presentation over published numbers, and it is where a port
// quietly drifts: these pin the formats the server-rendered pages used, so
// the shell shows the same strings the old screens did.

describe("formatTimestamp", () => {
  it("renders an epoch stamp as local YYYY-MM-DD HH:mm:ss", () => {
    // 2025-08-26 00:00:00 local time, as seconds since the epoch.
    const ts = new Date(2025, 7, 26, 0, 0, 0).getTime() / 1000;
    expect(formatTimestamp(ts)).toBe("2025-08-26 00:00:00");
  });
});

describe("formatDuration", () => {
  it("shows seconds alone below a minute", () => {
    expect(formatDuration(42)).toBe("42s");
  });

  it("shows minutes and zero-padded seconds below an hour", () => {
    expect(formatDuration(2 * 60 + 5)).toBe("2m 05s");
  });

  it("shows hours, minutes and seconds above one", () => {
    expect(formatDuration(3 * 3600 + 2 * 60 + 5)).toBe("3h 02m 05s");
  });
});

describe("shortRevision", () => {
  it("truncates a pinned revision to twelve characters", () => {
    expect(shortRevision("a".repeat(40))).toBe("a".repeat(12));
  });

  it("leaves nothing to truncate when it is already short", () => {
    expect(shortRevision("abc")).toBe("abc");
    expect(shortRevision(null)).toBe("");
  });
});

describe("formatDurationRange", () => {
  it("shows both ends of a range, never a point", () => {
    expect(formatDurationRange(60, 120)).toBe("1m 00s–2m 00s");
  });

  it("says when a phase is not estimable", () => {
    expect(formatDurationRange(null, 120)).toBe("not estimable");
    expect(formatDurationRange(60, undefined)).toBe("not estimable");
  });
});

describe("formatMinorCost", () => {
  it("shows a minor-unit cost with its currency, to two places", () => {
    // 4131 paisa = ₹41.31, the measured L4 hourly rate.
    expect(formatMinorCost(4131, "INR", 100)).toBe("INR 41.31");
  });

  it("respects the published minor unit, not a formatting assumption", () => {
    expect(formatMinorCost(4131, "USD", 100)).toBe("USD 41.31");
  });

  it("renders an absent cost as a dash", () => {
    expect(formatMinorCost(null, "INR", 100)).toBe("—");
  });
});

describe("failureExplanation", () => {
  it("explains the safety limits in plain language", () => {
    expect(failureExplanation("gpu_stalled")).toMatch(/stopped by a safety limit/);
    expect(failureExplanation("gpu_max_duration_exceeded")).toMatch(
      /safety limit/,
    );
  });

  it("has nothing to add when the message already carries the reason", () => {
    expect(failureExplanation("training_failed")).toBeNull();
    expect(failureExplanation(undefined)).toBeNull();
  });
});
