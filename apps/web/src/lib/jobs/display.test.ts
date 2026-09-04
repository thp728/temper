import { describe, expect, it } from "vitest";
import {
  failureExplanation,
  formatDuration,
  formatDurationRange,
  formatExactTimestamp,
  formatMinorCost,
  formatMinorCostRange,
  formatTimestamp,
  shortRevision,
} from "@/lib/jobs/display";

// Formatting is presentation over published numbers, and it is where a port
// quietly drifts: these pin the formats the server-rendered pages used, so
// the shell shows the same strings the old screens did.

describe("formatTimestamp", () => {
  // The clock is pinned in UTC: expectations hold in whatever timezone the
  // suite runs under, the same guarantee the component gets from `timeZone:
  // "UTC"` in the absolute branch.
  const NOW = Date.UTC(2026, 8, 4, 12, 0, 0) / 1000;

  it("reads relative while the stamp is fresh", () => {
    expect(formatTimestamp(NOW - 30, NOW)).toBe("just now");
    expect(formatTimestamp(NOW - 3 * 60, NOW)).toBe("3 min ago");
    expect(formatTimestamp(NOW - 60, NOW)).toBe("1 min ago");
    expect(formatTimestamp(NOW - 4 * 3600, NOW)).toBe("4 hrs ago");
    expect(formatTimestamp(NOW - 3600, NOW)).toBe("1 hr ago");
  });

  it("mirrors the future the same way", () => {
    expect(formatTimestamp(NOW + 30, NOW)).toBe("just now");
    expect(formatTimestamp(NOW + 3 * 60, NOW)).toBe("in 3 min");
    expect(formatTimestamp(NOW + 4 * 3600, NOW)).toBe("in 4 hrs");
  });

  it("falls back to a short absolute date once older than a day", () => {
    // 2025-08-26 12:00:00 UTC, as seconds since the epoch.
    const ts = Date.UTC(2025, 7, 26, 12, 0, 0) / 1000;
    expect(formatTimestamp(ts, NOW)).toBe("Aug 26, 2025");
    expect(formatTimestamp(NOW - 25 * 3600, NOW)).toBe("Sep 3, 2026");
  });

  it("renders an absent stamp as a dash", () => {
    expect(formatTimestamp(null, NOW)).toBe("—");
    expect(formatTimestamp(undefined, NOW)).toBe("—");
  });
});

describe("formatExactTimestamp", () => {
  it("renders a locale-free UTC stamp, identical on server and client", () => {
    expect(formatExactTimestamp(Date.UTC(2026, 8, 2, 12, 7, 7) / 1000)).toBe(
      "2026-09-02 12:07:07 UTC",
    );
    expect(formatExactTimestamp(null)).toBe("—");
  });
});

describe("formatDuration", () => {
  it("renders a zero-padded HH:MM:SS clock", () => {
    expect(formatDuration(42)).toBe("00:00:42");
    expect(formatDuration(134)).toBe("00:02:14");
    expect(formatDuration(725)).toBe("00:12:05");
    expect(formatDuration(6312)).toBe("01:45:12");
    expect(formatDuration(12044)).toBe("03:20:44");
  });

  it("renders an absent duration as a dash", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(undefined)).toBe("—");
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
    expect(formatDurationRange(60, 120)).toBe("00:01:00–00:02:00");
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

describe("formatMinorCostRange", () => {
  it("shares one currency prefix across the range", () => {
    expect(formatMinorCostRange(210, 567, "INR", 100)).toBe("INR 2.10–5.67");
  });

  it("renders an absent end as a dash rather than inventing a number", () => {
    expect(formatMinorCostRange(null, 567, "INR", 100)).toBe("—");
    expect(formatMinorCostRange(210, undefined, "INR", 100)).toBe("—");
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
