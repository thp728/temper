import { describe, expect, it } from "vitest";
import {
  failureExplanation,
  formatDuration,
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
