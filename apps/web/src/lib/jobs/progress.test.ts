import { describe, expect, it } from "vitest";
import type { JobProgress } from "@/lib/api/generated/client";
import {
  formatBytes,
  formatEta,
  formatRate,
  outputByPhase,
  proportion,
  supersedeProgress,
} from "@/lib/jobs/progress";

describe("formatBytes", () => {
  it("uses the same SI base docker and tqdm scale on", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(567)).toBe("567 B");
    expect(formatBytes(15190000)).toBe("15.2 MB");
    expect(formatBytes(400000000)).toBe("400.0 MB");
    expect(formatBytes(4_000_000_000)).toBe("4.0 GB");
  });
});

describe("formatRate", () => {
  it("renders a measured rate and says when none exists", () => {
    expect(formatRate(12_400_000)).toBe("12.4 MB/s");
    expect(formatRate(null)).toBe("measuring…");
  });
});

describe("formatEta", () => {
  it("reuses the app's one duration clock", () => {
    expect(formatEta(45)).toBe("00:00:45");
    expect(formatEta(125)).toBe("00:02:05");
    expect(formatEta(3725)).toBe("01:02:05");
    expect(formatEta(null)).toBeNull();
  });
});

describe("proportion", () => {
  it("divides done by total, clamped to the bar", () => {
    expect(proportion(50, 100)).toBe(0.5);
    expect(proportion(100, 100)).toBe(1);
    expect(proportion(200, 100)).toBe(1);
  });

  it("is null without a total", () => {
    expect(proportion(50, null)).toBeNull();
    expect(proportion(null, 100)).toBeNull();
  });
});

describe("supersedeProgress", () => {
  it("replaces a phase's record rather than accumulating it", () => {
    const a: JobProgress = {
      phase: "image pull",
      done: 15_000_000,
      total: 42_000_000,
      rate: 8_000_000,
      eta_s: 3,
      ts: 1,
    };
    const b: JobProgress = {
      phase: "model download",
      done: 400_000_000,
      total: 4_000_000_000,
      rate: 10_000_000,
      eta_s: 360,
      ts: 2,
    };
    const newer: JobProgress = {
      phase: "image pull",
      done: 28_000_000,
      total: 42_000_000,
      rate: 9_000_000,
      eta_s: 2,
      ts: 3,
    };
    const merged = supersedeProgress([a, b], [newer]);
    expect(merged).toHaveLength(2);
    const pull = merged.find((p) => p.phase === "image pull")!;
    expect(pull.done).toBe(28_000_000);
    expect(pull.rate).toBe(9_000_000);
  });
});

describe("outputByPhase", () => {
  it("groups the retained raw lines under their phase", () => {
    const grouped = outputByPhase([
      { id: 1, phase: "image pull", line: "a: Pulling fs layer" },
      { id: 2, phase: "image pull", line: "a: Pull complete" },
      { id: 3, phase: "model download", line: "model.safetensors: 10%" },
    ]);
    expect(grouped.get("image pull")).toHaveLength(2);
    expect(grouped.get("model download")).toHaveLength(1);
  });
});
