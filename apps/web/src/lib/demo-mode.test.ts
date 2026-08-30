import { afterEach, describe, expect, it } from "vitest";
import { isZeroCostMode } from "@/lib/demo-mode";

// The mode is one setting read by every process (ADR-0071), and the shell
// parses it with the same rule as the control plane's config reader: an
// explicit `0`/`false` must read as the real tier, or a deployment that turns
// the fake off would still have every page marked as a demonstration.

afterEach(() => {
  delete process.env.TEMPER_FAKE_PROVIDER;
});

describe("isZeroCostMode", () => {
  it.each(["1", "true", "TRUE", "yes", "on"])(
    "is on for %s",
    (value) => {
      process.env.TEMPER_FAKE_PROVIDER = value;
      expect(isZeroCostMode()).toBe(true);
    },
  );

  it.each(["0", "false", "no", "off"])(
    "is off for %s",
    (value) => {
      process.env.TEMPER_FAKE_PROVIDER = value;
      expect(isZeroCostMode()).toBe(false);
    },
  );

  it("is off when the setting is unset", () => {
    delete process.env.TEMPER_FAKE_PROVIDER;
    expect(isZeroCostMode()).toBe(false);
  });

  it("is off for a value neither half recognises", () => {
    process.env.TEMPER_FAKE_PROVIDER = "perhaps";
    expect(isZeroCostMode()).toBe(false);
  });
});
