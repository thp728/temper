import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(cleanup);

// Radix Popper (which the shadcn tooltip is built on) observes its anchor
// and content size; jsdom ships no ResizeObserver, so positioning crashes
// without one. A no-op stub is enough here: tests assert what the tooltip
// says, never where it floats.
if (typeof globalThis.ResizeObserver === "undefined") {
  class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  globalThis.ResizeObserver =
    ResizeObserver as unknown as typeof globalThis.ResizeObserver;
}
