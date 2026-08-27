import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import QuoteView from "@/components/QuoteView";
import type { Quote } from "@/lib/api/generated/client";

// The plan's cost-and-time estimate (issue #72). The assertions are what a
// user reads: a duration range (never a point), a per-phase cost breakdown in
// the account's currency, the token count, and the estimate label.

function quote(overrides: Partial<Quote> = {}): Quote {
  return {
    currency: "INR",
    minor_unit: 100,
    dataset_id: "ds_abc123",
    dataset_created_at: 1756160400,
    base_revision: "a".repeat(40),
    token_count: 1_234_567,
    expires_at: 1756160400 + 86400,
    phases: [
      {
        name: "provisioning",
        duration_low_s: 13,
        duration_high_s: 17,
        cost_low_minor: 15,
        cost_high_minor: 20,
      },
      {
        name: "training",
        duration_low_s: 80,
        duration_high_s: 800,
        cost_low_minor: 92,
        cost_high_minor: 918,
      },
    ],
    duration_low_s: 93,
    duration_high_s: 817,
    cost_low_minor: 107,
    cost_high_minor: 938,
    storage_cost_usd_per_hour: 0.0137,
    storage_cost_usd_total_low_minor: 1,
    storage_cost_usd_total_high_minor: 4,
    is_estimate: true,
    ...overrides,
  };
}

describe("QuoteView", () => {
  it("shows a duration range and a cost range, never points", () => {
    render(<QuoteView quote={quote()} />);
    expect(screen.getByText(/1m 33s–13m 37s/)).toBeVisible();
    expect(screen.getByText(/INR 1\.07 – INR 9\.38/)).toBeVisible();
  });

  it("labels itself an estimate that never blocks", () => {
    render(<QuoteView quote={quote()} />);
    expect(screen.getByText(/An estimate, not a guarantee/i)).toBeVisible();
    expect(screen.getByText(/never blocks a launch/i)).toBeVisible();
  });

  it("shows the token count and the currency", () => {
    render(<QuoteView quote={quote()} />);
    // Grouping varies with the runtime locale; the number itself is pinned.
    expect(screen.getByText(/1,?2,?3,?4,?5,?6,?7/)).toBeVisible();
    expect(screen.getByText("INR")).toBeVisible();
  });

  it("breaks the cost down per phase", () => {
    render(<QuoteView quote={quote()} />);
    for (const phase of ["provisioning", "training"]) {
      expect(screen.getByText(phase)).toBeVisible();
    }
  });

  it("pins what it was computed against and when it expires", () => {
    render(<QuoteView quote={quote()} />);
    expect(screen.getByText(/Estimated against dataset/i)).toBeVisible();
    expect(screen.getByText(/model revision/i)).toBeVisible();
    expect(screen.getByText(/expires/i)).toBeVisible();
  });

  it("says when a phase is not estimable rather than inventing a number", () => {
    const q = quote();
    q.phases = [
      {
        name: "training",
        duration_low_s: null,
        duration_high_s: null,
        cost_low_minor: null,
        cost_high_minor: null,
      },
    ];
    render(<QuoteView quote={q} />);
    expect(screen.getByText("not estimable")).toBeVisible();
  });
});
