import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import QuoteView from "@/components/QuoteView";
import type { Quote } from "@/lib/api/generated/client";

// The plan's cost-and-time estimate (issue #72). The assertions are what a
// user reads: a duration range (never a point), a per-phase cost breakdown in
// the account's currency, the token count, and the estimate label. Since #76
// each decision is also shown with its reason visible by default and its
// alternatives one interaction away.

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
    decisions: [
      {
        decision: "hardware",
        chosen: "L4",
        constraint:
          "predicted peak is 5.4 GB; the L4 (24.0 GB) is the cheapest card currently available that holds it with 18.6 GB to spare.",
        alternatives: [
          {
            value: "H100",
            cost: "INR 250.00/hr",
            constraint: "fits, but costs more than the L4.",
          },
        ],
      },
      {
        decision: "method",
        chosen: "qlora",
        constraint:
          "the trainer can execute qlora today, and selection picks the cheapest executable method that fits.",
        alternatives: [
          {
            value: "lora",
            cost: "needs 11.4 GB peak",
            constraint: "the trainer cannot execute LoRA yet.",
          },
        ],
      },
    ],
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

  it("shows storage on its own line, not folded into the account currency", () => {
    render(<QuoteView quote={quote()} />);
    // ADR-0030: storage bills separately and is labelled USD, never silently
    // converted into the account's currency.
    expect(screen.getByText(/storage \(USD, separate line\)/)).toBeVisible();
    expect(screen.getByText(/USD 0\.01 – USD 0\.04/)).toBeVisible();
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

  it("shows each decision with its reason visible by default", () => {
    render(<QuoteView quote={quote()} />);
    expect(
      screen.getByRole("heading", { name: "Why this configuration" }),
    ).toBeVisible();
    expect(screen.getByRole("heading", { name: "hardware" })).toBeVisible();
    expect(screen.getByText("L4")).toBeVisible();
    expect(
      screen.getByText(/cheapest card currently available/i),
    ).toBeVisible();
    expect(screen.getByRole("heading", { name: "method" })).toBeVisible();
    expect(screen.getByText("qlora")).toBeVisible();
  });

  it("keeps alternatives one interaction away, never hidden or always shown", async () => {
    const user = userEvent.setup();
    render(<QuoteView quote={quote()} />);
    // Collapsed by default: each decision's alternatives sit behind its
    // disclosure, not on the page. jsdom keeps closed <details> content in
    // the DOM, so the honest assertion is on the disclosure's own state.
    const summaries = screen.getAllByText("Alternatives considered (1)");
    expect(summaries.length).toBeGreaterThan(0);
    const first = summaries[0]!;
    const disclosure = first.closest("details") as HTMLDetailsElement;
    expect(disclosure.open).toBe(false);
    // One interaction reveals them.
    await user.click(first);
    expect(disclosure.open).toBe(true);
    expect(screen.getByText("H100")).toBeVisible();
    expect(screen.getByText(/INR 250\.00\/hr/)).toBeVisible();
  });

  it("renders no decision section when the quote carries none", () => {
    render(<QuoteView quote={quote({ decisions: [] })} />);
    expect(
      screen.queryByRole("heading", { name: "Why this configuration" }),
    ).toBeNull();
  });
});
