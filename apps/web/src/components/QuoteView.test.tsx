import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import QuoteView from "@/components/QuoteView";
import type { Quote } from "@/lib/api/generated/client";

// The plan's cost-and-time estimate (issue #72). The assertions are what a
// user reads: a duration range (never a point), a per-phase cost breakdown in
// the account's currency, the token count, and the estimate label. Since #76
// each decision is also shown with its reason visible by default and its
// alternatives one interaction away. Since #79 the plan is editable: each
// decision carries a control beside its explanation, and an overridden one is
// marked.

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
        overridden: false,
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
        overridden: false,
      },
      {
        decision: "sequence length",
        chosen: "2048",
        constraint: "the trainer default (2048, trainer-defaults.json).",
        alternatives: [
          {
            value: "4096",
            cost: "doubles activation memory",
            constraint: "nothing in the dataset has asked for a longer window.",
          },
        ],
        overridden: false,
      },
    ],
    override_options: {
      method: ["qlora", "lora", "full"],
      hardware: ["A100-80GB", "H100", "H200", "L4", "RTX-PRO6000"],
      precision: ["nf4 (4-bit)", "bf16 (no quantisation)"],
    },
    ...overrides,
  };
}

describe("QuoteView", () => {
  it("shows a duration range and a cost range, never points", () => {
    render(<QuoteView quote={quote()} />);
    expect(screen.getByText(/00:01:33–00:13:37/)).toBeVisible();
    expect(screen.getByText(/INR 1\.07 – INR 9\.38/)).toBeVisible();
  });

  it("labels itself an estimate", () => {
    render(<QuoteView quote={quote()} />);
    expect(screen.getByRole("heading", { name: /Cost and time estimate/i })).toBeVisible();
    expect(screen.getByText(/Estimated against dataset/i)).toBeVisible();
  });

  it("prints no disclaimer beside the heading — the heading says estimate", () => {
    const { rerender } = render(<QuoteView quote={quote()} />);
    expect(screen.getByRole("heading", { name: /Cost and time estimate/i })).toBeVisible();
    expect(screen.queryByText(/never blocks a launch/i)).not.toBeInTheDocument();
    rerender(<QuoteView quote={quote()} editable overrides={[]} onOverridesChange={() => {}} />);
    expect(screen.queryByText(/never blocks a launch/i)).not.toBeInTheDocument();
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

  // --- the plan is editable (issue #79) -------------------------------------

  it("sits a control beside every explanation when editable", () => {
    render(
      <QuoteView
        quote={quote()}
        editable
        overrides={[]}
        onOverridesChange={() => {}}
      />,
    );
    expect(
      screen.getByRole("combobox", { name: "hardware override" }),
    ).toBeVisible();
    expect(
      screen.getByRole("combobox", { name: "method override" }),
    ).toBeVisible();
    // Free-form decisions get an input, not a fixed list.
    expect(
      screen.getByRole("spinbutton", { name: "sequence length override" }),
    ).toBeVisible();
  });

  it("renders no controls on a finished job", () => {
    render(<QuoteView quote={quote()} />);
    expect(
      screen.queryByRole("combobox", { name: "method override" }),
    ).toBeNull();
  });

  it("marks a decision the user overrode", () => {
    const q = quote();
    const base = q.decisions ?? [];
    q.decisions = [
      { ...base[0]!, overridden: true },
      base[1]!,
      base[2]!,
    ];
    render(
      <QuoteView
        quote={q}
        editable
        overrides={[{ decision: "hardware", value: "H100" }]}
        onOverridesChange={() => {}}
      />,
    );
    expect(screen.getByText("you changed this")).toBeVisible();
  });

  it("re-requests the plan when a control changes", async () => {
    const user = userEvent.setup();
    const onOverride = vi.fn();
    render(
      <QuoteView
        quote={quote()}
        editable
        overrides={[]}
        onOverridesChange={onOverride}
      />,
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "method override" }),
      "lora",
    );
    expect(onOverride).toHaveBeenCalledWith([
      { decision: "method", value: "lora" },
    ]);
  });

  it("clears an override when the predictor's choice is reselected", async () => {
    const user = userEvent.setup();
    const onOverride = vi.fn();
    const overrides = [{ decision: "method", value: "lora" }];
    render(
      <QuoteView
        quote={quote()}
        editable
        overrides={overrides}
        onOverridesChange={onOverride}
      />,
    );
    // The select always shows the effective value: reselecting the
    // predictor's own choice unpins rather than pinning it.
    await user.selectOptions(
      screen.getByRole("combobox", { name: "method override" }),
      "qlora",
    );
    expect(onOverride).toHaveBeenCalledWith([]);
  });

  it("commits a free-form decision from its input", async () => {
    const user = userEvent.setup();
    const onOverride = vi.fn();
    render(
      <QuoteView
        quote={quote()}
        editable
        overrides={[]}
        onOverridesChange={onOverride}
      />,
    );
    const input = screen.getByRole("spinbutton", {
      name: "sequence length override",
    });
    await user.clear(input);
    await user.type(input, "4096");
    await user.tab();
    expect(onOverride).toHaveBeenCalledWith([
      { decision: "sequence length", value: "4096" },
    ]);
  });

  it("offers the predictor's choice to undo a free-form override", async () => {
    const user = userEvent.setup();
    const onOverride = vi.fn();
    const overrides = [{ decision: "sequence length", value: "4096" }];
    render(
      <QuoteView
        quote={quote()}
        editable
        overrides={overrides}
        onOverridesChange={onOverride}
      />,
    );
    await user.click(
      screen.getByRole("button", { name: /Use the predictor's choice/ }),
    );
    expect(onOverride).toHaveBeenCalledWith([]);
  });
  it("shows a refusal beside the decisions when one cannot be honoured", () => {
    render(
      <QuoteView
        quote={quote()}
        editable
        overrides={[]}
        onOverridesChange={() => {}}
        refusal={{
          code: "configuration_does_not_fit",
          message: "full on a L4 predicts 66.9 GB peak, which the 24.0 GB L4 cannot hold.",
        }}
      />,
    );
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("configuration_does_not_fit");
    expect(alert).toHaveTextContent("66.9 GB peak");
  });

  // --- explanations behind "?" (the launch wizard) --------------------------

  it("keeps the card to label, value and control with the reason behind a '?'", async () => {
    const user = userEvent.setup();
    render(
      <QuoteView
        quote={quote()}
        editable
        overrides={[]}
        onOverridesChange={() => {}}
        explain="dialog"
      />,
    );
    // The card keeps its heading, its chosen value and its control...
    // (the chosen badge and the combobox option share the word "qlora",
    // so the badge is read through its card, not by bare text).
    expect(screen.getByRole("heading", { name: "method" })).toBeVisible();
    const card = screen
      .getByRole("heading", { name: "method" })
      .closest("div")?.parentElement;
    expect(card).toHaveTextContent("qlora");
    expect(
      screen.getByRole("combobox", { name: "method override" }),
    ).toBeVisible();
    // ...while the reason and the alternatives stay out of the page.
    expect(
      screen.queryByText(/cheapest executable method/i),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("Alternatives considered (1)"),
    ).not.toBeInTheDocument();
    // One click opens them — scoped to the dialog, since the combobox
    // options share the same words.
    await user.click(
      screen.getByRole("button", { name: "About the method decision" }),
    );
    const dialog = await screen.findByRole("dialog");
    expect(
      within(dialog).getByText(/cheapest executable method/i),
    ).toBeVisible();
    expect(
      within(dialog).getByText("Alternatives considered (1)"),
    ).toBeVisible();
    expect(within(dialog).getByText("lora")).toBeVisible();
  });

  it("marks an overridden decision in dialog mode too", () => {
    const q = quote();
    const base = q.decisions ?? [];
    q.decisions = [
      { ...base[0]!, overridden: true },
      base[1]!,
      base[2]!,
    ];
    render(
      <QuoteView
        quote={q}
        editable
        overrides={[{ decision: "hardware", value: "H100" }]}
        onOverridesChange={() => {}}
        explain="dialog"
      />,
    );
    expect(screen.getByText("you changed this")).toBeVisible();
    expect(
      screen.getByRole("combobox", { name: "hardware override" }),
    ).toBeVisible();
  });

  // --- option cards (the hardware choice) ------------------------------------

  it("renders one radio per published option with the predictor's pick recommended", () => {
    render(
      <QuoteView
        quote={quote()}
        editable
        overrides={[]}
        onOverridesChange={() => {}}
        explain="dialog"
        bareDecisions
        cardDecisions={["hardware"]}
      />,
    );
    // No select, no section chrome: radios named by their option.
    expect(
      screen.queryByRole("combobox", { name: "hardware override" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Why this configuration" }),
    ).not.toBeInTheDocument();
    const group = screen.getByRole("radiogroup", { name: "hardware options" });
    const radios = within(group).getAllByRole("radio");
    expect(radios.map((r) => (r as HTMLInputElement).value)).toEqual([
      "A100-80GB",
      "H100",
      "H200",
      "L4",
      "RTX-PRO6000",
    ]);
    // The predictor's default starts selected and badged...
    expect(
      within(group).getByRole("radio", { name: /L4/ }),
    ).toBeChecked();
    expect(within(group).getByText("Recommended")).toBeVisible();
    // ...with the losing option's own cost beside it.
    expect(within(group).getByText("INR 250.00/hr")).toBeVisible();
  });

  it("pins an option and unpins by reselecting the predictor's choice", async () => {
    const user = userEvent.setup();
    const onOverride = vi.fn();
    render(
      <QuoteView
        quote={quote()}
        editable
        overrides={[]}
        onOverridesChange={onOverride}
        explain="dialog"
        bareDecisions
        cardDecisions={["hardware"]}
      />,
    );
    const group = screen.getByRole("radiogroup", { name: "hardware options" });
    await user.click(within(group).getByRole("radio", { name: /H100/ }));
    expect(onOverride).toHaveBeenCalledWith([
      { decision: "hardware", value: "H100" },
    ]);
  });

  it("leaves non-card decisions on their controls", () => {
    render(
      <QuoteView
        quote={quote()}
        editable
        overrides={[]}
        onOverridesChange={() => {}}
        explain="dialog"
        bareDecisions
        cardDecisions={["hardware"]}
      />,
    );
    // Method keeps its select; sequence length keeps its input.
    expect(
      screen.getByRole("combobox", { name: "method override" }),
    ).toBeVisible();
    expect(
      screen.getByRole("spinbutton", { name: "sequence length override" }),
    ).toBeVisible();
  });

  it("badges and unpins against the predictor's default, not the pinned choice", async () => {
    const user = userEvent.setup();
    const onOverride = vi.fn();
    // A pinned quote: the choice echoes the pin, marked overridden.
    const q = quote({
      decisions: [
        {
          decision: "hardware",
          chosen: "H100",
          constraint: "you chose H100.",
          alternatives: [],
          overridden: true,
        },
      ],
    });
    render(
      <QuoteView
        quote={q}
        editable
        overrides={[{ decision: "hardware", value: "H100" }]}
        onOverridesChange={onOverride}
        explain="dialog"
        bareDecisions
        cardDecisions={["hardware"]}
        defaultChoices={{ hardware: "L4" }}
      />,
    );
    const group = screen.getByRole("radiogroup", { name: "hardware options" });
    // The pin is selected, but Recommended stays on the default.
    expect(within(group).getByRole("radio", { name: "H100" })).toBeChecked();
    expect(
      within(group).getByRole("radio", { name: "L4 Recommended" }),
    ).not.toBeChecked();
    // Reselecting the default unpins instead of re-pinning it.
    await user.click(within(group).getByRole("radio", { name: "L4 Recommended" }));
    expect(onOverride).toHaveBeenCalledWith([]);
  });
});
