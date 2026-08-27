import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import AdvancedSurface from "@/components/AdvancedSurface";
import type { AdvancedSurface as AdvancedSurfaceModel } from "@/lib/api/generated/client";

// The advanced surface (issue #80), generated from the trainer's own schema
// and published through `GET /v1/surface`: overridable settings behind an
// explicit disclosure with the specific thing that goes wrong beside each,
// the trainer's known-but-unsupported settings visible with their reasons,
// and the adjustable-versus-refused distinction explained.

function surface(): AdvancedSurfaceModel {
  return {
    source_image: "axolotlai/axolotl:main@sha256:abc",
    axolotl_version: "0.19.0.dev0",
    config_model: "AxolotlInputConfig",
    known_keys: ["learning_rate", "num_epochs", "wandb_project"],
    platform_internal_keys: ["simulated_failure_code"],
    overrideable_keys: ["learning_rate", "num_epochs"],
    counts: {
      calculated: 1,
      exposed_with_named_failure_mode: 2,
      known_but_unsupported: 1,
    },
    tiers: {
      calculated: {
        seed: {
          tier: "calculated",
          reason: "Set by the platform (42) so runs are reproducible.",
        },
      },
      exposed_with_named_failure_mode: {
        learning_rate: {
          tier: "exposed_with_named_failure_mode",
          reason: "The peak learning rate for the cosine schedule.",
          failure_mode:
            "Too high and the loss diverges to NaN partway through a paid run.",
          type: "float",
        },
        num_epochs: {
          tier: "exposed_with_named_failure_mode",
          reason: "How many passes over the training set.",
          failure_mode:
            "Too few and the adapter underfits; too many on a small dataset and it overfits.",
          type: "float",
        },
      },
      known_but_unsupported: {
        wandb_project: {
          tier: "known_but_unsupported",
          reason:
            "wandb_project: experiment-tracking integration. The platform runs no such integration.",
        },
      },
    },
    runtime_only_validators: {
      model_validators: ["check_fsdp_deepspeed"],
      field_validator_count: 27,
    },
  };
}

const defaults = { learning_rate: 0.0002, num_epochs: 3 };

describe("AdvancedSurface", () => {
  it("hides the exposed settings behind an explicit disclosure", () => {
    render(
      <AdvancedSurface
        surface={surface()}
        defaults={defaults}
        values={{}}
        onChange={() => {}}
      />,
    );
    // The disclosure is present and closed: opening the advanced surface is a
    // deliberate act, not a wall of dials. (jsdom does not hide a closed
    // details' children from role queries, so the not-reachable-until-opened
    // guarantee is asserted in the real-browser journey instead.)
    const details = screen.getByRole("group", { name: "Advanced settings" });
    expect(details).toBeInTheDocument();
    expect((details as HTMLDetailsElement).open).toBe(false);
  });

  it("shows each exposed setting with its specific failure mode inline", async () => {
    const user = userEvent.setup();
    render(
      <AdvancedSurface
        surface={surface()}
        defaults={defaults}
        values={{}}
        onChange={() => {}}
      />,
    );
    await user.click(
      screen.getByRole("group", { name: "Advanced settings" }).querySelector(
        "summary",
      ) as HTMLElement,
    );
    const lr = screen.getByRole("spinbutton", { name: "learning_rate override" });
    await expect(lr).toHaveValue(0.0002);
    // The failure mode is the specific thing that goes wrong, inline, not a
    // general caution -- and the reason sits beside it.
    expect(
      screen.getByText(/diverges to NaN partway through a paid run/i),
    ).toBeVisible();
    expect(screen.getByText(/peak learning rate/i)).toBeVisible();
  });

  it("shows a trainer setting this product does not offer, with its reason", async () => {
    const user = userEvent.setup();
    render(
      <AdvancedSurface
        surface={surface()}
        defaults={defaults}
        values={{}}
        onChange={() => {}}
      />,
    );
    await user.click(
      screen.getByRole("group", { name: "Advanced settings" }).querySelector(
        "summary",
      ) as HTMLElement,
    );
    // The unsupported set is searchable: find the setting, read its reason.
    await user.type(
      screen.getByRole("searchbox", {
        name: "Search the trainer's settings Temper does not offer",
      }),
      "wandb_project",
    );
    expect(screen.getByText("wandb_project")).toBeVisible();
    expect(
      screen.getByText(/experiment-tracking integration/i),
    ).toBeVisible();
  });

  it("explains the distinction between an adjustable setting and a refused input", async () => {
    const user = userEvent.setup();
    render(
      <AdvancedSurface
        surface={surface()}
        defaults={defaults}
        values={{}}
        onChange={() => {}}
      />,
    );
    await user.click(
      screen.getByRole("group", { name: "Advanced settings" }).querySelector(
        "summary",
      ) as HTMLElement,
    );
    expect(
      screen.getByText(
        /Why some settings are adjustable and others are refused/i,
      ),
    ).toBeVisible();
    // Both named examples: the mixed thinking-mode dataset and the below-floor
    // dataset are refused inputs, not hidden controls.
    expect(
      screen.getByText(/mixes reasoning traces with plain answers/i),
    ).toBeVisible();
    expect(
      screen.getByText(/below the minimum usable row count/i),
    ).toBeVisible();
  });

  it("records a changed setting through the callback", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <AdvancedSurface
        surface={surface()}
        defaults={defaults}
        values={{}}
        onChange={onChange}
      />,
    );
    await user.click(
      screen.getByRole("group", { name: "Advanced settings" }).querySelector(
        "summary",
      ) as HTMLElement,
    );
    const lr = screen.getByRole("spinbutton", { name: "learning_rate override" });
    await user.clear(lr);
    await user.type(lr, "0.0001");
    await user.tab();
    expect(onChange).toHaveBeenLastCalledWith({ learning_rate: "0.0001" });
  });

  it("reverts a changed setting to the default", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <AdvancedSurface
        surface={surface()}
        defaults={defaults}
        values={{ learning_rate: "0.0001" }}
        onChange={onChange}
      />,
    );
    await user.click(
      screen.getByRole("group", { name: "Advanced settings" }).querySelector(
        "summary",
      ) as HTMLElement,
    );
    // The override is marked, and offers its way back to the default.
    expect(screen.getByText("you changed this")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Use the default" }));
    expect(onChange).toHaveBeenLastCalledWith({});
  });
});
