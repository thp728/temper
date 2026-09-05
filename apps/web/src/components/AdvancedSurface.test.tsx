import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import AdvancedSurface from "@/components/AdvancedSurface";
import type { AdvancedSurface as AdvancedSurfaceModel } from "@/lib/api/generated/client";

// The advanced surface (issue #80), generated from the trainer's own schema
// and published through `GET /v1/surface`. Since the step-2 cards edit every
// exposed key in place, this disclosure carries only what the cards cannot:
// the trainer settings Temper does not support yet, visible with their
// reasons and searchable -- never wondered about as overlooked.

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

describe("AdvancedSurface", () => {
  it("hides the unsupported settings behind an explicit disclosure", () => {
    render(<AdvancedSurface surface={surface()} />);
    // The disclosure is present and closed: opening it is a deliberate act,
    // not a wall of dials. (jsdom does not hide a closed details' children
    // from role queries, so the not-reachable-until-opened guarantee is
    // asserted in the real-browser journey instead.)
    const details = screen.getByRole("group", { name: "Advanced settings" });
    expect(details).toBeInTheDocument();
    expect((details as HTMLDetailsElement).open).toBe(false);
  });

  it("carries no editors: exposed keys are edited in the step-2 cards", () => {
    render(<AdvancedSurface surface={surface()} />);
    // One editor per key, and it lives in the hyperparameter cards -- a
    // second "learning_rate override" here would double every control name.
    expect(
      screen.queryByRole("spinbutton", { name: "learning_rate override" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("spinbutton", { name: "num_epochs override" }),
    ).not.toBeInTheDocument();
  });

  it("says the trainer settings are not yet supported, with support coming", async () => {
    const user = userEvent.setup();
    render(<AdvancedSurface surface={surface()} />);
    await user.click(
      screen.getByRole("group", { name: "Advanced settings" }).querySelector(
        "summary",
      ) as HTMLElement,
    );
    expect(
      screen.getByText(/Axolotl settings Temper doesn't support yet/i),
    ).toBeVisible();
    expect(screen.getByText(/support will be added/i)).toBeVisible();
  });

  it("shows a trainer setting this product does not offer, with its reason", async () => {
    const user = userEvent.setup();
    render(<AdvancedSurface surface={surface()} />);
    await user.click(
      screen.getByRole("group", { name: "Advanced settings" }).querySelector(
        "summary",
      ) as HTMLElement,
    );
    // The unsupported set is searchable: find the setting, read its reason.
    await user.type(
      screen.getByRole("searchbox", {
        name: "Search unsupported Axolotl settings",
      }),
      "wandb_project",
    );
    expect(screen.getByText("wandb_project")).toBeVisible();
    expect(
      screen.getByText(/experiment-tracking integration/i),
    ).toBeVisible();
  });
});
