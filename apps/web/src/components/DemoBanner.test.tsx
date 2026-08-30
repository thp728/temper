import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import DemoBanner from "@/components/DemoBanner";

// Spec 012's honesty rule, at the component level: the marking says what the
// mode is and what it cannot do, and there is no way to turn it off from the
// page — a banner a visitor can dismiss is a banner that can be missing from
// the screenshot meant to prove the mode was labelled.

describe("DemoBanner", () => {
  it("says it is a demonstration and what that means", () => {
    render(<DemoBanner />);
    expect(screen.getByLabelText("Demonstration mode")).toBeVisible();
    expect(screen.getByText("Demonstration mode")).toBeVisible();
    expect(screen.getByText(/simulated machine/i)).toBeVisible();
    expect(screen.getByText(/spends money/i)).toBeVisible();
  });

  it("cannot be dismissed", () => {
    render(<DemoBanner />);
    const banner = screen.getByLabelText("Demonstration mode");
    expect(banner.querySelector("button")).toBeNull();
    expect(banner.querySelector("a")).toBeNull();
  });
});
