import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ValidationProgressView from "@/components/ValidationProgressView";
import type { DatasetRecord } from "@/lib/api/generated/client";

// What the user reads while validation runs: the filename, a proportion
// complete, and the promise that the page will resolve itself. Assertions are
// what a user sees, never markup structure -- with one exception, the meta
// refresh, which is the mechanism that makes the page resolve itself and is
// itself a user-visible behaviour (a tab that reloads).
function validatingRecord(
  overrides: Partial<DatasetRecord> = {},
): DatasetRecord {
  return {
    id: "ds_abc123",
    filename: "big.jsonl",
    created_at: 1756160400,
    status: "validating",
    report: null,
    progress: { bytes_read: 500, bytes_total: 1000, rows: 42 },
    ...overrides,
  };
}

describe("ValidationProgressView", () => {
  it("shows the dataset being validated with a proportion complete", () => {
    render(<ValidationProgressView record={validatingRecord()} />);
    expect(screen.getByText("Validating your dataset…")).toBeVisible();
    expect(screen.getByText(/50%/)).toBeVisible();
    expect(screen.getByText(/42 rows read/)).toBeVisible();
    expect(screen.getByText("big.jsonl")).toBeVisible();
  });

  it("shows an indeterminate state before any progress is reported", () => {
    render(
      <ValidationProgressView record={validatingRecord({ progress: null })} />,
    );
    expect(screen.getByText("Reading rows…")).toBeVisible();
  });

  it("offers a way back while waiting", () => {
    render(<ValidationProgressView record={validatingRecord()} />);
    expect(
      screen.getByRole("link", { name: "Back to upload" }),
    ).toBeVisible();
  });

  it("reloads itself until validation finishes", () => {
    render(<ValidationProgressView record={validatingRecord()} />);
    // React 19 hoists the tag into the document head, where a browser honours
    // it -- the same mechanism the job record uses while a job is running.
    const refresh = document.querySelector('meta[http-equiv="refresh"]');
    expect(refresh).toHaveAttribute("content", "2");
  });
});
