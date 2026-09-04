import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import TokenCountView from "@/components/TokenCountView";
import type { DatasetRecord, TokenDistribution } from "@/lib/api/generated/client";

// The counting phase (issue #42) has its own state on the record: while it
// runs the view shows progress, when it lands it shows the total, the
// distribution across rows and the rows that would be truncated, and when it
// fails it says so without blocking the report. A record that was never
// counted (an invalid dataset) shows nothing.

function dist(overrides: Partial<TokenDistribution> = {}): TokenDistribution {
  return {
    total_tokens: 1234,
    rows_counted: 3,
    sequence_len: 2048,
    truncated_rows: 1,
    max_row_tokens: 3000,
    histogram: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    histogram_edges: [
      0, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192, 16384, 32768,
      65536, 131072,
    ],
    ...overrides,
  };
}

function record(overrides: Partial<DatasetRecord> = {}): DatasetRecord {
  return {
    id: "ds_abc123",
    filename: "d.jsonl",
    created_at: 1756160400,
    status: "valid",
    report: {
      valid: true,
      row_count: 3,
      usable_rows: 3,
      schema_type: "chat",
      enable_thinking: false,
      errors: [],
      warnings: [],
      preview: [],
    },
    ...overrides,
  };
}

describe("TokenCountView", () => {
  it("shows progress while the count is being produced", () => {
    render(
      <TokenCountView
        record={record({
          token_count_status: "counting",
          counting_progress: {
            bytes_read: 50,
            bytes_total: 100,
            rows: 1,
          },
        })}
      />,
    );
    expect(screen.getByText("Counting tokens…")).toBeVisible();
    expect(screen.getByText(/50%/)).toBeVisible();
    expect(screen.getByRole("progressbar")).toBeVisible();
  });

  it("shows the total, the distribution and the truncation when done", () => {
    render(
      <TokenCountView
        record={record({
          token_count_status: "done",
          report: {
            ...record().report!,
            token_count: 1234,
            token_distribution: dist(),
          },
        })}
      />,
    );
    expect(screen.getByText("Token count")).toBeVisible();
    expect(screen.getByText("1,234")).toBeVisible();
    expect(screen.getByText("Longest row")).toBeVisible();
    expect(screen.getByText("3,000")).toBeVisible();
    expect(screen.getByText("Would be truncated")).toBeVisible();
    expect(
      screen.getByText(/1 row is longer than the 2048-token sequence/),
    ).toBeVisible();
    expect(
      screen.getByRole("img", { name: /distribution/i }),
    ).toBeVisible();
  });

  it("pluralises the truncation note", () => {
    render(
      <TokenCountView
        record={record({
          token_count_status: "done",
          report: {
            ...record().report!,
            token_count: 1234,
            token_distribution: dist({ truncated_rows: 2 }),
          },
        })}
      />,
    );
    expect(
      screen.getByText(/2 rows are longer than the 2048-token sequence/),
    ).toBeVisible();
  });

  it("caps the distribution at 7 bars, collapsing everything past the cutoff into one", () => {
    render(
      <TokenCountView
        record={record({
          token_count_status: "done",
          report: {
            ...record().report!,
            token_count: 1234,
            // Rows land in the first bin, at the cutoff, and in the far tail
            // (131072+); with 14 histogram edges that would be 14 separate
            // bars if none were collapsed.
            token_distribution: dist({
              rows_counted: 3,
              histogram: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            }),
          },
        })}
      />,
    );
    // The two rows at or past the 2048 cutoff read as one bar, not two.
    expect(screen.getByText("2048+")).toBeVisible();
    expect(screen.getByText("2 (67%)")).toBeVisible();
    // Nothing from the collapsed tail keeps its own bin.
    expect(screen.queryByText(/3072/)).not.toBeInTheDocument();
    expect(screen.queryByText(/131072/)).not.toBeInTheDocument();
  });

  it("says the count is unavailable when the phase failed", () => {
    render(
      <TokenCountView
        record={record({ token_count_status: "failed" })}
      />,
    );
    expect(screen.getByText("Token count unavailable")).toBeVisible();
    expect(
      screen.getByText(/you can still proceed/i),
    ).toBeVisible();
  });

  it("renders nothing for a dataset that was never counted", () => {
    const { container } = render(
      <TokenCountView record={record()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
