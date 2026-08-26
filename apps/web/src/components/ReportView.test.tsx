import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ReportView from "@/components/ReportView";
import type { DatasetRecord } from "@/lib/api/generated/client";

// The assertions are what a user reads: a rejected line's number, an error's
// stable code, the counts, the proceed action. Never markup structure.
function record(overrides: Partial<DatasetRecord> = {}): DatasetRecord {
  return {
    id: "ds_abc123",
    filename: "d.jsonl",
    created_at: 1756160400,
    status: "valid",
    report: {
      valid: true,
      row_count: 12,
      usable_rows: 12,
      schema_type: "chat",
      enable_thinking: false,
      errors: [],
      warnings: [
        {
          line: null,
          code: "few_rows",
          message:
            "12 usable rows. Training will run, but results are typically weak below ~50 examples.",
        },
      ],
      preview: [
        {
          messages: [
            { role: "user", content: "q0" },
            { role: "assistant", content: "a0" },
          ],
        },
      ],
    },
    ...overrides,
  };
}

describe("ReportView", () => {
  // The stat cards pair a label with its value; read them as pairs rather
  // than matching bare numbers, which repeat across cards.
  function statValue(name: string): string | null | undefined {
    return screen.getByText(name).nextElementSibling?.textContent;
  }

  it("shows row count, usable rows, schema and thinking mode in plain language", () => {
    render(<ReportView record={record()} />);
    expect(statValue("Rows found")).toBe("12");
    expect(statValue("Usable rows")).toBe("12");
    expect(statValue("Schema")).toBe("chat");
    expect(screen.getByText("Not detected")).toBeVisible();
    // The label alone is a fact; the explanation says what it means for the
    // run -- the old page's behaviour, moved.
    expect(
      screen.getByText(/trained to answer directly/),
    ).toBeVisible();
  });

  it("moves focus to the report heading on arrival", () => {
    render(<ReportView record={record()} />);
    const heading = screen.getByRole("heading", { level: 1 });
    expect(heading).toHaveAttribute("tabindex", "-1");
    expect(heading).toHaveFocus();
  });

  it("shows a preview of how the rows were understood", () => {
    render(<ReportView record={record()} />);
    expect(screen.getByText(/q0/)).toBeVisible();
    expect(screen.getByText(/a0/)).toBeVisible();
    expect(screen.getByText(/user:/i)).toBeVisible();
    expect(screen.getByText(/assistant:/i)).toBeVisible();
  });

  it("offers proceeding to a valid dataset even when warnings exist", () => {
    render(<ReportView record={record()} />);
    // The warning is on the page with its stable code...
    expect(screen.getByText("few_rows")).toBeVisible();
    expect(
      screen.getByText(/do not block/i, { selector: "p" }),
    ).toBeVisible();
    // ...and the journey continues anyway.
    expect(
      screen.getByRole("link", { name: "Choose a model and continue" }),
    ).toBeVisible();
  });

  it("names each problem against its line for a rejected dataset", () => {
    const rejected = record({
      status: "invalid",
      report: {
        valid: false,
        row_count: 13,
        usable_rows: 11,
        schema_type: "chat",
        enable_thinking: false,
        errors: [
          {
            line: 12,
            code: "invalid_json",
            message: "Not valid JSON: Expecting property name at column 2.",
          },
          {
            line: 13,
            code: "empty_target",
            message: "Final assistant turn is empty.",
          },
        ],
        warnings: [],
        preview: [],
      },
    });
    render(<ReportView record={rejected} />);

    expect(
      screen.getByText(/this dataset was rejected/i),
    ).toBeVisible();
    expect(screen.getByText("Line 12")).toBeVisible();
    expect(screen.getByText("invalid_json")).toBeVisible();
    expect(screen.getByText("Line 13")).toBeVisible();
    expect(screen.getByText("empty_target")).toBeVisible();

    const problems = screen.getByRole("region", {
      name: /problems \(2\)/i,
    });
    expect(problems).toBeVisible();
  });

  it("does not offer proceeding to a rejected dataset", () => {
    const rejected = record({
      status: "invalid",
      report: {
        ...record().report!,
        valid: false,
        errors: [
          { line: null, code: "too_few_rows", message: "1 usable row(s)." },
        ],
      },
    });
    render(<ReportView record={rejected} />);
    expect(
      screen.queryByRole("link", { name: "Choose a model and continue" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Back to upload" }),
    ).toBeVisible();
  });

  it("describes a thinking-mode dataset as detected", () => {
    const r = record({ report: { ...record().report!, enable_thinking: true } });
    render(<ReportView record={r} />);
    expect(screen.getByText("Detected")).toBeVisible();
    expect(
      screen.getByText(/thinking mode enabled/),
    ).toBeVisible();
  });

  it("renders a file-level error without inventing a line number", () => {
    const r = record({
      status: "invalid",
      report: {
        ...record().report!,
        valid: false,
        errors: [
          { line: null, code: "empty", message: "File contains no rows." },
        ],
        warnings: [],
      },
    });
    render(<ReportView record={r} />);
    expect(screen.getByText("Whole file")).toBeVisible();
    expect(screen.queryByText(/^line \d+$/i)).not.toBeInTheDocument();
  });
});
