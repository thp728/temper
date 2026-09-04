import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ReportView from "@/components/ReportView";
import type { DatasetRecord } from "@/lib/api/generated/client";

// ReportView renders DatasetActionsMenu for its rename/delete affordance,
// which reaches for the app router and the generated client's mutations --
// neither exists in this render-only environment, so both are stubbed the
// same way DatasetActionsMenu.test.tsx stubs them.
vi.mock("next/navigation", () => ({
  useRouter: () => ({ refresh: vi.fn(), push: vi.fn() }),
}));
vi.mock("@/lib/api/generated/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/generated/client")>()),
  deleteDatasetV1DatasetsDatasetIdDelete: vi.fn(),
  renameDatasetV1DatasetsDatasetIdPatch: vi.fn(),
}));

// The assertions are what a user reads: a rejected line's number, an error's
// stable code, the counts, the status badge. Never markup structure.
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

  it("shows warnings on an otherwise valid dataset without blocking it", () => {
    render(<ReportView record={record()} />);
    expect(screen.getByText("few_rows")).toBeVisible();
    expect(
      screen.getByText(/do not block/i, { selector: "p" }),
    ).toBeVisible();
    expect(screen.getByText("Ready")).toBeVisible();
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

    const errorsRegion = screen.getByRole("region", {
      name: /errors \(2\)/i,
    });
    expect(errorsRegion).toBeVisible();
  });

  it("marks a rejected dataset as needing fixes", () => {
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
    expect(screen.getByText("Needs fixes")).toBeVisible();
  });

  it("explains a mixed thinking-mode block on the report", () => {
    const mixed = record({
      status: "invalid",
      report: {
        ...record().report!,
        valid: false,
        errors: [
          {
            line: null,
            code: "mixed_thinking",
            message:
              "Dataset mixes reasoning traces with plain responses: every row must be consistent.",
          },
        ],
        warnings: [],
        preview: [],
      },
    });
    render(<ReportView record={mixed} />);
    // The block is explained in plain language, with its stable code.
    expect(screen.getByText("mixed_thinking")).toBeVisible();
    expect(screen.getByText(/mixes reasoning traces/i)).toBeVisible();
  });

  it("describes a thinking-mode dataset as detected", () => {
    const r = record({ report: { ...record().report!, enable_thinking: true } });
    render(<ReportView record={r} />);
    expect(screen.getByText("Detected")).toBeVisible();
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

  it("renders the accepted schema as a code block for an unrecognised schema", () => {
    const r = record({
      status: "invalid",
      report: {
        ...record().report!,
        valid: false,
        errors: [
          {
            line: null,
            code: "unrecognised_schema",
            message:
              'No row has a \'messages\' list. Temper accepts chat-format JSONL: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}. Keys found instead: [\'input\', \'instruction\', \'output\'].',
          },
        ],
        warnings: [],
      },
    });
    render(<ReportView record={r} />);
    // The prose and the found-keys line stay readable text.
    expect(
      screen.getByText(/temper accepts chat-format jsonl:/i),
    ).toBeVisible();
    expect(screen.getByText(/keys found instead/i)).toBeVisible();
    // The example itself renders as a code block, reformatted for
    // readability, not as one long line buried in the sentence.
    const code = document.querySelector("pre code");
    expect(code).not.toBeNull();
    expect(code!.textContent).toContain('"messages"');
    expect(code!.textContent).toContain("\n");
  });
});
