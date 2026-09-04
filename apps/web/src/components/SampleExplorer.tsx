"use client";

import { useState } from "react";
import { ChevronLeft, ChevronRight, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { PreviewRow, PreviewTurn } from "@/lib/api/generated/client";

// Wireframe 2's "Sample Explorer" scaled to what the report actually carries:
// the validator caps its preview at three rows (temper_core.validation.
// MAX_PREVIEW), so "Sample 1 of 3" is a real bound, not a fabricated total --
// there is no per-row token count, domain tag or quality score published,
// so this shows only role, content and a raw-JSON view of the same row.

type ViewMode = "rendered" | "json";

// A colour-coded left rule instead of a bordered, headered box per turn --
// one line of separation rather than three (box border, header divider,
// header background), which was more segmentation than a chat transcript
// needs.
const ROLE_BORDER: Record<string, string> = {
  system: "border-secondary",
  user: "border-primary",
  assistant: "border-success",
};

function TurnBlock({ turn }: { turn: PreviewTurn }) {
  const role = turn.role ?? "(no role)";
  return (
    <div
      className={`border-l-2 py-0.5 pl-3 ${ROLE_BORDER[role] ?? "border-muted-foreground"}`}
    >
      <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {role}:
      </span>
      <p className="mt-0.5 whitespace-pre-wrap text-sm text-muted-foreground">
        {turn.content}
      </p>
    </div>
  );
}

async function copyRow(row: PreviewRow) {
  try {
    await navigator.clipboard.writeText(JSON.stringify(row, null, 2));
  } catch {
    // Clipboard access can be refused by the browser; the button has
    // nothing useful to fall back to, so the attempt is silently dropped.
  }
}

export default function SampleExplorer({ rows }: { rows: PreviewRow[] }) {
  const [index, setIndex] = useState(0);
  const [view, setView] = useState<ViewMode>("rendered");

  if (rows.length === 0) return null;
  // `rows.length` is checked above, so this index is always in bounds --
  // the array access still reads as possibly-undefined under
  // noUncheckedIndexedAccess.
  const row = rows[Math.min(index, rows.length - 1)]!;

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-muted/20 px-4 py-2">
        <span
          className="rounded bg-muted px-2 py-0.5 font-mono text-xs text-muted-foreground"
          title="The validator keeps only the first few rows for preview -- this is not the whole file."
        >
          Sample {index + 1} of {rows.length}
        </span>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="outline"
              size="icon-sm"
              aria-label="Previous sample"
              disabled={index === 0}
              onClick={() => setIndex((i) => Math.max(0, i - 1))}
            >
              <ChevronLeft aria-hidden="true" />
            </Button>
            <Button
              type="button"
              variant="outline"
              size="icon-sm"
              aria-label="Next sample"
              disabled={index === rows.length - 1}
              onClick={() => setIndex((i) => Math.min(rows.length - 1, i + 1))}
            >
              <ChevronRight aria-hidden="true" />
            </Button>
          </div>
          <div className="flex items-center rounded bg-muted p-0.5 text-xs">
            <button
              type="button"
              aria-pressed={view === "rendered"}
              onClick={() => setView("rendered")}
              className={`rounded px-2 py-0.5 ${
                view === "rendered"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Rendered
            </button>
            <button
              type="button"
              aria-pressed={view === "json"}
              onClick={() => setView("json")}
              className={`rounded px-2 py-0.5 ${
                view === "json"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Raw
            </button>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => copyRow(row)}
          >
            <Copy className="size-3.5" aria-hidden="true" />
            Copy
          </Button>
        </div>
      </div>

      <div className="h-[420px] overflow-y-auto p-4">
        {(row.messages?.length ?? 0) === 0 ? (
          <p className="text-sm">No messages list found in this row.</p>
        ) : view === "json" ? (
          <pre className="overflow-x-auto rounded-lg border bg-background/60 p-3 font-mono text-xs text-muted-foreground">
            {JSON.stringify(row, null, 2)}
          </pre>
        ) : (
          <div className="space-y-2">
            {(row.messages ?? []).map((turn, j) => (
              <TurnBlock key={j} turn={turn} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
