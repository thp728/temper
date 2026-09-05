"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowDown, Check, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { JobEvent } from "@/lib/api/generated/client";

// The console box itself -- dark, monospace, scrollable, with copy and
// jump-to-bottom -- shared by the finished record's paginated log
// (`EventLog`) and the running view's live tail (issue: non-terminal/
// terminal parity). Paging is each caller's own concern (a finished job
// pages through a fixed history; a live one has nothing to page, only more
// arriving), so this component knows only how to show the lines it is given
// and let a reader copy or jump to the newest one -- exactly the two things
// that stay true whether the record is still growing or never will again.
export default function LogConsole({
  events,
  disclosure,
  extraAction,
}: {
  events: JobEvent[];
  disclosure: React.ReactNode;
  extraAction?: React.ReactNode;
}) {
  const [copied, setCopied] = useState(false);
  const [atBottom, setAtBottom] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  function checkAtBottom() {
    const el = logRef.current;
    if (!el) return;
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
  }

  // A page of loaded history, or a stream appending in place, can turn a
  // short log into a scrollable one; re-checked whenever the rendered lines
  // change, not just on scroll.
  useEffect(checkAtBottom, [events]);

  function scrollToBottom() {
    const el = logRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }

  async function copyLog() {
    await navigator.clipboard.writeText(events.map((e) => e.message).join("\n"));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="overflow-hidden rounded-[12px] border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b bg-muted/30 px-3 py-2">
        <div className="text-sm text-muted-foreground" aria-live="polite">
          {disclosure}
        </div>
        <div className="flex items-center gap-2">
          {extraAction}
          {events.length > 0 && (
            <Button variant="ghost" size="sm" onClick={() => void copyLog()}>
              {copied ? (
                <Check aria-hidden className="size-3.5" />
              ) : (
                <Copy aria-hidden className="size-3.5" />
              )}
              {copied ? "Copied" : "Copy"}
            </Button>
          )}
        </div>
      </div>
      <div className="relative">
        <div
          ref={logRef}
          onScroll={checkAtBottom}
          role="log"
          aria-label="Logs"
          tabIndex={0}
          className="max-h-[28rem] overflow-y-auto bg-[#0a0a0c] p-3 font-mono text-xs leading-relaxed text-muted-foreground"
        >
          {events.length === 0 ? (
            <div className="font-sans text-muted-foreground">Nothing recorded yet.</div>
          ) : (
            events.map((e) => (
              <div key={e.id} className="whitespace-pre-wrap">
                {e.message}
              </div>
            ))
          )}
        </div>
        {!atBottom && events.length > 0 && (
          <button
            type="button"
            onClick={scrollToBottom}
            className="absolute right-3 bottom-3 inline-flex items-center gap-1.5 rounded-full bg-foreground px-3 py-1.5 text-xs font-medium text-background shadow-lg transition-opacity hover:opacity-90"
          >
            <ArrowDown aria-hidden className="size-3.5" />
            Jump to bottom
          </button>
        )}
      </div>
    </div>
  );
}
