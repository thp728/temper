"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowDown, Check, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getEventsV1JobsJobIdEventsGet } from "@/lib/api/generated/client";
import type { JobEvent } from "@/lib/api/generated/client";

// The finished job's event log with #56's guardrail: when the stored history
// is larger than the page the API returns (cap 500), the interface says what
// it is showing and of how many, and the full record remains reachable by
// paging with `after`. A silent truncation is the defect; a stated one is a
// feature. The typical promoted run (~20 events) never truncates, so the
// disclosure reads "Showing 21 of 21" and no paging control appears; a
// genuinely large run shows "Showing 500 of 734" with a control to page.
//
// A terminal job's log is a fixed record, not a live tail: there is no
// worker to filter by (one job runs on one machine) and nothing still
// arriving to call "live", so this console skips both rather than fake
// them. "Jump to bottom" and "Copy" stay, because they need nothing beyond
// what is already rendered.
export default function EventLog({
  jobId,
  initialEvents,
  initialTotal,
}: {
  jobId: string;
  initialEvents: JobEvent[];
  initialTotal: number;
}) {
  const [events, setEvents] = useState<JobEvent[]>(initialEvents);
  const [total, setTotal] = useState<number>(initialTotal);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [atBottom, setAtBottom] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  const hasMore = events.length < total;
  const lastId = events.length > 0 ? events[events.length - 1]!.id : 0;

  const loadMore = async () => {
    setLoading(true);
    setError(null);
    try {
      const page = await getEventsV1JobsJobIdEventsGet(jobId, {
        after: lastId,
        limit: 500,
      });
      // The page is the next oldest-first window; append in order so the
      // history stays chronological. `total` travels on every page, so keep it
      // fresh in case the job grew, though a terminal job's total is frozen.
      setEvents((prev) => [...prev, ...page.events]);
      setTotal(page.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load more events");
    } finally {
      setLoading(false);
    }
  };

  function checkAtBottom() {
    const el = logRef.current;
    if (!el) return;
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
  }

  // A page of loaded history can turn a short log into a scrollable one;
  // re-checked whenever the rendered lines change, not just on scroll.
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
    <section aria-labelledby="output-heading" className="space-y-2">
      <h2 id="output-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
        Logs
      </h2>
      <div className="overflow-hidden rounded-[12px] border bg-card">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b bg-muted/30 px-3 py-2">
          {/* Disclosure: where a limit still applies the page says what it is
              showing and of how many (issue #56). When not truncated this is
              still honest: "Showing 21 of 21 events". */}
          <p
            className="text-sm text-muted-foreground"
            aria-live="polite"
            data-testid="event-disclosure"
          >
            {total > 0
              ? `Showing ${events.length} of ${total} events`
              : events.length === 0
                ? "No events yet"
                : `Showing ${events.length} events`}
            {hasMore ? " (paginated, limit 500). The full record remains reachable." : ""}
          </p>
          <div className="flex items-center gap-2">
            {error && <span className="text-sm text-destructive">{error}</span>}
            {hasMore && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => void loadMore()}
                disabled={loading}
                aria-label="Load more events"
              >
                {loading ? "Loading…" : `Load more events (${total - events.length} remaining)`}
              </Button>
            )}
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
    </section>
  );
}
