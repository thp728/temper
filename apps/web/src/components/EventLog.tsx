"use client";

import { useState } from "react";
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

  return (
    <section aria-labelledby="output-heading" className="space-y-2">
      <h2 id="output-heading" className="text-lg font-semibold">
        Output
      </h2>
      {/* Disclosure: where a limit still applies the page says what it is
          showing and of how many (issue #56). When not truncated this is still
          honest: "Showing 21 of 21 events". */}
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
      <div
        role="log"
        aria-label="Output"
        tabIndex={0}
        className="max-h-80 overflow-y-auto rounded-lg border bg-card p-3 font-mono text-xs"
      >
        {events.length === 0 ? (
          <div className="font-sans text-muted-foreground">Nothing recorded yet.</div>
        ) : (
          events.map((e) => <div key={e.id}>{e.message}</div>)
        )}
      </div>
      {hasMore && (
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => void loadMore()}
            disabled={loading}
            aria-label="Load more events"
          >
            {loading ? "Loading…" : `Load more events (${total - events.length} remaining)`}
          </Button>
          {error && <span className="text-sm text-destructive">{error}</span>}
        </div>
      )}
    </section>
  );
}
