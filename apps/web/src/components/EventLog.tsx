"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import LogConsole from "@/components/LogConsole";
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
// them. The console box itself -- copy, jump-to-bottom, dark scrollback --
// is `LogConsole`, shared with the running view's live tail so the Logs tab
// looks like the same surface before and after the job ends; paging is this
// component's own concern, layered on top as `LogConsole`'s `extraAction`.
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
      <h2 id="output-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
        Logs
      </h2>
      <LogConsole
        events={events}
        disclosure={
          <span data-testid="event-disclosure">
            {total > 0
              ? `Showing ${events.length} of ${total} events`
              : events.length === 0
                ? "No events yet"
                : `Showing ${events.length} events`}
            {hasMore ? " (paginated, limit 500). The full record remains reachable." : ""}
          </span>
        }
        extraAction={
          <>
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
          </>
        }
      />
    </section>
  );
}
