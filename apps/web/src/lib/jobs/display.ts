// Presentation over published numbers: the formats the server-rendered pages
// showed, defined once so the list and the record cannot drift apart. No
// component knowledge, no React -- these are functions over the contract.

// An epoch stamp as local YYYY-MM-DD HH:mm:ss, which is what the old pages'
// `datetimeformat` filter produced and what a record is read against.
export function formatTimestamp(epochSeconds: number): string {
  // "sv-SE" formats exactly this shape across runtimes; the date is built
  // from epoch seconds so the value stays the server's own clock reading.
  return new Date(epochSeconds * 1000).toLocaleString("sv-SE");
}

// Ported from `_fmt_duration`: hours, minutes and zero-padded seconds.
export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  const secs = s % 60;
  const two = (n: number) => String(n).padStart(2, "0");
  if (hours) return `${hours}h ${two(minutes)}m ${two(secs)}s`;
  if (minutes) return `${minutes}m ${two(secs)}s`;
  return `${secs}s`;
}

// A pinned revision is forty characters; twelve is what fits beside a model
// name and what the old pages showed.
export function shortRevision(revision: string | null | undefined): string {
  return (revision ?? "").slice(0, 12);
}

// The statuses at which a record stops changing -- `db.TERMINAL_STATES` seen
// from the browser, where no database import can reach. One copy, so a
// status added server-side surfaces here as exactly one edit.
export const TERMINAL_STATUSES = ["complete", "failed", "cancelled"];

// What the stable codes mean, where the API's message alone does not say it:
// both safety limits read as defects unless someone explains that they are
// circuit breakers working as designed. Everything else carries its reason
// in its own message.
const EXPLANATIONS: Record<string, string> = {
  gpu_stalled:
    "The job was stopped by a safety limit: it produced no output for longer than the stall timeout allows.",
  gpu_max_duration_exceeded:
    "The job was stopped by a safety limit: it ran past the maximum duration a single job may use.",
};

export function failureExplanation(code: string | null | undefined): string | null {
  return (code && EXPLANATIONS[code]) || null;
}

// The clock, for a page that must state how long something has been going.
// Server Components run once per request on the server, where reading the
// clock is legitimate; the React purity lint cannot see that boundary, so
// the read lives behind this named seam rather than inline in a component.
export function epochNow(): number {
  return Date.now() / 1000;
}
