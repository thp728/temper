// Presentation over the published progress figures (issue #49): how a phase's
// bytes, measured rate and estimate are rendered. One definition each, so the
// live view and the finished record cannot drift apart. Pure functions over
// the contract -- no component knowledge, no React.

import type { JobOutputLine, JobProgress } from "@/lib/api/generated/client";

// A byte count as docker/tqdm write one: SI base (1000), one decimal for
// anything above a kilobyte -- `15.2 MB`, `4.0 GB` -- matching the same
// convention `temper_core.progress` parses, so a number means the same figure
// wherever it is drawn.
export function formatBytes(bytes: number): string {
  const units = ["B", "kB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  if (unit === 0) return `${Math.round(value)} ${units[unit]}`;
  return `${value.toFixed(1)} ${units[unit]}`;
}

// A measured rate in bytes per second, as the live figure the interface
// shows: `12.4 MB/s`. Null when nothing has been measured yet.
export function formatRate(bytesPerSecond: number | null | undefined): string {
  if (bytesPerSecond == null) return "measuring…";
  return `${formatBytes(bytesPerSecond)}/s`;
}

// An estimate as a human interval, reusing the duration format the record
// already uses so the two never disagree about how long a minute is. Null
// when there is no estimate to render.
export function formatEta(etaSeconds: number | null | undefined): string | null {
  if (etaSeconds == null) return null;
  const total = Math.max(0, Math.round(etaSeconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  if (minutes < 60) {
    return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

// How far a phase has got, 0–1, or null when there is no total to divide by.
// The proportion a bar draws from; the finished record and the live view read
// the same function so they cannot draw different widths.
export function proportion(
  done: number | null | undefined,
  total: number | null | undefined,
): number | null {
  if (done == null || total == null || total <= 0) return null;
  return Math.min(1, Math.max(0, done / total));
}

// The retained raw lines grouped by the phase they were promoted into, so the
// collapsed detail is offered per phase and nothing is discarded.
export function outputByPhase(output: JobOutputLine[]): Map<string, JobOutputLine[]> {
  const grouped = new Map<string, JobOutputLine[]>();
  for (const line of output) {
    const bucket = grouped.get(line.phase) ?? [];
    bucket.push(line);
    grouped.set(line.phase, bucket);
  }
  return grouped;
}

// The latest per phase, replacing rather than appending (issue #49): a newer
// progress record for a phase supersedes the older one, whatever order they
// arrive in off the stream.
export function supersedeProgress(
  current: JobProgress[],
  incoming: JobProgress[],
): JobProgress[] {
  const byPhase = new Map(current.map((p) => [p.phase, p]));
  for (const p of incoming) {
    byPhase.set(p.phase, p);
  }
  return [...byPhase.values()];
}
