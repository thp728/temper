// Presentation over published numbers: the formats the server-rendered pages
// showed, defined once so the list and the record cannot drift apart. No
// component knowledge, no React -- these are functions over the contract.

import type { JobEvent } from "@/lib/api/generated/client";

// A timestamp as a person reads it: relative while it is fresh ("just now",
// "3 min ago", "4 hrs ago"), an absolute "Sep 4, 2026" once it is older than
// a day. `nowSeconds` is a parameter so tests pin the clock -- and so a
// server-rendered page passes its own reading down (the dashboard's `now`,
// the jobs page's `now`): hydration must never see a clock the client
// computed for itself (`RunningJobView`'s rule), so components rendered on
// the server take `now` as a prop and only purely client-side surfaces fall
// back to the live clock. A future stamp (a quote expiry, a clock skew)
// mirrors the past -- "in 3 min" -- and an absent stamp is a dash, the same
// rule `formatMinorCost` applies.
const DAY_S = 24 * 3600;

export function formatTimestamp(
  epochSeconds: number | null | undefined,
  nowSeconds: number = Date.now() / 1000,
): string {
  if (epochSeconds == null) {
    return "—";
  }
  const delta = nowSeconds - epochSeconds;
  if (Math.abs(delta) < 60) {
    return "just now";
  }
  if (delta > 0 && delta < 3600) {
    const minutes = Math.floor(delta / 60);
    return `${minutes} min ago`;
  }
  if (delta > 0 && delta < DAY_S) {
    const hours = Math.floor(delta / 3600);
    return `${hours} hr${hours === 1 ? "" : "s"} ago`;
  }
  if (delta < 0 && delta > -3600) {
    const minutes = Math.floor(-delta / 60);
    return `in ${minutes} min`;
  }
  if (delta < 0 && delta > -DAY_S) {
    const hours = Math.floor(-delta / 3600);
    return `in ${hours} hr${hours === 1 ? "" : "s"}`;
  }
  // "en-US" with a short month is exactly this shape across runtimes, and
  // the day is read in UTC so a server in one timezone and a browser in
  // another agree on it: the value stays a deterministic function of the
  // epoch seconds, which is what keeps server-rendered HTML and hydration
  // from disagreeing about what yesterday means.
  return new Date(epochSeconds * 1000).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

// The exact stamp behind a friendly date, for tooltips: UTC, locale-free,
// and identical on server and client, so it never breaks hydration the way
// a bare `toLocaleString()` does (the server's locale is not the browser's).
export function formatExactTimestamp(
  epochSeconds: number | null | undefined,
): string {
  if (epochSeconds == null) {
    return "—";
  }
  return (
    new Date(epochSeconds * 1000)
      .toISOString()
      .replace("T", " ")
      .replace(/\.\d+Z$/, " UTC")
  );
}

// A duration as a zero-padded HH:MM:SS clock ("00:02:14") -- the one duration
// format everywhere in the app, from the jobs table to elapsed times, ETAs,
// quote ranges and endpoint lifetimes. One shape so aligned columns scan and
// no two surfaces disagree about how long a minute is.
export function formatDuration(
  seconds: number | null | undefined,
): string {
  if (seconds == null) {
    return "—";
  }
  const s = Math.max(0, Math.floor(seconds));
  const two = (n: number) => String(n).padStart(2, "0");
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  return `${two(hours)}:${two(minutes)}:${two(s % 60)}`;
}

// A pinned revision is forty characters; twelve is what fits beside a model
// name and what the old pages showed.
export function shortRevision(revision: string | null | undefined): string {
  return (revision ?? "").slice(0, 12);
}

// A duration range as "low–high": never a point, because the throughput the
// quote rests on is the softest number in the model. Both ends use the same
// `formatDuration` so a reader sees comparable units.
export function formatDurationRange(
  low: number | null | undefined,
  high: number | null | undefined,
): string {
  if (low == null || high == null) {
    return "not estimable";
  }
  return `${formatDuration(low)}–${formatDuration(high)}`;
}

// A cost in a currency's smallest unit (paisa for INR, cent for USD), shown
// with the currency it is denominated in. The minor unit is a published
// number (the quote's `minor_unit`), never a formatting assumption: the
// decimal places derive from it (100 → 2 places, 1000 → 3), so a currency
// whose smallest unit is not a hundredth still prints correctly.
export function formatMinorCost(
  minor: number | null | undefined,
  currency: string,
  minorUnit: number,
): string {
  if (minor == null) {
    return "—";
  }
  const amount = minor / minorUnit;
  // 100 → 2, 1000 → 3; anything else falls back to a sensible two places.
  const decimals = Number.isInteger(Math.log10(minorUnit))
    ? Math.log10(minorUnit)
    : 2;
  const formatted = new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(amount);
  return `${currency} ${formatted}`;
}

// The statuses at which a record stops changing -- `db.TERMINAL_STATES` seen
// from the browser, where no database import can reach. One copy, so a
// status added server-side surfaces here as exactly one edit.
export const TERMINAL_STATUSES = ["complete", "failed", "cancelled"];

// The latest measured loss in a job's history: the newest metric event wins,
// whatever order older events arrive in. One definition, so the live view and
// the finished record agree about what "latest" means.
export function latestLoss(
  events: JobEvent[],
): { loss: number; step?: number } | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (!e || e.kind !== "metric" || !e.data) continue;
    const loss = e.data["loss"];
    if (typeof loss !== "number") continue;
    const step = e.data["step"];
    return { loss, step: typeof step === "number" ? step : undefined };
  }
  return null;
}

// What the stable codes mean, where the API's message alone does not say it:
// both safety limits read as defects unless someone explains that they are
// circuit breakers working as designed. Everything else carries its reason
// in its own message.
const EXPLANATIONS: Record<string, string> = {
  gpu_stalled:
    "The job was stopped by a safety limit: it produced no output for longer than the stall timeout allows.",
  gpu_max_duration_exceeded:
    "The job was stopped by a safety limit: it ran past the maximum duration a single job may use.",
  training_diverged:
    "Training diverged, and the loss became meaningless (NaN or exploding) and the run was stopped early so you are not billed for hours that cannot produce anything. Try a lower learning rate or check your data. A single retry at half the learning rate is available as a choice.",
  training_instability:
    "Training instability detected: loss is spiking well above its recent average. This may be early divergence; the run is continuing, but the signal is worth watching and a lower learning rate may help if it continues.",
  // Issue #35: an out-of-memory failure is a memory recovery's business, not
  // a divergence abort. The stable codes keep the two distinguishable on the
  // page, and this explanation says what a genuine exhaustion means.
  training_oom:
    "The job ran out of device memory during training. It was retried automatically with the effective batch preserved: the per-step batch was halved and accumulation doubled, so your training did not change.",
  memory_retries_exhausted:
    "The job ran out of device memory repeatedly, and every memory reduction the platform can apply was tried (per-step batch, gradient checkpointing, sequence length, more capable hardware). The last attempt still did not fit, so the run was stopped rather than retried forever. The attempts are recorded on this job. Try a smaller configuration, a shorter sequence length, or a model that predicts a smaller peak.",
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
