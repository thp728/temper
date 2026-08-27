import type { JobEvent } from "@/lib/api/generated/client";

// The live channel for a running job. Transport-only, like mutator.ts: it
// owns how the stream is addressed and parsed, and knows nothing about the
// job domain beyond the JobEvent shape the contract publishes.
//
// The generated client cannot carry this -- an EventSource is not a fetch,
// and `apiFetch` would try to JSON-parse the stream -- so the path lives here,
// the same way the artifact's download URL lives in the record view. The path
// is the API's own (`GET /v1/jobs/{id}/stream`), published in the contract,
// and the journeys exercise it end to end, so drift fails a test rather than
// a page.

export function jobStreamUrl(jobId: string, after: number): string {
  return `/v1/jobs/${jobId}/stream?after=${after}`;
}

// One event's JSON off the wire, refused if it is not a job event. The
// server only ever sends this shape; a malformed line is ignored rather than
// let crash the listener (an uncaught throw inside an EventSource handler
// would otherwise take the whole live view down with it).
export function parseJobEvent(raw: string): JobEvent {
  const parsed: unknown = JSON.parse(raw);
  const candidate = parsed as { id?: unknown; job_id?: unknown; kind?: unknown };
  if (
    typeof parsed !== "object" ||
    parsed === null ||
    typeof candidate.id !== "number" ||
    typeof candidate.job_id !== "string" ||
    typeof candidate.kind !== "string"
  ) {
    throw new Error("the stream delivered something that is not a job event");
  }
  return parsed as JobEvent;
}
