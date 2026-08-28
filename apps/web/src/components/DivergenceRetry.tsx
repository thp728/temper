"use client";

import * as React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import type { JobRecord } from "@/lib/api/generated/client";

export default function DivergenceRetry({ job }: { job: JobRecord }) {
  const canRetry = job.error_code === "training_diverged";
  const [retryState, setRetryState] = React.useState<"idle" | "loading" | "done" | "error">("idle");
  const [retryError, setRetryError] = React.useState<string | null>(null);
  const [retryId, setRetryId] = React.useState<string | null>(null);

  if (!canRetry) return null;

  const doRetry = async () => {
    setRetryState("loading");
    setRetryError(null);
    try {
      const res = await fetch(`/v1/jobs/${job.id}/retry`, { method: "POST" });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = body?.detail?.message || body?.detail || `Retry failed (${res.status})`;
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
      const newId = body?.id || body?.job_id || "";
      setRetryId(newId);
      setRetryState("done");
    } catch (e) {
      setRetryError(e instanceof Error ? e.message : String(e));
      setRetryState("error");
    }
  };

  return (
    <div className="space-y-2 rounded border bg-card p-3">
      <p className="text-sm text-muted-foreground">
        This run diverged — the loss became meaningless. You can retry once at half the learning
        rate as a choice, not an automatic rerun, because a diverging run usually means the data or
        the rate is wrong and repeating it is rarely the answer.
      </p>
      {retryState === "idle" && (
        <Button variant="outline" size="sm" onClick={() => void doRetry()}>
          Retry at half learning rate
        </Button>
      )}
      {retryState === "loading" && <p className="text-sm">Creating retry…</p>}
      {retryState === "done" && retryId && (
        <p className="text-sm">
          Retry created:{" "}
          <Link href={`/jobs/${retryId}`} className="underline hover:no-underline">
            {retryId}
          </Link>
        </p>
      )}
      {retryState === "error" && retryError && <p className="text-sm text-destructive">{retryError}</p>}
      {job.retry_from && (
        <p className="text-xs text-muted-foreground">
          This job was itself a retry of{" "}
          <Link href={`/jobs/${job.retry_from}`} className="underline">
            {job.retry_from}
          </Link>
          .
        </p>
      )}
    </div>
  );
}
