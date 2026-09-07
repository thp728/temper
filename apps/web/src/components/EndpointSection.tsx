"use client";

import { useEffect, useState } from "react";
import { Play, Square, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ApiError } from "@/lib/api/mutator";
import { formatDuration, formatTimestamp } from "@/lib/jobs/display";
import {
  createEndpointV1JobsJobIdEndpointPost,
  deleteEndpointV1JobsJobIdEndpointDelete,
  endpointPreviewV1JobsJobIdEndpointPreviewGet,
  getEndpointV1JobsJobIdEndpointGet,
  inferEndpointV1JobsJobIdEndpointInferPost,
} from "@/lib/api/generated/client";
import type {
  EndpointCreated,
  EndpointPreview,
  EndpointRecord,
} from "@/lib/api/generated/client";

function EndpointStat({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <dt className="font-mono text-[11px] tracking-wide text-muted-foreground uppercase">
        {label}
      </dt>
      <dd className="text-sm font-semibold tabular-nums">{children}</dd>
    </div>
  );
}

export default function EndpointSection({
  jobId,
  jobStatus,
}: {
  jobId: string;
  jobStatus: string;
}) {
  const [endpoint, setEndpoint] = useState<EndpointRecord | null>(null);
  const [preview, setPreview] = useState<EndpointPreview | null>(null);
  const [created, setCreated] = useState<EndpointCreated | null>(null);
  const [prompt, setPrompt] = useState("");
  const [completion, setCompletion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [inferLoading, setInferLoading] = useState(false);

  const isComplete = jobStatus === "complete";

  // This poll never clears `error`. It used to, on the 404 that means "no
  // endpoint yet" -- which is the ordinary state of every unserved job, and
  // fires every fifteen seconds. So a failed Start showed its reason and
  // then had it wiped by the next tick, and the button read as doing
  // nothing at all. That is how `endpoint_provision_failed` went unnoticed
  // against real hardware. Only a user action clears the error now, at the
  // moment it starts; a background poll may raise one but never erases what
  // the user was told.
  async function fetchEndpoint() {
    try {
      const res = await getEndpointV1JobsJobIdEndpointGet(jobId);
      const data = (res as unknown as { data?: unknown }).data ?? (res as unknown as EndpointRecord);
      // orval returns {data: EndpointRecord}
      // For safety, handle both shapes
      if (data && (data as EndpointRecord).id) {
        setEndpoint(data as EndpointRecord);
      } else if ((res as unknown as EndpointRecord).id) {
        setEndpoint(res as unknown as EndpointRecord);
      }
    } catch (e: unknown) {
      // 404 endpoint_not_found means no endpoint yet -- the ordinary state
      // of every job nobody has served yet, not an error to show the user.
      // Read the stable code/status structurally, never by substring-matching
      // a human message that can be reworded without anyone realising control
      // flow depended on it.
      if (e instanceof ApiError && (e.code === "endpoint_not_found" || e.status === 404)) {
        setEndpoint(null);
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    }
  }

  async function fetchPreview() {
    try {
      const res = await endpointPreviewV1JobsJobIdEndpointPreviewGet(jobId);
      const data = (res as unknown as { data?: unknown }).data ?? (res as unknown as EndpointPreview);
      setPreview(data as EndpointPreview);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    if (!isComplete) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void fetchEndpoint();
    void fetchPreview();
    const id = setInterval(() => void fetchEndpoint(), 15_000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, isComplete]);

  if (!isComplete) return null;

  async function onStart() {
    setLoading(true);
    setError(null);
    setCreated(null);
    try {
      const res = await createEndpointV1JobsJobIdEndpointPost(jobId);
      const data = (res as unknown as { data?: unknown }).data ?? (res as unknown as EndpointCreated);
      const createdData = (data as EndpointCreated).api_key
        ? (data as EndpointCreated)
        : (res as unknown as EndpointCreated);
      setCreated(createdData as EndpointCreated);
      setEndpoint(createdData as unknown as EndpointRecord);
      setCompletion(null);
    } catch (e: unknown) {
      if (e instanceof ApiError) {
        setError(`${e.code}: ${e.message}`);
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setLoading(false);
    }
  }

  async function onStop() {
    setLoading(true);
    setError(null);
    try {
      await deleteEndpointV1JobsJobIdEndpointDelete(jobId);
      setEndpoint(null);
      setCreated(null);
      setCompletion(null);
      // refresh preview for next start
      await fetchPreview();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onInfer() {
    if (!endpoint) return;
    if (!prompt.trim()) {
      setError("Prompt must not be empty.");
      return;
    }
    // Use the key from creation if we have it, otherwise prompt user -- for
    // the demo we reuse the prefix is not enough, so we need the full key.
    // In the UI after creation we hold the full key in `created`; if the
    // user reloads, the key is gone (stored hashed) and they must have saved
    // it -- we show the prefix and ask them to paste the full key.
    let key = created?.api_key;
    if (!key) {
      // Ask for key if not held (after reload the key is not retrievable)
      const pasted = window.prompt(
        "Enter the endpoint API key (shown once at creation, prefix " +
          (endpoint.api_key_prefix || "") +
          "). Paste the full key:",
      );
      if (!pasted) return;
      key = pasted;
    }
    setInferLoading(true);
    setError(null);
    setCompletion(null);
    try {
      const res = await inferEndpointV1JobsJobIdEndpointInferPost(jobId, { prompt }, {
        headers: { "X-API-Key": key },
      } as unknown as object);
      const data = (res as unknown as { data?: unknown }).data ?? (res as unknown as { completion: string; expires_at: number });
      setCompletion((data as { completion: string }).completion);
      // refresh expiry after use
      await fetchEndpoint();
    } catch (e: unknown) {
      if (e instanceof ApiError) {
        setError(`${e.code}: ${e.message}`);
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setInferLoading(false);
    }
  }

  // If an endpoint is running, show it; otherwise show the preview + start
  if (endpoint) {
    const isExpired = endpoint.status !== "running";
    return (
      <section aria-labelledby="endpoint-heading" className="space-y-3">
        <Card className="space-y-4 p-5">
          <div className="flex flex-wrap items-start justify-between gap-4 border-b pb-4">
            <div className="flex items-start gap-2">
              <Zap aria-hidden className="mt-0.5 size-5 shrink-0 text-primary" />
              <div>
                <h2 id="endpoint-heading" className="font-semibold">
                  Endpoint {endpoint.status}
                  {endpoint.stop_reason ? ` (${endpoint.stop_reason})` : ""}
                </h2>
                <p className="text-sm text-muted-foreground">
                  {isExpired
                    ? "This endpoint cannot serve prompts anymore."
                    : "Send it a prompt below, or stop it now."}
                </p>
              </div>
            </div>
            {/* One stop control, not the two the pre-tabs layout carried (the
                prompt panel's own button and this header's were both offered
                whenever the key from creation was gone) -- same action, one
                place to find it. */}
            {!isExpired && (
              <Button onClick={onStop} disabled={loading} variant="outline" size="sm">
                <Square aria-hidden className="size-3.5" />
                Stop endpoint now
              </Button>
            )}
          </div>

          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <EndpointStat label="Cost">
              {endpoint.price_per_hour != null && endpoint.currency
                ? `${endpoint.price_per_hour} ${endpoint.currency}/hr`
                : "—"}
            </EndpointStat>
            <EndpointStat label="Idle stop">
              {formatDuration(endpoint.idle_timeout_s)}
            </EndpointStat>
            <EndpointStat label="Stops at">
              {formatTimestamp(endpoint.expires_at)}
            </EndpointStat>
            <EndpointStat label="Hard stop at">
              {formatTimestamp(endpoint.max_expires_at)}
            </EndpointStat>
          </dl>

          <dl className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-xs text-muted-foreground">Key prefix</dt>
              <dd>
                <code className="rounded bg-muted px-1">{endpoint.api_key_prefix}</code>
              </dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Machine</dt>
              <dd>{endpoint.machine_id ?? "—"}</dd>
            </div>
          </dl>

          {created?.api_key && (
            <Alert>
              <AlertTitle>API key, shown once</AlertTitle>
              <AlertDescription className="space-y-2">
                <p className="break-all">
                  <code className="rounded bg-muted px-1">{created.api_key}</code>
                </p>
                <p className="text-xs text-muted-foreground">
                  This key is stored hashed and cannot be shown again. Copy it now. It is required
                  in the X-API-Key header for inference.
                </p>
              </AlertDescription>
            </Alert>
          )}

          {!isExpired && (
            <div className="space-y-2">
              <label htmlFor="endpoint-prompt" className="text-sm font-medium">
                Prompt
              </label>
              <textarea
                id="endpoint-prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Ask your tuned model something..."
                className="min-h-20 w-full rounded-[8px] border bg-background p-2 text-sm"
              />
              <div className="flex gap-2">
                <Button onClick={onInfer} disabled={inferLoading} size="sm">
                  {inferLoading ? "Thinking…" : "Send prompt"}
                </Button>
                {!created && (
                  <Button
                    onClick={fetchEndpoint}
                    disabled={loading}
                    variant="ghost"
                    size="sm"
                  >
                    Refresh
                  </Button>
                )}
              </div>
              {completion && (
                <div className="rounded-[8px] border bg-muted p-3 text-sm whitespace-pre-wrap">
                  {completion}
                </div>
              )}
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3 rounded-[8px] border bg-muted/30 px-4 py-3 text-sm text-muted-foreground">
            <span>
              This endpoint extends its idle expiry on each use, but still
              stops at the hard ceiling even if you keep using it, because a
              busy endpoint cannot be kept alive forever.
            </span>
            <span className="shrink-0 font-mono text-primary">
              Max duration: {formatDuration(endpoint.max_lifetime_s)}
            </span>
          </div>
        </Card>
      </section>
    );
  }

  // No endpoint yet -- show preview and start button
  return (
    <section aria-labelledby="endpoint-heading" className="space-y-3">
      <Card className="space-y-4 p-5">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b pb-4">
          <div className="flex items-start gap-2">
            <Zap aria-hidden className="mt-0.5 size-5 shrink-0 text-primary" />
            <div>
              <h2 id="endpoint-heading" className="font-semibold">
                Try your model
              </h2>
              <p className="text-sm text-muted-foreground">
                Start a temporary endpoint to try your tuned model without
                downloading anything. Requires an API key, auto-stops when
                idle.
              </p>
            </div>
          </div>
          <Button onClick={onStart} disabled={loading || !isComplete}>
            <Play aria-hidden className="size-3.5" />
            {loading ? "Starting…" : "Start endpoint"}
          </Button>
        </div>

        {preview ? (
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <EndpointStat label="Cost">
              {preview.price_per_hour} {preview.currency}/hr
            </EndpointStat>
            <EndpointStat label="Idle stop">
              {formatDuration(preview.idle_timeout_s)} after last use
            </EndpointStat>
            <EndpointStat label="Max duration">
              {formatDuration(preview.max_lifetime_s)}
            </EndpointStat>
          </dl>
        ) : (
          <p className="text-sm text-muted-foreground">Loading preview…</p>
        )}

        <div className="flex flex-wrap items-center justify-between gap-3 rounded-[8px] border bg-muted/30 px-4 py-3 text-sm text-muted-foreground">
          <span>
            The endpoint carries its own expiry from the moment it starts,
            extends on use, and stops itself via a timer.
          </span>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </Card>
    </section>
  );
}
