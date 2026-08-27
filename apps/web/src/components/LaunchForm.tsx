"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import QuoteView from "@/components/QuoteView";
import {
  createJobV1JobsPost,
  getQuoteV1QuotesGet,
  type JobSpecPreview,
  type ModelCatalog,
  type Quote,
} from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

// The launch step of the journey, and the point of commitment: choose a base
// model from the catalog, read everything the job will train with -- the
// effective specification, any feasibility warning and the duration/cost
// quote -- and start it with one action. The specification shown here is what
// POST /v1/jobs freezes; the screen launches with no overrides of its own, so
// the preview and the job cannot disagree.
//
// The quote is fetched here, after the page has rendered, for whichever model
// is selected: an estimate never blocks the surface it appears on (spec 005),
// so the plan draws immediately and the numbers fill in when they arrive.

function specEntries(preview: JobSpecPreview): [string, string][] {
  return Object.entries(preview.hyperparameters ?? {}).map(([k, v]) => [
    k,
    String(v),
  ]);
}

export default function LaunchForm({
  catalog,
  preview,
}: {
  catalog: ModelCatalog;
  preview: JobSpecPreview;
}) {
  const router = useRouter();
  const [selected, setSelected] = useState(catalog.default);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [refusal, setRefusal] = useState<ApiError | null>(null);
  const [quote, setQuote] = useState<Quote | null>(null);
  // Starts loading: the effect fetches the default model's quote on mount.
  const [quoteLoading, setQuoteLoading] = useState(true);

  // The quote depends on which model is selected (a bigger model downloads
  // more and may need different hardware), so it is re-fetched on every
  // change -- server-computed, never guessed at on the client. Only the fetch
  // lives in the effect; the reset happens in the change handler, because a
  // synchronous reset here would cascade renders for no user-visible reason.
  useEffect(() => {
    let cancelled = false;
    getQuoteV1QuotesGet({
      dataset_id: preview.dataset.id,
      base_model: selected,
    })
      .then((q) => {
        if (!cancelled) setQuote(q);
      })
      .catch(() => {
        // An estimate that cannot be fetched is shown as absent, never as an
        // error that blocks the page.
        if (!cancelled) setQuote(null);
      })
      .finally(() => {
        if (!cancelled) setQuoteLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected, preview.dataset.id]);

  function onModelChange(modelId: string) {
    setSelected(modelId);
    setQuote(null);
    setQuoteLoading(true);
  }

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const chosen = new FormData(event.currentTarget).get("base_model");
    if (typeof chosen !== "string" || !chosen) {
      setRefusal(
        new ApiError(0, "no_model", "Choose a base model to train from."),
      );
      return;
    }
    setBusy(true);
    setRefusal(null);
    setStatus("Launching your job…");
    try {
      // No overrides: this form launches exactly the specification shown.
      const job = await createJobV1JobsPost({
        dataset_id: preview.dataset.id,
        base_model: chosen,
        hyperparameters: {},
      });
      setStatus("Job launched. Opening it…");
      router.push(`/jobs/${job.id}`);
    } catch (err) {
      setStatus("");
      setRefusal(err instanceof ApiError ? err : NETWORK_ERROR);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-6">
      <fieldset className="space-y-3" disabled={busy}>
        <legend className="text-lg font-semibold">Base model</legend>
        <p className="text-sm text-muted-foreground">
          Every model here has been tested end to end. Each is pinned to an
          exact revision, so a completed run can always say what it trained
          against.
        </p>
        <div className="space-y-3">
          {catalog.models.map((m) => (
            <Label
              key={m.id}
              htmlFor={`model-${m.id}`}
              className="flex cursor-pointer items-start gap-3 rounded-lg border p-4 hover:bg-muted/50"
            >
              <Input
                id={`model-${m.id}`}
                type="radio"
                name="base_model"
                value={m.id}
                defaultChecked={m.id === catalog.default}
                onChange={() => onModelChange(m.id)}
                className="mt-1 size-4"
              />
              {/* Label/value pairs stay a real description list: that
                  adjacency is what a screen reader announces. */}
              <span className="space-y-1">
                <span className="block font-medium">
                  {m.repo} — {m.good_for}
                </span>
                <dl className="text-sm text-muted-foreground">
                  <div>
                    <dt className="inline">Licence </dt>
                    <dd className="inline">{m.license}</dd>
                    <dt className="inline"> · revision </dt>
                    <dd className="inline">
                      <code className="rounded bg-muted px-1">{m.revision}</code>
                    </dd>
                  </div>
                  <div>
                    <dt className="inline">Parameters </dt>
                    <dd className="inline">{m.params_b}B</dd>
                    <dt className="inline"> · context </dt>
                    <dd className="inline">{m.context_length}</dd>
                    <dt className="inline"> · needs at least a </dt>
                    <dd className="inline">{m.min_gpu}</dd>
                  </div>
                  <div>
                    <dt className="inline">Predicted peak VRAM </dt>
                    <dd className="inline">
                      {m.peak_memory.total_gb.toFixed(2)} GB
                    </dd>
                    <dt className="inline"> · headroom on the {m.peak_memory.gpu_type} </dt>
                    <dd className="inline">
                      {m.peak_memory.headroom_gb.toFixed(2)} GB of{" "}
                      {m.peak_memory.gpu_capacity_gb} GB
                    </dd>
                    <dt className="inline"> </dt>
                    <dd className="inline">
                      (estimate, ±
                      {Math.round(m.peak_memory.tolerance * 100)}%)
                    </dd>
                  </div>
                </dl>
              </span>
            </Label>
          ))}
        </div>
      </fieldset>

      <section aria-labelledby="spec-h" className="space-y-2">
        <h2 id="spec-h" className="text-lg font-semibold">
          The job specification
        </h2>
        <p className="text-sm text-muted-foreground">
          This job will train with the following hyperparameters. They are{" "}
          <strong>frozen at launch</strong>: the job runs with exactly these,{" "}
          <strong>and they cannot be changed afterwards</strong>. A later
          change to defaults never retroactively alters what a run did.
        </p>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-1 rounded-lg border bg-card p-4 sm:grid-cols-3">
          {specEntries(preview).map(([key, value]) => (
            <div key={key}>
              <dt className="text-sm text-muted-foreground">
                <code>{key}</code>
              </dt>
              <dd className="font-medium">{value}</dd>
            </div>
          ))}
        </dl>
      </section>

      {/* The quote for the selected model: a duration range and a per-phase
          cost breakdown, both labelled an estimate. It loads after the page
          renders and never blocks anything -- the estimate warns, it does
          not refuse (spec 005). */}
      {quote ? (
        <QuoteView quote={quote} />
      ) : (
        <p
          aria-live="polite"
          className="text-sm text-muted-foreground"
        >
          {quoteLoading
            ? "Loading the cost and time estimate…"
            : "A cost and time estimate could not be computed for this model right now. Launching will still work; you just will not see the numbers first."}
        </p>
      )}

      <div className="flex items-center gap-3">
        <Button type="submit" disabled={busy} size="lg">
          {busy ? "Launching…" : "Launch job"}
        </Button>
        <p aria-live="polite" className="text-sm text-muted-foreground">
          {busy ? status : ""}
        </p>
      </div>

      {refusal && (
        <Alert variant="destructive">
          <AlertTitle>
            The job could not be launched.{" "}
            <code className="rounded bg-muted px-1 text-xs">
              {refusal.code}
            </code>
          </AlertTitle>
          <AlertDescription>{refusal.message}</AlertDescription>
        </Alert>
      )}
    </form>
  );
}

