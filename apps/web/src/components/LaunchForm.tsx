"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import AdvancedSurface from "@/components/AdvancedSurface";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import QuoteView, { type OverrideRefusal } from "@/components/QuoteView";
import {
  createJobV1JobsPost,
  getQuoteV1QuotesGet,
  recomputeQuoteV1QuotesPost,
  type AdvancedSurface as AdvancedSurfaceModel,
  type DecisionOverride,
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
//
// The plan is editable (issue #79): changing one decision re-requests the
// plan from the server rather than mutating it locally, so the recomputation
// rules live in one place. The overrides the user pins are passed to the
// launch, which freezes them into the job spec.

export default function LaunchForm({
  catalog,
  preview,
  surface,
}: {
  catalog: ModelCatalog;
  preview: JobSpecPreview;
  surface: AdvancedSurfaceModel | null;
}) {
  const router = useRouter();
  const [selected, setSelected] = useState(catalog.default);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [refusal, setRefusal] = useState<ApiError | null>(null);
  const [quote, setQuote] = useState<Quote | null>(null);
  const [overrides, setOverrides] = useState<DecisionOverride[]>([]);
  // The advanced-surface overrides (issue #80): name -> value as typed, for
  // the settings the trainer exposes behind the disclosure. They ride the
  // same recompute as the plan decisions and are frozen into the launch.
  const [hyperparameters, setHyperparameters] = useState<
    Record<string, string>
  >({});
  const [planRefusal, setPlanRefusal] = useState<OverrideRefusal | null>(null);
  // Starts loading: the effect fetches the default model's quote on mount.
  const [quoteLoading, setQuoteLoading] = useState(true);
  // The last set of overrides the server accepted, so a refused change can be
  // reverted and the plan always describes a configuration that can launch.
  const lastGood = useRef<{
    overrides: DecisionOverride[];
    hyperparameters: Record<string, string>;
  }>({ overrides: [], hyperparameters: {} });

  // The quote depends on which model is selected (a bigger model downloads
  // more and may need different hardware) and on the pinned decisions, so it
  // is re-fetched on every change -- server-computed, never guessed at on the
  // client. An override is re-requested, not applied locally (issue #79); the
  // advanced-surface overrides ride on the same request (issue #80).
  useEffect(() => {
    let cancelled = false;
    const hasDemands =
      overrides.length > 0 || Object.keys(hyperparameters).length > 0;
    const request = hasDemands
      ? recomputeQuoteV1QuotesPost({
          dataset_id: preview.dataset.id,
          base_model: selected,
          overrides,
          hyperparameters,
        })
      : getQuoteV1QuotesGet({
          dataset_id: preview.dataset.id,
          base_model: selected,
        });
    request
      .then((q) => {
        if (cancelled) return;
        setQuote(q);
        lastGood.current = { overrides, hyperparameters };
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 400) {
          // An override that cannot be honoured is refused with the same
          // arithmetic the predictor used (issue #79, extended to the
          // advanced surface by issue #80): shown beside the plan, and the
          // plan reverts to the last valid configuration. The refusal is only
          // cleared by the user's next action, never by the refetch this
          // revert triggers.
          setOverrides(lastGood.current.overrides);
          setHyperparameters(lastGood.current.hyperparameters);
          setPlanRefusal({ code: err.code, message: err.message });
        } else {
          // An estimate that cannot be fetched is shown as absent, never as an
          // error that blocks the page.
          setQuote(null);
        }
      })
      .finally(() => {
        if (!cancelled) setQuoteLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected, preview.dataset.id, overrides, hyperparameters]);

  function onModelChange(modelId: string) {
    setSelected(modelId);
    setQuote(null);
    setQuoteLoading(true);
    // A pinned decision belongs to the configuration it was pinned against;
    // switching models starts a fresh plan.
    setOverrides([]);
    setHyperparameters({});
    setPlanRefusal(null);
    lastGood.current = { overrides: [], hyperparameters: {} };
  }

  function handleOverridesChange(next: DecisionOverride[]) {
    setQuoteLoading(true);
    setPlanRefusal(null);
    setOverrides(next);
  }

  function handleHyperparametersChange(next: Record<string, string>) {
    setQuoteLoading(true);
    setPlanRefusal(null);
    setHyperparameters(next);
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
      const job = await createJobV1JobsPost({
        dataset_id: preview.dataset.id,
        base_model: chosen,
        hyperparameters,
        overrides,
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
          {specEntries(preview, hyperparameters).map(([key, value]) => (
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
          not refuse (spec 005). On the plan every decision carries a control
          (issue #79): change one and the rest recomputes from the server. */}
      {quote ? (
        <QuoteView
          quote={quote}
          editable
          overrides={overrides}
          onOverridesChange={handleOverridesChange}
          refusal={planRefusal}
        />
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

      {/* The advanced surface (issue #80): every dial the pinned trainer
          exposes, behind an explicit disclosure, each naming what goes wrong.
          It is generated from the trainer's own schema and published via
          `GET /v1/surface`; a change re-requests the plan like any other
          override, and a change that makes the job infeasible is refused
          before launch. When the surface cannot be loaded the job is still
          launchable -- with the defaults, which is what a first-time user
          gets anyway. */}
      {surface && (
        <AdvancedSurface
          surface={surface}
          defaults={preview.hyperparameters ?? {}}
          values={hyperparameters}
          onChange={handleHyperparametersChange}
        />
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

function specEntries(
  preview: JobSpecPreview,
  advanced: Record<string, string>,
): [string, string][] {
  // The specification section shows what a launch would freeze: the effective
  // defaults the server resolved, with any advanced-surface overrides (issue
  // #80) merged on top. The server remains the source of truth at launch;
  // this is the same preview the page promised before the advanced controls
  // existed, kept honest as they are changed.
  const merged = { ...(preview.hyperparameters ?? {}), ...advanced };
  return Object.entries(merged).map(([k, v]) => [k, String(v)]);
}
