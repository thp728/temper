"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  createJobV1JobsPost,
  type JobSpecPreview,
  type ModelCatalog,
} from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

// The launch step of the journey, and the point of commitment: choose a base
// model from the catalog, read everything the job will train with -- the
// effective specification and any feasibility warning arrive beside it --
// and start it with one action. The specification shown here is what
// POST /v1/jobs freezes; the screen launches with no overrides of its own,
// so the preview and the job cannot disagree.

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
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [refusal, setRefusal] = useState<ApiError | null>(null);

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
