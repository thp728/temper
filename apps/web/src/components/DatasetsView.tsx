"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { FileJson, Search, Upload } from "lucide-react";
import ImportForm from "@/components/ImportForm";
import UploadForm from "@/components/UploadForm";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import DatasetActionsMenu from "@/components/DatasetActionsMenu";
import { formatTimestamp } from "@/lib/jobs/display";
import type { DatasetRecord } from "@/lib/api/generated/client";

// Matches docs/wireframes/datasets.html: action bar + large dashed drop zone
// + RECENT DATASETS 3-up grid. Built only from what the API publishes (AGENTS.md:
// "Build only the parts which the current api supports already"). The product
// only ingests JSONL, so "Format" earned no card row -- every card would read
// the same. The timestamp reads "Updated" (`updated_at`, falling back to
// `created_at` for a dataset older than that column): rename is the one
// update a dataset can have.

function StatusBadge({ record }: { record: DatasetRecord }) {
  const report = record.report;
  if (record.status === "importing") {
    return (
      <span className="inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wide text-amber-700 dark:border-amber-900/50 dark:bg-amber-950/40 dark:text-amber-300">
        Importing
      </span>
    );
  }
  if (record.status === "validating") {
    return (
      <span className="inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wide text-amber-700 dark:border-amber-900/50 dark:bg-amber-950/40 dark:text-amber-300">
        Validating
      </span>
    );
  }
  if (record.token_count_status === "counting") {
    return (
      <span className="inline-flex items-center rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wide text-sky-700 dark:border-sky-900/50 dark:bg-sky-950/40 dark:text-sky-300">
        Counting tokens
      </span>
    );
  }
  if (report && !report.valid) {
    return (
      <span className="inline-flex items-center rounded-full border border-destructive/30 bg-destructive/10 px-2 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wide text-destructive">
        Invalid
      </span>
    );
  }
  if (report?.valid) {
    return (
      <span className="inline-flex items-center rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wide text-emerald-700 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300">
        Ready
      </span>
    );
  }
  return (
    <span className="inline-flex items-center rounded-full border bg-muted px-2 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
      Stored
    </span>
  );
}

function FileIcon() {
  return <FileJson className="size-4 text-primary" aria-hidden="true" />;
}

function DatasetCard({ record }: { record: DatasetRecord }) {
  const report = record.report;
  return (
    <div className="group relative flex flex-col rounded-lg border bg-card p-4 transition-colors hover:border-ring/50 hover:bg-muted/20">
      <div
        className="absolute inset-x-0 top-0 h-px bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity rounded-t-lg"
        aria-hidden="true"
      />
      <div className="flex items-start justify-between gap-2">
        <span className="flex min-w-0 items-center gap-2">
          <FileIcon />
          <Link
            href={`/datasets/${record.id}`}
            className="truncate rounded font-medium text-sm leading-tight hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {record.filename}
          </Link>
        </span>
        {/* The corner is the context menu's alone now -- status moved into
            the details list below, so this is the one thing left up here. */}
        <DatasetActionsMenu id={record.id} filename={record.filename} />
      </div>
      <dl className="mt-4 space-y-2 font-mono text-xs text-muted-foreground">
        <div className="flex items-center justify-between">
          <dt>Status</dt>
          <dd>
            <StatusBadge record={record} />
          </dd>
        </div>
        <div className="flex items-center justify-between">
          <dt>Rows</dt>
          <dd className="text-foreground tabular-nums">
            {report ? report.row_count.toLocaleString() : "—"}
          </dd>
        </div>
        <div className="flex items-center justify-between">
          <dt>Usable</dt>
          <dd className="text-foreground tabular-nums">
            {report ? report.usable_rows.toLocaleString() : "—"}
          </dd>
        </div>
        <div className="flex items-center justify-between">
          <dt>Tokens</dt>
          <dd className="text-foreground tabular-nums">
            {report?.token_count != null
              ? report.token_count.toLocaleString()
              : "—"}
          </dd>
        </div>
        <div className="flex items-center justify-between border-t pt-2 mt-2">
          <dt>Updated</dt>
          <dd className="text-muted-foreground">
            {formatTimestamp(record.updated_at ?? record.created_at)}
          </dd>
        </div>
      </dl>
    </div>
  );
}

export default function DatasetsView({
  datasets,
  loadError,
}: {
  datasets: DatasetRecord[];
  loadError?: { code?: string; message: string } | null;
}) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return datasets;
    return datasets.filter(
      (d) =>
        d.filename.toLowerCase().includes(q) || d.id.toLowerCase().includes(q),
    );
  }, [datasets, query]);

  return (
    <div className="space-y-8">
      {/* Page heading + Import: aligned on one row per feedback */}
      <div className="space-y-2">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <h1
            id="upload-heading"
            className="text-xl font-semibold tracking-tight text-balance sm:text-2xl"
          >
            Manage and prepare your raw data for fine-tuning jobs.
          </h1>
          <Dialog>
            <DialogTrigger asChild>
              <Button className="shrink-0">
                <Upload className="size-4" aria-hidden="true" />
                Import Dataset
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[640px]">
              <DialogHeader>
                <DialogTitle>Import from Hugging Face</DialogTitle>
                <DialogDescription>
                  Only public datasets are supported at the moment.
                </DialogDescription>
              </DialogHeader>
              <ImportForm />
            </DialogContent>
          </Dialog>
        </div>
        <p className="text-sm text-muted-foreground">
          Upload a file or import from Hugging Face
        </p>
      </div>

      {/* Upload area: wireframe datasets.html:355. The form itself carries the
          a11y label “Dataset file (.jsonl)” and the submit name
          “Upload and validate” that the e2e journeys assert on. */}
      <div>
        <UploadForm />
      </div>

      {/* Load error: keep stable code visible per web/AGENTS.md */}
      {loadError && (
        <p className="text-sm text-muted-foreground">
          <code className="rounded bg-muted px-1">{loadError.code}</code> —{" "}
          {loadError.message}
        </p>
      )}

      {/* Recent datasets: search aligned with this h2, not the page h1 */}
      <section aria-labelledby="recent-datasets-heading" className="space-y-3">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <h2
            id="recent-datasets-heading"
            className="text-xs font-medium tracking-widest uppercase text-foreground"
          >
            Recent Datasets
          </h2>
          <div className="relative w-full sm:w-64">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden="true"
            />
            <Input
              aria-label="Search datasets"
              placeholder="Search datasets..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="pl-8"
            />
          </div>
        </div>
        {filtered.length === 0 ? (
          <div className="rounded-lg border border-dashed bg-card/50 px-6 py-12 text-center">
            <p className="font-medium">
              {datasets.length === 0
                ? "No datasets yet"
                : "No matching datasets"}
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              {datasets.length === 0
                ? "Upload a JSONL file above. Its validation report comes back before anything is spent."
                : `No dataset matches “${query}”.`}
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filtered.map((d) => (
              <DatasetCard key={d.id} record={d} />
            ))}
          </div>
        )}
        {/* Only shown while search has actually narrowed the list -- there
            is no pagination here, so stating a count against the total
            outside of that context just raises "is this paginated?". */}
        {query.trim() !== "" && filtered.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Showing {filtered.length} of {datasets.length} dataset
            {datasets.length === 1 ? "" : "s"}.
          </p>
        )}
      </section>

      {/* Import now lives in the modal above; the trigger is the
          “Import Dataset” button in the action bar. No inline card. */}
    </div>
  );
}
