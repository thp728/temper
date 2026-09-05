"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { CloudUpload } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { uploadDatasetV1DatasetsPost } from "@/lib/api/generated/client";
import type { DatasetAccepted } from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";
import { cn } from "@/lib/utils";

// The upload step of the journey: choose a file, send it to the one ingest
// path the API publishes, land on its validation report. A refusal -- wrong
// extension, oversized body, server unreachable -- is shown here with its
// stable code, exactly as the API states it, never laundered into "error".
//
// Inside the launch wizard the form instead hands the new record back
// through onUploaded, so the wizard can watch validation land and select it
// without leaving the screen. Absent, the form keeps its standalone
// behavior: navigate to the new dataset's report.
export default function UploadForm({
  onUploaded,
}: {
  onUploaded?: (dataset: DatasetAccepted) => void;
}) {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [refusal, setRefusal] = useState<ApiError | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const file = inputRef.current?.files?.[0];
    if (!file) {
      setRefusal(
        new ApiError(0, "no_file", "Choose a .jsonl dataset file first."),
      );
      return;
    }
    setBusy(true);
    setRefusal(null);
    setStatus("Uploading your dataset…");
    try {
      const uploaded = await uploadDatasetV1DatasetsPost({ file });
      if (onUploaded) {
        onUploaded(uploaded);
        return;
      }
      setStatus("Dataset received. Opening the report…");
      router.push(`/datasets/${uploaded.id}`);
    } catch (err) {
      setStatus("");
      setRefusal(err instanceof ApiError ? err : NETWORK_ERROR);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-4">
      <div className="space-y-2">
        {/* Label stays for a11y/journeys (getByLabelText) but is visually the drop-zone's context */}
        <Label htmlFor="dataset-file" className="sr-only">
          Dataset file (.jsonl)
        </Label>
        {/* Wireframe upload area: dashed, hover -> primary/50, icon scales. Kept the
             accessible input covering the whole area so drag & click both work.
             Text matches docs/wireframes/datasets.html:372 (5GB) and
             apps/control-plane/src/temper_control_plane/config.py:230
             DEFAULT_MAX_DATASET_MB=5120, ADR-0036. We only ingest JSONL. */}
        <div
          className={cn(
            "group relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed bg-card px-6 py-10 text-center transition-colors",
            "hover:border-primary/50 hover:bg-muted/30",
            "has-[input:focus-visible]:border-ring has-[input:focus-visible]:ring-3 has-[input:focus-visible]:ring-ring/50",
            dragOver ? "border-primary/50 bg-muted/30" : "border-border",
            busy ? "opacity-60" : "",
          )}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={() => setDragOver(false)}
        >
          <input
            ref={inputRef}
            id="dataset-file"
            name="file"
            type="file"
            accept=".jsonl"
            className="absolute inset-0 z-0 cursor-pointer opacity-0 disabled:cursor-not-allowed"
            disabled={busy}
            onChange={(event) =>
              setFileName(event.target.files?.[0]?.name ?? null)
            }
          />
          <span
            className={cn(
              "flex size-14 items-center justify-center rounded-full bg-muted transition-transform duration-300 group-hover:scale-105",
              dragOver ? "scale-105" : "",
            )}
            aria-hidden="true"
          >
            <CloudUpload className="size-7 text-muted-foreground group-hover:text-primary transition-colors" />
          </span>
          <span className="pointer-events-none space-y-1">
            <span className="block font-medium tracking-tight">
              {fileName ?? "Drop files to upload"}
            </span>
            <span className="block font-mono text-xs text-muted-foreground">
              {fileName ? "Click to choose a different file" : "Supports JSONL up to 5GB."}
            </span>
          </span>
          {/* CTA centered inside dropzone: deactivated until a file is chosen */}
          <Button
            type="submit"
            disabled={busy || !fileName}
            className="relative z-10 mt-2"
            aria-disabled={busy || !fileName}
          >
            {busy ? "Validating…" : "Upload and validate"}
          </Button>
        </div>
      </div>

      <p className="text-sm text-muted-foreground">
        Chat-format JSONL: one JSON object per line with a{" "}
        <code className="rounded bg-muted px-1">messages</code> list.
      </p>

      <p aria-live="polite" className="text-sm text-muted-foreground">
        {busy ? status : ""}
      </p>

      {refusal && (
        <Alert variant="destructive">
          <AlertTitle>
            The upload was refused.{" "}
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
