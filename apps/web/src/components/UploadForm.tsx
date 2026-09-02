"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { uploadDatasetV1DatasetsPost } from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

// The upload step of the journey: choose a file, send it to the one ingest
// path the API publishes, land on its validation report. A refusal -- wrong
// extension, oversized body, server unreachable -- is shown here with its
// stable code, exactly as the API states it, never laundered into "error".
export default function UploadForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [refusal, setRefusal] = useState<ApiError | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

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
        <Label htmlFor="dataset-file">Dataset file (.jsonl)</Label>
        <div className="relative flex h-24 flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-input bg-muted/30 text-center transition-colors hover:border-primary/50 hover:bg-muted/50 has-[input:focus-visible]:border-ring has-[input:focus-visible]:ring-3 has-[input:focus-visible]:ring-ring/50">
          <input
            ref={inputRef}
            id="dataset-file"
            name="file"
            type="file"
            accept=".jsonl,.json"
            className="absolute inset-0 cursor-pointer opacity-0"
            onChange={(event) =>
              setFileName(event.target.files?.[0]?.name ?? null)
            }
          />
          <span className="pointer-events-none font-mono text-xs font-medium tracking-wide text-foreground">
            {fileName ?? "Choose a .jsonl file"}
          </span>
          <span className="pointer-events-none text-xs text-muted-foreground">
            {fileName ? "Click to choose a different file" : "Click to browse"}
          </span>
        </div>
        <p className="text-sm text-muted-foreground">
          Chat-format JSONL: one JSON object per line with a{" "}
          <code className="rounded bg-muted px-1">messages</code> list.
        </p>
      </div>

      <Button type="submit" disabled={busy}>
        {busy ? "Validating…" : "Upload and validate"}
      </Button>

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
