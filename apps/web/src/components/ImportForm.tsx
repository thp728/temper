"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const HF_PREFIX = "https://huggingface.co/";
import { importDatasetV1DatasetsImportPost } from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

// The import step of the journey (issue #45): start from a public dataset
// repository instead of preparing a file. A repository, optionally a
// configuration (subset) and a split, is sent to the one import path the API
// publishes; the fetched rows validate through exactly the same path an
// upload's do, so the report that comes back is the same shape. A refusal --
// repository unfetchable, split empty, server unreachable -- is shown here
// with its stable code, exactly as the API states it.
export default function ImportForm() {
  const router = useRouter();
  const repoRef = useRef<HTMLInputElement>(null);
  const configRef = useRef<HTMLInputElement>(null);
  const splitRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [refusal, setRefusal] = useState<ApiError | null>(null);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const repo = repoRef.current?.value.trim();
    if (!repo) {
      setRefusal(
        new ApiError(
          0,
          "no_repo",
          "Enter the public repository to import from.",
        ),
      );
      return;
    }
    setBusy(true);
    setRefusal(null);
    setStatus("Importing your dataset…");
    try {
      const config = configRef.current?.value.trim();
      const split = splitRef.current?.value.trim();
      const imported = await importDatasetV1DatasetsImportPost({
        repo,
        ...(config ? { config } : {}),
        ...(split ? { split } : {}),
      });
      setStatus("Dataset received. Opening the report…");
      router.push(`/datasets/${imported.id}`);
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
        <Label htmlFor="import-repo">Public repository</Label>
        <div className="flex h-9 items-stretch overflow-hidden rounded-md border border-input bg-muted/50 has-[input:focus-visible]:border-ring has-[input:focus-visible]:ring-3 has-[input:focus-visible]:ring-ring/50">
          <span className="flex shrink-0 items-center border-r border-input px-3 text-sm text-muted-foreground select-none">
            {HF_PREFIX}
          </span>
          <input
            ref={repoRef}
            id="import-repo"
            name="repo"
            type="text"
            placeholder="meta-llama/Llama-3.2-3B-Instruct"
            className="min-w-0 flex-1 bg-transparent px-3 text-base text-foreground placeholder:text-muted-foreground outline-none md:text-sm"
          />
        </div>
        <p className="text-sm text-muted-foreground">
          A public dataset repository. Rows are fetched and validated exactly
          like an upload — nothing gets a shortcut for arriving over a
          network.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="import-config">Configuration (optional)</Label>
          <Input
            ref={configRef}
            id="import-config"
            name="config"
            type="text"
            placeholder="default"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="import-split">Split (optional)</Label>
          <Input
            ref={splitRef}
            id="import-split"
            name="split"
            type="text"
            placeholder="train"
          />
        </div>
      </div>

      <Button type="submit" disabled={busy}>
        {busy ? "Importing…" : "Import and validate"}
      </Button>

      <p aria-live="polite" className="text-sm text-muted-foreground">
        {busy ? status : ""}
      </p>

      {refusal && (
        <Alert variant="destructive">
          <AlertTitle>
            The import was refused.{" "}
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
