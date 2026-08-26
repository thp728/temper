"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
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
    setStatus("Validating your dataset…");
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
      <div>
        <label
          htmlFor="dataset-file"
          className="block text-sm font-medium text-neutral-800"
        >
          Dataset file (.jsonl)
        </label>
        <input
          ref={inputRef}
          id="dataset-file"
          name="file"
          type="file"
          accept=".jsonl,.json"
          className="mt-1 block w-full cursor-pointer rounded-md border border-neutral-300 bg-white px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-neutral-100 file:px-3 file:py-1 hover:file:bg-neutral-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900"
        />
        <p className="mt-1 text-sm text-neutral-500">
          Chat-format JSONL: one JSON object per line with a{" "}
          <code className="rounded bg-neutral-100 px-1">messages</code> list.
        </p>
      </div>

      <button
        type="submit"
        disabled={busy}
        className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {busy ? "Validating…" : "Upload and validate"}
      </button>

      <p aria-live="polite" className="text-sm text-neutral-600">
        {busy ? status : ""}
      </p>

      {refusal && (
        <div
          role="alert"
          className="rounded-md border border-red-300 bg-red-50 p-4"
        >
          <p className="font-medium text-red-900">
            The upload was refused.{" "}
            <code className="rounded bg-red-100 px-1 text-sm">
              {refusal.code}
            </code>
          </p>
          <p className="mt-1 text-sm text-red-800">{refusal.message}</p>
        </div>
      )}
    </form>
  );
}
