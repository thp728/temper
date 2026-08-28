"use client";

import { useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  probeModelV1ModelsProbePost,
  type AdmittedModel,
} from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

// The boundary coming down (issue #58): any model repository may be used at a
// pinned revision once a compatibility probe reports on it. This form is
// where the probing happens in front of the user -- a repository and its
// pinned revision go to the one probe path the API publishes, and the
// persisted result (verdict and findings) is what lets a job be created
// against it. A refusal -- an unpinned revision, a reference that does not
// resolve -- is shown with its stable code, exactly as the API states it.
export default function AdmitModelForm({
  onAdmitted,
}: {
  onAdmitted: (record: AdmittedModel) => void;
}) {
  const repoRef = useRef<HTMLInputElement>(null);
  const revisionRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [refusal, setRefusal] = useState<ApiError | null>(null);

  async function probe() {
    const repo = repoRef.current?.value.trim();
    const revision = revisionRef.current?.value.trim();
    if (!repo || !revision) {
      setRefusal(
        new ApiError(
          0,
          "no_reference",
          "Enter the public repository and its pinned revision.",
        ),
      );
      return;
    }
    setBusy(true);
    setRefusal(null);
    setStatus("Probing the model…");
    try {
      const record = await probeModelV1ModelsProbePost({ repo, revision });
      setStatus("Probe complete.");
      onAdmitted(record);
    } catch (err) {
      setStatus("");
      setRefusal(err instanceof ApiError ? err : NETWORK_ERROR);
    } finally {
      setBusy(false);
    }
  }

  // A `div`, deliberately not a `<form>`: this disclosure sits inside the
  // launch form, and a nested form submits natively with no handler -- it
  // would navigate with its own fields as query parameters and drop the
  // dataset the launch was built around. Enter still probes.
  function onEnter(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter" && !busy) {
      event.preventDefault();
      void probe();
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="admit-repo">Public repository</Label>
          <Input
            ref={repoRef}
            id="admit-repo"
            name="repo"
            type="text"
            placeholder="e.g. Qwen/Qwen3-8B"
            className="cursor-text"
            onKeyDown={onEnter}
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="admit-revision">Pinned revision</Label>
          <Input
            ref={revisionRef}
            id="admit-revision"
            name="revision"
            type="text"
            placeholder="40-character commit SHA"
            className="cursor-text"
            onKeyDown={onEnter}
          />
        </div>
      </div>
      <p className="text-sm text-muted-foreground">
        The model must be pinned to a resolved commit, never a branch name, so
        a completed run describes a model that cannot change afterwards. It is
        probed through the same facts the cost and memory predictions use; the
        result is shown here before it can launch.
      </p>

      <Button type="button" onClick={() => void probe()} disabled={busy}>
        {busy ? "Probing…" : "Probe and admit"}
      </Button>

      <p aria-live="polite" className="text-sm text-muted-foreground">
        {busy ? status : ""}
      </p>

      {refusal && (
        <Alert variant="destructive">
          <AlertTitle>
            The model could not be admitted.{" "}
            <code className="rounded bg-muted px-1 text-xs">
              {refusal.code}
            </code>
          </AlertTitle>
          <AlertDescription>{refusal.message}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
