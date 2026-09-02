import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Models",
};

// The model explorer (browse the catalog, admit models from Hugging Face) is
// a later slice of the rework. This route exists so the sidebar's "Models"
// item has somewhere to land; the catalog itself is already reachable the way
// it always was, at the moment a run is launched.
export default function ModelsPage() {
  return (
    <section aria-labelledby="models-heading" className="space-y-4">
      <h1 id="models-heading" className="text-2xl font-semibold tracking-tight">
        Models
      </h1>
      <p className="text-muted-foreground">
        The model explorer is coming in a later slice. In the meantime, the
        base-model catalog and Hugging Face imports are offered where you need
        them: when you launch a run.
      </p>
    </section>
  );
}
