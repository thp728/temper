import { BadgeCheck, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { JobRecord } from "@/lib/api/generated/client";

// Every retained checkpoint the run has produced so far, and which one is
// currently the best by held-out loss (issue #62). Shared by the running
// view and the finished record (issue: non-terminal/terminal parity):
// checkpoints land during training, not only at the end (#37), so a job
// still running can already have a history here -- the same section, same
// wording, just growing.
export default function CheckpointSection({ job }: { job: JobRecord }) {
  const checkpoints = job.checkpoints ?? [];
  if (checkpoints.length === 0) return null;
  const best = job.best_checkpoint;
  return (
    <section aria-labelledby="checkpoints-heading" className="space-y-3">
      <h2
        id="checkpoints-heading"
        className="text-xs font-medium tracking-widest uppercase text-foreground"
      >
        Retained checkpoints
      </h2>
      {best && best.step != null && (
        <div className="flex items-start gap-2 px-4 py-3 font-mono text-sm">
          <BadgeCheck aria-hidden className="mt-0.5 size-4 shrink-0 text-primary" />
          <p>
            <strong className="font-semibold text-primary">
              Best checkpoint: step {best.step}
              {best.held_out_loss != null && (
                <>
                  {" "}
                  (held-out loss{" "}
                  <span className="text-success">{best.held_out_loss}</span>)
                </>
              )}
            </strong>
            {best.reason && (
              <>
                {" — "}
                <span className="font-sans text-muted-foreground">
                  {best.reason}
                </span>
              </>
            )}
          </p>
        </div>
      )}
      <div className="overflow-hidden rounded-[12px] border bg-card font-mono text-sm">
        <ul className="divide-y">
          {checkpoints.map((c) => (
            <li
              key={c.step}
              className="flex flex-wrap items-center justify-between gap-3 px-4 py-3"
            >
              <div className="flex items-center gap-2">
                <span className="font-semibold">Step {c.step}</span>
                {c.selected && (
                  <span className="rounded bg-primary/15 px-1.5 py-0.5 text-[10px] font-semibold tracking-wider text-primary uppercase">
                    chosen result
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3 text-muted-foreground">
                <span>
                  {c.held_out_loss != null ? (
                    <>
                      held-out loss:{" "}
                      <span
                        className={
                          c.selected
                            ? "font-semibold text-success"
                            : "text-foreground"
                        }
                      >
                        {c.held_out_loss}
                      </span>
                    </>
                  ) : (
                    "no held-out loss recorded"
                  )}
                </span>
                {c.verified ? (
                  <Button variant="outline" size="sm" asChild>
                    <a href={`/v1/jobs/${job.id}/checkpoints/${c.step}`}>
                      <Download aria-hidden className="size-3.5" />
                      Download
                    </a>
                  </Button>
                ) : (
                  <span className="text-xs">not retained</span>
                )}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
