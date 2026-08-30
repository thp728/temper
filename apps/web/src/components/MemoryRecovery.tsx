import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { JobRecord } from "@/lib/api/generated/client";

// A job whose execution was retried automatically (issue #35): an
// out-of-memory failure halves the per-step batch and doubles accumulation so
// the effective batch is preserved and the optimisation does not change. The
// attempts list records what actually happened -- each execution, its own
// machine, its own spec and its own outcome -- so the user is told a recovery
// happened and what changed, and the history says what actually happened
// rather than one continuous run that was not.
//
// Deliberately distinct from a divergence retry (ADR-0055): that one is a
// choice a user makes, never an automatic rerun, and it renders its own
// control in the failure section. This banner exists only when the job's
// executions are plural AND the last one completed AND an attempt actually
// carried a memory `recovery` -- the shape a memory recovery leaves behind.
// The last clause is what keeps a resumed run (issue #60) out of this banner:
// resumption also makes attempts plural and ends complete, but its attempts
// carry no `recovery` record -- the run was interrupted and continued from a
// checkpoint, never retried for memory, and must not be presented as one.
// An exhausted recovery (all attempts failed) is a failure, explained by the
// record's failure section, not a recovery to present as one.
export default function MemoryRecovery({ job }: { job: JobRecord }) {
  const attempts = job.attempts ?? [];
  if (attempts.length <= 1) return null;
  if (attempts[attempts.length - 1]?.outcome !== "complete") return null;
  if (!attempts.some((a) => a.recovery)) return null;

  return (
    <Alert data-testid="memory-recovery">
      <AlertTitle>Memory recovery</AlertTitle>
      <AlertDescription className="space-y-3">
        <p>
          This job ran out of device memory during training and was retried
          automatically. The effective batch was preserved — the per-step
          batch was halved and accumulation doubled — so your training did not
          change. This is a memory recovery, not a divergence retry: a
          diverging run offers a half-rate retry as a choice instead.
        </p>
        <ul className="divide-y divide-border rounded border bg-card">
          {attempts.map((a) => (
            <li
              key={a.attempt}
              className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-3 py-2 text-sm"
            >
              <span className="font-medium">Attempt {a.attempt}</span>
              <span>{a.outcome}</span>
              {a.error_code && (
                <code className="rounded bg-muted px-1 text-xs">
                  {a.error_code}
                </code>
              )}
              {a.recovery?.action ? (
                <span className="text-muted-foreground">
                  {String(a.recovery.action)}
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  );
}
