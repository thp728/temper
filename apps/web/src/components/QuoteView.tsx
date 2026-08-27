"use client";

import {
  formatDurationRange,
  formatMinorCost,
  shortRevision,
} from "@/lib/jobs/display";
import type { Quote } from "@/lib/api/generated/client";

// The plan's duration-and-cost prediction, shown before anything is spent
// (issue #72). Everything here is an estimate and says so; the quote is what
// a launch freezes into the job spec, so this is the last moment the numbers
// shown can be acted on. Label/value pairs stay real dt/dd pairs -- that
// adjacency is what a screen reader announces.

function PhaseRow({ phase, quote }: { phase: Quote["phases"][number]; quote: Quote }) {
  return (
    <div className="grid grid-cols-[1fr_auto] gap-x-6 gap-y-0.5 py-1 sm:grid-cols-[1fr_auto_auto]">
      <dt className="text-sm text-muted-foreground">{phase.name}</dt>
      <dd className="text-sm font-medium text-right">
        {formatDurationRange(phase.duration_low_s, phase.duration_high_s)}
      </dd>
      <dd className="text-sm text-right text-muted-foreground sm:min-w-28">
        {formatMinorCost(
          phase.cost_low_minor,
          quote.currency,
          quote.minor_unit,
        )}
        {" – "}
        {formatMinorCost(
          phase.cost_high_minor,
          quote.currency,
          quote.minor_unit,
        )}
      </dd>
    </div>
  );
}

export default function QuoteView({ quote }: { quote: Quote }) {
  return (
    <section aria-labelledby="quote-heading" className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="quote-heading" className="text-lg font-semibold">
          Cost and time estimate
        </h2>
        <p className="text-sm text-muted-foreground">
          An estimate, not a guarantee — it never blocks a launch.
        </p>
      </div>

      <dl className="grid grid-cols-2 gap-x-6 gap-y-1 rounded-lg border bg-card p-4 sm:grid-cols-4">
        <div>
          <dt className="text-sm text-muted-foreground">Duration</dt>
          <dd className="font-medium">
            {formatDurationRange(quote.duration_low_s, quote.duration_high_s)}
          </dd>
        </div>
        <div>
          <dt className="text-sm text-muted-foreground">Cost</dt>
          <dd className="font-medium">
            {formatMinorCost(quote.cost_low_minor, quote.currency, quote.minor_unit)}
            {" – "}
            {formatMinorCost(quote.cost_high_minor, quote.currency, quote.minor_unit)}
          </dd>
        </div>
        <div>
          <dt className="text-sm text-muted-foreground">Tokens</dt>
          <dd className="font-medium">
            {quote.token_count != null ? quote.token_count.toLocaleString() : "—"}
          </dd>
        </div>
        <div>
          <dt className="text-sm text-muted-foreground">Currency</dt>
          <dd className="font-medium">{quote.currency}</dd>
        </div>
      </dl>

      {/* The phase-by-phase breakdown is the point: a long cold start on a
          large model is visible rather than buried in one blended rate. */}
      <div className="rounded-lg border bg-card p-4">
        <dl className="divide-y divide-border">
          {quote.phases.map((phase) => (
            <PhaseRow key={phase.name} phase={phase} quote={quote} />
          ))}
          {/* Storage bills on its own line, in USD, never folded into the
              account-currency phases (ADR-0030): an invisible line is exactly
              the hidden cost the quote exists to remove. */}
          <div className="grid grid-cols-[1fr_auto] gap-x-6 gap-y-0.5 py-1 sm:grid-cols-[1fr_auto_auto]">
            <dt className="text-sm text-muted-foreground">
              storage (USD, separate line)
            </dt>
            <dd className="text-sm text-right text-muted-foreground sm:col-start-3 sm:min-w-28">
              {formatMinorCost(
                quote.storage_cost_usd_total_low_minor,
                "USD",
                100,
              )}
              {" – "}
              {formatMinorCost(
                quote.storage_cost_usd_total_high_minor,
                "USD",
                100,
              )}
            </dd>
          </div>
        </dl>
        <p className="mt-2 text-xs text-muted-foreground">
          Estimated against dataset{" "}
          <code className="rounded bg-muted px-1">{shortRevision(quote.dataset_id)}</code>
          {" · "}model revision{" "}
          <code className="rounded bg-muted px-1">{shortRevision(quote.base_revision)}</code>
          {" · "}expires{" "}
          {new Date(quote.expires_at * 1000).toLocaleString("sv-SE")}.
        </p>
      </div>
    </section>
  );
}
