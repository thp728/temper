"use client";

import { useState } from "react";
import {
  formatDurationRange,
  formatMinorCost,
  formatTimestamp,
  shortRevision,
} from "@/lib/jobs/display";
import type {
  DecisionOverride,
  Quote,
  QuoteDecision,
} from "@/lib/api/generated/client";

// The plan's duration-and-cost prediction, shown before anything is spent
// (issue #72). Everything here is an estimate and says so; the quote is what
// a launch freezes into the job spec, so this is the last moment the numbers
// shown can be acted on. Label/value pairs stay real dt/dd pairs -- that
// adjacency is what a screen reader announces.
//
// Below the numbers, each decision the predictor made for the user is shown
// with its reason visible by default and its alternatives one interaction
// away (issue #76): the reason is the product, and the same records ride on
// the frozen quote, so a finished job explains itself as completely as a
// planned one.
//
// In the plan (editable), the controls sit beside the explanations (issue
// #79): each decision carries a control to pin it, an overridden decision is
// marked, and changing one re-requests the plan rather than mutating it
// locally -- the caller owns the recompute and passes the new quote back. On
// a finished job the same component renders without the controls: the quote
// is frozen, and an explanation that only existed while the page was open
// would be the black box this is built to remove.

export interface OverrideRefusal {
  code: string;
  message: string;
}

// The free-form decisions accept any positive number, expressed in the
// decision's own vocabulary ("N GB" for disk, a bare integer otherwise).
function parseChosen(d: QuoteDecision): {
  n: number;
  format: (n: number) => string;
} {
  if (d.decision === "disk") {
    const m = d.chosen.match(/(\d+)/);
    return {
      n: m && m[1] ? parseInt(m[1], 10) : NaN,
      format: (n) => `${n} GB`,
    };
  }
  return { n: parseInt(d.chosen, 10), format: (n) => String(n) };
}

function DecisionControl({
  d,
  options,
  overrides,
  onOverridesChange,
}: {
  d: QuoteDecision;
  options?: string[];
  overrides: DecisionOverride[];
  onOverridesChange: (overrides: DecisionOverride[]) => void;
}) {
  const current = overrides.find((o) => o.decision === d.decision);

  function setValue(value: string | null) {
    const rest = overrides.filter((o) => o.decision !== d.decision);
    onOverridesChange(
      value == null || value === "" ? rest : [...rest, { decision: d.decision, value }],
    );
  }

  // Select-style decisions carry a server-published vocabulary; the free-form
  // ones accept any positive number, so they get an input instead. Both sit
  // beside the explanation they edit (spec 005: one surface, not a beginner
  // mode and an expert mode).
  if (options && options.length > 0) {
    return (
      <select
        aria-label={`${d.decision} override`}
        value={current?.value ?? ""}
        onChange={(e) => setValue(e.target.value)}
        className="h-9 rounded-md border bg-transparent px-2 text-sm"
      >
        <option value="">Use the predictor&apos;s choice</option>
        {options.map((v) => (
          <option key={v} value={v}>
            {v}
          </option>
        ))}
      </select>
    );
  }

  return <NumericControl d={d} overrides={overrides} onOverridesChange={onOverridesChange} />;
}

function NumericControl({
  d,
  overrides,
  onOverridesChange,
}: {
  d: QuoteDecision;
  overrides: DecisionOverride[];
  onOverridesChange: (overrides: DecisionOverride[]) => void;
}) {
  const current = overrides.find((o) => o.decision === d.decision);
  const parsed = parseChosen(d);
  const [text, setText] = useState(String(parsed.n));
  // Remount when the effective value changes (a recompute landed), so the
  // input always reflects the plan it edits rather than a stale keystroke.
  const key = current?.value ?? d.chosen;

  function commit() {
    const n = Number(text);
    if (!Number.isInteger(n) || n < 1) {
      setText(parsed.n.toString());
      return;
    }
    const rest = overrides.filter((o) => o.decision !== d.decision);
    onOverridesChange([...rest, { decision: d.decision, value: parsed.format(n) }]);
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <input
        key={key}
        type="number"
        min={1}
        aria-label={`${d.decision} override`}
        defaultValue={parsed.n}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
        }}
        className="h-9 w-24 rounded-md border bg-transparent px-2 text-sm"
      />
      {current && (
        <button
          type="button"
          onClick={() =>
            onOverridesChange(overrides.filter((o) => o.decision !== d.decision))
          }
          className="text-sm text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          Use the predictor&apos;s choice
        </button>
      )}
    </div>
  );
}

function DecisionCard({
  d,
  options,
  editable,
  overrides,
  onOverridesChange,
}: {
  d: QuoteDecision;
  options?: string[];
  editable?: boolean;
  overrides: DecisionOverride[];
  onOverridesChange?: (overrides: DecisionOverride[]) => void;
}) {
  const alternatives = d.alternatives ?? [];
  return (
    <div className="rounded-[12px] border bg-card p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-xs font-medium tracking-widest uppercase text-muted-foreground">
          {d.decision}
        </h3>
        <span className="shrink-0 rounded-md border bg-muted px-2 py-0.5 font-mono text-xs font-medium text-foreground">
          {d.chosen}
        </span>
      </div>
      {d.overridden && (
        <span className="mt-2 inline-block rounded bg-amber-100 px-1.5 py-0.5 text-xs font-medium text-amber-800">
          you changed this
        </span>
      )}
      <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{d.constraint}</p>
      {editable && onOverridesChange && (
        <div className="mt-3">
          <DecisionControl
            d={d}
            options={options}
            overrides={overrides}
            onOverridesChange={onOverridesChange}
          />
        </div>
      )}
      {alternatives.length > 0 && (
        <details className="group mt-4 border-t pt-3">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 font-mono text-xs text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
            <span aria-hidden className="text-[10px] transition-transform group-open:rotate-90">
              ▶
            </span>
            Alternatives considered ({alternatives.length})
          </summary>
          <ul className="mt-3 space-y-2">
            {alternatives.map((a, i) => (
              <li key={i} className="rounded-md border bg-muted/40 px-3 py-2">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="font-mono text-xs font-medium">{a.value}</span>
                  <span className="font-mono text-xs text-muted-foreground">{a.cost}</span>
                </div>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                  {a.constraint}
                </p>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function PhaseRow({ phase, quote }: { phase: Quote["phases"][number]; quote: Quote }) {
  return (
    <div className="grid grid-cols-[1fr_auto] items-baseline gap-x-6 gap-y-1 py-3.5 font-mono text-sm sm:grid-cols-[1fr_auto_auto]">
      <dt className="text-foreground">{phase.name}</dt>
      <dd className="text-right text-muted-foreground">
        {formatDurationRange(phase.duration_low_s, phase.duration_high_s)}
      </dd>
      <dd className="text-right text-muted-foreground sm:min-w-36">
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

export default function QuoteView({
  quote,
  editable = false,
  overrides = [],
  onOverridesChange,
  refusal,
}: {
  quote: Quote;
  editable?: boolean;
  overrides?: DecisionOverride[];
  onOverridesChange?: (overrides: DecisionOverride[]) => void;
  refusal?: OverrideRefusal | null;
}) {
  return (
    <>
      <section aria-labelledby="quote-heading" className="space-y-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="quote-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
            Cost and time estimate
          </h2>
          {editable && (
            <p className="text-sm text-muted-foreground">
              An estimate rather than a guarantee. It never blocks a launch.
            </p>
          )}
        </div>

      <dl className="grid grid-cols-2 gap-x-6 gap-y-4 rounded-[12px] border bg-card p-5 sm:grid-cols-4">
        <div>
          <dt className="text-xs font-medium tracking-widest uppercase text-muted-foreground">
            Duration
          </dt>
          <dd className="mt-1 font-mono text-base font-semibold tabular-nums">
            {formatDurationRange(quote.duration_low_s, quote.duration_high_s)}
          </dd>
        </div>
        <div>
          <dt className="text-xs font-medium tracking-widest uppercase text-muted-foreground">
            Cost
          </dt>
          <dd className="mt-1 font-mono text-base font-semibold tabular-nums">
            {formatMinorCost(quote.cost_low_minor, quote.currency, quote.minor_unit)}
            {" – "}
            {formatMinorCost(quote.cost_high_minor, quote.currency, quote.minor_unit)}
          </dd>
        </div>
        <div>
          <dt className="text-xs font-medium tracking-widest uppercase text-muted-foreground">
            Tokens
          </dt>
          <dd className="mt-1 font-mono text-base font-semibold tabular-nums">
            {quote.token_count != null ? quote.token_count.toLocaleString("en-US") : "—"}
          </dd>
        </div>
        <div>
          <dt className="text-xs font-medium tracking-widest uppercase text-muted-foreground">
            Currency
          </dt>
          <dd className="mt-1 font-mono text-base font-semibold tabular-nums">{quote.currency}</dd>
        </div>
      </dl>

      {/* The phase-by-phase breakdown is the point: a long cold start on a
          large model is visible rather than buried in one blended rate. */}
      <div className="rounded-[12px] border bg-card px-6">
        <dl className="divide-y divide-border">
          {quote.phases.map((phase) => (
            <PhaseRow key={phase.name} phase={phase} quote={quote} />
          ))}
          {/* Storage bills on its own line, in USD, never folded into the
              account-currency phases (ADR-0030): an invisible line is exactly
              the hidden cost the quote exists to remove. */}
          <div className="grid grid-cols-[1fr_auto] items-baseline gap-x-6 gap-y-1 py-3.5 font-mono text-sm sm:grid-cols-[1fr_auto_auto]">
            <dt className="text-foreground">storage (USD, separate line)</dt>
            <dd className="hidden text-right text-muted-foreground sm:col-start-2 sm:block">—</dd>
            <dd className="text-right text-muted-foreground sm:col-start-3 sm:min-w-36">
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
        <p className="border-t px-6 py-3 font-mono text-xs text-muted-foreground">
          Estimated against dataset {shortRevision(quote.dataset_id)}
          {" · "}model revision {shortRevision(quote.base_revision)}
          {" · "}expires {formatTimestamp(quote.expires_at)}.
        </p>
      </div>
      </section>

      {/* The reasons are the product (issue #76): each decision the predictor
          made is shown with its reason visible by default and its
          alternatives one interaction away -- never hidden, never always
          shown. The records are the same ones frozen into the job spec, so
          a finished job explains itself as completely as a planned one. In
          the plan the controls sit beside the explanations (issue #79), and
          a change re-requests the plan: one surface, not a beginner mode and
          an expert mode. */}
      {quote.decisions && quote.decisions.length > 0 && (
        <section aria-labelledby="decisions-heading" className="space-y-3">
          <h2 id="decisions-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
            Why this configuration
          </h2>
          <p className="text-sm text-muted-foreground">
            {editable
              ? "These are the decisions Temper made for you. Change any line "
                + "and the rest recomputes; the configuration below is what a "
                + "launch would freeze."
              : "These are the decisions Temper made for you, and the "
                + "alternatives that lost; the configuration this job froze."}
          </p>
          {refusal && (
            <p role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800">
              <strong>{refusal.code}:</strong> {refusal.message}
            </p>
          )}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {quote.decisions.map((d) => (
              <DecisionCard
                key={d.decision}
                d={d}
                options={quote.override_options?.[d.decision]}
                editable={editable}
                overrides={overrides}
                onOverridesChange={onOverridesChange}
              />
            ))}
          </div>
        </section>
      )}
    </>
  );
}
