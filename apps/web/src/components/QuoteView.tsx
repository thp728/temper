"use client";

import { useState } from "react";
import { Check, ChevronDown, CircleHelp } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
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
//
// The three sections render independently (all on by default, which is the
// finished-job record and what the existing tests assert): the launch wizard
// spreads them across its steps — training-shape decisions beside the
// hyperparameters, the estimate and the where-it-runs decisions beside the
// hardware — while each section keeps the same headings and accessible names
// wherever it appears. The wizard passes explain="dialog" so a form stays a
// form: label, value and control on the card, reason and alternatives behind
// a "?". The record keeps the inline default.

export interface OverrideRefusal {
  code: string;
  message: string;
}

// Pinning and unpinning share one rule wherever the control lives (select,
// radios): reselecting the predictor's *default* clears the override,
// because the controls always show the effective value and there is no
// placeholder to confuse with a value. The default is passed in — after a
// pin the recomputed quote's `chosen` *is* the pinned value, so reading it
// back would make the original default unselectable.
function applyDecisionValue(
  d: QuoteDecision,
  overrides: DecisionOverride[],
  value: string | null,
  defaultChoice: string,
): DecisionOverride[] {
  const rest = overrides.filter((o) => o.decision !== d.decision);
  return value == null || value === "" || value === defaultChoice
    ? rest
    : [...rest, { decision: d.decision, value }];
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
  defaultChoice,
}: {
  d: QuoteDecision;
  options?: string[];
  overrides: DecisionOverride[];
  onOverridesChange: (overrides: DecisionOverride[]) => void;
  defaultChoice: string;
}) {
  const current = overrides.find((o) => o.decision === d.decision);

  function setValue(value: string | null) {
    onOverridesChange(applyDecisionValue(d, overrides, value, defaultChoice));
  }

  // Select-style decisions carry a server-published vocabulary; the free-form
  // ones accept any positive number, so they get an input instead. Both sit
  // beside the explanation they edit (spec 005: one surface, not a beginner
  // mode and an expert mode).
  if (options && options.length > 0) {
    return (
      <span className="relative inline-flex items-center">
        <select
          aria-label={`${d.decision} override`}
          value={current?.value ?? d.chosen}
          onChange={(e) => setValue(e.target.value)}
          className="h-9 appearance-none rounded-md border bg-transparent pr-8 pl-2 text-sm"
        >
          {options.map((v) => (
            <option key={v} value={v} className="bg-card text-foreground">
              {v === defaultChoice ? `${v} (predictor's choice)` : v}
            </option>
          ))}
        </select>
        <ChevronDown
          aria-hidden="true"
          className="pointer-events-none absolute top-1/2 right-2 size-4 -translate-y-1/2 text-muted-foreground"
        />
      </span>
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

// One radio card per published option, for a decision the user picks rather
// than types. The predictor's choice carries its badge; the effective value
// carries the selected ring. Only names and the alternatives' own cost lines
// appear here -- per-card specs the contract does not publish (VRAM, vCPUs,
// hourly rates, availability) are not invented to fill the mock's boxes.
function DecisionOptionCards({
  d,
  options,
  overrides,
  onOverridesChange,
  editable,
  defaultChoice,
}: {
  d: QuoteDecision;
  options: string[];
  overrides: DecisionOverride[];
  onOverridesChange?: (overrides: DecisionOverride[]) => void;
  editable?: boolean;
  defaultChoice: string;
}) {
  const canEdit = editable && onOverridesChange ? true : false;
  const effective = overrides.find((o) => o.decision === d.decision)?.value ?? d.chosen;
  const costs = new Map(
    (d.alternatives ?? []).map((a) => [a.value, a.cost] as const),
  );

  function pick(value: string) {
    if (!canEdit) return;
    onOverridesChange?.(applyDecisionValue(d, overrides, value, defaultChoice));
  }

  return (
    <div
      role="radiogroup"
      aria-label={`${d.decision} options`}
      className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2"
    >
      {options.map((v) => {
        const checked = effective === v;
        const recommended = v === defaultChoice;
        return (
          <label
            key={v}
            className={
              "flex cursor-pointer items-center gap-2.5 rounded-lg border bg-card p-3 transition-colors hover:bg-muted/50 focus-within:ring-1 focus-within:ring-ring " +
              (checked ? "border-primary/60 ring-1 ring-primary/40" : "")
            }
          >
            <input
              type="radio"
              name={`decision-${d.decision}`}
              value={v}
              checked={checked}
              disabled={!canEdit}
              onChange={() => pick(v)}
              className="sr-only"
            />
            <span
              aria-hidden="true"
              className={
                "flex size-5 shrink-0 items-center justify-center rounded-full border transition-colors " +
                (checked
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-muted-foreground/40 text-transparent")
              }
            >
              <Check className="size-3.5" />
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex flex-wrap items-center gap-1.5">
                <span className="truncate text-sm font-semibold">{v}</span>{" "}
                {recommended && (
                  <span className="shrink-0 rounded-md border border-primary/30 bg-primary/15 px-1.5 py-px font-mono text-[10px] font-medium text-primary">
                    Recommended
                  </span>
                )}
              </span>
              {costs.get(v) && (
                <span className="block truncate font-mono text-xs text-muted-foreground">
                  {costs.get(v)}
                </span>
              )}
            </span>
          </label>
        );
      })}
    </div>
  );
}

function DecisionCard({
  d,
  options,
  editable,
  overrides,
  onOverridesChange,
  explain = "inline",
  layout = "control",
  defaultChoice,
}: {
  d: QuoteDecision;
  options?: string[];
  editable?: boolean;
  overrides: DecisionOverride[];
  onOverridesChange?: (overrides: DecisionOverride[]) => void;
  // Where the reason and the losing alternatives live. "inline" prints them
  // on the card (the finished-job record, where the explanation is the
  // product); "dialog" keeps the card to label, value and control with the
  // explanation one "?" click away (the launch wizard, which is a form).
  explain?: "inline" | "dialog";
  // How a select-vocabulary decision is picked. "control" is the compact
  // select beside the explanation; "cards" is one radio card per option, for
  // the hardware choice Temper defaults and the user overrides. Cards need
  // the server-published vocabulary, so anything else falls back to control.
  layout?: "control" | "cards";
  // The predictor's default for this decision: what reselecting unpins and
  // what the Recommended badge marks. Falls back to the current choice,
  // which is identical until the first pin.
  defaultChoice?: string;
}) {
  const alternatives = d.alternatives ?? [];
  const base = defaultChoice ?? d.chosen;
  const isCards =
    layout === "cards" && options && options.length > 0 ? true : false;
  return (
    <div
      className={
        isCards
          ? "rounded-[12px] border bg-card p-5 sm:col-span-2"
          : "rounded-[12px] border bg-card p-5"
      }
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-xs font-medium tracking-widest uppercase text-muted-foreground">
          {d.decision}
        </h3>
        <span className="flex shrink-0 items-center gap-1.5">
          <span className="rounded-md border bg-muted px-2 py-0.5 font-mono text-xs font-medium text-foreground">
            {d.chosen}
          </span>
          {isCards && <DecisionExplanationDialog d={d} />}
        </span>
      </div>
      {d.overridden && (
        <span className="mt-2 inline-block rounded bg-amber-100 px-1.5 py-0.5 text-xs font-medium text-amber-800">
          you changed this
        </span>
      )}
      {explain === "inline" && (
        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{d.constraint}</p>
      )}
      {isCards ? (
        <DecisionOptionCards
          d={d}
          options={options ?? []}
          overrides={overrides}
          onOverridesChange={onOverridesChange}
          editable={editable}
          defaultChoice={base}
        />
      ) : explain === "dialog" ? (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {editable && onOverridesChange && (
            <DecisionControl
              d={d}
              options={options}
              overrides={overrides}
              onOverridesChange={onOverridesChange}
              defaultChoice={base}
            />
          )}
          <DecisionExplanationDialog d={d} />
        </div>
      ) : (
        <>
          {editable && onOverridesChange && (
            <div className="mt-3">
              <DecisionControl
                d={d}
                options={options}
                overrides={overrides}
                onOverridesChange={onOverridesChange}
                defaultChoice={base}
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
              <AlternativesList alternatives={alternatives} />
            </details>
          )}
        </>
      )}
    </div>
  );
}

// The losing alternatives, shared by the inline disclosure and the dialog:
// value, what it would have cost, and why it lost.
function AlternativesList({
  alternatives,
}: {
  alternatives: NonNullable<QuoteDecision["alternatives"]>;
}) {
  return (
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
  );
}

// The reason and the losing alternatives behind a "?" button, for surfaces
// that are forms first: the card keeps label, value and control, and the
// explanation opens on demand.
function DecisionExplanationDialog({ d }: { d: QuoteDecision }) {
  const alternatives = d.alternatives ?? [];
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label={`About the ${d.decision} decision`}
          className="text-muted-foreground hover:text-foreground"
        >
          <CircleHelp className="size-4" aria-hidden="true" />
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle className="text-xs font-medium tracking-widest uppercase text-muted-foreground">
          {d.decision}
        </DialogTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          {d.constraint}
        </p>
        {alternatives.length > 0 && (
          <div className="border-t pt-3">
            <p className="font-mono text-xs text-muted-foreground">
              Alternatives considered ({alternatives.length})
            </p>
            <AlternativesList alternatives={alternatives} />
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function PhaseRow({ phase, quote }: { phase: Quote["phases"][number]; quote: Quote }) {
  return (
    <tr>
      <td className="px-6 py-3.5 text-foreground">{phase.name}</td>
      <td className="px-6 py-3.5 text-right text-muted-foreground">
        {formatDurationRange(phase.duration_low_s, phase.duration_high_s)}
      </td>
      <td className="px-6 py-3.5 text-right text-muted-foreground sm:min-w-36">
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
      </td>
    </tr>
  );
}

export default function QuoteView({
  quote,
  editable = false,
  overrides = [],
  onOverridesChange,
  refusal,
  showEstimate = true,
  showPhases = true,
  showDecisions = true,
  decisions,
  explain = "inline",
  bareDecisions = false,
  cardDecisions = [],
  defaultChoices = {},
}: {
  quote: Quote;
  editable?: boolean;
  overrides?: DecisionOverride[];
  onOverridesChange?: (overrides: DecisionOverride[]) => void;
  refusal?: OverrideRefusal | null;
  showEstimate?: boolean;
  showPhases?: boolean;
  showDecisions?: boolean;
  // Allowlist of decision names to show. Absent means every decision the
  // predictor recorded — the six names in temper_core.decisions, of which the
  // wizard shows the training-shape ones (method, precision,
  // "sequence length") beside the hyperparameters and the where-it-runs ones
  // (hardware, "device count", disk) beside the compute choice.
  decisions?: string[];
  // Where each decision's reason and losing alternatives live: "inline"
  // prints them on the card (the finished-job record), "dialog" keeps the
  // card to label, value and control with the explanation behind a "?"
  // (the launch wizard, which is a form first).
  explain?: "inline" | "dialog";
  // Drop the decisions section chrome (heading and intro) and render only
  // the refusal and the cards, for embedding inside a form card that names
  // the subject itself. Headings, controls and names inside are unchanged.
  bareDecisions?: boolean;
  // Decisions picked from option cards instead of the compact select: the
  // hardware choice Temper defaults and the user overrides, one radio card
  // per server-published option.
  cardDecisions?: string[];
  // The predictor's default per decision, for pin/unpin and badges. After a
  // pin the recomputed quote's `chosen` is the pinned value, so without this
  // the original default would become unselectable. Absent entries fall back
  // to the current choice.
  defaultChoices?: Record<string, string>;
}) {
  const shownDecisions =
    decisions && decisions.length > 0
      ? (quote.decisions ?? []).filter((d) => decisions.includes(d.decision))
      : (quote.decisions ?? []);
  return (
    <>
      {showEstimate && (
      <section aria-labelledby="quote-heading" className="space-y-3">
        <h2 id="quote-heading" className="text-xs font-medium tracking-widest uppercase text-foreground">
          Cost and time estimate
        </h2>

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
      {showPhases && (
      <div className="overflow-x-auto rounded-[12px] border bg-card">
        <table className="w-full font-mono text-sm tabular-nums">
          <caption className="sr-only">
            Estimated duration and cost by phase
          </caption>
          <thead>
            <tr className="border-b text-left text-xs tracking-wider text-muted-foreground uppercase">
              <th scope="col" className="px-6 py-3 font-medium">
                Phase
              </th>
              <th scope="col" className="px-6 py-3 text-right font-medium">
                Duration
              </th>
              <th scope="col" className="px-6 py-3 text-right font-medium">
                Cost
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {quote.phases.map((phase) => (
              <PhaseRow key={phase.name} phase={phase} quote={quote} />
            ))}
            {/* Storage bills on its own line, in USD, never folded into the
                account-currency phases (ADR-0030): an invisible line is exactly
                the hidden cost the quote exists to remove. */}
            <tr>
              <td className="px-6 py-3.5 text-foreground">
                storage (USD, separate line)
              </td>
              <td className="px-6 py-3.5 text-right text-muted-foreground">—</td>
              <td className="px-6 py-3.5 text-right text-muted-foreground sm:min-w-36">
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
              </td>
            </tr>
          </tbody>
        </table>
        <p className="border-t px-6 py-3 font-mono text-xs text-muted-foreground">
          Estimated against dataset {shortRevision(quote.dataset_id)}
          {" · "}model revision {shortRevision(quote.base_revision)}
          {" · "}expires {formatTimestamp(quote.expires_at)}.
        </p>
      </div>
      )}
      </section>
      )}

      {/* The reasons are the product (issue #76): each decision the predictor
          made is shown with its reason visible by default and its
          alternatives one interaction away -- never hidden, never always
          shown. The records are the same ones frozen into the job spec, so
          a finished job explains itself as completely as a planned one. In
          the plan the controls sit beside the explanations (issue #79), and
          a change re-requests the plan: one surface, not a beginner mode and
          an expert mode. */}
      {showDecisions && shownDecisions.length > 0 && (
        bareDecisions ? (
          <>
            {refusal && (
              <p role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800">
                <strong>{refusal.code}:</strong> {refusal.message}
              </p>
            )}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {shownDecisions.map((d) => (
                <DecisionCard
                  key={d.decision}
                  d={d}
                  options={quote.override_options?.[d.decision]}
                  editable={editable}
                  overrides={overrides}
                  onOverridesChange={onOverridesChange}
                  explain={explain}
                layout={cardDecisions.includes(d.decision) ? "cards" : "control"}
                defaultChoice={defaultChoices[d.decision]}
                />
              ))}
            </div>
          </>
        ) : (
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
            {shownDecisions.map((d) => (
              <DecisionCard
                key={d.decision}
                d={d}
                options={quote.override_options?.[d.decision]}
                editable={editable}
                overrides={overrides}
                onOverridesChange={onOverridesChange}
                explain={explain}
                layout={cardDecisions.includes(d.decision) ? "cards" : "control"}
                defaultChoice={defaultChoices[d.decision]}
              />
            ))}
          </div>
        </section>
        )
      )}
    </>
  );
}
