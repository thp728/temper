"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import type {
  AdvancedSurface,
  SurfaceField,
} from "@/lib/api/generated/client";

// The advanced surface (issue #80): every setting the pinned trainer exposes,
// behind an explicit disclosure, each with the specific thing that goes wrong
// if it is set badly. It is generated from the trainer's own schema (issue
// #33) and rendered here from the published `GET /v1/surface` document --
// never hand-listed, so the panel cannot drift from what the pre-launch gate
// refuses.
//
// Three things sit in here, because Spec 009 says the distinction is the
// product, not a footnote:
//
// * **Adjustable.** Each exposed field carries its reason and, beside it, the
//   specific failure mode -- a correct value exists, Temper chose one, and an
//   informed user may choose otherwise. Changing one re-requests the plan
//   from the server (the caller owns the recompute), and a change that makes
//   the job infeasible is refused before launch.
// * **Known but unsupported here.** A setting the trainer supports but this
//   product does not is visible with its reason rather than absent -- shown,
//   searchable, never wondered about as overlooked.
// * **Refused inputs.** Some things are not adjustable at all: a dataset that
//   mixes reasoning traces with plain answers, and one below the minimum
//   usable row count, have no correct value that any control could express.
//   They are refused at validation and cannot be overridden -- that is an
//   absent control, not a hidden one, and the copy below says so.

export default function AdvancedSurface({
  surface,
  defaults,
  values,
  onChange,
}: {
  surface: AdvancedSurface;
  defaults: Record<string, unknown>;
  values: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
}) {
  const [query, setQuery] = useState("");
  const exposed = surface.tiers.exposed_with_named_failure_mode ?? {};
  const unsupported = surface.tiers.known_but_unsupported ?? {};

  const matches = Object.entries(unsupported).filter(
    ([name, field]) =>
      query.trim() === "" ||
      name.includes(query.trim()) ||
      (field.reason ?? "").toLowerCase().includes(query.trim().toLowerCase()),
  );

  function setValue(name: string, value: string | null) {
    const next = { ...values };
    if (value == null || value === "") {
      delete next[name];
    } else {
      next[name] = value;
    }
    onChange(next);
  }

  const overriddenCount = Object.keys(values).length;

  return (
    <details
      className="group rounded-lg border bg-card"
      aria-label="Advanced settings"
    >
      <summary className="flex cursor-pointer items-center justify-between p-4 text-base font-semibold">
        <span>Advanced settings</span>
        <span className="text-sm font-normal text-muted-foreground">
          {overriddenCount > 0
            ? `${overriddenCount} changed`
            : "every exposed dial of the pinned trainer, with what goes wrong"}
        </span>
      </summary>

      <div className="space-y-5 border-t p-4">
        {/* The adjustable settings: each carries its failure mode inline. */}
        {Object.entries(exposed).map(([name, field]) => (
          <ExposedField
            key={name}
            name={name}
            field={field}
            default={defaults[name] ?? ""}
            value={values[name]}
            onSet={(v) => setValue(name, v)}
            onRevert={() => setValue(name, null)}
          />
        ))}

        {/* Why some things are adjustable and others are refused: the
            distinction Spec 009 makes load-bearing. */}
        <section
          aria-labelledby="adjustable-vs-refused"
          className="rounded-lg bg-muted/40 p-4"
        >
          <h3 id="adjustable-vs-refused" className="font-medium">
            Why some settings are adjustable and others are refused
          </h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            <li>
              A setting you can change above has a correct value: Temper chose
              one for you, and an informed user may choose otherwise. Each one
              says what goes wrong if it is set badly.
            </li>
            <li>
              A refused input has no correct value at all. A dataset that mixes
              reasoning traces with plain answers is ambiguous by construction,
              and a dataset below the minimum usable row count cannot train a
              meaningful adapter. No control resolves either, so neither can be
              overridden. That is an absent control, not a hidden one.
            </li>
            <li>
              A setting the trainer knows but Temper does not offer is refused
              with its reason, and is shown below rather than absent.
            </li>
          </ul>
        </section>

        {/* Known but unsupported here: visible with its reason, searchable. */}
        <section aria-labelledby="unsupported-heading" className="space-y-2">
          <h3 id="unsupported-heading" className="font-medium">
            The trainer supports these; Temper does not offer them (
            {Object.keys(unsupported).length})
          </h3>
          <p className="text-sm text-muted-foreground">
            Each is visible with the reason it is not offered, rather than
            absent. Passing one to a launch is refused with that reason.
          </p>
          <Input
            type="search"
            aria-label="Search the trainer's settings Temper does not offer"
            placeholder="Search by name or reason"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="h-9"
          />
          <ul className="max-h-64 space-y-2 overflow-y-auto rounded-lg border p-3 text-sm">
            {matches.length === 0 ? (
              <li className="text-muted-foreground">
                {query.trim()
                  ? "No setting matches that search."
                  : "Type to see a setting and the reason it is not offered."}
              </li>
            ) : (
              matches.map(([name, field]) => (
                <li key={name}>
                  <p>
                    <code className="rounded bg-muted px-1">{name}</code>
                  </p>
                  <p className="text-muted-foreground">{field.reason}</p>
                </li>
              ))
            )}
          </ul>
        </section>
      </div>
    </details>
  );
}

function ExposedField({
  name,
  field,
  default: defaultValue,
  value,
  onSet,
  onRevert,
}: {
  name: string;
  field: SurfaceField;
  default: unknown;
  value?: string;
  onSet: (value: string) => void;
  onRevert: () => void;
}) {
  const current = value ?? String(defaultValue);
  const [text, setText] = useState(current);
  // Remount when the effective value changes (a recompute or a revert
  // landed), so the input always reflects the plan it edits.
  const key = value ?? "default";
  const numeric = field.type === "int" || field.type === "float";

  function commit() {
    const trimmed = text.trim();
    if (trimmed === "" || trimmed === String(defaultValue)) {
      onRevert();
    } else {
      onSet(trimmed);
    }
  }

  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="font-medium">
          <code className="rounded bg-muted px-1">{name}</code>
        </h4>
        {value !== undefined && (
          <span className="rounded bg-amber-100 px-1.5 py-0.5 text-xs font-medium text-amber-800">
            you changed this
          </span>
        )}
      </div>
      <p className="mt-1 text-sm text-muted-foreground">{field.reason}</p>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Input
          key={key}
          type={numeric ? "number" : "text"}
          step={field.type === "int" ? "1" : "any"}
          aria-label={`${name} override`}
          defaultValue={current}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
          }}
          className="h-9 w-40"
        />
        {value !== undefined && (
          <button
            type="button"
            onClick={onRevert}
            className="text-sm text-muted-foreground underline underline-offset-2 hover:text-foreground"
          >
            Use the default
          </button>
        )}
      </div>
      {field.failure_mode && (
        <p className="mt-2 text-sm">
          <span className="font-medium">What goes wrong: </span>
          <span className="text-muted-foreground">{field.failure_mode}</span>
        </p>
      )}
    </div>
  );
}
