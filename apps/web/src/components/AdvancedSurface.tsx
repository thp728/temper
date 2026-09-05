"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Input } from "@/components/ui/input";
import type { AdvancedSurface } from "@/lib/api/generated/client";

// The advanced surface (issue #80), generated from the trainer's own schema
// (issue #33) and rendered here from the published `GET /v1/surface` document
// -- never hand-listed, so the panel cannot drift from what the pre-launch
// gate refuses.
//
// Since the step-2 cards edit every exposed key in place, this disclosure
// carries only what the cards cannot: the trainer settings Temper does not
// support yet. Each is visible with its reason rather than absent -- shown,
// searchable, never wondered about as overlooked. Axolotl exposes these
// parameters; support in Temper will be added, and passing one to a launch
// is refused for now, with its reason.

export default function AdvancedSurface({
  surface,
}: {
  surface: AdvancedSurface;
}) {
  const [query, setQuery] = useState("");
  const unsupported = surface.tiers.known_but_unsupported ?? {};

  const matches = Object.entries(unsupported).filter(
    ([name, field]) =>
      query.trim() === "" ||
      name.includes(query.trim()) ||
      (field.reason ?? "").toLowerCase().includes(query.trim().toLowerCase()),
  );

  return (
    <details
      className="group rounded-lg border bg-card"
      aria-label="Advanced settings"
    >
      <summary className="flex cursor-pointer items-center justify-between gap-2 p-4 text-base font-semibold">
        <span>Advanced settings</span>
        <ChevronDown
          aria-hidden="true"
          className="size-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180"
        />
      </summary>

      <div className="space-y-5 border-t p-4">
        {/* Known but unsupported here: visible with its reason, searchable. */}
        <section aria-labelledby="unsupported-heading" className="space-y-2">
          <h3 id="unsupported-heading" className="font-medium">
            Axolotl settings Temper doesn&apos;t support yet (
            {Object.keys(unsupported).length})
          </h3>
          <p className="text-sm text-muted-foreground">
            Axolotl exposes these parameters, but Temper doesn&apos;t support
            them yet — support will be added. Passing one to a launch is
            refused for now, with its reason.
          </p>
          <Input
            type="search"
            aria-label="Search unsupported Axolotl settings"
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
