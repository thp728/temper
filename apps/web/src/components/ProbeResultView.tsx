"use client";

import type { ProbeResult } from "@/lib/api/generated/client";

// The probe result, shown rather than merely enforced (issue #58): a model
// that passes with warnings is usable, and this is where the user learns what
// they took on. Each finding names its severity -- a `block` is why the model
// cannot launch, a `warn` is what the user accepted by launching anyway -- and
// the memory line carries the same arithmetic the launch-time refusal (#54)
// would refuse with.
export default function ProbeResultView({ probe }: { probe: ProbeResult }) {
  const verdictLabel =
    probe.verdict === "blocked"
      ? "Blocked"
      : probe.verdict === "usable_with_warnings"
        ? "Usable with warnings"
        : "Usable";

  const findings = probe.findings ?? [];

  return (
    <div className="space-y-2 text-sm" data-testid="probe-result">
      <p>
        <span className="font-medium">Compatibility probe: </span>
        <span
          className={
            probe.verdict === "blocked"
              ? "font-medium text-red-700"
              : probe.verdict === "usable_with_warnings"
                ? "font-medium text-amber-700"
                : "font-medium text-green-700"
          }
        >
          {verdictLabel}
        </span>
      </p>

      {findings.length > 0 && (
        <ul className="space-y-1">
          {findings.map((f) => (
            <li key={f.code} className="text-muted-foreground">
              <code className="rounded bg-muted px-1 text-xs">
                {f.severity === "block" ? "blocks" : "warning"}
              </code>{" "}
              {f.message}
            </li>
          ))}
        </ul>
      )}

      {probe.memory && (
        <dl className="flex flex-wrap gap-x-4 gap-y-1">
          <div>
            <dt className="inline text-muted-foreground">Predicted peak </dt>
            <dd className="inline">
              {probe.memory.peak_gb !== null &&
              probe.memory.peak_gb !== undefined
                ? `${probe.memory.peak_gb.toFixed(2)} GB`
                : "unknown"}
            </dd>
          </div>
          <div>
            <dt className="inline text-muted-foreground"> · smallest card </dt>
            <dd className="inline">{probe.memory.card ?? "none"}</dd>
          </div>
          {probe.memory.headroom_gb !== null &&
            probe.memory.headroom_gb !== undefined && (
              <div>
                <dt className="inline text-muted-foreground"> · headroom </dt>
                <dd className="inline">
                  {probe.memory.headroom_gb.toFixed(2)} GB
                </dd>
              </div>
            )}
          {probe.memory.note && (
            <div className="basis-full">
              <dd className="text-muted-foreground">{probe.memory.note}</dd>
            </div>
          )}
        </dl>
      )}
    </div>
  );
}
