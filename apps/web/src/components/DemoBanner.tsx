// Spec 012's honesty rule: the zero-cost tier must say what it is in the
// interface, on every page it serves, and the marking cannot be turned off in
// that mode. This is that marking. It is server-rendered by the root layout,
// so a screenshot of any page includes it, and there is deliberately no
// dismiss control — a banner a visitor can dismiss is a banner that can be
// missing from the very screenshot meant to prove the mode was labelled.
export default function DemoBanner() {
  return (
    <div
      aria-label="Demonstration mode"
      className="border-b border-amber-600 bg-amber-100 px-4 py-2 text-sm text-amber-950"
    >
      <div className="mx-auto flex w-full max-w-3xl flex-wrap items-baseline gap-x-2 gap-y-1">
        <strong>Demonstration mode</strong>
        <span className="text-amber-900">
          Jobs run against a simulated machine. Nothing here touches real
          compute, crosses a connection, or spends money.
        </span>
      </div>
    </div>
  );
}
