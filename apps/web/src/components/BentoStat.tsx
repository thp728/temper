// One tile of the job record's status row, shared by the running view and
// the finished record (issue: non-terminal/terminal parity) so a job's page
// reloading from live to finished swaps data, never the shape holding it.
//
// Each tile is its own `dl`, so the label/value pair stays a real dt/dd
// adjacency (that's what a screen reader announces, and what the tests read)
// while still living inside its own bordered card. Every tile holds to the
// same two lines -- label, then value -- so the row stays level; a third
// line on one card alone (issue: device count as its own hint) is what broke
// that before.
export default function BentoStat({
  icon: Icon,
  iconClassName,
  label,
  valueClassName = "mt-1 text-2xl font-semibold tracking-tight tabular-nums",
  children,
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  iconClassName?: string;
  label: string;
  valueClassName?: string;
  children: React.ReactNode;
}) {
  return (
    <dl className="rounded-[12px] border bg-card p-4">
      <Icon
        aria-hidden
        className={iconClassName ?? "mb-2 size-4 text-muted-foreground"}
      />
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className={valueClassName}>{children}</dd>
    </dl>
  );
}
