import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

// The empty state every list-shaped section shares: a centred icon in a soft
// well, a heading, one line of supporting copy, and -- where the section is
// the page's own next step -- an action beneath. The look was set by the
// dashboard's active-jobs panel; reading it back here means one empty state
// has one look everywhere, instead of each screen improvising its own. The
// icon is decorative; the text carries the meaning.
export default function EmptyStatePanel({
  icon: Icon,
  heading,
  headingAs: Heading = "h3",
  children,
  action,
}: {
  icon: LucideIcon;
  heading: string;
  /** Sections under an h2 render their panel heading as h3; pages under the
      h1 title render it as h2. Kept explicit so the outline never skips. */
  headingAs?: "h2" | "h3";
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-[12px] border bg-card px-6 py-12 text-center">
      <span className="flex size-12 items-center justify-center rounded-full bg-foreground/5">
        <Icon aria-hidden="true" className="size-5 text-muted-foreground" />
      </span>
      <Heading className="font-medium">{heading}</Heading>
      <p className="max-w-sm text-sm text-muted-foreground">{children}</p>
      {action}
    </div>
  );
}
