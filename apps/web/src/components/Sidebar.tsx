"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Database,
  LayoutDashboard,
  Network,
  Plus,
  Rocket,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import TemperMark from "@/components/TemperMark";
import { cn } from "@/lib/utils";

// The main navigation, on the left (the rework's shell): dashboard, datasets,
// models, jobs. A client component because only the browser knows which route
// it is on -- the active item is marked with `aria-current="page"`, so a
// screen reader announces where the user is, and the styling rides the same
// flag rather than a second, visual-only attribute.
//
// The brand lockup (mark, wordmark, subtitle) is one link to the dashboard --
// one accessible name, never two adjacent links to the same place. The new-
// job action is pinned to the bottom of the nav, as the wireframe has it.
// "New job", not the wireframe's "New Run": CONTEXT.md's Job entry avoids
// "run".

const NAV_ITEMS: { href: string; label: string; icon: LucideIcon }[] = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/models", label: "Models", icon: Network },
  { href: "/jobs", label: "Jobs", icon: Rocket },
];

// The dashboard owns the exact root; every other item is also active on its
// descendants (a job record is still "Jobs").
function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r bg-card md:flex">
      <div className="px-4 pt-5 pb-6">
        <Link
          href="/"
          className="flex items-center gap-3 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {/* Sized to the exact pixel height of the text stack beside it --
              text-2xl/leading-tight (30px) plus text-xs's default line
              height (16px) -- rather than a size-* step, whose nearest
              values (44px, 48px) sit a couple of pixels off either way.
              The mark's own viewBox padding is left untouched: it already
              roughly matches the half-leading either line of text carries
              above and below its own ink, which is what lines the two up. */}
          <TemperMark className="block h-[46px] w-[46px] shrink-0 text-primary" />
          <span>
            <span className="block text-2xl leading-tight font-semibold tracking-tight">
              Temper
            </span>
            <span className="block text-xs text-muted-foreground">
              Fine-tuning platform
            </span>
          </span>
        </Link>
      </div>
      <nav aria-label="Main" className="flex-1 space-y-1 px-3">
        {NAV_ITEMS.map((item) => {
          const active = isActive(pathname, item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex items-center gap-2.5 rounded-md px-3 py-2.5 text-[15px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                active
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
            >
              <Icon aria-hidden="true" className="size-[18px]" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="px-3 pb-4">
        <Button asChild className="w-full">
          <Link href="/datasets">
            <Plus aria-hidden="true" className="size-4" />
            New job
          </Link>
        </Button>
      </div>
    </aside>
  );
}
