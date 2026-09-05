import {
  Ban,
  CheckCircle2,
  Clock,
  Cpu,
  Download,
  Loader2,
  Package,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

// The status pill: the job's own status word in a rounded-full badge, a
// translucent tint of its hue behind the same hue at full strength. Colour
// carries meaning, so it appears only where there is an outcome to colour:
// complete is the design system's one semantic success, failed is the
// destructive token, and everything else -- cancelled, and every working
// state (queued, provisioning, preparing, training, packaging, the full
// vocabulary the control plane's db.py defines) -- stays neutral, because a
// job still in progress has no outcome yet. An unknown status falls back to
// neutral rather than crashing, so a state added server-side renders before
// this map learns about it.
//
// One component, not ternaries at the call site: the jobs list reuses it.

const NEUTRAL = "bg-foreground/10 text-muted-foreground";

const STATUS_TONES: Record<string, string> = {
  queued: NEUTRAL,
  provisioning: NEUTRAL,
  preparing: NEUTRAL,
  training: NEUTRAL,
  packaging: NEUTRAL,
  complete: "bg-success/10 text-success",
  failed: "bg-destructive/10 text-destructive",
  cancelled: NEUTRAL,
};

// Each stage gets the icon of what is actually happening on it -- hardware
// selection is a chip, a machine coming up is a download, packaging is a
// box -- so the state tile reads at a glance instead of leaning on the word
// alone. `training` is the one icon that spins: it is the only stage where
// something is visibly *happening* right now rather than being waited on,
// and the spin is what a reader notices before the tile reloads frozen into
// its terminal icon.
const STATUS_ICONS: Record<
  string,
  React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>
> = {
  queued: Clock,
  provisioning: Cpu,
  preparing: Download,
  training: Loader2,
  packaging: Package,
  complete: CheckCircle2,
  failed: XCircle,
  cancelled: Ban,
};

export function statusIcon(
  status: string,
): React.ComponentType<{ className?: string; "aria-hidden"?: boolean }> {
  return STATUS_ICONS[status] ?? Ban;
}

// Only `training`'s icon (a bare spinner) reads sensibly in motion; every
// other stage's icon is a static glyph that would look broken spinning.
export function statusIconSpins(status: string): boolean {
  return status === "training";
}

export default function StatusPill({ status }: { status: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 font-mono text-xs font-medium uppercase",
        STATUS_TONES[status] ?? NEUTRAL,
      )}
    >
      {status}
    </span>
  );
}
