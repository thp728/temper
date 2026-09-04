"use client";

import { useId, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import type { MetricPoint } from "@/lib/jobs/loss";

// The two loss curves every run gets read against (issue #53), each on its
// own axis rather than one shared one. Training loss is measured every step
// and held-out loss every so many epochs (the trainer prints eval rows by
// epoch, not step) -- forcing them onto one shared x meant "measurement #5"
// could be step 50 on one curve and epoch 0.5 on the other, coincidentally
// sharing a position only because that many metric events had landed by
// then. Splitting the plots means each x-axis says what it actually is.
//
// No chart library -- a line over one series is the whole need, and a
// dependency bought for two small charts is a dependency maintained forever.
// Every tick label and the point tooltip render as HTML siblings of the SVG,
// not SVG <text>: an SVG text element is sized in the viewBox's own
// coordinate space, so it scales with the chart's rendered width instead of
// staying a fixed, legible size the way the rest of the page's type does.

const WIDTH = 300;
const HEIGHT = 170;
const PAD = { top: 12, right: 12, bottom: 20, left: 40 };

const plotWidth = WIDTH - PAD.left - PAD.right;
const plotHeight = HEIGHT - PAD.top - PAD.bottom;

const pctX = (x: number) => (x / WIDTH) * 100;
const pctY = (y: number) => (y / HEIGHT) * 100;

function formatLoss(n: number): string {
  return n < 0.1 ? n.toFixed(4) : n.toFixed(3);
}

// Evenly spaced x-axis ticks over the range actually recorded, capped so a
// long run doesn't crowd the axis with a label per point.
function ticksOver(minX: number, maxX: number, count = 4): number[] {
  if (maxX <= minX) return [minX];
  const n = Math.min(count, maxX - minX + 1);
  const out = new Set<number>();
  for (let i = 0; i < n; i++) {
    out.add(Math.round(minX + ((maxX - minX) * i) / (n - 1)));
  }
  return [...out].sort((a, b) => a - b);
}

interface Point {
  x: number;
  loss: number;
}

function DeltaChip({ from, to }: { from: number; to: number }) {
  // Loss is the one metric where down is the win -- green for a drop, the
  // destructive tone for a rise, the same delta-badge language the token
  // histogram and the capability comparison already use elsewhere on this
  // page.
  const percent = from !== 0 ? ((to - from) / Math.abs(from)) * 100 : 0;
  const improved = to < from;
  const Icon = improved ? ArrowDown : ArrowUp;
  return (
    <span
      className={`inline-flex items-center gap-0.5 text-xs font-medium tabular-nums ${
        improved ? "text-success" : "text-destructive"
      }`}
    >
      <Icon aria-hidden className="size-3" />
      {Math.abs(percent).toFixed(0)}% since the first measurement
    </span>
  );
}

// One series, one axis: training loss against step, or held-out loss against
// epoch. Never both on the same plot, so the x-axis it draws is always the
// unit its own label names.
function SeriesChart({
  title,
  unit,
  points,
  color,
  dashed,
  areaFill,
  bestX,
  bestLabel,
}: {
  title: string;
  /** What the x-axis actually is: "step" or "epoch". */
  unit: string;
  points: Point[];
  color: string;
  dashed?: boolean;
  areaFill?: boolean;
  /** The x-value of the checkpoint this run kept, if it falls on this series. */
  bestX?: number;
  bestLabel?: string;
}) {
  const gradientId = useId();
  const [hovered, setHovered] = useState<Point | null>(null);

  if (points.length === 0) {
    return (
      <div className="rounded-[12px] border bg-card p-5">
        <p className="text-xs text-muted-foreground">{title}</p>
        <p className="mt-3 text-sm text-muted-foreground">
          Not measured yet. It appears here as the run goes.
        </p>
      </div>
    );
  }

  const start = points[0]!.loss;
  const latest = points[points.length - 1]!.loss;

  const minX = points[0]!.x;
  const maxX = points[points.length - 1]!.x;
  const xScale = (x: number) =>
    PAD.left + (maxX === minX ? plotWidth / 2 : ((x - minX) / (maxX - minX)) * plotWidth);

  const values = points.map((p) => p.loss);
  let minY = Math.min(...values);
  let maxY = Math.max(...values);
  const padY = Math.max((maxY - minY) * 0.1, 0.05);
  minY -= padY;
  maxY += padY;
  const yScale = (y: number) =>
    PAD.top + (1 - (y - minY) / (maxY - minY)) * plotHeight;

  const line =
    points.length > 1
      ? points.map((p) => `${xScale(p.x)},${yScale(p.loss)}`).join(" ")
      : null;
  const area =
    line != null
      ? `${xScale(minX)},${HEIGHT - PAD.bottom} ${line} ${xScale(maxX)},${HEIGHT - PAD.bottom}`
      : null;

  const yTicks = [0, 1 / 3, 2 / 3, 1].map((f) => minY + (maxY - minY) * (1 - f));
  const xAxisTicks = ticksOver(minX, maxX);

  return (
    <figure className="rounded-[12px] border bg-card p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs text-muted-foreground">{title}</p>
          <p className="mt-0.5 text-2xl font-semibold tracking-tight tabular-nums">
            {formatLoss(latest)}
          </p>
          {points.length > 1 && <DeltaChip from={start} to={latest} />}
        </div>
        {bestX !== undefined && bestLabel && (
          <div className="text-right">
            <p className="text-xs text-muted-foreground">Kept checkpoint</p>
            <p className="mt-0.5 text-sm font-medium">{bestLabel}</p>
          </div>
        )}
      </div>

      <div className="relative mt-4 select-none">
        {/* Y-axis tick labels, positioned as HTML so they hold a fixed,
            legible size regardless of the chart's rendered width. Placed in
            the same left margin the plot itself reserves (PAD.left). */}
        {yTicks.map((t, i) => (
          <span
            key={i}
            className="absolute left-0 -translate-y-1/2 text-right text-[10px] whitespace-nowrap text-muted-foreground tabular-nums"
            style={{ top: `${pctY(yScale(t))}%`, width: `${pctX(PAD.left - 6)}%` }}
          >
            {formatLoss(t)}
          </span>
        ))}

        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="h-auto w-full overflow-visible"
          role="img"
          aria-label={`Chart of ${title.toLowerCase()} against ${unit}`}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.22" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>

          {yTicks.map((t, i) => (
            <line
              key={i}
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={yScale(t)}
              y2={yScale(t)}
              stroke="var(--color-border)"
              strokeWidth={1}
            />
          ))}

          {areaFill && area && <polygon points={area} fill={`url(#${gradientId})`} />}
          {line && (
            <polyline
              points={line}
              fill="none"
              stroke={color}
              strokeWidth={2}
              strokeDasharray={dashed ? "4 3" : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          )}

          {/* A visible dot at every measurement, plus a larger invisible hit
              target so hovering or focusing it names the exact value. */}
          {points.map((p) => (
            <g key={p.x}>
              <circle cx={xScale(p.x)} cy={yScale(p.loss)} r={3} fill={color} />
              <circle
                cx={xScale(p.x)}
                cy={yScale(p.loss)}
                r={9}
                fill="transparent"
                tabIndex={0}
                role="img"
                aria-label={`${formatLoss(p.loss)}, ${unit} ${p.x}`}
                onMouseEnter={() => setHovered(p)}
                onFocus={() => setHovered(p)}
                onMouseLeave={() => setHovered(null)}
                onBlur={() => setHovered(null)}
                className="cursor-pointer outline-none"
              />
            </g>
          ))}

          {bestX !== undefined && (
            <circle
              cx={xScale(bestX)}
              cy={yScale(points.find((p) => p.x === bestX)?.loss ?? latest)}
              r={5}
              fill="var(--color-card)"
              stroke="var(--color-primary)"
              strokeWidth={2}
            />
          )}

          <line
            x1={PAD.left}
            y1={HEIGHT - PAD.bottom}
            x2={WIDTH - PAD.right}
            y2={HEIGHT - PAD.bottom}
            stroke="var(--color-border)"
          />

          {hovered && (
            <circle
              cx={xScale(hovered.x)}
              cy={yScale(hovered.loss)}
              r={5.5}
              fill="none"
              stroke={color}
              strokeWidth={1.5}
              pointerEvents="none"
            />
          )}
        </svg>

        {/* X-axis tick labels: real values in the axis's own unit. */}
        <div className="relative" style={{ height: 14 }}>
          {xAxisTicks.map((x) => (
            <span
              key={x}
              className="absolute top-0 -translate-x-1/2 text-[10px] whitespace-nowrap text-muted-foreground tabular-nums first:translate-x-0 last:-translate-x-full"
              style={{ left: `${pctX(xScale(x))}%` }}
            >
              {x}
            </span>
          ))}
        </div>

        {hovered && (
          <div
            className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-md border bg-popover px-2 py-1 text-xs whitespace-nowrap text-popover-foreground shadow-sm"
            style={{
              left: `${pctX(xScale(hovered.x))}%`,
              top: `${Math.max(0, pctY(yScale(hovered.loss)) - 4)}%`,
            }}
          >
            <span className="font-medium tabular-nums">{formatLoss(hovered.loss)}</span>
            <span className="text-muted-foreground">
              {" · "}
              {unit} {hovered.x}
            </span>
          </div>
        )}
      </div>
      <p className="mt-1 text-[10px] text-muted-foreground capitalize">{unit}</p>

      {points.length < 2 && (
        <p className="mt-2 text-xs text-muted-foreground">
          One measurement recorded so far. A line appears once there are two.
        </p>
      )}
    </figure>
  );
}

export default function LossChart({
  training,
  heldOut,
  best,
}: {
  training: MetricPoint[];
  heldOut: MetricPoint[];
  /** The checkpoint this run kept, so its point can be marked on the curve. */
  best?: { step?: number | null; held_out_loss?: number | null } | null;
}) {
  const trainingPts: Point[] = training
    .filter((p) => p.loss !== undefined)
    .map((p, i) => ({ x: p.step ?? i, loss: p.loss! }))
    .sort((a, b) => a.x - b.x);
  const heldOutPts: Point[] = heldOut
    .filter((p) => p.heldOutLoss !== undefined)
    .map((p, i) => ({ x: p.epoch ?? i, loss: p.heldOutLoss! }))
    .sort((a, b) => a.x - b.x);

  if (trainingPts.length === 0 && heldOutPts.length === 0) {
    return (
      <div className="rounded-[12px] border bg-card p-6 text-sm text-muted-foreground">
        No loss measured yet. It appears here as the run goes.
      </div>
    );
  }

  // The checkpoint this run kept: matched onto each series by the value that
  // series actually carries -- the training curve by the step it names, the
  // held-out curve by the loss it was chosen from (issue #62's record,
  // cross-read rather than re-derived).
  const bestTrainX =
    best?.step != null && trainingPts.some((p) => p.x === best.step)
      ? best.step
      : undefined;
  const bestHeldOutPt =
    best?.held_out_loss != null
      ? heldOutPts.find((p) => p.loss === best.held_out_loss)
      : undefined;

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <SeriesChart
        title="Training loss"
        unit="step"
        points={trainingPts}
        color="var(--color-primary)"
        areaFill
        bestX={bestTrainX}
        bestLabel={bestTrainX !== undefined ? `step ${bestTrainX}` : undefined}
      />
      <SeriesChart
        title="Held-out loss"
        unit="epoch"
        points={heldOutPts}
        color="var(--color-muted-foreground)"
        dashed
        bestX={bestHeldOutPt?.x}
        bestLabel={bestHeldOutPt ? `epoch ${bestHeldOutPt.x}` : undefined}
      />
    </div>
  );
}
