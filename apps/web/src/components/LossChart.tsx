"use client";

import type { MetricPoint } from "@/lib/jobs/loss";

// The one chart the running-job view carries (issue #53): training loss and
// held-out loss on the same axes, so a user can watch them diverge. No chart
// library -- two polylines over a shared domain is the whole need, and a
// dependency bought for one screen is a dependency maintained forever.
//
// The shared x-axis is the order the measurements were recorded in (the index
// in the combined metric stream), because the held-out rows do not carry a
// step number. The y-axis spans the losses actually seen, padded so a run
// that starts at 1.9 and ends at 0.5 is still readable while it is early.

const WIDTH = 640;
const HEIGHT = 220;
const PAD = { top: 12, right: 14, bottom: 26, left: 46 };

const plotWidth = WIDTH - PAD.left - PAD.right;
const plotHeight = HEIGHT - PAD.top - PAD.bottom;

function value(p: MetricPoint): number | undefined {
  return p.loss ?? p.heldOutLoss;
}

function measured(points: MetricPoint[]): number[] {
  return points
    .map(value)
    .filter((v): v is number => typeof v === "number");
}

function polyline(
  points: MetricPoint[],
  xScale: (x: number) => number,
  yScale: (y: number) => number,
): string | null {
  const pts = points.filter((p) => value(p) !== undefined);
  if (pts.length < 2) return null;
  return pts.map((p) => `${xScale(p.x)},${yScale(value(p)!)}`).join(" ");
}

export default function LossChart({
  training,
  heldOut,
}: {
  training: MetricPoint[];
  heldOut: MetricPoint[];
}) {
  const all = measured([...training, ...heldOut]);
  if (all.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-4 text-sm text-muted-foreground">
        No loss measured yet. It appears here as the run goes.
      </div>
    );
  }

  const minX = 0;
  const maxX = Math.max(0, ...[...training, ...heldOut].map((p) => p.x));
  const xScale = (x: number) =>
    PAD.left + (maxX === minX ? 0 : ((x - minX) / (maxX - minX)) * plotWidth);

  let minY = Math.min(...all);
  let maxY = Math.max(...all);
  const padY = Math.max((maxY - minY) * 0.1, 0.05);
  minY -= padY;
  maxY += padY;
  const yScale = (y: number) =>
    PAD.top + (1 - (y - minY) / (maxY - minY)) * plotHeight;

  const trainingLine = polyline(training, xScale, yScale);
  const heldOutLine = polyline(heldOut, xScale, yScale);

  return (
    <figure className="rounded-lg border bg-card p-4">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full"
        role="img"
        aria-label="Chart of training loss and held-out loss over the course of the run"
      >
        {trainingLine && (
          <polyline
            points={trainingLine}
            fill="none"
            stroke="var(--color-sky-600)"
            strokeWidth={2}
          />
        )}
        {heldOutLine && (
          <polyline
            points={heldOutLine}
            fill="none"
            stroke="var(--color-rose-600)"
            strokeWidth={2}
            strokeDasharray="4 3"
          />
        )}
        {/* Lone points are still visible before a series has two of them. */}
        {training
          .filter((p) => value(p) !== undefined && training.length < 2)
          .map((p) => (
            <circle
              key={`t${p.x}`}
              cx={xScale(p.x)}
              cy={yScale(value(p)!)}
              r={3}
              fill="var(--color-sky-600)"
            />
          ))}
        {heldOut
          .filter((p) => value(p) !== undefined && heldOut.length < 2)
          .map((p) => (
            <circle
              key={`h${p.x}`}
              cx={xScale(p.x)}
              cy={yScale(value(p)!)}
              r={3}
              fill="var(--color-rose-600)"
            />
          ))}
        <line
          x1={PAD.left}
          y1={HEIGHT - PAD.bottom}
          x2={WIDTH - PAD.right}
          y2={HEIGHT - PAD.bottom}
          stroke="var(--color-border)"
        />
        <text
          x={(PAD.left + WIDTH - PAD.right) / 2}
          y={HEIGHT - 8}
          textAnchor="middle"
          className="fill-muted-foreground text-[10px]"
        >
          measurement, in the order it was recorded
        </text>
      </svg>
      <figcaption className="mt-2 flex gap-4 text-sm">
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden
            className="inline-block h-0.5 w-4 rounded"
            style={{ backgroundColor: "var(--color-sky-600)" }}
          />
          Training loss
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden
            className="inline-block h-0.5 w-4 rounded"
            style={{
              backgroundColor: "var(--color-rose-600)",
              borderTop: "2px dashed var(--color-rose-600)",
              backgroundClip: "content-box",
            }}
          />
          Held-out loss
        </span>
      </figcaption>
    </figure>
  );
}
