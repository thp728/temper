// The prediction-vs-measurement comparison (issue #77), as presentation over
// the published quote and actuals.
//
// The numbers -- ratio = actual / predicted midpoint, and direction = where
// the actual lands on the predicted range -- mirror `temper_core.calibration`
// in packages/core. Each half computes them for its own surface (the finished
// job page here, the aggregate view there), and both are tested against the
// same expectation, so a drift would fail a test rather than silently change
// what a reader is told.
//
// No component knowledge, no React: functions over the published shapes.

export type Direction = "under" | "inside" | "over";

// The middle of a predicted range, or null when either end is missing. The
// quote predicts a range, never a point (spec 005: the throughput figures are
// the softest numbers in the model); the midpoint is the single number a
// ratio against the actual needs.
export function midpoint(
  low: number | null | undefined,
  high: number | null | undefined,
): number | null {
  if (low == null || high == null) return null;
  return (low + high) / 2;
}

// Actual over predicted: >1 means under-predicted, <1 over-predicted. Null
// when either side is missing, so a comparison never invents a number.
export function ratio(
  actual: number | null | undefined,
  predicted: number | null | undefined,
): number | null {
  if (actual == null || predicted == null || predicted === 0) return null;
  return actual / predicted;
}

// Where the actual landed relative to the predicted range. "over" and "under"
// are the systematic-error signals: an estimate that always lands "over" is
// wrong in a particular, fixable way.
export function rangeDirection(
  actual: number | null | undefined,
  low: number | null | undefined,
  high: number | null | undefined,
): Direction | null {
  if (actual == null || low == null || high == null) return null;
  if (actual < low) return "under";
  if (actual > high) return "over";
  return "inside";
}

export function formatGigabytes(gb: number | null | undefined): string {
  if (gb == null) return "—";
  return `${gb.toFixed(2)} GB`;
}

// A ratio as "1.24×": the number a reader compares against 1, never a bare
// decimal that reads like a probability.
export function formatRatio(r: number | null | undefined): string {
  if (r == null) return "—";
  return `${r.toFixed(2)}×`;
}

export function directionLabel(d: Direction | null | undefined): string {
  switch (d) {
    case "under":
      return "came in under the predicted range";
    case "inside":
      return "landed inside the predicted range";
    case "over":
      return "overran the predicted range";
    default:
      return "could not be compared";
  }
}
