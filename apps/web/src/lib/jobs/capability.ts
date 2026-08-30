// Presentation over the published general-capability slice (issue #73).
//
// This is a smoke test for catastrophic forgetting, not a benchmark: a fixed,
// versioned slice of general questions answered by the base model and the
// tuned model, reported as a delta with the sample size beside the number and
// the uncertainty stated. The numbers -- the scores, the change, the
// per-question weight, the standard error -- are read from the record, never
// recomputed or invented here; the one definition of the slice and its
// threshold lives in the trainer's `capability.py` and travels recorded on the
// result, so the machine and the interface cannot drift about what a number
// means.
//
// No component knowledge, no React: functions over the published shape.

import type { BestCheckpoint, Capability } from "@/lib/api/generated/client";

// "6 of 8" -- the fraction, with the sample size literally beside the number
// (Spec 011: the interface states the sample size beside the number).
export function fractionText(
  correct: number | null | undefined,
  total: number | null | undefined,
): string {
  if (!total) return "—";
  return `${correct ?? 0} of ${total}`;
}

// "75%" -- the score as a whole-number percentage.
export function percentText(
  correct: number | null | undefined,
  total: number | null | undefined,
): string {
  if (!total) return "";
  return `${Math.round(((correct ?? 0) / total) * 100)}%`;
}

// "6 of 8 (75%)" -- the number a reader sees, sample size beside it.
export function scoreText(
  correct: number | null | undefined,
  total: number | null | undefined,
): string {
  if (!total) return "—";
  const pct = percentText(correct, total);
  return pct ? `${fractionText(correct, total)} (${pct})` : fractionText(correct, total);
}

// The change, tuned minus base, as "−2 of 8 (−25%)". Negative is a drop --
// the smoke test's signal. Read from the recorded correct counts, with the
// recorded delta (a fraction) as the percentage.
export function changeText(capability: Capability): string {
  const total = capability.total ?? 0;
  if (!total) return "—";
  const diff = (capability.tuned_correct ?? 0) - (capability.base_correct ?? 0);
  const pct = Math.round((capability.delta ?? 0) * 100);
  const sign = diff > 0 ? "+" : diff < 0 ? "−" : "";
  return `${sign}${Math.abs(diff)} of ${total} (${pct >= 0 ? "+" : "−"}${Math.abs(pct)}%)`;
}

// "On 8 questions, each is worth 12.5% of the score." -- the weight that makes
// a small slice legible: with so few questions, one question is a large chunk.
export function weightSentence(total: number | null | undefined): string {
  if (!total) return "";
  const pct = 100 / total;
  const shown = Number.isInteger(pct) ? `${pct}%` : `${pct.toFixed(1)}%`;
  return `On ${total} question${total === 1 ? "" : "s"}, each is worth ${shown} of the score.`;
}

// "The standard error of the change is ±1.2 questions (±15%)." -- the stated
// uncertainty, in the units a reader already holds (questions and percent).
// The standard error is recorded as a fraction of the score (delta_se); the
// interface converts it, it never decides it.
export function uncertaintySentence(capability: Capability): string {
  const total = capability.total ?? 0;
  const se = capability.delta_se ?? 0;
  if (!total) return "";
  const questions = (se * total).toFixed(1);
  const pct = Math.round(se * 100);
  return `The standard error of the change is ±${questions} question${
    questions === "1.0" ? "" : "s"
  } (±${pct}%).`;
}

// "Tuned model (chosen checkpoint, step 20)" -- which model the tuned side
// actually was, read from the recorded selection (issue #62's rule).
export function tunedModelLabel(
  selection: BestCheckpoint | null | undefined,
): string {
  return selection?.step != null
    ? `Tuned model (chosen checkpoint, step ${selection.step})`
    : "Tuned model";
}
