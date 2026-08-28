// Presentation over the published side-by-side comparison (issue #69): the
// decoding settings and the shape of a row, as the finished record shows
// them. The settings are read from the record, never defined here -- the one
// definition lives in the trainer's `comparison.py` and travels recorded on
// the result, so the generator and the interface cannot drift about what a
// comparison was made under.
//
// No component knowledge, no React: functions over the published shape.

import type { ComparisonRow } from "@/lib/api/generated/client";

// The conversation up to the last user turn is what was actually generated
// from (the held-out answer is never fed to either model); for display, the
// user's question is the prompt a reader recognises.
export function promptText(
  prompt: ComparisonRow["prompt"] | null | undefined,
): string {
  if (!prompt) return "";
  for (let i = prompt.length - 1; i >= 0; i--) {
    const turn = prompt[i];
    if (turn?.role === "user") return turn.content ?? "";
  }
  return "";
}

// The decoding settings as a sentence a non-specialist can read, so a reader
// can tell whether two outputs are comparable. Built from the recorded keys
// only -- an unknown key is ignored, never guessed.
export function decodingSentence(
  decoding: Record<string, unknown> | null | undefined,
): string {
  if (!decoding) return "";
  const parts: string[] = [];
  if (typeof decoding.temperature === "number") {
    parts.push(`temperature ${decoding.temperature}`);
  }
  if (typeof decoding.max_new_tokens === "number") {
    parts.push(`up to ${decoding.max_new_tokens} new tokens`);
  }
  if (decoding.do_sample === true) {
    parts.push("sampling on");
  }
  if (parts.length === 0) return "";
  return `Both models answered with ${parts.join(", ")}.`;
}
