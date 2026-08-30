import { describe, expect, it } from "vitest";
import {
  changeText,
  fractionText,
  percentText,
  scoreText,
  tunedModelLabel,
  uncertaintySentence,
  weightSentence,
} from "@/lib/jobs/capability";

// The capability slice's presentation (issue #73): a smoke test, not a
// benchmark -- the sample size beside the number, the change read from the
// record, and the uncertainty stated in the units a reader already holds.
// Everything reads the published record; nothing re-decides the threshold.

const capability = (over: Record<string, unknown> = {}) => ({
  ok: true,
  version: 1,
  total: 8,
  base_correct: 6,
  tuned_correct: 4,
  base_score: 0.75,
  tuned_score: 0.5,
  delta: -0.25,
  delta_se: 0.152,
  large_regression: true,
  decoding: { temperature: 0.7, max_new_tokens: 128, do_sample: true },
  rows: [],
  ...over,
});

describe("capability presentation", () => {
  it("shows the sample size beside the number", () => {
    expect(scoreText(6, 8)).toBe("6 of 8 (75%)");
    expect(scoreText(4, 8)).toBe("4 of 8 (50%)");
    expect(scoreText(0, 8)).toBe("0 of 8 (0%)");
    expect(scoreText(6, 0)).toBe("—");
    expect(scoreText(null, null)).toBe("—");
  });

  it("reads the change from the record, tuned minus base", () => {
    expect(changeText(capability())).toBe("−2 of 8 (−25%)");
    // A tuned model that improves is a positive change.
    expect(
      changeText(capability({ base_correct: 4, tuned_correct: 6, delta: 0.25 })),
    ).toBe("+2 of 8 (+25%)");
    // No change at all.
    expect(
      changeText(capability({ base_correct: 6, tuned_correct: 6, delta: 0 })),
    ).toBe("0 of 8 (+0%)");
    expect(changeText(capability({ total: 0 }))).toBe("—");
  });

  it("states each question's weight in plain language", () => {
    expect(weightSentence(8)).toBe("On 8 questions, each is worth 12.5% of the score.");
    expect(weightSentence(4)).toBe("On 4 questions, each is worth 25% of the score.");
    expect(weightSentence(null)).toBe("");
  });

  it("states the uncertainty in questions and percent", () => {
    expect(uncertaintySentence(capability())).toBe(
      "The standard error of the change is ±1.2 questions (±15%).",
    );
    // No uncertainty (both sides identical) states zero rather than hiding.
    expect(
      uncertaintySentence(capability({ delta_se: 0 })),
    ).toBe("The standard error of the change is ±0.0 questions (±0%).");
    expect(uncertaintySentence(capability({ total: 0 }))).toBe("");
  });

  it("names which checkpoint the tuned side answered through", () => {
    expect(
      tunedModelLabel({ step: 20, basis: "best_held_out_loss", reason: "lowest" }),
    ).toBe("Tuned model (chosen checkpoint, step 20)");
    expect(tunedModelLabel(null)).toBe("Tuned model");
  });

  it("keeps the raw fraction and percent helpers honest", () => {
    expect(fractionText(6, 8)).toBe("6 of 8");
    expect(percentText(6, 8)).toBe("75%");
    expect(percentText(6, 0)).toBe("");
  });
});
