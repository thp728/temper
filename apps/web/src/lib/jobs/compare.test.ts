import { describe, expect, it } from "vitest";
import { decodingSentence, promptText } from "@/lib/jobs/compare";

// The comparison's presentation (issue #69): both sides per prompt, the
// decoding settings read from the record (never defined here -- the one
// definition lives in the trainer), and the honest absence when a comparison
// failed. The settings sentence is what tells a reader whether two outputs
// are comparable, which is Spec 011's "recorded means stored and shown".

describe("comparison presentation", () => {
  it("reads the user's question from the recorded prompt", () => {
    expect(
      promptText([{ role: "user", content: "q?" }]),
    ).toBe("q?");
    // The last user turn wins, whatever came before it.
    expect(
      promptText([
        { role: "system", content: "be terse" },
        { role: "user", content: "q1" },
      ]),
    ).toBe("q1");
    expect(promptText([])).toBe("");
    expect(promptText(null)).toBe("");
  });

  it("turns the recorded decoding settings into a readable sentence", () => {
    expect(
      decodingSentence({ temperature: 0.7, max_new_tokens: 128, do_sample: true }),
    ).toBe(
      "Both models answered with temperature 0.7, up to 128 new tokens, sampling on.",
    );
    // An unknown setting is ignored, never guessed.
    expect(decodingSentence({ temperature: 0.7 })).toBe(
      "Both models answered with temperature 0.7.",
    );
    expect(decodingSentence(null)).toBe("");
  });
});
