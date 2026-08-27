import { describe, expect, it } from "vitest";
import { PLATEAU_PATIENCE, heldOutPlateau } from "@/lib/jobs/plateau";

// The plateau is a patience rule over the held-out series, and the wording is
// the deliverable: a non-specialist has to be able to read "the model is no
// longer improving on data it has not seen" and know what to do with it.

describe("heldOutPlateau", () => {
  it("says nothing until there is enough history", () => {
    expect(heldOutPlateau([])).toBeNull();
    expect(heldOutPlateau([1.0])).toBeNull();
    // Patience 2 needs at least three points before it can mean anything.
    expect(heldOutPlateau([1.0, 1.0])).toBeNull();
  });

  it("reports a plateau when the last readings stop improving", () => {
    const note = heldOutPlateau([1.0, 1.0, 1.0]);
    expect(note).not.toBeNull();
    expect(note!.bestHeldOutLoss).toBe(1.0);
    expect(note!.evaluationsSinceImprovement).toBe(2);
    expect(note!.message).toContain("overfitting");
    expect(note!.message).toContain("1.0000");
  });

  it("reports a plateau when a rise after an early improvement persists", () => {
    const note = heldOutPlateau([1.0, 0.5, 0.9, 0.9]);
    expect(note).not.toBeNull();
    expect(note!.bestHeldOutLoss).toBe(0.5);
    expect(note!.evaluationsSinceImprovement).toBe(2);
  });

  it("says nothing while the held-out loss is still improving", () => {
    expect(heldOutPlateau([1.0, 0.5, 0.3])).toBeNull();
  });

  it("treats an equal reading as not improving", () => {
    // A flat run is exactly the plateau the signal exists to catch.
    expect(heldOutPlateau([1.0, 1.0, 1.0, 1.0])).not.toBeNull();
  });

  it("forgives one worse reading when the next one improves", () => {
    expect(heldOutPlateau([1.0, 0.5, 0.9, 0.4])).toBeNull();
  });

  it("respects a caller-chosen patience", () => {
    // A single worse reading is enough at patience 1.
    expect(heldOutPlateau([1.0, 1.0], 1)).not.toBeNull();
    expect(PLATEAU_PATIENCE).toBe(2);
  });
});
