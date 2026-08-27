// A held-out loss that stops improving, said in language a non-specialist can
// read (issue #53). The rule is a patience counter over the held-out series: a
// loss that has not beaten the best seen so far for N consecutive evaluations
// is a plateau, and that is the first visible sign of overfitting. The wording
// is presentation -- the surface the reader is looking at -- so it lives with
// the view, as a pure function a test can pin.

export interface PlateauNote {
  message: string;
  bestHeldOutLoss: number;
  evaluationsSinceImprovement: number;
  totalEvaluations: number;
}

// How many consecutive evaluations without improvement count as a plateau.
// Two: with fewer than three evaluation points there is not enough history to
// say a run has stopped improving, and a single worse reading after two good
// ones is noise, not a signal.
export const PLATEAU_PATIENCE = 2;

export function heldOutPlateau(
  heldOutLosses: number[],
  patience: number = PLATEAU_PATIENCE,
): PlateauNote | null {
  if (heldOutLosses.length < patience + 1) return null;
  let best = Infinity;
  let sinceImprovement = 0;
  for (const loss of heldOutLosses) {
    if (loss < best) {
      // A strictly lower loss is an improvement; an equal one is not.
      best = loss;
      sinceImprovement = 0;
    } else {
      sinceImprovement += 1;
    }
  }
  if (sinceImprovement < patience) return null;
  return {
    bestHeldOutLoss: best,
    evaluationsSinceImprovement: sinceImprovement,
    totalEvaluations: heldOutLosses.length,
    message: `Held-out loss has not improved for the last ${sinceImprovement} evaluations. This is often the first sign of overfitting: the model keeps getting better on the training data it has seen, but no longer on examples it has not. The best held-out loss so far is ${best.toFixed(4)}.`,
  };
}
