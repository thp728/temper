# ADR-0032 — Every calculated default carries its reason

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/005-predictor-and-quote.md`
- **Issue:** [#76](https://github.com/thp728/temper/issues/76)

## Context

The predictor (ADR-0028, 0029, 0030, 0031) decides almost everything about a
job on the user's behalf: the method, the card, how many of them, the disk,
the training precision and the sequence length. Until now it returned the
decisions but not the reasoning -- a quote said *what* but not *why*, and a
user who disagreed with a line could not tell whether it was a decision made
from their job's own numbers or a default nobody had looked at. A product that
silently picks an L4 is not obviously better than one that asks; one that says
which card, because of which constraint, and what the alternatives would have
cost has taught the user something and earned the right to decide for them
(spec 005's thesis, and this record's whole reason to exist).

The raw material already existed: `selection` already computes whether each
candidate configuration fits, `memory` already produces the per-pool peak for
any method and sequence length, `disk` already breaks the requirement into its
pools, and the measured anchors were already recorded. What was missing was a
shape that carried the reasoning beside the answer, and the discipline that a
reason, once shown, survives the job.

## Decision

**Every decision the predictor makes is returned as a structured record** --
the decision, the value chosen, the constraint that forced it, and the
alternatives with what each would have cost. This is deliberately the same
shape as this project's own ADRs, turned into a product surface, and it is
structured rather than rendered so the API, the interface and the artifact
manifest all read from one source.

**Six decisions each carry one record**, in the order the acceptance criteria
name them: method, hardware, device count, disk, precision and sequence length.
All six are computed from the same inputs the quote is built from -- model
facts, the effective hyperparameters, the chosen hardware plan and the disk
plan -- so a reason can never drift from the configuration it explains.

**`selection` returns the fitting configurations it rejected.** The search is
where alternatives are discovered: every configuration that passed the same
fit check and lost (higher price, or a tied price with more devices) rides back
on `HardwarePlan.alternatives`. The method, precision and sequence-length
alternatives cost in memory rather than price, so those are computed in
`temper_core.decisions` from the model's facts -- the same `memory` arithmetic
that decided the fit.

**The reasons ride on the quote, and persist with it.** The records are a
field of the quote, frozen into the job spec at launch exactly like the rest
of it and never updated, so a completed job explains itself as completely as a
planned one. They are never regenerated on read.

**The interface shows the reason by default and the alternatives one
interaction away.** The plan renders each decision's chosen value and its
constraint directly; the alternatives sit behind a disclosure, collapsed --
never hidden, never always shown, because the page serves a first-time user
who reads it as an explanation and an expert who reads it to find what to
override.

## Alternatives considered

**Render the reasons as prose, regenerated from a template on read.**
Rejected for the same reason the frozen job spec exists: an explanation
rebuilt from current numbers can drift from the configuration it claims to
explain after a default changes, and it cannot describe a completed job whose
inputs are gone. The reason must be a record that survives with the quote, not
a sentence recomputed to fit.

**Show the alternatives always, inline.**
Rejected: the plan is one surface for a first-time user and an expert, not two
modes. Always-shown alternatives bury the reasons under the rejections;
hidden ones make the plan feel like a black box. A disclosure is the middle
that satisfies both -- "alternatives are one interaction away rather than
hidden or always shown" is the acceptance criterion itself.

**Let the quote re-derive the hardware alternatives from availability.**
Rejected: the fit check and the selection rule live in `selection`, and a
second copy of that loop in the quote path would be exactly the kind of
drift this repo exists to prevent. The search returns what it considered; the
quote arranges it, it does not recompute it.

## Consequences

- `temper_core.decisions` is new: `Decision`, `DecisionAlternative`, `decide`
  (the six records) and `to_dict` (one serialization).
- `selection.HardwarePlan` gains `alternatives` -- the fitting configurations
  it rejected -- and the search records every candidate rather than only the
  winner.
- `temper_core.quote.Quote` carries `decisions`; the contract and API publish
  `QuoteDecision`/`QuoteDecisionAlternative`, so the web client is generated
  for them.
- The plan screen and the finished-job page both render the reasons; the
  finished-job page reads them from the frozen quote, which is the same data
  the plan showed.
- An alternative-less decision (nothing else fit) is still a record: an honest
  "there was no other option" is itself a reason.
- The recorded costs keep spec 005's measured-versus-derived honesty: memory
  figures are `memory`'s arithmetic over model facts, prices are the
  provider's live hourly rate in the account's currency, and both are labelled
  where they are not exact.
