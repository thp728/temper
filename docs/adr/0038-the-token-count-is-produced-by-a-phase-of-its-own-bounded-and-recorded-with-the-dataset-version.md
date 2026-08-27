# ADR-0038 — The token count is produced by a phase of its own, bounded, and recorded with the dataset version

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/006-streaming-transport-progress-and-artifacts.md`
- **Issue:** [#42](https://github.com/thp728/temper/issues/42)

## Context

Cost is quoted per training token, so a dataset's token count is a hard
prerequisite of the quote -- and tokenising is the expensive half of
validation. The ticket's fork was explicit: count in the same pass as
validation, *unless* measurement showed tokenising dominates, in which case
counting runs as its own phase with its own state rather than blocking the
report. Which branch the measurement puts the product on decides where the
count is produced, so it was measured before anything was written.

Spike 9 had already measured a floor of **3.4 MB/s** for a validate-plus-
tokenise pass against **21.6 MB/s** for validation alone (a ~6x slowdown),
and its sensitivity analysis showed the decision did not depend on the
contended figure: tokenising would have to be an order of magnitude faster to
fit the tolerable wait. That measurement predated the streaming validator's
rewrite (#31), so it was re-measured against the shipped validator at a size
one run can afford. The re-measurement came back the same way: **1.8 MB/s
same-pass against 12.4 MB/s validate alone** (a ~7x slowdown), and at the
1.3 GB upload ceiling the same pass would take **~720 s against the 60 s
tolerable wait**. The measurement lands the ticket on **SPLIT**.

The second constraint was the memory guarantee #31 just landed: validation
streams one row at a time and holds nothing that grows with the file, and
that guarantee is guarded by tests. Token counting is the easiest possible way
to reintroduce materialisation -- holding every row's token ids, or a list of
per-row counts, would grow with the file and break the flat-memory claim. The
acceptance criterion asks for *the distribution across rows*, which must
therefore have a bounded representation.

## Decision

**Token counting runs as its own phase with its own state, after validation,
and its result is recorded with the dataset version.**

* **The report is never blocked.** Validation finishes and the report lands
  exactly as it did before #42. The count arrives afterwards, on the same
  dataset record, under a state of the counting phase's own: `counting` |
  `done` | `failed`. A count that cannot be produced (the tokenizer could not
  be downloaded, the row vanished) records `failed` and the dataset stays
  valid and launchable -- an absent count renders absent, the estimate posture
  ADR-0031 already documented for the nullable field.
* **The distribution is bounded.** Per-row token counts go into a histogram
  over a fixed set of edges (`temper_core.counting.HISTOGRAM_EDGES`), never
  one entry per row, so nothing retained scales with the file. The histogram's
  2048 edge is the trainer's default sequence length on purpose. The exact
  number of rows that would be truncated at that length is counted during the
  pass, not derived afterwards.
* **The count is recorded with the version, and the quote reads it without
  recomputation.** `temper_core.quote.Quote.token_count` was already a
  nullable field (ADR-0031) and `apps/control-plane/quote.py` already read
  `report.token_count`; this work fills that seam. The count and distribution
  are stored on the dataset row and merged into the published report, so the
  quote never re-tokenises.
* **The tokenizer is a seam, not a core dependency.** `temper_core` takes a
  `dict -> int` `count_row` callable and stays dependency-free; the control
  plane owns the real tokenizer (the default model's, pinned), loaded once and
  cached, beside the other things that touch Hugging Face. The deterministic
  fake (one token per character) replaces it in tests and in `FAKE_PROVIDER`
  journeys, so no test or journey needs the network.

## Why the measurement branch was taken

The re-measurement at `.scratch42/findings-measure42.json` (a 200 MB
synthetic dataset, the shipped validator, Qwen3-4B's tokenizer):

| pass | MB/s | 1.3 GB ceiling time |
| --- | --- | --- |
| validate (shipped #31 rewrite) | 12.4 | 105 s |
| validate + tokenise (same pass) | 1.8 | 722 s |
| count only (split phase) | 2.0 | 650 s |

Same-pass tokenising is ~7x slower than validation alone, and would push the
largest upload the product accepts to twelve minutes of synchronous wait.
Even at a generous 4x faster machine the same pass stays ~3x over the budget,
so the split recommendation does not depend on the contended figure -- the
same robustness argument spike 9 made. **The measurement put the ticket on
SPLIT, and counting blocks nothing.**

## Alternatives considered

**Count in the same validation pass.** Rejected by the measurement: ~7x
slower, ~12 minutes at the ceiling against a 60 s budget. It would also
couple the report to the tokenizer's availability -- a tokenizer that fails
to load would block the report validation finished producing.

**Hold the distribution as one entry per row.** Rejected: it is the exact
materialisation the flat-memory guarantee exists to forbid, and the
`retained_objects`-style test would fail on it. A histogram over fixed edges
carries the same "distribution across rows" information in constant memory.

**Recompute the count at quote time.** Rejected: tokenising is the expensive
half, and redoing it per quote (per model, per plan-screen render) is the cost
the whole ticket exists to avoid. The count is a property of the dataset
under a named tokenizer and sequence length, produced once and read.

**Make the count a blocking part of the report surface.** Rejected: an
invalid dataset cannot launch, so counting it wastes the expensive pass; a
dataset that is valid but not yet counted must be launchable immediately
(the quote renders the count absent). Blocking the report or the launch on
the count would make an estimate gate a run, which spec 005 refuses.

## Consequences

- `temper_core.counting` provides the streaming, bounded counting pass and
  its flat-memory tests; the control plane runs it as a background phase and
  records the result with the dataset version.
- `DatasetReport` carries `token_count` and `token_distribution`;
  `DatasetRecord` carries the counting phase's `token_count_status` and
  `counting_progress`, so the report page can show progress, the result, or
  the failure -- and the quote's existing `token_count` seam is filled.
- The upload ceiling (ADR-0036) is unchanged: it was derived from the
  *validate* throughput, which is what the synchronous upload waits on. The
  counting phase runs after, asynchronously, and does not extend the wait.
- Tests that upload a valid dataset spawn a counting thread; the suite joins
  it at teardown so the phase never outlives its test's database.

## Rollback

Restore the previous validator and quote: revert `temper_core/counting.py`,
the token-count fields on the contract and `db`, and the counting phase in
`datasets.py`. The report and quote render the count absent exactly as they
did before this ticket -- the nullable field was the documented state.
