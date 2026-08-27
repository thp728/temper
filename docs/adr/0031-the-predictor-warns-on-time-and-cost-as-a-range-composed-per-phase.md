# ADR-0031 — The predictor warns on time and cost, as a range, composed per phase

- **Status:** accepted
- **Date:** 2026-08-27
- **Spec:** `docs/specs/005-predictor-and-quote.md`
- **Issue:** [#72](https://github.com/thp728/temper/issues/72)

## Context

Nothing in the product computed what a job would cost before it ran. Cost was
derived afterwards, from the event log against a stored hourly price, and the
README said so plainly. Duration was the one prediction the product made, and
it was a single point from a single measured run, applied to everything. The
commercial baseline showed neither figure up front either, which made the
upfront quote the clearest differentiator available -- and the hidden hosting
bill the loudest complaint against that baseline.

The blockers this record stands on are all measured and recorded: `memory`
predicts peak VRAM from a model-facts seam (ADR-0028), `selection` picks the
cheapest fitting hardware from live availability and price (ADR-0029), `disk`
sizes the machine's storage from model facts with storage billed separately in
USD (ADR-0030), and spikes 5 and 9 supplied the download rate, the token-count
cost, and the disk ceiling. What was left was the half of spec 005 that turns
those into numbers a user sees before committing: **time and cost, as a range,
composed per phase, in the account's own currency**.

## Decision

**`temper_core.quote` computes the quote: pure arithmetic, no I/O, no clock.**
It takes model facts, hyperparameters, a selected hardware price and currency,
the disk's storage line, the dataset's row and token counts, and the reference
the quote was computed against (`dataset_id`/`dataset_created_at`,
`base_revision`), and returns a `Quote`: a duration and cost range per phase,
totals, and an expiry. It takes the wall clock and the price as arguments, so
the arithmetic is testable without a network, a GPU, or a clock -- the same
discipline ADR-0028/0029/0030 established and `packages/core`'s zero-I/O rule
requires.

**Time and cost warn; they never block.** This is ADR-0007's posture inherited
exactly: the quote is labelled an estimate wherever it appears, and a launch is
never refused because a quote is missing. A configuration that cannot be priced
-- provider unreachable, nothing free, disk over the ceiling, a model that will
not resolve -- produces **no quote**, not an error, on both the plan screen and
the launch path. Memory is the half that blocks, and it already does, in
`selection` (ADR-0029).

**Every duration is a range, never a point.** The throughput figures behind the
quote are the softest numbers in the model -- one measured download (364 MB/s
mean, spike 5) and one measured training run (1.19 row-passes/s, 2026-08-19) --
so every phase carries a low and a high, the low is never the high, and the
range is what is shown. The training phase is widened to a full order of
magnitude (`sqrt(10)` each way) because the spec's risk note states the
published range for the underlying efficiency figure spans nearly an order of
magnitude across workloads -- a DERIVED spread, labelled as such, that
calibration (#77) will tighten as real runs accumulate.

**Cost is composed per phase, not as one blended rate.** Provisioning,
readiness, image pull, model download, training and teardown have different
durations, and the first four are largely independent of the dataset. The
download phase uses the measured rate, so a large model's cold start is visible
as a phase of its own rather than buried -- the difference between a rounding
error and half the bill that a single blended rate exists to hide.

**Currency travels with the amount, in the smallest unit.** The account bills
in INR, read live from the provider on every quote (spike 5); costs are
integers in the currency's smallest unit (paisa for INR, cent for USD) with the
currency code alongside -- never a formatting choice applied after the fact.
An unknown currency is refused, not assumed, because misreporting every figure
by a made-up factor is worse than asking. Storage remains a separate USD line,
unconverted, per ADR-0030: no live per-account storage price exists to convert
it against, and a labelled USD figure is more honest than a silently assumed
conversion.

**The quote is persisted, immutable, and expires.** It pins the dataset version
(the immutable stored object, by `dataset_id` and `dataset_created_at`) and the
model revision it was computed against, so it cannot outlive its inputs. It
carries `expires_at` (default 24h, configurable), because "a price I was shown
yesterday is not silently honoured against hardware that has changed" is the
reason it expires at all. **Launching copies the quote into the job spec** --
`jobs.quote_json`, frozen at creation beside the hyperparameters and warnings,
written once and never updated -- so a completed job can still say what it was
predicted to cost and how long it was predicted to take, which is what the
predicted-versus-actual recording (#77) compares against.

**The plan screen shows the quote per model.** The preview endpoint computes a
quote for every catalog model, keyed by model id, because which model is chosen
changes what the job costs (a bigger model downloads more and may need
different hardware). The launch screen shows the selected model's quote, and
the finished-job page shows the frozen one. The seams are the existing ones --
the `models` seam and a provider seam that tests replace with the same
`FakeProvider` the rest of the suite uses -- so the whole quote path is
exercised with no network and no GPU.

## Alternatives considered

**Reuse `feasibility`'s single-point estimate and multiply it by a spread.**
Rejected: `feasibility` is a warning mechanism answering one question ("could
this dataset plainly not finish?"), and its single point exists to be compared
against a ceiling. The quote is a different surface answering a different
question ("what will this cost and how long?"), needs the per-phase breakdown
the warning never had, and must never show a point. They share the measured
training throughput (`feasibility.ROWS_PER_SECOND`, read not retyped) and
nothing else.

**Fold storage into the account-currency cost at a fixed exchange rate.**
Rejected for exactly the reason ADR-0030 rejected it: an invented exchange rate
would look measured and is not. The quote carries the documented USD figure,
labelled, and the account-currency figure for everything the account actually
bills in.

**Quote a single blended rate.**
Rejected: this is the exact number a large model's user most wants, and hiding
it is the complaint the ticket exists to answer. Per-phase composition is the
point of the feature, not an implementation detail.

**Block a launch when the quote cannot be computed.**
Rejected: ADR-0007's posture is inherited deliberately. A wrong block on an
estimate is worse than a wrong warning, and the memory half already blocks
through selection.

## Consequences

- `temper_core.quote` is new: pure arithmetic, `PHASES`, measured constants
  with their citations, `Quote`/`PhaseEstimate`, and `estimate(...)`.
- `apps/control-plane/quote.py` assembles the quote from the seams; `main.py`
  and `web.py`'s create path freeze it onto the job via `jobs.create(quote=...)`.
- `jobs` rows gain `quote_json`, frozen at creation and never updated;
  `JobRecord` publishes it; `JobSpecPreview` carries `quotes` per model.
- The plan screen shows the quote (duration range + per-phase cost + token
  count + currency + expiry + references), and the finished-job page shows the
  frozen quote.
- A quote that cannot be priced renders absent, never as an error, and a
  launch proceeds without one.
- The token count is carried and shown when the dataset has one. Producing it
  is Spec 006's job (#42); until then the field is null and the training
  phase is anchored to the measured row-pass throughput, which is the honest
  number available.
- Recording predicted-versus-actual is a separate ticket (#77), now unblocked:
  the frozen quote on the job row is exactly what it needs to compare against.
