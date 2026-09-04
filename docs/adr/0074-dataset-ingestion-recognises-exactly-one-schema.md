# ADR-0074 — Dataset ingestion recognises exactly one schema

- **Status:** accepted
- **Date:** 2026-09-04
- **Supersedes:** nothing
- **Spec:** none — no spec ever scoped dataset format support; see Context.
- **Issue:** none — surfaced during a dataset-detail UI review, not filed as a
  ticket.

## Context

`temper_core.validation` has, since its first line, understood exactly one
row shape: a top-level `messages` field holding `[{role, content}, ...]`
dicts with at least one `assistant` turn — the same shape OpenAI's chat
fine-tuning API uses. `schema_type` is hard-coded to the literal string
`"chat"` and no other value has ever been produced; every other shape —
Alpaca's `instruction`/`input`/`output`, ShareGPT's `conversations`, plain
`prompt`/`completion`, raw `text`, preference pairs — falls through to
`unrecognised_schema` and is refused.

Nothing in the spec suite or the ADR log ever named this as a scoped
decision. Spec 003 (Phase A UI) assumes the schema already exists ("I want
to see a preview... to confirm the schema was read correctly") without
discussing alternatives. Spec 002 covers size and transport, not format.
No ADR mentions Alpaca, ShareGPT, completion-style, or preference-pair
formats anywhere. The closest thing to a rationale on record is
`docs/research-reports/report-c.md`'s competitive survey, which found real
platforms differ widely here — OpenAI ships a single chat-JSONL format,
others support four (conversational/instruction/preference/generic) — and
recommended, as the highest-leverage investment, *"Dataset upload with real
validation feedback — line-level errors, format detection, row/token
counts,"* not breadth of accepted formats. That recommendation is what
shaped the validator's depth (line-numbered errors, capped-but-counted
issue lists, thinking-mode detection) — but it was never turned into an
explicit statement that format breadth was *out* of scope. It was simply
never revisited once the first schema shipped.

This surfaced now while auditing the dataset-detail page against real
permutations: asked how many schemas exist across the industry, how many
the pinned trainer's stack (Axolotl) can consume, and how many Temper
recognises, the honest answer to the third was *"one, and nobody decided
that on purpose."* That is exactly the kind of gap this project's own
standard says gets recorded rather than left implicit — the brief's "expect
to be grilled" bar means an unexamined boundary like this one is a question
waiting to be asked.

## Decision

**Temper's dataset ingestion — upload and import alike (ADR-0047 guarantees
they share one path) — recognises exactly one schema, on purpose, going
forward: the OpenAI-style `messages` list.** Any other shape is refused with
`unrecognised_schema`, naming the field it expected and the keys it found
instead. Nothing is silently coerced or partially accepted.

This is adopted, not merely inherited: single-schema depth is the right
trade against this build's actual constraint — time to submission, not
breadth of user formats — and it keeps the validation surface (one set of
line-numbered error codes, one preview renderer, one thinking-mode
detector) small enough to be exercised as thoroughly as report-c's own
recommendation asks for.

## Alternatives considered

**Add Alpaca-style `instruction`/`input`/`output` support alongside chat.**
Rejected: it roughly doubles the validation surface — a second set of error
codes, a second preview shape, and a thinking-mode detector that has no
multi-turn concept to hang off — for a format the industry is moving away
from as chat-style fine-tuning becomes the default (report-c's own survey
shows this trend).

**A generic field-mapping/template layer**, mirroring Axolotl's
`chat_template` mechanism, that lets a user point arbitrary column names at
`role`/`content`. Rejected for now: genuinely more flexible, but it turns a
fixed, fully-testable contract into a per-user configuration surface — the
exact hazard `docs/research-reports/report-d-axolotl-utilization.md` flags
in Axolotl's own config schema, where 88% of constraints live in opaque
validator hooks a form can't anticipate. Trading a small, well-understood
contract for that flexibility is a real option later, not a default now.

**Silently detect and coerce a known alternate schema** (e.g. recognise
ShareGPT's `conversations`/`from`/`value` and remap it internally).
Rejected: a coercion a user cannot see is a second validation path wearing
a trenchcoat — the exact failure mode ADR-0047 exists to rule out for
imports. It would also require deciding role semantics and thinking-mode
detection for row shapes the validator has never been tested against.

**Leave it undocumented.** Rejected — this record is what replaces that
option.

## Consequences

- A user bringing an Alpaca- or ShareGPT-shaped file gets a coded, legible
  rejection (`unrecognised_schema`) naming the missing field and the keys
  it found, not a confusing partial ingest.
- Any future decision to widen format support is a new ADR, not a quiet
  addition to `validation.py` — the same discipline every other decision in
  this log gets.
- The gap against Axolotl (which consumes nearly every shape above through
  its named loaders and its schema-agnostic `chat_template` path) is now a
  named, deliberate scope boundary instead of an implicit one, worth
  restating directly in the submission's "what's out of scope" material.
- No code changes land with this record — it documents behaviour
  (`schema_type = "chat"`, the `unrecognised_schema` refusal) that already
  exists. Nothing is migrated or rolled back by it landing.

## Rollback

Not applicable in the usual sense — nothing new was deployed. Reversing
this decision means implementing multi-schema support, which is itself a
new ADR when it happens, not a revert of this one.
