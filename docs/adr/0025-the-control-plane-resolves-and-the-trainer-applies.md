# ADR-0025 — The control plane resolves; the trainer applies

- **Status:** accepted
- **Date:** 2026-08-26
- **Spec:** `docs/specs/009-models-methods-and-advanced-mode.md`
- **Issue:** [#83](https://github.com/thp728/temper/issues/83)

## Context

Every hyperparameter used to be resolved twice. `apps/trainer/entrypoint.py`
held a `DEFAULTS` table and applied user overrides over it inside the job
container; `packages/core/hyperparams.py` held a hand-mirrored copy of the same
table and resolved again on the control-plane side, so the create-job page
could promise what the run would do. The mirror was pinned together by a test
asserting literal equality of two dicts in two processes that could not see
each other.

Two resolvers means two places a value is chosen, and they can disagree while
every test stays green — the pin compared the *tables*, not the *behaviour*.
Worse, the resolver that actually decided a paid training run was the one
nobody could see: the trainer's copy ran last, on the machine, after launch,
and its choices surfaced only by reading `config.yaml` afterwards.

[ADR-0010's appendix](0010-the-repository-is-laid-out-as-apps-and-packages.md)
already recorded the duplication as a known defect and pointed at two tickets:
#82 collapses where the defaults table lives, #83 removes the second resolver.
This record covers the second move; it deliberately does not touch where the
table is declared, which stays in `temper_core.hyperparams` until #82 lands.

## Decision

**The job specification written at launch carries every hyperparameter already
resolved, and the trainer resolves nothing.** `_remote_script` writes
`hyperparams.effective(overrides)` into the machine's copy of the spec whole —
defaults the user never mentioned, alpha recomputed from a moved rank, rsLoRA
inferred at rank >= 32. The trainer's `DEFAULTS` table, its alpha-recompute and
its rsLoRA inference are deleted. What remains on the trainer side is the
guard:

- **A missing required value fails loudly** (`IncompleteJobSpec`, surfaced as
  error code `spec_incomplete`, keys named) instead of falling back to a number
  nobody chose. Training on an unchosen number is worse than not training.
- **Unknown keys are still refused loudly and echoed back**, at the top level
  and inside `hyperparameters`, under `rejected_overrides` in `result.json`.
  Until #33 generates the known-key set from the pinned image's own schema,
  "known" means read by the entrypoint when rendering `config.yaml`.
- **The trainer derives nothing else.** Whatever alpha–r pairing or rsLoRA flag
  the spec carries is what trains, even when visibly strange; second-guessing
  the spec is the second resolver back through the wall.

**An unknown key is refused at creation, not left for the machine.** Because
resolution now happens before launch, a key outside the overridable set would
be dropped by `effective()` without ever reaching the trainer's guard — a
silent regression of the refuse-loudly rule. `jobs.create` refuses them with a
stable code (`unknown_hyperparameter`) naming each offending key, before
anything is provisioned. The trainer keeps its guard as defence in depth for
callers that bypass the API.

**The agreement pin changes shape to match.** The old equality-of-two-tables
test becomes: the set of keys the trainer requires equals exactly what
`hyperparams.effective({})` produces. If a default is added to the resolver
without the trainer learning to read it, the suite fails here rather than
failing every launch on a paid machine.

## Alternatives considered

**Keep the trainer's resolver as the authoritative one and have the control
plane call into it.** Rejected: the control plane does not import application
code (ADR-0010), and shipping the trainer's module into the control plane to
borrow one function is the coupling the app/package split exists to prevent.

**Move the defaults table into `packages/contracts/` in this change too.**
Rejected as out of scope: that is #82's move, a sibling branch carries it, and
both PRs touching the same seam with two different restructurings guarantees a
conflict nobody can review honestly. This record leaves the table exactly
where it was.

**Let unknown keys fall through resolution and rely on the trainer's echo.**
Rejected: the drop happens silently inside `effective()`, so the echo would
never fire and the caller would believe their override was in effect — the
exact failure the refuse-loudly rule exists to prevent.

**Fail the whole job when the trainer sees an unknown key.** Deferred: refusal
plus echo preserves today's contract, and hard-failing is better placed
alongside #33's generated surface, where "unknown" becomes unknown-to-the-
schema rather than unknown-to-a hand-written list.

## Consequences

- One resolver, and it is the visible one: its output lands in the job's event
  history via the remote script and in `result.json`'s rendered config.
- The create-page promise and the launch payload are the same expression now
  (`effective(...)`), not two implementations pinned by a test.
- Standalone users of the image must supply a complete resolved spec;
  `job.example.json` is that complete example, and the README documents the
  path. Missing keys fail with named keys, never silently.
- [ADR-0010](0010-the-repository-is-laid-out-as-apps-and-packages.md)'s
  appendix describes `DEFAULTS`/`ALLOWED_OVERRIDES` sitting in the trainer as
  current state; this record supersedes that clause — the copies are gone, and
  only #82's relocation of the surviving table remains open.
