# Temper

A fine-tuning platform: upload a JSONL dataset, pick a base model, get a trained LoRA adapter.

An open-source fine-tuning platform. The bar is *"70-80% of what commercial products offer"*,
*"cannot be the hello-world level of fine-tuning"*, and the one that shapes everything here:
**every decision is explainable, with alternatives and tradeoffs on record.**

Auth and billing are out of scope. Every other flow is meant to be complete.

## Environment

**Anything that SSHes runs through PowerShell, never Git Bash.** Git Bash ships its own `ssh` and cannot
see the Windows `ssh-agent` service, so a passphrase-protected key fails auth and the failure
surfaces as `"ssh ready — no answer within 240s"`, which looks exactly like a dead VM.

```powershell
& ".venv\Scripts\python.exe" -u spike/spike4.py
```

Before any GPU work, `ssh-add -l` must list one ED25519 key. If it does not, `ssh-add ~/.ssh/id_ed25519`.

Interpreter: the repo's `.venv` (`uv sync` creates it). `python -u` for backgrounded runs;
`PYTHONIOENCODING=utf-8` for anything writing emoji, since Windows defaults to cp1252 and raises.
Use the Write tool over bash heredocs past ~50 lines, which have silently produced no file and no
error. Commit with `git commit -F -`, never `-m` with backticks, which get command-substituted.

**Money.** GPU work bills per minute. The reference account bills in **INR**, so read
`account.currency()` rather than assuming USD. Cheapest VM-capable GPU observed here is L4 at ₹41.31/hr
(prices move; re-check before provisioning). **Every code path that creates a VM destroys it in a
`finally` block and then confirms by listing instances.** A destroy call's return value is not
evidence. An orphaned GPU bills until somebody notices.

**Secrets.** `spike/.env` holds `JL_API_KEY`, git-ignored at two levels. Never print it, never
commit it, never paste it into a message.

## Layout and tasks

One rule, from [ADR-0010](docs/adr/0010-the-repository-is-laid-out-as-apps-and-packages.md): **if it
ships it is an app, if it is imported it is a package.** `apps/` holds `control-plane`,
`worker`, `trainer` and `web`; `packages/` holds `core` (pure domain, no framework imports) and `contracts`
(artifacts crossing a boundary where import is impossible: some generated, some hand-kept). `spike/` is a documented
throwaway; code graduating out of it takes its tests along.

**A value two components must agree on is defined once and read, never retyped.**

`just` runs every task, and the pipeline invokes the same recipes. `just --list` is the index.
Recipes hold no logic, so anyone without `just` reads the line and runs it.
[ADR-0011](docs/adr/0011-one-command-runs-every-task-and-one-defines-green.md) has the gate
contract: `just check` is the definition of green, pre-commit is a fast filter rather than the gate,
and `main` is protected by the pipeline check.

Work reaches `main` through a branch and a pull request, one per issue, squash-merged, `Closes #N`
in the body. Minimum ceremony: no templates, no required reviewers.

## Phases

Phase A (to 2026-08-21) proved the loop on a deliberately scrappy stack. Phase B is
the larger half: Postgres, Temporal, Redis, MinIO, Next.js, quality gates. Phase A was written so
Phase B would be a migration, and **the domain logic carries over unchanged.** Anywhere it does not,
the seam was leakier than claimed, and that gets recorded rather than patched over.

## How to work

- **Direct and concise.** Lead with what needs action or a decision, not a status recap.
- **Label assumptions.** If something is extrapolated rather than measured, say so in the code and
  the docs. Measured numbers beat estimated ones everywhere here; mark which is which.
- **Prove it on real hardware before believing it.** Four spikes found eleven wrong assumptions in a
  document that looked authoritative.
- **Record what turned out wrong, visibly.** The standard is *"here's what I got wrong and how I
  caught it"*, which lands better than a clean chart. Corrections are an asset here.
- **Tests do not cost money.** Stub the provider. A suite that provisions a GPU per run does not get
  run.
- **Commits:** short imperative subject, then a body explaining **why**, including tradeoffs
  accepted and alternatives rejected. The log is part of the defensibility story.

## Decisions

New decisions are ADRs in `docs/adr/`, **written as the change lands, not afterwards.** An entry
written during the build records reasoning; one written after reconstructs it, and reconstruction is
what fails under questioning. Numbers are assigned when a record lands, never reserved. A decision
is not edited once accepted; superseding it means a new record.

New decisions land here only. See `docs/adr/` for the index and numbering rules.

## Where things are

`CONTEXT.md` is the domain glossary and fixes one vocabulary for code, specs and records.
`docs/adr/` holds decision records, `docs/specs/` the twelve specs tickets come from, `docs/agents/`
the issue tracker, triage labels and domain docs. Each module carries its own `AGENTS.md` with the
rules specific to it.

`uv` owns Python dependencies, `pnpm` will own JavaScript ones, `just` owns verbs. One workspace,
one `uv.lock`. `just check` before you push; `just --list` for everything else.
