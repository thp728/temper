# Contracts

**Generated artifacts that cross a boundary where importing is impossible.**
Nothing here is written by hand, and nothing here is a Python type that two
Python processes could simply import: those live in `packages/core`.

| File | Produced by | Consumed by |
| --- | --- | --- |
| `openapi.json` | `just contracts`, from the FastAPI application | the generated TypeScript client in `apps/web` |
| `axolotl-schema.json` | the introspection of the pinned trainer image (spike 7; see its `_comment`) | `temper_core.surface` (the known-key universe), and the trainer entrypoint, which receives it via the image `COPY` |
| `axolotl-field-tiers.json` | `temper_core.surface` reads it; the classification itself is issue #33's checked-in judgment | the generated advanced surface and the pre-launch refusal gate |
| `advanced-surface.json` | `just contracts`, from `temper_core.surface` over the two files above | the interface's advanced surface (#80) and the refusal gate; drift here fails `just check` |
| `trainer-defaults.json` | hand-maintained; each value's reasoning lives in the wiki and git history | `temper_core.hyperparams` (so the control plane), and the trainer entrypoint, which receives it via the image `COPY` |

`trainer-defaults.json` is the one definition of the trainer's defaults (#82).
The trainer image never installs a Python package, so this is the boundary
importing cannot cross: both sides read the file, and
`test_agreement_with_the_domain.py` fails if any module declares the values
as literals again. Which defaults a user may *override* is no longer declared
here -- since #33 that is the exposed tier of `axolotl-field-tiers.json`,
read through `temper_core.surface`, so the reachable set and the refusal gate
cannot drift.

`axolotl-schema.json` is the configuration schema of the pinned trainer image,
introspected from `AxolotlInputConfig`. It is the universe the advanced
surface is generated from: deterministic for a given image digest, which is
what makes the generated surface stable. A key not listed here is a key the
trainer does not know, and is refused loudly and echoed back at every level --
*unknown* means unknown to the trainer, not absent from a hand-written list.

`axolotl-field-tiers.json` classifies every one of those fields into exactly
one tier (`calculated`, `exposed_with_named_failure_mode`,
`known_but_unsupported`), with a reason on every entry and a named failure
mode on every exposed one. It is data rather than code so a classification is
reviewable in a diff; `temper_core.surface` refuses to import with a field in
no tier (or a tier entry that is not a field), so a hole is a build failure.

## The rule this directory exists for

**A value two components must agree on is defined once and read, never
retyped.** The trainer's defaults were the founding counter-example: Python
literals in `apps/trainer/entrypoint.py`, hand-mirrored in three more files,
two of which said so in a comment. #82 collapsed them into
`trainer-defaults.json`, here because "the image cannot import" is precisely
a boundary where sharing must happen as data. Issue #33 applies the same rule
to the trainer's configuration surface: the schema is read once (as a
snapshot of the pinned image), the classification is read once (as a data
file), and the exposed surface is generated from the two rather than
hand-listed.

See [ADR-0010](../../docs/adr/0010-the-repository-is-laid-out-as-apps-and-packages.md).
