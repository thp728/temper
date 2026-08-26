# Contracts

**Generated artifacts that cross a boundary where importing is impossible.**
Nothing here is written by hand, and nothing here is a Python type that two
Python processes could simply import: those live in `packages/core`.

| File | Produced by | Consumed by |
| --- | --- | --- |
| `openapi.json` | `just contracts`, from the FastAPI application | the generated TypeScript client in `apps/web` |
| `axolotl-field-tiers.json` | spike 7, from the pinned trainer image | the advanced configuration surface (#33) |
| `trainer-defaults.json` | hand-maintained; each value's reasoning lives in the wiki and git history | `temper_core.hyperparams` (so the control plane), and the trainer entrypoint, which receives it via the image `COPY` |

`trainer-defaults.json` is the one definition of the trainer's defaults and
overridable keys (#82). The trainer image never installs a Python package, so
this is the boundary importing cannot cross: both sides read the file, and
`test_agreement_with_the_domain.py` fails if any module declares the values
as literals again.

`axolotl-field-tiers.json` is still marked `SAMPLE ONLY` and covers twenty
fields. Issue #33 generates the full classification from the pinned image, at
which point every field lands in exactly one tier and a field in no tier fails a
completeness check.

## The rule this directory exists for

**A value two components must agree on is defined once and read, never
retyped.** The trainer's defaults were the founding counter-example: Python
literals in `apps/trainer/entrypoint.py`, hand-mirrored in three more files,
two of which said so in a comment. #82 collapsed them into
`trainer-defaults.json`, here because "the image cannot import" is precisely
a boundary where sharing must happen as data.

See [ADR-0010](../../docs/adr/0010-the-repository-is-laid-out-as-apps-and-packages.md).
