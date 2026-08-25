# Contracts

**Generated artifacts that cross a boundary where importing is impossible.**
Nothing here is written by hand, and nothing here is a Python type that two
Python processes could simply import: those live in `packages/core`.

| File | Produced by | Consumed by |
| --- | --- | --- |
| `openapi.json` | `just contracts`, from the FastAPI application | the generated TypeScript client in `apps/web` |
| `axolotl-field-tiers.json` | spike 7, from the pinned trainer image | the advanced configuration surface (#33) |

`axolotl-field-tiers.json` is still marked `SAMPLE ONLY` and covers twenty
fields. Issue #33 generates the full classification from the pinned image, at
which point every field lands in exactly one tier and a field in no tier fails a
completeness check.

## The rule this directory exists for

**A value two components must agree on is defined once and read, never
retyped.** The trainer's defaults are currently the counter-example: they are
Python literals in `apps/trainer/entrypoint.py`, hand-mirrored in three
control-plane files, two of which say so in a comment. Issue #82 collapses them
into a data file here, read by the control plane and copied into the image at
build time.

See [ADR-0010](../../docs/adr/0010-the-repository-is-laid-out-as-apps-and-packages.md).
