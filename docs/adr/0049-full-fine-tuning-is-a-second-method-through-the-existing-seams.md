# ADR-0049 — Full fine-tuning is a second method through the existing seams

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/009-models-methods-and-advanced-mode.md`
- **Issue:** [#66](https://github.com/thp728/temper/issues/66)

## Context

Issue #66 makes full fine-tuning a real second method: selectable, chosen by
the predictor when it is the right call, its learning rate keyed off the
method rather than a global constant, and its artifact produced and delivered
at whole-model size. Spec 009's whole point is that a platform limited to the
cheap method cannot be described as covering what commercial products offer —
but the work here was deliberately framed as a second method *through the
existing seams*, not new infrastructure. Four of those seams already exist and
are unchanged in shape:

- **Artifact kinds** (#32): `temper_core.artifacts` already derives the kind
  from the method — `full` produces a `full_model`, never an adapter — and the
  download path already serves any member set without special-casing. The
  kind-derivation and the manifest were written before any full-model run
  existed.
- **The machine-write path** (#50, #37): the machine writes its own artifact
  to one scoped grant (ADR-0009) and the control plane verifies what landed by
  streaming it back through the checksum, flat in memory. That is exactly the
  path a whole model — larger than any adapter — rides.
- **The predictor** (#48, #55): peak VRAM is arithmetic over a model-facts
  seam, and hardware is chosen cheapest-fit-first with every decision carrying
  its reason (#76) and any decision overridable (#79). Method was already one
  of those decisions; it was merely restricted to what the trainer could run.
- **The defaults table** (#82): `packages/contracts/trainer-defaults.json` is
  the one definition both the resolver and the trainer read. A second method
  with a different learning rate had to be a *second footing in that same
  table*, never a second table and never a branch at the use site.

The one piece that genuinely did not exist was the trainer executing a full
fine-tune and collecting a whole model as its artifact.

## Decision

**Full fine-tuning is a second executable method, delivered as one streamed
archive through the existing artifact path, with its defaults keyed off the
method in the one contract.**

- **`EXECUTABLE_METHODS` becomes `("full", "qlora")`.** The trainer can now
  run both, single-device. The method is not chosen separately from the GPU
  and reconciled; it falls out of the existing cheapest-fit-first search,
  which already breaks a tied price toward the more capable method that fits
  (`METHODS_BEST_FIRST`). So "chosen when it is the right call" is not a new
  rule — it is what the existing decision already does once `full` is in the
  executable set: on a card a full fine-tune fits where nothing cheaper does,
  the predictor picks it; where an adapter fits cheaper, the adapter wins. LoRA
  stays unexecutable (its own spec-009 ticket).

- **The defaults table gains a per-method footing, not a second table.**
  `trainer-defaults.json` now carries `defaults` (the adapter footing,
  unchanged) and `by_method` — today one entry, `full`, overriding the
  learning rate to `1e-5` against the adapter's `2e-4`. A full fine-tune
  updates every weight; its learning rate must be lower or the base model is
  destabilised, and the difference is data in the one contract both sides
  read, not a constant read by both with a branch at the use site (ADR-0010).
  `hyperparams.effective(overrides, method=...)` overlays the footing before
  user overrides, so an explicit learning rate always beats the method
  default. Every `by_method` key is also a `defaults` key, which is what keeps
  the trainer's required-key set complete for every method.

- **The trainer branches on the method in its config, and nowhere else.** The
  job spec written at provisioning carries `method` and the method-resolved
  hyperparameters (the control plane resolves; the trainer applies, #83).
  `build_config` for `full` configures no PEFT adapter and no 4-bit
  quantisation (`load_in_4bit: False`, explicit so the config cannot silently
  drift into an adapter shape) and drops the adapter-only hyperparameters from
  the YAML. The learning rate arrives resolved in the spec; the trainer never
  recomputes it. A spec with no `method` reads as `qlora` — the only method
  that ever ran before the key existed, the same reading `artifacts.kind_for`
  gives a missing method.

- **A full model's artifact is one streamed archive.** The machine's write
  grant is for one object (ADR-0009), and a whole model's file set is unknown
  at mint time — its size, shards and member names exist only after training.
  So the trainer tars the trained model directory into `model.tar.gz`
  (streaming, flat in memory — the same guarantee the artifact path already
  guards) and uploads that one object. The control plane verifies it against
  the machine's checksum exactly as it does an adapter, and the download
  serves it as a member named by what it is, with the manifest declaring
  `full_model` and its load path. A second archive-in-a-zip extraction step is
  the honest price of the one-grant path, and the loading instructions say so.

- **The refusal arithmetic names the lightest executable method.** Three
  call sites used `EXECUTABLE_METHODS[0]` as "the cheapest/lightest method".
  That assumption was invisible while the set had one entry and breaks the
  moment `full` leads it. A `lightest_method` helper (the method with the
  smallest per-param weight footprint) now answers that question, so an
  unpinned refusal with a card free names the adapter's peak — if even the
  cheapest-to-fit method cannot fit, nothing can, and naming the heaviest
  would overstate what the user has to beat.

- **The result document and artifact record use the honest name.** The
  trainer's result fields were renamed from `adapter_*` to `artifact_*`
  because a full model's artifact is not an adapter, and a field named
  `adapter_sha256` for a whole model would be a satisfied-looking lie. The
  adapter's own config stays `adapter_config`, adapter-only.

## Alternatives considered

**Ship the whole model as multiple stored objects, one grant per file.**
Rejected: the grant count and the file names exist only after training, and
the machine has no inbound channel to request more grants (ADR-0004, it
publishes no ports). Pre-minting a guess at the file set would be the same
guess a tar avoids, and checkpoints (#37) work because their slot count is
known ahead of time. One archive through the existing one-object grant is the
seam that already exists.

**Extract the archive into member objects on the control plane at collect
time.**
Rejected: that is new infrastructure — tar extraction, path-traversal
hardening, per-member storage — exactly the "second method through the seams"
scope this issue refuses. The download already zips any member set, so a
one-member archive needs no new code anywhere on the serving side.

**Let the trainer keep the `adapter_*` result names for the shared "primary
object".**
Rejected: the names are the trainer-to-orchestrator contract, and naming a
whole model's checksum `adapter_sha256` is the kind of claim that fails the
"explain every decision" bar. The rename is mechanical and bounded; the lie
was not.

**Give full fine-tuning its own learning-rate default inside the trainer's
config builder.**
Rejected: that is a second resolver — the trainer derives nothing (#83), and a
learning rate chosen on the machine is one the control plane's frozen spec
cannot show. The method-resolved value must live in the one contract the
resolver reads.

**Keep `EXECUTABLE_METHODS[0]` as "the lightest" and order the set
cheapest-first.**
Rejected: the search takes the first executable method that fits at each price
point (the free-quality-upgrade tie-break), so the set must be best-first and
`full` must lead it. "Lightest" is a different question from "best", and the
two stopped coinciding the moment the set grew past one entry.

## Consequences

- `full` is selectable (a plan override, refused only where it cannot fit) and
  predictor-chosen where it is the cheapest configuration that fits; the
  method decision shows the adapter alternative with what it would have cost.
- A full job's spec carries its own learning rate and no adapter claims; the
  artifact is one `model.tar.gz`, verified against the machine's checksum,
  downloaded with a manifest declaring `full_model` and its two-step load
  path.
- The machine now tars the model directory, so a full job needs disk for both
  the model and its archive transiently — inside the working-space allowance
  the disk arithmetic already carries as a labelled assumption, and confirmed
  (or corrected) by the hardware run.
- Two acceptance criteria cannot be finished without GPUs and are stated as
  outstanding rather than quietly reinterpreted: a real full fine-tuning job
  on real hardware whose artifact loads, and the memory prediction for the
  method anchored against that run. Everything else is exercised end to end
  against the fake provider.

## Rollback

Revert `EXECUTABLE_METHODS` to `("qlora",)`, remove the `by_method` footing
(and the `method=` parameter from the resolver), and the trainer runs its
pre-#66 adapter-only shape; the artifact renames are cosmetic but would need
their callers reverted with them.
