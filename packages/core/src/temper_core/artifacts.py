"""The deliverable and its kinds: what a user downloads, and how to load it.

Issue #32 / spec 006. The glossary used to call the adapter the deliverable;
full fine-tuning produces no adapter, so that definition is about to become
false while eleven tickets assume one or the other. This module fixes the
vocabulary in code before the code assumes otherwise: **the artifact is the
canonical deliverable, and the adapter is one kind of it.**

Everything here is pure data over strings -- no framework, no I/O -- because
the API, the download path, the interface and the decision records must all
read the same vocabulary from one source (ADR-0010: a value two components
must agree on is defined once and read, never retyped).

**Kind is derived from the job's method, never stored.** The job row's
`method` is frozen at creation and immutable, so a kind derived from it is
immutable too: it cannot drift from the method that produced the artifact, and
existing rows -- written before this module existed -- read correctly the
moment this code deploys, with no migration. Storing a `kind` column would
introduce exactly the disagreement this derivation removes, and would leave
every pre-existing row needing a backfill whose answer would be this same
function. The artifact *record* exposes the derived kind wherever the artifact
is described; it is computed, never persisted.

`kind_for(None)` returns the adapter kind because that is the honest reading of
a job row written before the method column existed: every pre-method run was a
QLoRA adapter (the trainer has only ever executed qlora). Treating the absence
of a method as "adapter" is a description of history, not a default that could
mask a future full-model row.
"""

from __future__ import annotations

# The kinds. These strings are the vocabulary the API, the download manifest
# and the interface share; a kind is never re-spelled anywhere else.
ARTIFACT_KIND_ADAPTER = "adapter"
ARTIFACT_KIND_FULL_MODEL = "full_model"
ARTIFACT_KIND_MERGED_MODEL = "merged_model"

ARTIFACT_KINDS: tuple[str, ...] = (
    ARTIFACT_KIND_ADAPTER,
    ARTIFACT_KIND_FULL_MODEL,
    ARTIFACT_KIND_MERGED_MODEL,
)

# method -> the kind of artifact that method produces. QLoRA and LoRA both
# produce a PEFT adapter; full fine-tuning rewrites the whole model, so it
# produces a fully trained model instead. A merged model is not produced by any
# training method -- merging is an artifact-time step the platform does not run
# yet (spec 011's territory) -- so no method maps to it.
KIND_BY_METHOD: dict[str, str] = {
    "qlora": ARTIFACT_KIND_ADAPTER,
    "lora": ARTIFACT_KIND_ADAPTER,
    "full": ARTIFACT_KIND_FULL_MODEL,
}

# The canonical member file names of an adapter artifact: the weights and the
# config that makes them loadable. Named once here because the control plane
# records them, the download serves them and the legacy-row fallback derives
# them -- one definition, read everywhere. The two other kinds carry no fixed
# member set here: a full model's files are whatever the trainer reports (its
# own config, tokenizer and weight shards), so their members are recorded with
# the artifact when such a run exists, not guessed at in advance.
ADAPTER_MEMBER_NAMES: tuple[str, ...] = (
    "adapter_model.safetensors",
    "adapter_config.json",
)

# The manifest written into every download. The spec names it twice: the
# manifest that travels with an artifact records which kind it is, because
# loading one differs from loading another (spec 006) -- evaluation and lineage
# content stay out of scope (spec 011).
MANIFEST_NAME = "temper-artifact.json"


class UnknownMethod(ValueError):
    """A method that maps to no artifact kind.

    Raised rather than guessed: a training method this module does not know
    cannot be said to produce any artifact, and inventing a kind would hand a
    user a download whose loading instructions were fabricated.
    """


class UnknownArtifactKind(ValueError):
    """A kind this module does not define. Raised by `loading_instructions`."""


def kind_for(method: str | None) -> str:
    """The kind of artifact `method` produces, derived -- never stored.

    `None` reads as the adapter kind: the pre-method-era job rows are all
    QLoRA adapters (see the module docstring), so the derivation must describe
    them correctly rather than refuse the history it was written after.
    """
    if method is None:
        return ARTIFACT_KIND_ADAPTER
    try:
        return KIND_BY_METHOD[method]
    except KeyError:
        raise UnknownMethod(
            f"method {method!r} produces no known artifact kind. "
            f"Known methods: {', '.join(sorted(KIND_BY_METHOD))}."
        ) from None


# Loading instructions, one per kind. They differ because the load path
# differs: an adapter has to be applied to a base model, a fully trained or
# merged model is complete on its own. This is the concrete difference the
# issue means by "loading instructions differ by kind".
LOADING_INSTRUCTIONS: dict[str, str] = {
    ARTIFACT_KIND_ADAPTER: (
        "A PEFT adapter that modifies the base model without altering it. "
        "Load the base model with AutoModelForCausalLM.from_pretrained, then "
        "apply these weights with PeftModel.from_pretrained(base_model, "
        "<unzipped folder>); the folder carries adapter_config.json, which "
        "records the rank, alpha and target modules."
    ),
    ARTIFACT_KIND_FULL_MODEL: (
        "A fully fine-tuned model: complete weights and configuration, "
        "shipped as one archive (model.tar.gz). Extract the download, then "
        "extract model.tar.gz -- it contains a single model/ folder -- and "
        "load that folder directly with "
        "AutoModelForCausalLM.from_pretrained(<the model folder>); no base "
        "model or adapter step is needed."
    ),
    ARTIFACT_KIND_MERGED_MODEL: (
        "A base model with its trained change merged into the full weights. "
        "Load the unzipped folder directly with "
        "AutoModelForCausalLM.from_pretrained(<unzipped folder>), like any "
        "complete model."
    ),
}


def loading_instructions(kind: str) -> str:
    """How to load an artifact of `kind` -- the per-kind load path.

    Refuses an unknown kind loudly rather than returning an instruction for
    one this module has not defined.
    """
    try:
        return LOADING_INSTRUCTIONS[kind]
    except KeyError:
        raise UnknownArtifactKind(
            f"{kind!r} is not a defined artifact kind. "
            f"Known kinds: {', '.join(ARTIFACT_KINDS)}."
        ) from None
