"""The side-by-side comparison (Spec 011, issue #69).

The same held-out prompts run through the base model and through the tuned
model, shown side by side, so a user without a background in this can judge
the difference for themselves. Nothing about this is a benchmark and the
interface must not pretend otherwise; it is the most directly useful evidence
the platform can produce, and it is produced on the machine that is already
warm with both models loaded.

Three decisions this module makes, and why:

* **The prompts are the platform's held-out rows, never a second sampling.**
  Issue #53 splits the dataset after deduplication, and that ordering is
  load-bearing: a duplicated row can never land on both sides. Prompt
  selection therefore reads the trainer's own held-out file -- the same rows
  the held-out loss was measured on -- and picks a fixed, deterministic slice
  of it. A second sample of the dataset would reintroduce the overlap the
  split exists to prevent, and the user would be shown answers the model
  memorised in exactly the case that matters.

* **Decoding settings are fixed and recorded.** A comparison run at a
  different temperature is not a comparison, so the settings are a named
  module-level constant, read by the generator and recorded onto the result
  so the interface can display them without retyping them (ADR-0010's "a
  value two components must agree on is defined once"). Recorded means stored
  on the result and shown, so a reader can tell whether two outputs are
  comparable -- that needs a test, not a docstring.

* **Evaluation failure never fails the run.** The training result is the
  deliverable; the comparison is what the platform adds. ``run_comparison``
  never raises: a generator that throws becomes ``ok: false`` with the reason
  recorded, and the entrypoint applies the same rule to model loading, so a
  paid run is never lost to the extra step.

The module is pure -- no I/O, no framework imports. The generators are a seam
(the entrypoint wires real transformers/peft models on the machine); the host
suite injects doubles, exactly as ``template_probe.py`` does with its
tokenizer.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

# Fixed decoding settings for the comparison generation, defined once
# (ADR-0010) and recorded onto the result. They are read by the generator on
# the machine and displayed by the interface from the record -- never
# retyped. Fixed rather than exposed because a comparison run at a different
# temperature is not a comparison: these are the settings under which "base
# answered X, tuned answered Y" means anything.
COMPARISON_DECODING: dict[str, float | int | bool] = {
    "temperature": 0.7,
    "do_sample": True,
    "max_new_tokens": 128,
}

# How many held-out rows are generated from on each side. Fixed and small:
# each row costs warm-machine time, and the slice exists to let a person read
# the difference, not to be a benchmark (Spec 011's small-fixed-versioned
# discipline applied to the comparison instead of the capability check).
COMPARISON_PROMPT_COUNT = 3

# The seed that fixes which held-out rows are selected. A fixed constant
# rather than the run's own, so the slice is the same shape for every job and
# reproducible from the record: the same split plus the same seed selects the
# same prompts. The split itself is fixed under the run's seed (issue #53),
# so the whole comparison is reproducible end to end.
COMPARISON_PROMPT_SEED = 0


class PromptGenerator(Protocol):
    """The slice of a loaded model the comparison needs: render and generate.

    The entrypoint wires a real transformers model -- and an adapter or a
    whole checkpoint for the tuned side -- with this module's decoding
    settings fixed inside it; tests inject a double. ``generate`` takes a
    conversation (a held-out row's messages, minus the held-out answer) and
    returns the model's completion as text.
    """

    def generate(self, conversation: Sequence[Mapping[str, Any]]) -> str: ...


def prompt_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The conversation up to the last user turn, so the model answers fresh.

    The held-out rows carry the reference assistant answer; feeding it to the
    model would be handing it the target it was never shown in training. The
    comparison prompt is the context up to the last user message, so both
    models answer the same question and the page shows their answers -- not
    the held-out answer. Trailing assistant turns are dropped; anything before
    the last user turn stays as context.
    """
    trimmed = [dict(m) for m in messages]
    while trimmed and trimmed[-1].get("role") == "assistant":
        trimmed.pop()
    return trimmed


def select_prompts(
    rows: Sequence[Mapping[str, Any]],
    count: int = COMPARISON_PROMPT_COUNT,
    seed: int = COMPARISON_PROMPT_SEED,
) -> list[dict[str, Any]]:
    """A fixed, deterministic slice of the held-out rows, in dataset order.

    ``count`` distinct rows are chosen by a seeded shuffle -- the same
    mechanism the split itself uses -- then returned in the order they appear
    in the dataset, so what the page shows is the dataset's own ordering.
    Fewer than ``count`` rows selects all of them. Determinism is the point:
    the same split and seed select the same prompts, so a comparison can be
    reproduced from the run's record.
    """
    n = min(count, len(rows))
    if n <= 0:
        return []
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)  # noqa: S311 - reproducible, not secret
    chosen = set(order[:n])
    return [dict(rows[i]) for i in range(len(rows)) if i in chosen]


@dataclass(frozen=True)
class ComparisonOutcome:
    """What the comparison produced, shaped for the result document.

    ``ok`` is whether the comparison was produced at all -- ``rows`` exist.
    On a failure ``reason`` names it and ``rows`` is empty; the decoding
    settings are recorded either way, because "fixed and recorded" holds even
    when the generation could not run, and ``selection`` names which
    checkpoint the tuned side compared (issue #62's rule, applied on the
    machine so it is the model selection actually chose).
    """

    ok: bool
    decoding: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    selection: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "decoding": self.decoding,
            "rows": self.rows,
        }
        if self.reason is not None:
            out["reason"] = self.reason
        if self.selection is not None:
            out["selection"] = self.selection
        return out


def run_comparison(
    *,
    conversations: Sequence[Sequence[Mapping[str, Any]]],
    base: PromptGenerator,
    tuned: PromptGenerator,
    decoding: Mapping[str, Any] = COMPARISON_DECODING,
    selection: Mapping[str, Any] | None = None,
) -> ComparisonOutcome:
    """Generate from both models for each prompt, and shape the record.

    **Never raises.** A generator that throws becomes ``ok: false`` with the
    reason recorded, so evaluation failure never fails the run (Spec 011) --
    the negative test for this lives in ``test_comparison.py``. On a failure
    any partial rows are dropped rather than left half-complete, because a
    reader cannot tell which rows of a broken run are trustworthy.

    The decoding settings actually used are copied onto the result, so a
    reader can tell whether two outputs are comparable. ``selection`` names
    which checkpoint the tuned side compared, so the record says the model
    selection actually chose rather than whatever happened to be last.
    """
    try:
        rows: list[dict[str, Any]] = []
        for conversation in conversations:
            # The held-out answer is trimmed before generation, so neither
            # model is handed the target it was never shown in training; the
            # recorded prompt is exactly what was generated from.
            prompt = prompt_messages(conversation)
            rows.append(
                {
                    "prompt": prompt,
                    "base": base.generate(prompt),
                    "tuned": tuned.generate(prompt),
                }
            )
        return ComparisonOutcome(
            ok=True,
            decoding=dict(decoding),
            rows=rows,
            selection=None if selection is None else dict(selection),
        )
    except Exception as e:  # noqa: BLE001 - the run must survive the extra step
        return ComparisonOutcome(
            ok=False,
            decoding=dict(decoding),
            reason=f"{type(e).__name__}: {e}",
            selection=None if selection is None else dict(selection),
        )
