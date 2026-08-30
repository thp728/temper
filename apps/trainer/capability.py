"""The general-capability slice (Spec 011, issue #73).

Fine-tuning degrades general capability, the effect is real, and no other
signal in the product catches it. This is a smoke test for catastrophic
forgetting, not an evaluation harness: a fixed, versioned slice of general
questions runs through the base model and through the tuned model on the
already-warm machine, and the difference is reported as a delta with its
sample size and uncertainty stated. The interface says it is a smoke test,
because a small slice honestly labelled is useful and the same slice dressed
as a benchmark is misleading (Spec 011).

Decisions this module makes, and why:

* **The slice is fixed, small and versioned.** The questions are a module
  constant -- the same set for every job -- and carry a version number, so a
  result is comparable to another result made under the same slice, and a
  reader knows which slice a number came from. The slice exists to catch a
  large regression, not to rank models; a general evaluation harness is
  Spec 011's explicit out-of-scope.

* **The questions are general knowledge, not the user's task.** The point is
  to detect what fine-tuning can destroy, so the slice is deliberately
  unrelated to whatever the user fine-tuned on. A model that learned the task
  and forgot arithmetic or geography is exactly what this catches.

* **Scoring is deterministic and stated.** Each question is multiple-choice;
  the model's answer is parsed for the option letter. A response that names
  no letter is *wrong* (counted against the model) rather than dropped,
  because a model that cannot produce an answer letter is itself evidence.

* **The delta is reported with its uncertainty.** The delta is the tuned
  score minus the base score over the *same* questions -- a paired
  before/after. Its standard error is computed from the per-question
  differences, and the sample size and the per-question weight (1/n) travel
  with the result, so a reader can see that a single question is worth 1/n of
  the score.

* **A large regression is a flag on the record, not a judgement call in the
  interface.** The threshold -- the tuned model answered at least two fewer
  questions correctly than the base -- is defined here and recorded on the
  result, so the interface surfaces it from the record rather than deciding
  for itself (ADR-0010: a value one side decides on is defined once).

* **Evaluation failure never fails the run.** Like the side-by-side comparison
  (#69), ``run_capability_slice`` never raises: a generator that throws
  becomes ``ok: false`` with the reason recorded, so a paid run is never lost
  to the extra step.

The module is pure -- no I/O, no framework imports. The generators are a seam
exactly as ``comparison.py``'s are: the entrypoint wires the real loaded
models on the machine; the host suite injects doubles.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from comparison import COMPARISON_DECODING, PromptGenerator

# The version of the fixed slice. Bumping this means the questions changed,
# and a number from slice v1 cannot be compared to a number from slice v2 as
# if they were the same measurement -- which is exactly why the version
# travels with every result.
CAPABILITY_SLICE_VERSION = 1

# The fixed slice itself. One set for every job, deliberately small (each
# question costs warm-machine time and the slice is a smoke test, not a
# benchmark), deliberately general (never the user's task -- the point is to
# catch what fine-tuning can destroy). Each item carries its domain and a
# multiple-choice prompt whose correct letter is in `answer`. The prompts are
# written out in full rather than assembled, so the exact bytes a model was
# asked are versioned data, not a construction.
CAPABILITY_QUESTIONS: list[dict[str, Any]] = [
    {
        "domain": "astronomy",
        "answer": "A",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer this general-knowledge question. Reply with only "
                    "the letter of the correct option, such as 'B'.\n\n"
                    "Which planet is the largest in the solar system?\n"
                    "A. Jupiter\nB. Saturn\nC. Neptune\nD. Uranus"
                ),
            }
        ],
    },
    {
        "domain": "geography",
        "answer": "C",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer this general-knowledge question. Reply with only "
                    "the letter of the correct option, such as 'B'.\n\n"
                    "What is the capital of Japan?\n"
                    "A. Beijing\nB. Seoul\nC. Tokyo\nD. Bangkok"
                ),
            }
        ],
    },
    {
        "domain": "history",
        "answer": "B",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer this general-knowledge question. Reply with only "
                    "the letter of the correct option, such as 'B'.\n\n"
                    "In which year did World War I begin?\n"
                    "A. 1905\nB. 1914\nC. 1920\nD. 1939"
                ),
            }
        ],
    },
    {
        "domain": "arithmetic",
        "answer": "A",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer this general-knowledge question. Reply with only "
                    "the letter of the correct option, such as 'B'.\n\n"
                    "What is 12 divided by 4?\n"
                    "A. 3\nB. 4\nC. 6\nD. 8"
                ),
            }
        ],
    },
    {
        "domain": "geography",
        "answer": "D",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer this general-knowledge question. Reply with only "
                    "the letter of the correct option, such as 'B'.\n\n"
                    "Which of these countries is in South America?\n"
                    "A. Spain\nB. Portugal\nC. Nigeria\nD. Brazil"
                ),
            }
        ],
    },
    {
        "domain": "biology",
        "answer": "C",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer this general-knowledge question. Reply with only "
                    "the letter of the correct option, such as 'B'.\n\n"
                    "Which organ pumps blood through the human body?\n"
                    "A. Lungs\nB. Liver\nC. Heart\nD. Brain"
                ),
            }
        ],
    },
    {
        "domain": "language",
        "answer": "B",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer this general-knowledge question. Reply with only "
                    "the letter of the correct option, such as 'B'.\n\n"
                    "Which of these words is a synonym of 'enormous'?\n"
                    "A. tiny\nB. huge\nC. slow\nD. bright"
                ),
            }
        ],
    },
    {
        "domain": "history",
        "answer": "A",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer this general-knowledge question. Reply with only "
                    "the letter of the correct option, such as 'B'.\n\n"
                    "Which ancient civilisation built the pyramids of Giza?\n"
                    "A. The Ancient Egyptians\nB. The Romans\nC. The Greeks\nD. The Vikings"
                ),
            }
        ],
    },
]

# How many more correct answers the tuned model must lose, relative to the
# base, before the run is flagged as a large regression. With n questions each
# worth 1/n of the score, losing two is more than the smoke test's noise
# plausibly explains and is the smallest drop a person cannot shrug off; it is
# recorded on the result, and the interface surfaces the flag prominently from
# the record rather than re-deciding it (ADR-0010).
LARGE_REGRESSION_QUESTIONS = 2


def parse_answer(text: str) -> str | None:
    """The option letter the model's answer names, or None when it names none.

    The prompt asks for only the letter, so a bare letter is the expected
    shape; the parser also accepts a letter anywhere in the response (e.g.
    "The answer is B.") by taking the first standalone option letter. A
    response that names no letter is None -- scored as wrong, not dropped,
    because failing to produce an answer letter is itself evidence.
    """
    if not text:
        return None
    upper = text.upper()
    bare = re.fullmatch(r"\s*([ABCD])[.)\]:]?\s*", upper)
    if bare:
        return bare.group(1)
    for match in re.finditer(r"\b([ABCD])\b", upper):
        return match.group(1)
    return None


def delta_standard_error(
    base_correct: Sequence[bool], tuned_correct: Sequence[bool]
) -> float:
    """The standard error of the paired delta, tuned minus base.

    Each question contributes a difference d_i = t_i - b_i in {-1, 0, 1}; the
    delta is their mean, and its standard error is the sample standard
    deviation of the differences over sqrt(n). Paired, not independent: the
    same questions ran through both models, so the paired standard error is
    the honest one for a before/after on the same items. Zero when every
    question answered the same on both sides; a single question yields no
    within-sample spread and therefore no standard error (n <= 1 -> 0.0).
    """
    n = len(base_correct)
    if n <= 1:
        return 0.0
    diffs = [
        int(bool(t)) - int(bool(b))
        for b, t in zip(base_correct, tuned_correct, strict=False)
    ]
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return math.sqrt(variance / n)


@dataclass(frozen=True)
class CapabilityOutcome:
    """What the general-capability slice produced, shaped for the result.

    ``ok`` is whether the slice ran at all -- ``rows`` exist. On a failure
    ``reason`` names it, the counts are zero, and the decoding settings are
    recorded either way, because "fixed decoding, recorded" holds even when
    the slice could not run. ``selection`` names which checkpoint the tuned
    side answered through (the run's own selection rule, applied on the
    machine so it is the model selection actually chose). ``large_regression``
    is the recorded flag the interface surfaces prominently.
    """

    ok: bool
    version: int = CAPABILITY_SLICE_VERSION
    regression_threshold: int = LARGE_REGRESSION_QUESTIONS
    total: int = 0
    base_correct: int = 0
    tuned_correct: int = 0
    delta: float = 0.0
    delta_se: float = 0.0
    large_regression: bool = False
    decoding: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    selection: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "version": self.version,
            "regression_threshold": self.regression_threshold,
            "total": self.total,
            "base_correct": self.base_correct,
            "tuned_correct": self.tuned_correct,
            "base_score": (
                round(self.base_correct / self.total, 4) if self.total else 0.0
            ),
            "tuned_score": (
                round(self.tuned_correct / self.total, 4)
                if self.total
                else 0.0
            ),
            "delta": round(self.delta, 4),
            "delta_se": round(self.delta_se, 4),
            "large_regression": self.large_regression,
            "decoding": self.decoding,
            "rows": self.rows,
        }
        if self.reason is not None:
            out["reason"] = self.reason
        if self.selection is not None:
            out["selection"] = self.selection
        return out


def run_capability_slice(
    *,
    questions: Sequence[Mapping[str, Any]] = CAPABILITY_QUESTIONS,
    base: PromptGenerator,
    tuned: PromptGenerator,
    decoding: Mapping[str, Any] = COMPARISON_DECODING,
    selection: Mapping[str, Any] | None = None,
    version: int = CAPABILITY_SLICE_VERSION,
) -> CapabilityOutcome:
    """Answer every slice question with both models and shape the record.

    **Never raises.** A generator that throws becomes ``ok: false`` with the
    reason recorded, so evaluation failure never fails the run (Spec 011) --
    the negative test lives in ``test_capability.py``. On a failure any
    partial rows are dropped rather than left half-complete, for the same
    reason the comparison drops them: a reader cannot tell which rows of a
    broken run are trustworthy.

    The decoding settings actually used are copied onto the result, so a
    reader knows the conditions the score was measured under (the same fixed
    settings the comparison uses -- defined once in ``comparison.py``).
    ``selection`` names which checkpoint the tuned side answered through, so
    the record says the model selection actually chose rather than whatever
    happened to be last.
    """
    try:
        rows: list[dict[str, Any]] = []
        base_corrects: list[bool] = []
        tuned_corrects: list[bool] = []
        for question in questions:
            messages = question["messages"]
            base_text = base.generate(messages)
            tuned_text = tuned.generate(messages)
            base_answer = parse_answer(base_text)
            tuned_answer = parse_answer(tuned_text)
            base_ok = base_answer == question["answer"]
            tuned_ok = tuned_answer == question["answer"]
            base_corrects.append(base_ok)
            tuned_corrects.append(tuned_ok)
            rows.append(
                {
                    "prompt": [dict(m) for m in messages],
                    "domain": question.get("domain"),
                    "answer": question["answer"],
                    "base": base_text,
                    "tuned": tuned_text,
                    "base_parsed": base_answer,
                    "tuned_parsed": tuned_answer,
                    "base_correct": base_ok,
                    "tuned_correct": tuned_ok,
                }
            )
        total = len(rows)
        base_correct = sum(1 for ok in base_corrects if ok)
        tuned_correct = sum(1 for ok in tuned_corrects if ok)
        delta = (tuned_correct - base_correct) / total if total else 0.0
        se = delta_standard_error(base_corrects, tuned_corrects)
        return CapabilityOutcome(
            ok=True,
            version=version,
            total=total,
            base_correct=base_correct,
            tuned_correct=tuned_correct,
            delta=delta,
            delta_se=se,
            large_regression=(base_correct - tuned_correct)
            >= LARGE_REGRESSION_QUESTIONS,
            decoding=dict(decoding),
            rows=rows,
            selection=None if selection is None else dict(selection),
        )
    except Exception as e:  # noqa: BLE001 - the run must survive the extra step
        return CapabilityOutcome(
            ok=False,
            version=version,
            decoding=dict(decoding),
            reason=f"{type(e).__name__}: {e}",
            selection=None if selection is None else dict(selection),
        )
