"""Classify one line of a machine's output into an event.

A pure function over a string. No I/O, no database, no framework imports — so
it can be tested against real training output directly, and so Phase B can move
what calls it without moving what it does.

The job is narrow: the training framework says everything it knows in prose,
and a few of those numbers — loss, epoch, step — are the ones a user watching a
job is actually asking about. Lifting them into structured fields at the moment
the line is read means a chart is a read-only addition later rather than a
second pass over the text, and text is the worst thing to build a chart on: it
changes whenever the framework changes its mind about formatting, and by then
the job is over and the line is gone.

**Two kinds of log dict are promoted.** Once per logging step, transformers'
progress callback writes the training log dict as a Python repr —
``{'loss': 1.9042, 'grad_norm': 3.99, 'learning_rate': 4.5e-05, 'epoch': 0.13}``.
A payload counts as that line only if it carries a `'loss'` key, and the quotes
are load-bearing: evaluation writes `'eval_loss'` and the end-of-training
summary writes `'train_loss'`, so neither is mistaken for a training step, and
neither contributes the `'epoch'` sitting beside it. Requiring the braces is
what keeps prose that merely mentions a loss out of the metrics, and a line cut
in half by a chunked read has no closing brace and so matches nothing.

The evaluation row is the held-out signal (issue #53): the trainer evaluates
on the held-out split, so ``{'eval_loss': 1.2, 'eval_runtime': 4.0,
'epoch': 1.0}`` is promoted as a metric carrying **`held_out_loss`** -- the
second series a loss chart is built from. Its epoch is promoted with it,
deliberately: this is a second series, not an intruder in the first, and the
live view charts the two together. `'train_loss'` stays a log line: it is a
summary of the run, not a point in either series.

**Grad norm, learning rate and throughput are deliberately not promoted.** Each
is diagnostic rather than progress — a user watching a job wants to know how far
through it is and whether the model is learning; a learning-rate schedule is
something you read afterwards, when it went wrong. They stay in log output, and
promoting one later is a change to this file alone.

**The progress bar was rejected as a source of the step number, and that is a
correction.** The step is the one number a user asks for that the log dict may
not carry, and tqdm has it as ``n/total``. The first version of this file took
it from any bar drawn without a description, on the reasoning that tokenisation
and download bars all label theirs (`Map:`, `model.safetensors:`) while the
training bar does not. That is true and it is not enough: transformers builds
the **evaluation** bar without a description either, and this trainer sets
`val_set_size` to 0.05 by default, so an eval bar is drawn every epoch and is
textually identical to the training bar. Promoting it would have walked the step
counter backwards once per epoch and changed the total underneath it — a false
metric, which is worse than a missing one, because a user reading a chart cannot
tell that the number came from a line that only looked like a measurement. So a
step is promoted only when the framework states it as a field. Getting it
reliably otherwise means a structured callback inside the trainer rather than a
parser outside it, which is Phase B work.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

METRIC = "metric"
LOG = "log"

# Promoted into structured fields, as (source key on the line, field on the
# event): `step` and `global_step` are the same number under the two names the
# framework has used for it; whichever appears wins.
FLOAT_FIELDS = (("loss", "loss"), ("epoch", "epoch"))
# The evaluation row (issue #53): the framework writes `eval_loss`, and it is
# exposed as `held_out_loss` -- the domain's word for the signal, the one the
# checkpoint record already uses. Its epoch is promoted too: the held-out
# series is the chart's second line, not an intruder in the first.
EVAL_FLOAT_FIELDS = (("eval_loss", "held_out_loss"), ("epoch", "epoch"))
STEP_FIELDS = ("step", "global_step")

# A number as Python prints one: optional sign, optional decimals, optional
# exponent. `nan` and `inf` deliberately do not match — a non-finite loss is
# real information but it is not a point on a chart, and it survives in the log
# line either way. The other fields on that line are still promoted.
_NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

# The framework's log dict, as a repr. Non-nesting on purpose: a payload that
# contains a nested dict does not match and stays a log line. The training log
# dict is flat, and mis-parsing a nested one would cost more than missing it.
_PAYLOAD = re.compile(r"\{[^{}]*\}")

# What marks a payload as a *training* step's log line rather than an
# evaluation row or the end-of-training summary. See the module docstring.
_IS_TRAINING_LOG = re.compile(r"['\"]loss['\"]\s*:")
# What marks a payload as an evaluation row (issue #53): `eval_loss`, the
# measurement taken on the held-out split. `'train_loss'` matches neither.
_IS_EVAL_LOG = re.compile(r"['\"]eval_loss['\"]\s*:")


@dataclass(frozen=True)
class Event:
    """One classified line: what kind of event it is, and what it carried.

    `message` is always the original line. Structuring the numbers must not
    cost the reader the line they arrived in — the prose is what makes an
    unexpected number explicable.
    """

    kind: str
    message: str
    data: dict[str, Any] | None = None


def _finite_number(payload: str, key: str) -> float | None:
    """The value of `key`, or None if it is absent, malformed or non-finite.

    The number may be quoted or bare. Plain transformers writes the log dict
    with live values -- ``{'loss': 1.9042, 'epoch': 0.13}`` -- but Axolotl's
    callback formats every value to a fixed precision first and hands on the
    **strings**: ``{'loss': '0.7157', 'epoch': '0.1333'}``. Reading only the
    bare form promoted nothing at all on a real run while every unit test that
    fed it transformers-shaped lines went on passing.

    The quote is captured and required again after the number, so a value that
    merely starts with digits (``'0.7 (est)'``) matches nothing rather than
    being silently truncated into a measurement.
    """
    m = re.search(
        rf"['\"]{re.escape(key)}['\"]\s*:\s*(['\"]?)({_NUMBER})\1", payload
    )
    if not m:
        return None
    value = float(m.group(2))
    return value if math.isfinite(value) else None


def _training_log_payload(line: str, marker: re.Pattern[str]) -> str | None:
    for m in _PAYLOAD.finditer(line):
        if marker.search(m.group(0)):
            return m.group(0)
    return None


def _promote(
    payload: str, fields: tuple[tuple[str, str], ...]
) -> dict[str, Any]:
    """Lift the numbers a payload carries into structured fields.

    Each entry is (source key, field name): the framework writes `eval_loss`,
    the event carries `held_out_loss` -- the same number under the domain's
    name.
    """
    data: dict[str, Any] = {}
    for source, target in fields:
        value = _finite_number(payload, source)
        if value is not None:
            data[target] = value
    for key in STEP_FIELDS:
        value = _finite_number(payload, key)
        if value is not None:
            data["step"] = int(value)
            break
    return data


def classify(line: str) -> Event:
    """Map one output line to a `log` or `metric` event.

    A training log line is a metric carrying `loss`; an evaluation row is a
    metric carrying `held_out_loss` (issue #53). A payload that named a loss
    and stated no usable number — a truncated line, or a job whose loss has
    gone non-finite — stays a log line.
    """
    payload = _training_log_payload(line, _IS_TRAINING_LOG)
    if payload is None:
        payload = _training_log_payload(line, _IS_EVAL_LOG)
        if payload is None:
            return Event(LOG, line)
        data = _promote(payload, EVAL_FLOAT_FIELDS)
    else:
        data = _promote(payload, FLOAT_FIELDS)

    if not data:
        return Event(LOG, line)
    return Event(METRIC, line, data)
