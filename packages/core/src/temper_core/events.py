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

**Progress is promoted, not filtered (issue #49).** The machine's loudest
output — docker's layer-pull lines and the model-download bars, hundreds per
pull — was slated to be filtered at classification so it never became events, a
plan that discards evidence. Those lines carry bytes-so-far and bytes-total:
the noise *is* the progress data. They are promoted into a `progress` record
carrying phase and bytes, which supersedes rather than accumulates, while the
raw lines are retained as collapsed detail so nothing is discarded. The phase
that dominates a large job — downloading weights — gains the signal it
currently lacks entirely. See `temper_core.progress`.

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

from temper_core.progress import (
    PHASE_IMAGE_PULL,
    PHASE_MODEL_DOWNLOAD,
    parse_size,
)

METRIC = "metric"
LOG = "log"
PROGRESS = "progress"

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


# --- progress (issue #49) ---------------------------------------------------
#
# The machine's loud output is the progress data. Two shapes are promoted into
# a progress record instead of a log line, so the finished-job page stops
# drowning in them and the phases a large job is dominated by gain a signal:
#
# * Docker's classic pull output -- a 12-hex layer id, a status word, and for
#   the byte-carrying statuses a bar and a done/total pair. Layers pull in
#   parallel, so every line names one layer, and the phase's figures are the
#   aggregate across layers (see temper_core.progress).
# * huggingface_hub's tqdm download bars -- `model.safetensors: 10%|█ | 400M/
#   4.00G [00:05<00:45]`, the description naming a file and the n/total
#   carrying byte sizes.
#
# The delimiter that keeps a training bar (` 33%|███ | 10/30 [...]`) and the
# tokenization map (`Map: 45%|███ | 90/200 [...]`) out of progress is the
# file-name rule: a download bar names a file (the description contains a dot);
# a map or a training bar does not. Requiring the dot is what stops example
# counts from being mistaken for bytes.

_LAYER_ID = r"[0-9a-f]{12}"
_LAYER_STATUS = (
    r"(?:Pulling fs layer|Waiting|Downloading|Download complete|"
    r"Extracting|Verifying Checksum|Pull complete|Layer already exists)"
)
_LAYER_LINE = re.compile(rf"^{_LAYER_ID}: ({_LAYER_STATUS})")
# A done/total byte pair, as docker writes it after the bar: `15.19MB/42.42MB`.
# Both halves must parse as sizes for the pair to be promoted.
_BYTE_PAIR = re.compile(
    r"(?P<done>\d+(?:\.\d+)?(?:[kKMGTP]i?B|[kKMGTP]|B)?)/"
    r"(?P<total>\d+(?:\.\d+)?(?:[kKMGTP]i?B|[kKMGTP]|B)?)"
)
# A tqdm download bar: `name.ext: 10%|█ | 400M/4.00G [00:05<00:45]`. The
# description must contain a dot -- it is a file name, which is what separates
# a download from a tokenization map.
_MODEL_DOWNLOAD = re.compile(
    r"^(?P<desc>[\w./\-]+\.[A-Za-z0-9]+):\s+\d+%\|(?P<bar>[^|]*)\|\s*"
    r"(?P<done>\d+(?:\.\d+)?(?:[kKMGTP]i?B|[kKMGTP]|B)?)/"
    r"(?P<total>\d+(?:\.\d+)?(?:[kKMGTP]i?B|[kKMGTP]|B)?)"
)


def _classify_progress(line: str) -> Event | None:
    """A layer-pull or model-download line, or None when the line is neither.

    Layer-pull lines are promoted even when the byte pair is missing or was
    cut off mid-line -- the status word alone names the phase, and letting the
    line fall back to a log event would reintroduce the flood this promotion
    exists to fold. Only the byte pair is conditional, because it is the part
    that can be truncated.
    """
    m = _LAYER_LINE.match(line)
    if m:
        data: dict[str, Any] = {
            "phase": PHASE_IMAGE_PULL,
            "layer": line[: line.index(":")],
        }
        pair = _BYTE_PAIR.search(line)
        if pair:
            done = parse_size(pair.group("done"))
            total = parse_size(pair.group("total"))
            if done is not None and total is not None:
                data["done"] = done
                data["total"] = total
        return Event(PROGRESS, line, data)
    m = _MODEL_DOWNLOAD.match(line)
    if m:
        done = parse_size(m.group("done"))
        total = parse_size(m.group("total"))
        if done is not None and total is not None:
            return Event(
                PROGRESS,
                line,
                {"phase": PHASE_MODEL_DOWNLOAD, "done": done, "total": total},
            )
    return None


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
    """Map one output line to a `log`, `metric` or `progress` event.

    A training log line is a metric carrying `loss`; an evaluation row is a
    metric carrying `held_out_loss` (issue #53); a layer-pull or model-download
    line is a progress record carrying phase and bytes (issue #49). A payload
    that named a loss and stated no usable number -- a truncated line, or a job
    whose loss has gone non-finite -- stays a log line.
    """
    progress_event = _classify_progress(line)
    if progress_event is not None:
        return progress_event
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
