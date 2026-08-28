"""The progress domain: phases, byte sizes, a live measured rate, and
per-phase superseding (issue #49).

The machine's loud output *is* the progress data: image-pull and model-download
lines carry bytes-so-far and bytes-total, and promoting them into a progress
record is the whole point of the issue. This module owns the vocabulary those
records are built from:

* **The phases.** Defined once here and read everywhere the progress record
  crosses -- the classifier emits them, the orchestrator stores them, the
  contract publishes them, and the interface renders them. A value two
  components must agree on is defined once.
* **Byte sizes.** Docker and huggingface_hub disagree in formatting (`15.19MB`
  vs `400M`) and agree in meaning: SI, 1000-based. One `parse_size` reads both,
  so a figure is the same number wherever it is drawn.
* **The measured rate.** A line rarely carries its own rate (docker's pull
  lines never do, and the captured download bar omits it), so the rate is
  measured between consecutive readings -- the live figure the issue asks for,
  the thing that corrects itself when actual throughput differs from predicted.
* **Superseding.** Progress replaces rather than appends: the tracker holds one
  record per phase, and an update replaces it. That is why progress cannot be a
  log line -- it has no step and it does not accumulate.

Image pull has one more rule: docker pulls layers in parallel, so no single
line is the image's progress. Each line names one layer's done/total, and the
phase's figures are the aggregate across every layer seen so far -- which is
what "image pull: 57MB/130MB" means.

Pure, with an injectable clock: no I/O, no framework imports, so it can be
tested against real output directly and moved without moving what calls it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

PHASE_IMAGE_PULL = "image pull"
PHASE_MODEL_DOWNLOAD = "model download"

# A size as docker (`15.19MB`) and huggingface_hub's tqdm (`400M`) write one.
# SI base (1000), the base both tools scale on; the decimal point is optional.
# The suffix is `MB`, `M`, or `B` -- docker keeps the B, tqdm drops it, and a
# bare number (a tiny file shown as `567/567`) is bytes.
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)((?:[kKMGTP]i?B|[kKMGTP]|B))?\s*$")

_UNITS = {
    "": 1.0,
    "B": 1.0,
    "k": 1e3,
    "K": 1e3,
    "kB": 1e3,
    "KB": 1e3,
    "M": 1e6,
    "MB": 1e6,
    "G": 1e9,
    "GB": 1e9,
    "T": 1e12,
    "TB": 1e12,
    "P": 1e15,
    "PB": 1e15,
}


def parse_size(text: str) -> float | None:
    """The byte count `text` names, or None if it is not a size.

    Docker writes `15.19MB`; huggingface_hub's progress bars drop the B and
    write `400M`. Both are read here as the same SI figure. Something that is
    not a size -- `90/200`, `2.00s/it` -- returns None rather than being
    silently coerced into a measurement.
    """
    m = _SIZE_RE.match(text.strip())
    if not m:
        return None
    factor = _UNITS.get(m.group(2) or "")
    if factor is None:
        return None
    return float(m.group(1)) * factor


def measure_rate(
    prev_done: float, prev_ts: float, done: float, ts: float
) -> float | None:
    """Bytes per second between two readings, or None when the interval
    cannot be measured (a zero or backwards interval measures nothing).

    The rate is measured, never read off the line: the captured download bar
    carries no rate, and docker's pull lines carry none either.
    """
    if ts <= prev_ts:
        return None
    moved = done - prev_done
    if moved <= 0:
        return None
    return moved / (ts - prev_ts)


def eta_seconds(
    done: float | None, total: float | None, rate: float | None
) -> float | None:
    """Estimated seconds remaining for the phase, from the live rate.

    None when any of the inputs that would make an estimate meaningful is
    absent -- no total, no rate, or a rate that is not moving forward. Zero
    when nothing is left.
    """
    if done is None or total is None or rate is None or rate <= 0:
        return None
    remaining = total - done
    if remaining <= 0:
        return 0.0
    return remaining / rate


@dataclass(frozen=True)
class ProgressReading:
    """One classified progress line, before measurement.

    `layer` is set for image-pull lines -- docker pulls layers in parallel, so
    the phase aggregates across layers. `done`/`total` are the line's own
    figures and may be absent (a status line like `Pull complete` carries no
    bytes).
    """

    phase: str
    done: float | None = None
    total: float | None = None
    layer: str | None = None
    ts: float | None = None


@dataclass(frozen=True)
class ProgressRecord:
    """The phase's current progress: superseded figures plus the measured rate
    and the estimate derived from it."""

    phase: str
    done: float | None
    total: float | None
    rate: float | None
    eta_s: float | None
    ts: float


class ProgressTracker:
    """One record per phase, replaced on every update (issue #49).

    Supersedes rather than accumulates -- the interface renders the latest per
    phase. The rate is measured between consecutive readings of the same
    phase, and the estimate uses it, so both correct themselves when the real
    throughput differs from whatever came before.

    The clock is injectable so a test can drive the interval between readings
    exactly.
    """

    def __init__(self, now: Callable[[], float] = time.time) -> None:
        self._now = now
        # The previous aggregated done and its timestamp, per phase: the two
        # readings the rate is measured between.
        self._prev_done: dict[str, float] = {}
        self._prev_ts: dict[str, float] = {}
        self._rate: dict[str, float] = {}
        # Per-phase layer state (image pull) and plain state (model download).
        self._layers: dict[
            str, dict[str, tuple[float | None, float | None]]
        ] = {}
        self._plain: dict[str, tuple[float | None, float | None]] = {}

    def update(self, reading: ProgressReading) -> ProgressRecord:
        ts = reading.ts if reading.ts is not None else self._now()
        done, total = self._resolve(reading)
        rate = self._measure(reading.phase, done, ts)
        return ProgressRecord(
            phase=reading.phase,
            done=done,
            total=total,
            rate=rate,
            eta_s=eta_seconds(done, total, rate),
            ts=ts,
        )

    def _resolve(
        self, reading: ProgressReading
    ) -> tuple[float | None, float | None]:
        """The phase's figures after this reading.

        For image pull, the aggregate across every layer seen so far: a layer
        that reported bytes keeps them, and the phase's done/total are the
        sums. For a phase with no layers, the reading replaces the figures --
        or keeps the previous ones when the line carried none (a status line
        does not blank the proportion that was already there).
        """
        if reading.layer is not None:
            layers = self._layers.setdefault(reading.phase, {})
            prev_done, total = layers.get(reading.layer, (None, reading.total))
            # A layer's delivered bytes only grow -- docker reports a layer's
            # done rising, then extracting, then complete -- so the max is the
            # honest figure for how much of it has arrived.
            done = (
                reading.done
                if reading.done is None
                else max(d for d in (prev_done, reading.done) if d is not None)
            )
            if reading.total is not None:
                total = reading.total
            layers[reading.layer] = (done, total)
            done = sum(d for d, _ in layers.values() if d is not None)
            totals = [t for _, t in layers.values() if t is not None]
            total = sum(totals) if totals else None
            return done, total
        prev_done, prev_total = self._plain.get(reading.phase, (None, None))
        done = reading.done if reading.done is not None else prev_done
        total = reading.total if reading.total is not None else prev_total
        self._plain[reading.phase] = (done, total)
        return done, total

    def _measure(
        self, phase: str, done: float | None, ts: float
    ) -> float | None:
        if done is None:
            # A status line measured nothing; the phase's last rate stands.
            return self._rate.get(phase)
        prev_done = self._prev_done.get(phase)
        prev_ts = self._prev_ts.get(phase)
        self._prev_done[phase] = done
        self._prev_ts[phase] = ts
        if prev_done is None or prev_ts is None:
            return self._rate.get(phase)
        measured = measure_rate(prev_done, prev_ts, done, ts)
        if measured is None:
            return self._rate.get(phase)
        self._rate[phase] = measured
        return measured
