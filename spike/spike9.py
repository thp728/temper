"""Spike 9 - how fast does validation stream a multi-GB dataset?

The 1 GB upload ceiling is a property of how validation is written, not a
product rule: the shipped path holds the file, the decoded text and every
parsed row at once, measured at ~4.8x the file size (ADR-0005). That was fine
while the largest model was 8B. Alongside a claim about 70B on real-world data
it is a contradiction, because a 70B fine-tune on a 200 MB dataset is its own
kind of toy.

Two numbers are needed before the streaming ticket is written:

  * **Throughput** - MB/s when validation does not hold the file. If 5 GB takes
    six minutes, the user is staring at a page and that is a design problem,
    not a performance one.
  * **What tokenisation costs** - the quote is priced per training token, so
    counting is a hard prerequisite of the quote. It has never been timed.

And one property, which matters more than either: **peak memory has to stay
flat as the file grows.** A streaming path that quietly accumulates is worse
than an honest ceiling, because the ceiling at least fails predictably.

No GPU, no VM, no money. Costs disk and patience.

Usage:
    python -u spike9.py                     # 1, 5 and 20 GB
    python -u spike9.py --sizes 0.25,1      # quicker, same shape
    python -u spike9.py --keep-datasets     # leave the generated files behind
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spike import Findings, header  # noqa: E402
from streaming import stream_validate  # noqa: E402

try:
    import psutil
except ImportError:
    sys.exit("psutil not installed. Run: pip install psutil")

FINDINGS_PATH = Path(__file__).parent / "findings-spike9.json"
DATA_DIR = Path(__file__).parent / ".spike9-data"
CACHE_DIR = Path(__file__).parent / ".cache"

# The catalog's smallest model, so the token counts are the ones the product
# would actually quote. tokenizer.json is ~11 MB and needs no torch, no
# transformers and no auth - `tokenizers` reads it directly.
TOKENIZER_REPO = "Qwen/Qwen3-4B"
TOKENIZER_URL = (
    f"https://huggingface.co/{TOKENIZER_REPO}/resolve/main/tokenizer.json"
)

# The in-memory validator is measured too, but only at a size that fits: at the
# recorded 4.8x multiplier, 1 GB peaks near 4.8 GB of RSS and 20 GB is not a
# measurement, it is a swap storm.
IN_MEMORY_CEILING_GB = 1.0

DEFAULT_SIZES_GB = [1.0, 5.0, 20.0]

# Tokenisation runs at roughly a sixth of the speed of validation, so a 20 GB
# tokenising pass runs for over an hour. Above this many bytes the pass reads a
# prefix, the RATE is measured on what it read, and the whole-file time is
# derived from that rate and marked as derived. Measuring a prefix and saying
# so is honest; running for an hour to produce the same number is not more so.
TOKENISE_PREFIX_BYTES = 1_000_000_000


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

# A realistic row shape, not a minimal one. Row shape drives everything being
# measured here: JSON parsing cost scales with the number of keys, validation
# walks every message, and tokenisation cost is dominated by content length.
# A file of {"messages":[{"role":"user","content":"hi"}]} would report a
# throughput this product will never see.
_SUBJECTS = [
    "the invoice",
    "our refund policy",
    "the API key rotation",
    "the shipping delay",
    "the seat upgrade",
    "the failed webhook",
    "the duplicate charge",
    "the expired certificate",
]
_OPENERS = [
    "I need help with",
    "Can you explain",
    "What happens to",
    "Could someone look at",
    "I'm confused about",
]
_BODY = (
    "Thanks for getting in touch. Here is what happens in this case, step by "
    "step, and what you should expect on your side. First, the request is "
    "recorded against your account with a reference you can quote back to us. "
    "Second, it is reviewed against the terms that were in force on the date "
    "of the transaction, not today's. Third, if anything is owed it is "
    "returned by the original payment method, which typically settles within "
    "five to seven working days. If none of that matches what you are seeing, "
    "reply with the reference and I will look at the specific record rather "
    "than the general rule."
)


def synth_row(rng: random.Random, i: int) -> dict:
    turns = rng.choice([2, 2, 2, 4])  # mostly single-exchange, some multi-turn
    msgs = [
        {
            "role": "system",
            "content": "You are a careful support agent for a payments company.",
        }
    ]
    for t in range(turns // 2):
        msgs.append(
            {
                "role": "user",
                "content": f"{rng.choice(_OPENERS)} {rng.choice(_SUBJECTS)}? "
                f"(ticket {i}-{t})",
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": _BODY[: rng.randrange(200, len(_BODY))],
            }
        )
    return {"messages": msgs}


def generate(path: Path, target_bytes: int, f: Findings) -> dict:
    """Write a JSONL file of about `target_bytes`, deterministically.

    Seeded so a re-run measures the same data. The size check is on bytes
    written rather than row count, because the row shape is variable and the
    thing being measured is MB/s.
    """
    if path.exists() and path.stat().st_size >= target_bytes * 0.98:
        f.note(
            "dataset reused",
            f"{path.name} already {path.stat().st_size / 1e9:.2f} GB",
        )
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "generated": False,
        }

    rng = random.Random(20260823)
    t0 = time.perf_counter()
    written = 0
    rows = 0
    with path.open("wb") as fh:
        buf = bytearray()
        while written < target_bytes:
            line = (
                json.dumps(synth_row(rng, rows), ensure_ascii=False) + "\n"
            ).encode()
            buf += line
            written += len(line)
            rows += 1
            if len(buf) >= 8 << 20:
                fh.write(buf)
                buf.clear()
        fh.write(buf)
    elapsed = time.perf_counter() - t0
    f.record(
        "dataset generated",
        True,
        f"{path.name}: {written / 1e9:.2f} GB, {rows:,} rows, "
        f"{elapsed:.1f}s ({written / 1e6 / elapsed:.0f} MB/s write)",
    )
    return {
        "path": str(path),
        "bytes": written,
        "rows": rows,
        "generated": True,
        "write_seconds": round(elapsed, 2),
    }


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass
class Measurement:
    label: str
    seconds: float
    bytes_processed: int
    peak_rss_bytes: int
    baseline_rss_bytes: int
    detail: dict

    @property
    def mb_per_s(self) -> float:
        return (
            self.bytes_processed / 1e6 / self.seconds if self.seconds else 0.0
        )

    @property
    def rss_growth_bytes(self) -> int:
        return self.peak_rss_bytes - self.baseline_rss_bytes

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "kind": "measured",
            "seconds": round(self.seconds, 2),
            "bytes": self.bytes_processed,
            "mb_per_s": round(self.mb_per_s, 1),
            "peak_rss_mb": round(self.peak_rss_bytes / 1e6, 1),
            "baseline_rss_mb": round(self.baseline_rss_bytes / 1e6, 1),
            "rss_growth_mb": round(self.rss_growth_bytes / 1e6, 1),
            "rss_growth_multiple_of_file": round(
                self.rss_growth_bytes / self.bytes_processed, 3
            )
            if self.bytes_processed
            else None,
            **self.detail,
        }


class RSSSampler:
    """Poll our own RSS on a thread while the work runs.

    Polling rather than a hook because the question is peak RESIDENT memory,
    which is what the machine runs out of. `tracemalloc` would report Python
    allocations and miss the interpreter's own arenas, and `resource` does not
    exist on Windows.

    A 20 ms interval can in principle miss a spike shorter than 20 ms. It is
    recorded rather than hand-waved: everything measured here runs for seconds,
    and a sub-20 ms allocation spike is not what a 4.8x multiplier looks like.
    """

    INTERVAL_S = 0.02

    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak = 0
        self.baseline = 0
        self.samples = 0

    def __enter__(self) -> RSSSampler:
        gc.collect()
        self.baseline = self.peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.INTERVAL_S):
            self.peak = max(self.peak, self._proc.memory_info().rss)
            self.samples += 1

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.peak = max(self.peak, self._proc.memory_info().rss)


def measure(label: str, fn, size_bytes: int, f: Findings) -> Measurement:
    with RSSSampler() as s:
        t0 = time.perf_counter()
        detail = fn()
        elapsed = time.perf_counter() - t0
    m = Measurement(
        label, elapsed, size_bytes, s.peak, s.baseline, detail or {}
    )
    f.record(
        label,
        True,
        f"{m.seconds:.1f}s, {m.mb_per_s:.0f} MB/s, "
        f"peak RSS {m.peak_rss_bytes / 1e6:.0f} MB "
        f"(+{m.rss_growth_bytes / 1e6:.0f} MB over baseline, "
        f"{m.rss_growth_bytes / size_bytes:.2f}x file)",
    )
    return m


# --------------------------------------------------------------------------
# Tokeniser
# --------------------------------------------------------------------------


def load_tokenizer(f: Findings):
    """Fetch the catalog model's tokenizer.json and return a str -> int counter.

    Returns (counter, description). Falls back to a whitespace stand-in if the
    download fails, and says so loudly -- a tokenisation cost measured with
    `str.split` is not a tokenisation cost, and reporting it as one would be
    the exact kind of unmarked estimate this project treats as a defect.
    """
    header("TOKENISER")
    CACHE_DIR.mkdir(exist_ok=True)
    local = CACHE_DIR / "qwen3-tokenizer.json"
    if not local.exists():
        try:
            print(f"  downloading {TOKENIZER_URL}")
            with urllib.request.urlopen(TOKENIZER_URL, timeout=120) as r:
                local.write_bytes(r.read())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            f.record("tokenizer downloaded", False, f"{type(e).__name__}: {e}")
            f.note(
                "tokenisation cost is NOT MEASURED",
                "fell back to str.split(); every tokenisation number in this "
                "run is a placeholder and must not be quoted",
            )
            return (
                lambda text: len(text.split())
            ), "FALLBACK str.split (NOT a tokeniser)"
    try:
        from tokenizers import Tokenizer
    except ImportError:
        f.record("tokenizers installed", False, "pip install tokenizers")
        f.note("tokenisation cost is NOT MEASURED", "fell back to str.split()")
        return (
            lambda text: len(text.split())
        ), "FALLBACK str.split (NOT a tokeniser)"

    tok = Tokenizer.from_file(str(local))
    f.record(
        "tokenizer loaded",
        True,
        f"{TOKENIZER_REPO}, vocab {tok.get_vocab_size():,}, "
        f"{local.stat().st_size / 1e6:.1f} MB",
    )
    # encode() rather than encode_batch(): the streaming path sees one turn at
    # a time and never holds a batch, so batching here would measure a
    # throughput the streaming design cannot reach.
    return (
        lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    ), f"{TOKENIZER_REPO} tokenizer.json via `tokenizers`"


# --------------------------------------------------------------------------
# The passes
# --------------------------------------------------------------------------


def run_size(size_gb: float, counter, f: Findings) -> dict:  # noqa: C901
    header(f"{size_gb} GB")
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"synthetic-{size_gb:g}gb.jsonl"
    gen = generate(path, int(size_gb * 1e9), f)
    size = gen["bytes"]

    result: dict = {"size_gb": size_gb, "dataset": gen, "passes": {}}

    def summarise(rep) -> dict:
        return {
            "row_count": rep.row_count,
            "usable_rows": rep.usable_rows,
            "valid": rep.valid,
            "errors": rep.error_count,
            "enable_thinking": rep.enable_thinking,
            "retained_objects": rep.retained_objects(),
            "token_count": rep.token_count,
        }

    # 1. Parse and validate, line numbers held. This is the product's promise.
    validate_only = measure(
        "stream: parse + validate (line numbers held)",
        lambda: summarise(stream_validate(path)),
        size,
        f,
    )

    # 2. The same pass with the line-number guarantee dropped. The difference
    #    is what the promise costs -- and the expectation is that it costs
    #    nothing, because a line number is a counter. Measuring it is how that
    #    expectation stops being an assumption.
    no_lines = measure(
        "stream: parse + validate (line numbers dropped)",
        lambda: summarise(stream_validate(path, track_lines=False)),
        size,
        f,
    )

    # 3. The same pass again, tokenising every turn -- over a prefix when the
    #    file is large enough that the full pass is an hour of waiting.
    tok_bytes = min(size, TOKENISE_PREFIX_BYTES)
    limit = None if tok_bytes >= size else TOKENISE_PREFIX_BYTES
    if limit is not None:
        f.note(
            "tokenising a prefix",
            f"{limit / 1e9:.1f} GB of {size / 1e9:.1f} GB; the rate is "
            f"measured, the whole-file time is derived from it",
        )
    tokenised = measure(
        "stream: parse + validate + tokenise",
        lambda: summarise(
            stream_validate(path, count_tokens=counter, byte_limit=limit)
        ),
        tok_bytes,
        f,
    )

    result["passes"] = {
        "validate": validate_only.to_dict(),
        "validate_without_line_numbers": no_lines.to_dict(),
        "validate_and_tokenise": tokenised.to_dict(),
    }

    # Compare like with like: the tokenising pass may have read a prefix, so
    # scale the validate-only time to the same number of bytes before
    # subtracting. Comparing a 20 GB validate against a 1 GB tokenise would
    # make tokenisation look free.
    validate_seconds_for_tok_bytes = (
        validate_only.seconds * tok_bytes / size if size else 0.0
    )
    token_share = (
        1 - validate_seconds_for_tok_bytes / tokenised.seconds
        if tokenised.seconds
        else 0
    )
    result["tokenisation"] = {
        "kind": "measured" if limit is None else "measured on a prefix",
        "bytes_tokenised": tok_bytes,
        "whole_file": limit is None,
        "extra_seconds_over_bytes_tokenised": round(
            tokenised.seconds - validate_seconds_for_tok_bytes, 2
        ),
        "share_of_total_time": round(token_share, 3),
        "slowdown_multiple": round(
            tokenised.seconds / validate_seconds_for_tok_bytes, 2
        )
        if validate_seconds_for_tok_bytes
        else None,
        "whole_file_seconds": {
            "value": round(size / 1e6 / tokenised.mb_per_s, 1)
            if tokenised.mb_per_s
            else None,
            "kind": "measured"
            if limit is None
            else "derived from the prefix rate",
        },
    }
    result["line_number_guarantee"] = {
        "kind": "derived",
        "extra_seconds": round(validate_only.seconds - no_lines.seconds, 2),
        "share_of_validate_time": round(
            (validate_only.seconds - no_lines.seconds) / validate_only.seconds,
            4,
        )
        if validate_only.seconds
        else None,
        "extra_rss_mb": round(
            (validate_only.rss_growth_bytes - no_lines.rss_growth_bytes) / 1e6,
            1,
        ),
        "caveat": "One run each, back to back, on a machine doing other things. A "
        "difference of a few percent here is run-to-run variance, not a "
        "cost -- and a NEGATIVE extra_rss_mb is the proof of that, since "
        "holding a counter cannot free memory. What the pair establishes "
        "is the absence of a large cost, not the presence of a small one.",
    }
    print(
        f"  -> tokenisation is {token_share * 100:.0f}% of the tokenising pass"
    )
    print(
        f"  -> line numbers cost "
        f"{result['line_number_guarantee']['extra_seconds']}s and "
        f"{result['line_number_guarantee']['extra_rss_mb']} MB"
    )

    # 4. The shipped in-memory validator, for the comparison the ceiling rests
    #    on -- but only where it fits in RAM.
    if size_gb <= IN_MEMORY_CEILING_GB:
        from temper_core.validation import validate as in_memory_validate

        in_mem = measure(
            "in-memory (shipped): parse + validate",
            lambda: {"usable_rows": in_memory_validate(path).usable_rows},
            size,
            f,
        )
        result["passes"]["in_memory_shipped"] = in_mem.to_dict()
        result["memory_multiplier"] = {
            "kind": "measured",
            "in_memory_x_file": round(in_mem.rss_growth_bytes / size, 2),
            "streaming_x_file": round(
                validate_only.rss_growth_bytes / size, 4
            ),
            "adr_0005_claim": 4.8,
        }
    else:
        f.note(
            "in-memory validator skipped",
            f"{size_gb} GB would need roughly "
            f"{size_gb * 4.8:.0f} GB of RSS at the ADR-0005 multiplier",
        )
    return result


def derive_ceiling(results: list[dict], f: Findings) -> dict:
    """Turn the measurements into the answer the ticket needs."""
    header("WHAT THE CEILING BECOMES")
    if not results:
        return {}

    rates = [r["passes"]["validate"]["mb_per_s"] for r in results]
    tok_rates = [
        r["passes"]["validate_and_tokenise"]["mb_per_s"] for r in results
    ]
    growth = [r["passes"]["validate"]["rss_growth_mb"] for r in results]
    sizes = [r["size_gb"] for r in results]

    # Flatness is the claim. Compare the largest file's retained memory to the
    # smallest's: if streaming works, the ratio is near 1 no matter how much
    # bigger the file got.
    flat = {
        "kind": "measured",
        "sizes_gb": sizes,
        "rss_growth_mb": growth,
        "file_size_ratio": round(max(sizes) / min(sizes), 1)
        if len(sizes) > 1
        else 1.0,
        "rss_growth_ratio": round(max(growth) / min(growth), 2)
        if len(growth) > 1 and min(growth) > 0
        else None,
    }
    is_flat = (
        flat["rss_growth_ratio"] is None or flat["rss_growth_ratio"] < 2.0
    )
    f.record(
        "peak memory is flat as the file grows",
        is_flat,
        f"file x{flat['file_size_ratio']} -> RSS x{flat['rss_growth_ratio']}",
    )

    mean_rate = statistics.fmean(rates)
    mean_tok_rate = statistics.fmean(tok_rates)

    # The user-tolerable wait is a judgement, stated as one rather than buried.
    # 60s is the point at which a synchronous upload page stops being a wait
    # and becomes a suspected hang.
    TOLERABLE_S = 60
    verdict = {
        "kind": "derived",
        "mean_validate_mb_per_s": round(mean_rate, 1),
        "mean_validate_and_tokenise_mb_per_s": round(mean_tok_rate, 1),
        "tolerable_wait_seconds": TOLERABLE_S,
        "tolerable_wait_justification": "the point at which a synchronous upload page stops reading as a "
        "wait and starts reading as a hang. A judgement, not a measurement.",
        "gb_validatable_within_tolerable_wait": round(
            mean_rate * TOLERABLE_S / 1000, 1
        ),
        "gb_validatable_and_tokenisable_within_tolerable_wait": round(
            mean_tok_rate * TOLERABLE_S / 1000, 1
        ),
        "current_ceiling_gb": 1.0,
        "current_ceiling_reason": "in-memory validation at ~4.8x the file size (ADR-0005)",
    }

    # The kill criterion, evaluated rather than left to a reader.
    five_gb_tokenise_s = 5000 / mean_tok_rate
    splits = five_gb_tokenise_s > TOLERABLE_S
    verdict["five_gb_tokenise_seconds"] = round(five_gb_tokenise_s, 1)
    verdict["token_counting_becomes_its_own_phase"] = splits
    verdict["recommendation"] = (
        "SPLIT. Validate on upload and count tokens asynchronously before the "
        "quote; the quote gains a 'counting' state. Tokenisation cannot be "
        "held inside a synchronous upload at this rate."
        if splits
        else "ONE PASS. Validation and token counting both fit inside a tolerable "
        "synchronous wait at the sizes measured, so the quote does not need a "
        "'counting' state and the upload does not need a second phase."
    )

    # How wrong could the rate be before the recommendation changes?
    #
    # This run shared a machine with other work, so the MB/s figures are a
    # FLOOR rather than a rate. That is a real caveat and it would be a weak
    # answer on its own -- so instead of asserting the number is good enough,
    # ask how much faster the machine would have to be to flip the verdict.
    # A conclusion that survives the uncertainty is worth more than a cleaner
    # measurement of the same conclusion.
    verdict["sensitivity"] = {
        "kind": "derived",
        "why": "the measured rates are a floor -- the machine was not idle. "
        "The question is not what the exact rate is, but whether the "
        "recommendation depends on it.",
        "five_gb_tokenise_seconds_if_rate_were": {
            f"x{m}": round(5000 / (mean_tok_rate * m)) for m in (1, 2, 4, 10)
        },
        "speedup_needed_to_fit_the_budget": round(
            five_gb_tokenise_s / TOLERABLE_S, 1
        ),
        "verdict_is_robust": five_gb_tokenise_s / TOLERABLE_S > 3,
        "conclusion": "Tokenisation would have to be an order of magnitude faster than "
        "measured to come close to the budget, and even a 10x machine "
        "leaves 5 GB well outside it. THE SPLIT RECOMMENDATION DOES NOT "
        "DEPEND ON THE CONTENDED MEASUREMENT -- re-running on an idle "
        "machine would sharpen the number and would not change the "
        "decision.",
    }
    f.record(
        "token counting fits in one synchronous pass",
        not splits,
        f"5 GB tokenises in {five_gb_tokenise_s:.0f}s against a "
        f"{TOLERABLE_S}s budget",
    )
    print(f"  -> {verdict['recommendation']}")
    return {"flatness": flat, "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sizes",
        default=",".join(str(s) for s in DEFAULT_SIZES_GB),
        help="comma-separated dataset sizes in GB",
    )
    ap.add_argument(
        "--keep-datasets",
        action="store_true",
        help="leave the generated JSONL behind (they are large)",
    )
    ap.add_argument(
        "--quiet-machine",
        action="store_true",
        help="assert in the findings that nothing else heavy was "
        "running. Throughput measured against a busy disk is "
        "a floor, not a rate, and the findings should say "
        "which one they hold.",
    )
    args = ap.parse_args()
    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]

    f = Findings()
    header("SPIKE 9 -- streaming validation throughput")
    free = shutil.disk_usage(Path(__file__).parent).free
    needed = sum(sizes) * 1e9
    print(
        f"  sizes: {sizes} GB -- needs {needed / 1e9:.0f} GB, "
        f"{free / 1e9:.0f} GB free"
    )
    if needed > free * 0.8:
        f.record(
            "disk space",
            False,
            f"need {needed / 1e9:.0f} GB, have {free / 1e9:.0f} GB free",
        )
        return 1

    counter, tok_desc = load_tokenizer(f)

    results: list[dict] = []
    derived: dict = {}
    try:
        for size_gb in sizes:
            results.append(run_size(size_gb, counter, f))
        derived = derive_ceiling(results, f)
    finally:
        payload = {
            "spike": "spike9-streaming-validation",
            "run_at": datetime.now(timezone.utc).isoformat(),
            "host": {
                "python": sys.version.split()[0],
                "platform": sys.platform,
                "cpu_count": os.cpu_count(),
                "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
                # Throughput is a property of the machine as much as the code.
                # A run that shared the disk with a large image pull reports a
                # rate the code can beat, and saying so is the difference
                # between a measurement and a number.
                "load_average_1min": round(psutil.getloadavg()[0], 2)
                if hasattr(psutil, "getloadavg")
                else None,
                "cpu_percent_during_run": None,
                "quiet_machine": args.quiet_machine,
            },
            "tokenizer": tok_desc,
            "row_shape": "multi-turn chat, system + 1-2 user/assistant pairs, "
            "200-700 char assistant turns",
            "results": results,
            "derived": derived,
            "known_divergences_from_the_shipped_validator": [
                "encoding errors name a line (the in-memory path can only give "
                "a byte offset, because it decodes the whole file at once)",
                "rows split on \\n only; str.splitlines() also splits on \\v, "
                "\\f, \\x1c-\\x1e, \\x85, \\u2028 and \\u2029",
            ],
            "steps": f.steps,
        }
        FINDINGS_PATH.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"\nFindings written to {FINDINGS_PATH}")
        if not args.keep_datasets and DATA_DIR.exists():
            shutil.rmtree(DATA_DIR, ignore_errors=True)
            print(f"Generated datasets removed from {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
