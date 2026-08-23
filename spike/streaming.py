"""A streaming rewrite of `api.validation`, for spike 9 to measure.

**This is a probe, not the product.** It exists so spike 9 can put numbers on
two questions before the streaming-validation ticket is written: how fast
validation runs when it does not hold the file, and how much of that time is
tokenisation. If the numbers say the ticket is worth writing, this file is the
draft the ticket starts from -- it is deliberately shaped like product code
rather than like a benchmark, so the measurement is of the real work.

The shipped validator reads the file into `bytes`, decodes it into one `str`,
and builds a `list[dict]` of every row. That is where the 1 GB upload ceiling
comes from. ADR-0005 derived that ceiling from a 4.8x memory multiplier;
**spike 9 measured 5.93x at a real 1 GB file**, so the recorded figure was
extrapolated from small files and was optimistic. Either way the ceiling is a
property of this code, not a product rule, and the fix is to never hold more
than one row: measured here at +4 MB, flat from 1 GB to 20 GB.

Three things have to survive the rewrite, and each is pinned by a test in
`test_streaming.py`:

* **The same verdict.** A streaming pass that is fast because it checks less is
  not a measurement of anything. `assert_agrees` compares both validators row
  for row and code for code.
* **The line-number guarantee.** Every error names its line. That is free while
  streaming -- a counter -- but the SAMPLES are not: the shipped thinking-mode
  detector accumulates one line number per assistant turn before it caps them,
  which is bounded by the file. Capping as we go is the difference.
* **Flat memory.** Everything retained is capped: the first few errors, the
  first few warnings, a three-row preview, twenty key sets, and ten sample line
  numbers. Nothing else outlives its row.

Two deliberate divergences from the shipped validator, both recorded rather
than hidden:

* **Encoding errors name a line.** The in-memory path decodes the whole file at
  once, so it can only report a byte offset with `line: None`. Decoding a line
  at a time can say which line, and that is strictly better for the user.
* **Rows are split on `\\n` only.** `str.splitlines()` also splits on `\\v`,
  `\\f`, `\\x1c`-`\\x1e`, `\\x85`, `\\u2028` and `\\u2029`, so a JSON string
  containing a literal U+2028 is two rows to the shipped validator and one row
  here. JSONL is newline-delimited by definition; this is a divergence in the
  shipped code's favour of accident rather than intent, and it is noted in
  findings-spike9.json rather than quietly matched.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

sys.path.insert(0, str(Path(__file__).parent.parent / "trainer"))
from thinking import THINK_OPEN  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.validation import MIN_ROWS, RECOMMENDED_ROWS, MAX_PREVIEW  # noqa: E402

DEFAULT_CHUNK_BYTES = 1024 * 1024
DEFAULT_ERROR_CAP = 100
DEFAULT_SAMPLE_CAP = 5
SCHEMA_KEY_SAMPLE = 20


@dataclass
class Issue:
    line: int | None
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"line": self.line, "code": self.code, "message": self.message}


@dataclass
class StreamReport:
    """Everything the shipped Report carries, plus what capping costs.

    `error_count` and `errors_suppressed` exist because the list is capped: a
    user whose file is broken on every line needs to know it is every line, and
    printing every line is what the cap prevents.
    """

    valid: bool = False
    row_count: int = 0
    usable_rows: int = 0
    schema_type: str | None = None
    enable_thinking: bool | None = None
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    preview: list[dict] = field(default_factory=list)

    error_count: int = 0
    warning_count: int = 0
    errors_suppressed: int = 0
    warnings_suppressed: int = 0

    turns_with_think: int = 0
    turns_without_think: int = 0
    thinking_lines_with: list[int] = field(default_factory=list)
    thinking_lines_without: list[int] = field(default_factory=list)

    token_count: int | None = None
    bytes_read: int = 0
    # True when the pass stopped at a byte limit. A truncated pass has no
    # verdict -- it is a rate measurement and nothing else.
    truncated: bool = False

    def retained_objects(self) -> int:
        """How many objects the report holds. Must not depend on file size.

        Asserted in the tests rather than only measured in the spike run: RSS
        at gigabyte scale tells you the claim held for one file, and this tells
        you which change broke it.
        """
        return (len(self.errors) + len(self.warnings) + len(self.preview)
                + len(self.thinking_lines_with) + len(self.thinking_lines_without))

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "row_count": self.row_count,
            "usable_rows": self.usable_rows,
            "schema_type": self.schema_type,
            "enable_thinking": self.enable_thinking,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "preview": self.preview,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors_suppressed": self.errors_suppressed,
            "warnings_suppressed": self.warnings_suppressed,
            "turns_with_think": self.turns_with_think,
            "turns_without_think": self.turns_without_think,
            "token_count": self.token_count,
            "truncated": self.truncated,
        }


def _normalise(text: str) -> str:
    """NFC, and strip control characters except tab and newline.

    Copied from `api.validation` rather than imported so the measurement is of
    a self-contained pass. If this file becomes the ticket, the duplicate goes.
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text
                   if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def _iter_lines(fh: BinaryIO, chunk_bytes: int) -> Iterator[tuple[int, bytes]]:
    """Yield (1-indexed line number, raw line bytes) without holding the file.

    Splits on `\\n` and keeps the tail between reads -- a row straddling two
    reads is the classic bug in a hand-rolled streaming parser, and the tests
    drive this at chunk sizes down to one byte for exactly that reason.
    """
    BOM = b"\xef\xbb\xbf"
    line_no = 0
    tail = b""
    first = True
    while True:
        chunk = fh.read(chunk_bytes)
        if not chunk:
            break
        buf = tail + chunk
        if first:
            # utf-8-sig, one line at a time: strip the BOM from the very first
            # bytes rather than decoding the file twice to find out. The test
            # is against the ACCUMULATED buffer, not the chunk -- a one-byte
            # read splits the BOM itself, and testing the chunk alone would
            # leave two of its three bytes at the head of line 1.
            if len(buf) < len(BOM) and BOM.startswith(buf):
                tail = buf
                continue
            if buf.startswith(BOM):
                buf = buf[len(BOM):]
            first = False
        *lines, tail = buf.split(b"\n")
        for raw in lines:
            line_no += 1
            yield line_no, raw
    if tail:
        yield line_no + 1, tail


def stream_validate(
    path: Path,
    messages_field: str = "messages",
    *,
    count_tokens: Callable[[str], int] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    error_cap: int = DEFAULT_ERROR_CAP,
    sample_cap: int = DEFAULT_SAMPLE_CAP,
    track_lines: bool = True,
    byte_limit: int | None = None,
) -> StreamReport:
    """Validate a JSONL dataset in one pass, holding one row at a time.

    `count_tokens` is the seam spike 9 measures through: pass nothing and the
    pass is parse-and-validate alone; pass a tokeniser and the difference is
    what tokenisation costs. It is a plain `str -> int` so the spike can swap
    the real Qwen tokeniser in without this file importing it.

    `track_lines=False` exists only so the spike can price the line-number
    guarantee. **The product never sets it.** A rejection that does not say
    which line leaves the user guessing at a file they cannot see.

    `byte_limit` stops the pass early, and exists for the same reason: at the
    measured tokenisation rate a 20 GB tokenising pass runs for over an hour,
    so the spike measures a prefix and derives the rest -- marked as derived.
    The verdict from a truncated pass is meaningless and `rep.truncated` says
    so. **The product never sets this either.**
    """
    rep = StreamReport()

    # Errors are collected in two lists and concatenated at the end, so the
    # order matches the shipped validator's -- which reports every parse
    # failure before it reports anything about the rows that did parse.
    parse_errors: list[Issue] = []
    row_errors: list[Issue] = []

    def add(bucket: list[Issue], issue: Issue, is_error: bool = True) -> None:
        """Count every issue; keep at most `error_cap` of them.

        The count is the number the user needs -- "every line is broken" is a
        different problem from "line 4,102 is broken" -- and the cap is what
        keeps a 5 GB file of broken JSON from becoming a 5 GB error list.
        """
        if is_error:
            rep.error_count += 1
        else:
            rep.warning_count += 1
        if len(bucket) < error_cap:
            bucket.append(issue)

    def finish(errors: list[Issue]) -> StreamReport:
        """Cap the lists, reconcile the suppressed counts, set the verdict."""
        rep.errors = errors[:error_cap]
        rep.warnings = rep.warnings[:error_cap]
        rep.errors_suppressed = rep.error_count - len(rep.errors)
        rep.warnings_suppressed = rep.warning_count - len(rep.warnings)
        # A pass that stopped early has seen part of a file. "Valid" would be
        # a claim about rows it never read.
        rep.valid = rep.error_count == 0 and not rep.truncated
        return rep

    def at(line_no: int) -> int | None:
        return line_no if track_lines else None

    with_messages = 0
    schema_keys: set[str] = set()
    schema_key_rows = 0
    parsed_rows = 0  # the index `trainer.thinking.detect` counts by

    with path.open("rb") as fh:
        for line_no, raw in _iter_lines(fh, chunk_bytes):
            if byte_limit is not None and rep.bytes_read >= byte_limit:
                rep.truncated = True
                break
            rep.bytes_read += len(raw) + 1
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                add(parse_errors, Issue(
                    at(line_no), "encoding",
                    f"Line is not valid UTF-8 ({e.reason} at byte {e.start} of "
                    f"the line)."))
                # Matches the shipped validator's behaviour of abandoning the
                # file on a decode failure, but names the line while doing it.
                return finish(parse_errors)
            if not line.strip():
                continue
            rep.row_count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                add(parse_errors, Issue(at(line_no), "invalid_json",
                                        f"Not valid JSON: {e.msg} at column {e.colno}."))
                continue
            if not isinstance(obj, dict):
                add(parse_errors, Issue(
                    at(line_no), "not_an_object",
                    f"Expected a JSON object, got {type(obj).__name__}."))
                continue

            parsed_rows += 1
            if len(rep.preview) < MAX_PREVIEW:
                rep.preview.append(obj)
            if schema_key_rows < SCHEMA_KEY_SAMPLE:
                schema_keys |= set(obj)
                schema_key_rows += 1

            msgs = obj.get(messages_field)
            if isinstance(msgs, list):
                with_messages += 1

            _check_row(obj, msgs, messages_field, line_no, parsed_rows, rep,
                       row_errors, add, at, sample_cap, count_tokens)

    if rep.row_count == 0:
        add(parse_errors, Issue(None, "empty", "File contains no rows."))
        return finish(parse_errors)

    if with_messages == 0:
        # Same shape as the shipped message, and the same early return: with no
        # recognised schema there is nothing meaningful to say per row.
        add(parse_errors, Issue(
            None, "unrecognised_schema",
            f"No row has a '{messages_field}' list. Temper accepts chat-format "
            f"JSONL: {{\"{messages_field}\": [{{\"role\": \"user\", \"content\": ...}}, "
            f"{{\"role\": \"assistant\", \"content\": ...}}]}}. "
            f"Keys found instead: {sorted(schema_keys)}."))
        # The shipped validator returns before it fills the preview here, and
        # a preview of rows in a schema we could not recognise would be
        # misleading anyway.
        rep.preview = []
        return finish(parse_errors)
    rep.schema_type = "chat"

    _finish_thinking(rep, row_errors, add, sample_cap)
    _check_size_gates(rep, row_errors, add)

    return finish(parse_errors + row_errors)


def _check_row(obj, msgs, messages_field, line_no, parsed_rows, rep,
               row_errors, add, at, sample_cap, count_tokens) -> None:
    """Everything that can be decided from one row, deciding it now.

    Split out because the loop above is about reading and this is about
    validating, and the spike times them as one pass either way.
    """
    if count_tokens is not None:
        # Tokenise every turn, not just the target. The quote is priced per
        # training token and the prompt is in the context window, so counting
        # only the assistant turn would under-count what the run actually
        # processes.
        for m in msgs or []:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                rep.token_count = (rep.token_count or 0) + count_tokens(m["content"])

    # Thinking mode, counted incrementally and capped. The shipped detector
    # keeps every line number and caps at the end; that is the one place the
    # line-number guarantee costs memory proportional to the file.
    for m in msgs or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content") or ""
            if THINK_OPEN.search(content):
                rep.turns_with_think += 1
                if len(rep.thinking_lines_with) < sample_cap:
                    rep.thinking_lines_with.append(parsed_rows)
            else:
                rep.turns_without_think += 1
                if len(rep.thinking_lines_without) < sample_cap:
                    rep.thinking_lines_without.append(parsed_rows)

    if not isinstance(msgs, list):
        add(row_errors, Issue(at(line_no), "missing_messages",
                              f"Row has no '{messages_field}' list."))
        return
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if "assistant" not in roles:
        add(row_errors, Issue(
            at(line_no), "no_assistant_turn",
            "No assistant turn. The assistant turn is the training target, so "
            "this row teaches nothing."))
        return
    if "user" not in roles:
        add(rep.warnings, Issue(
            at(line_no), "no_user_turn",
            "No user turn; the model sees a response with no prompt."),
            is_error=False)
    last_assistant = [m for m in msgs
                      if isinstance(m, dict) and m.get("role") == "assistant"][-1]
    content = last_assistant.get("content")
    if not isinstance(content, str) or not content.strip():
        add(row_errors, Issue(
            at(line_no), "empty_target",
            "Final assistant turn is empty. Under assistant-only loss this row "
            "is a no-op."))
        return
    if _normalise(content) != content:
        add(rep.warnings, Issue(
            at(line_no), "normalised",
            "Content contained control characters or non-NFC sequences; it "
            "will be normalised."), is_error=False)
    rep.usable_rows += 1


def _finish_thinking(rep, row_errors, add, sample_cap) -> None:
    total = rep.turns_with_think + rep.turns_without_think
    if total == 0:
        rep.enable_thinking = False
        return
    if rep.turns_with_think and rep.turns_without_think:
        add(row_errors, Issue(
            None, "mixed_thinking",
            f"Dataset mixes reasoning traces with plain responses: "
            f"{rep.turns_with_think} assistant turn(s) contain <think> blocks "
            f"and {rep.turns_without_think} do not. Every row must be "
            f"consistent, because the chat template is applied to the whole "
            f"dataset -- a mixed set trains half the rows against the wrong "
            f"template. With <think>, e.g. lines "
            f"{rep.thinking_lines_with[:sample_cap]}; without, e.g. lines "
            f"{rep.thinking_lines_without[:sample_cap]}."))
        return
    rep.enable_thinking = bool(rep.turns_with_think)


def _check_size_gates(rep, row_errors, add) -> None:
    if rep.usable_rows < MIN_ROWS:
        add(row_errors, Issue(
            None, "too_few_rows",
            f"{rep.usable_rows} usable row(s); the minimum is {MIN_ROWS}. "
            f"Below that a run cannot produce a meaningful adapter, so it is "
            f"blocked rather than allowed to waste GPU time."))
    elif rep.usable_rows < RECOMMENDED_ROWS:
        add(rep.warnings, Issue(
            None, "few_rows",
            f"{rep.usable_rows} usable rows. Training will run, but results "
            f"are typically weak below ~{RECOMMENDED_ROWS} examples."),
            is_error=False)
