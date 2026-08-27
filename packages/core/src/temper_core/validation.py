"""Dataset validation. Line-numbered errors, or it is not validation.

Report C names dataset validation as where users churn first, and the reason is
always the same: a rejection that does not say *which line* leaves the user
guessing at a file they cannot see. Every error here carries a line number.

Validation is **streaming**: it reads the file in chunks and decides each row
before the next row is read, so peak memory stays flat as the dataset grows.
That property is what ADR-0005's 1 GB upload ceiling was standing in for --
the ceiling was derived from a validator that held the whole file (measured at
5.93x its size, spike 9), and it is gone with the rewrite. The ceiling that
replaces it is a product limit derived from measured throughput
([ADR-0036](../docs/adr/0036-the-dataset-size-limit-is-derived-from-measured-throughput.md)), not from how validation is written.

Three things survive the rewrite, each pinned by a test:

* **The same verdict.** A streaming pass that is fast because it checks less
  is not validation. The tests compare it against the pre-rewrite validator's
  behaviour row for row and code for code.
* **The line-number guarantee.** Every error names its line. That is free
  while streaming -- a counter -- but the *samples* are not: thinking-mode
  detection would accumulate one line number per assistant turn before capping
  if written the old way. It caps as it goes.
* **Flat memory.** Everything retained is capped: the first hundred errors and
  warnings, a three-row preview, twenty schema keys, and five sample line
  numbers per thinking direction. Nothing else outlives its row. The cap is
  not silent -- `error_count`/`errors_suppressed` carry the true totals, so a
  user whose file is broken on every line is told it is every line.

Two deliberate divergences from the in-memory path it replaces, both recorded
rather than hidden:

* **Encoding errors name a line.** The in-memory path decoded the whole file
  at once, so it could only report a byte offset with `line: None`. Decoding
  a line at a time can say which line, and that is strictly better for the
  user.
* **Rows are split on `\\n` only.** `str.splitlines()` also splits on `\\v`,
  `\\f`, `\\x1c`-`\\x1e`, `\\x85`, `\\u2028` and `\\u2029`, so a JSON string
  containing a literal U+2028 was two rows to the in-memory path and is one
  row here. JSONL is newline-delimited by definition; this is a divergence in
  the old code's favour of accident rather than intent.

Two things are deliberately not warnings:

* **Fewer than 10 usable rows blocks.** Adopted from OpenAI's enforced minimum
  rather than invented -- "we copied the floor a large platform enforces" is a
  better answer than a number someone picked.
* **A mixed thinking-mode dataset blocks.** Ambiguous by construction; see
  `temper_core/thinking.py`, which is also the module the trainer image runs.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from temper_core.thinking import THINK_OPEN, mixed_thinking_message

MIN_ROWS = 10  # hard floor: block
RECOMMENDED_ROWS = 50  # below this: warn
MAX_PREVIEW = 3

# The issue-collecting callback used throughout: add to a bucket, counting
# every issue and capping the list. `Callable[..., None]` because the closure
# carries a default (`is_error=True`) that a fully-parameterised Callable
# cannot express.
_AddIssue = Callable[..., None]

# The caps that keep memory flat. Each is a constant, not a tunable: the cap
# is what makes peak memory independent of file size, so a deployment knob
# would be a knob on a safety property.
MAX_ERRORS = 100  # most problems a report lists; the count is never capped
MAX_SAMPLE = 5  # thinking-mode example lines, per direction
SCHEMA_KEY_SAMPLE = (
    20  # rows whose keys contribute to the unrecognised-schema list
)
CHUNK_BYTES = 1024 * 1024  # how much of the file is held while reading
PROGRESS_EVERY_BYTES = 1024 * 1024  # how often `on_progress` may fire


@dataclass
class Issue:
    line: int | None
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"line": self.line, "code": self.code, "message": self.message}


@dataclass
class Report:
    valid: bool = False
    row_count: int = 0
    usable_rows: int = 0
    schema_type: str | None = None
    enable_thinking: bool | None = None
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    preview: list[dict[str, Any]] = field(default_factory=list)

    # The errors/warnings lists are capped to keep memory flat; these carry
    # the true totals so a capped report never silently understates how
    # broken a file is.
    error_count: int = 0
    warning_count: int = 0
    errors_suppressed: int = 0
    warnings_suppressed: int = 0

    # Thinking-mode tallies and example lines, capped as they are collected
    # (`MAX_SAMPLE` each). They are what the `mixed_thinking` message is built
    # from; they stay off the published report, which carries the same
    # information inside that message, exactly as the old report did.
    turns_with_think: int = 0
    turns_without_think: int = 0
    thinking_lines_with: list[int] = field(default_factory=list)
    thinking_lines_without: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
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
        }

    def retained_objects(self) -> int:
        """How many objects the report holds. Must not depend on file size.

        Asserted in the tests rather than only measured in the spike run: RSS
        at gigabyte scale tells you the claim held for one file, and this
        tells you which change broke it. A refactor that quietly re-accumulates
        every row (say, a list of all parsed rows feeding thinking detection)
        makes this number scale with the file, and the test fails.
        """
        return (
            len(self.errors)
            + len(self.warnings)
            + len(self.preview)
            + len(self.thinking_lines_with)
            + len(self.thinking_lines_without)
        )


@dataclass(frozen=True)
class ValidationProgress:
    """Where a streaming pass has got to. The product surfaces this so a large
    upload does not look like a frozen page."""

    bytes_read: int
    bytes_total: int | None
    rows: int


def _normalise(text: str) -> str:
    """NFC, and strip control characters except tab and newline.

    Not cosmetic: the same visual string existing as two different token
    sequences distorts length statistics and defeats deduplication.
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )


def _iter_lines(
    chunks: Iterable[bytes],
) -> Iterator[tuple[int, bytes, int]]:
    """Yield (1-indexed line number, raw line bytes, bytes read so far).

    Splits on `\\n` and keeps the tail between chunks -- a row straddling two
    reads is the classic bug in a hand-rolled streaming parser, and the tests
    drive this at chunk sizes down to one byte for exactly that reason. The
    third element is the **cumulative** input bytes consumed up to and
    including this line -- the BOM included -- which is what lets progress be
    measured in bytes and reach exactly 100% at the end of the file.
    """
    BOM = b"\xef\xbb\xbf"
    line_no = 0
    tail = b""
    first = True
    consumed = 0
    for chunk in chunks:
        consumed += len(chunk)
        buf = tail + chunk
        if first:
            # utf-8-sig, one line at a time: strip the BOM from the very
            # first bytes rather than decoding the file twice to find out.
            # The test is against the ACCUMULATED buffer, not the chunk -- a
            # one-byte read splits the BOM itself, and testing the chunk alone
            # would leave two of its three bytes at the head of line 1.
            if len(buf) < len(BOM) and BOM.startswith(buf):
                tail = buf
                continue
            if buf.startswith(BOM):
                buf = buf[len(BOM) :]
            first = False
        *lines, tail = buf.split(b"\n")
        for raw in lines:
            line_no += 1
            yield line_no, raw, consumed
    if tail:
        yield line_no + 1, tail, consumed


def _chunks_of(path: Path, chunk_bytes: int = CHUNK_BYTES) -> Iterator[bytes]:
    """Read `path` in bounded chunks. The file is held only for the read."""
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_bytes):
            yield chunk


def validate(
    path: Path,
    messages_field: str = "messages",
    *,
    on_progress: Callable[[ValidationProgress], None] | None = None,
    total_bytes: int | None = None,
) -> Report:
    """Validate the dataset at `path`, streaming one row at a time.

    `total_bytes` lets the caller state the file's size so `on_progress` can
    report a fraction; when it is None, progress reports bytes read without a
    total.
    """
    return validate_chunks(
        _chunks_of(path),
        messages_field,
        on_progress=on_progress,
        total_bytes=total_bytes,
    )


def validate_bytes(raw: bytes, messages_field: str = "messages") -> Report:
    """Validate bytes already held. The pure entry point.

    Deliberately a thin wrapper over the same streaming core as `validate`:
    there is exactly one validator, so a verdict never depends on which door
    the bytes arrived through. An upload no longer goes through here -- the
    ingest path streams the body to storage and validates the stored object,
    so the control plane does not hold the file either -- but the shape stays
    for callers that already hold bytes.
    """
    return validate_chunks((raw,), messages_field)


def validate_chunks(
    chunks: Iterable[bytes],
    messages_field: str = "messages",
    *,
    on_progress: Callable[[ValidationProgress], None] | None = None,
    total_bytes: int | None = None,
) -> Report:
    """Validate dataset bytes as they arrive, one row at a time.

    `chunks` is any iterable of bytes -- a file's chunks, a stored object's
    stream, or one whole `bytes` for `validate_bytes`. This is the single
    implementation behind every entry point; the upload path feeds it chunks
    straight off storage so the control plane's memory stays flat too.
    """
    rep = Report()

    # Errors are collected in two lists and concatenated at the end, so the
    # order matches the old validator's -- which reported every parse failure
    # before it reported anything about the rows that did parse.
    parse_errors: list[Issue] = []
    row_errors: list[Issue] = []

    def add(bucket: list[Issue], issue: Issue, is_error: bool = True) -> None:
        """Count every issue; keep at most `MAX_ERRORS` of them.

        The count is the number the user needs -- "every line is broken" is a
        different problem from "line 4,102 is broken" -- and the cap is what
        keeps a 5 GB file of broken JSON from becoming a 5 GB error list.
        """
        if is_error:
            rep.error_count += 1
        else:
            rep.warning_count += 1
        if len(bucket) < MAX_ERRORS:
            bucket.append(issue)

    def finish(errors: list[Issue]) -> Report:
        """Cap the lists, reconcile the suppressed counts, set the verdict."""
        rep.errors = errors[:MAX_ERRORS]
        rep.warnings = rep.warnings[:MAX_ERRORS]
        rep.errors_suppressed = rep.error_count - len(rep.errors)
        rep.warnings_suppressed = rep.warning_count - len(rep.warnings)
        rep.valid = rep.error_count == 0
        return rep

    with_messages = 0
    schema_keys: set[str] = set()
    schema_key_rows = 0
    parsed_rows = 0  # the index `thinking.detect` counted by, kept for parity
    bytes_read = 0

    next_report = PROGRESS_EVERY_BYTES

    def report() -> None:
        if on_progress is not None:
            on_progress(
                ValidationProgress(bytes_read, total_bytes, rep.row_count)
            )

    for line_no, raw, bytes_read in _iter_lines(chunks):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            add(
                parse_errors,
                Issue(
                    line_no,
                    "encoding",
                    f"Line is not valid UTF-8 ({e.reason} at byte {e.start} of "
                    f"the line).",
                ),
            )
            # Matches the old validator's behaviour of abandoning the file on
            # a decode failure, but names the line while doing it.
            report()
            return finish(parse_errors)
        if not line.strip():
            continue
        rep.row_count += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            add(
                parse_errors,
                Issue(
                    line_no,
                    "invalid_json",
                    f"Not valid JSON: {e.msg} at column {e.colno}.",
                ),
            )
            continue
        if not isinstance(obj, dict):
            add(
                parse_errors,
                Issue(
                    line_no,
                    "not_an_object",
                    f"Expected a JSON object, got {type(obj).__name__}.",
                ),
            )
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

        _check_row(
            obj,
            msgs,
            messages_field,
            line_no,
            parsed_rows,
            rep,
            row_errors,
            add,
        )

        if bytes_read >= next_report:
            report()
            next_report = bytes_read + PROGRESS_EVERY_BYTES

    if rep.row_count == 0:
        add(parse_errors, Issue(None, "empty", "File contains no rows."))
        report()
        return finish(parse_errors)

    if with_messages == 0:
        # Same shape as the old message, and the same early return: with no
        # recognised schema there is nothing meaningful to say per row.
        add(
            parse_errors,
            Issue(
                None,
                "unrecognised_schema",
                f"No row has a '{messages_field}' list. Temper accepts chat-format "
                f'JSONL: {{"{messages_field}": [{{"role": "user", "content": ...}}, '
                f'{{"role": "assistant", "content": ...}}]}}. '
                f"Keys found instead: {sorted(schema_keys)}.",
            ),
        )
        # The old validator returned before it filled the preview here, and a
        # preview of rows in a schema we could not recognise would be
        # misleading anyway.
        rep.preview = []
        report()
        return finish(parse_errors)
    rep.schema_type = "chat"

    _finish_thinking(rep, row_errors, add)
    _check_size_gates(rep, row_errors, add)

    report()
    return finish(parse_errors + row_errors)


def _check_row(
    obj: dict[str, Any],
    msgs: Any,
    messages_field: str,
    line_no: int,
    parsed_rows: int,
    rep: Report,
    row_errors: list[Issue],
    add: _AddIssue,
) -> None:
    """Everything that can be decided from one row, deciding it now.

    Split out because the loop above is about reading and this is about
    validating, and the two are timed as one pass either way.
    """

    # Thinking mode, counted incrementally and capped. The old detector kept
    # every line number and capped at the end; that was the one place the
    # line-number guarantee cost memory proportional to the file.
    for m in msgs or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content") or ""
            if THINK_OPEN.search(content):
                rep.turns_with_think += 1
                if len(rep.thinking_lines_with) < MAX_SAMPLE:
                    rep.thinking_lines_with.append(parsed_rows)
            else:
                rep.turns_without_think += 1
                if len(rep.thinking_lines_without) < MAX_SAMPLE:
                    rep.thinking_lines_without.append(parsed_rows)

    if not isinstance(msgs, list):
        add(
            row_errors,
            Issue(
                line_no,
                "missing_messages",
                f"Row has no '{messages_field}' list.",
            ),
        )
        return
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if "assistant" not in roles:
        add(
            row_errors,
            Issue(
                line_no,
                "no_assistant_turn",
                "No assistant turn. The assistant turn is the training target, so "
                "this row teaches nothing.",
            ),
        )
        return
    if "user" not in roles:
        add(
            rep.warnings,
            Issue(
                line_no,
                "no_user_turn",
                "No user turn; the model sees a response with no prompt.",
            ),
            is_error=False,
        )
    last_assistant = [
        m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"
    ][-1]
    content = last_assistant.get("content")
    if not isinstance(content, str) or not content.strip():
        add(
            row_errors,
            Issue(
                line_no,
                "empty_target",
                "Final assistant turn is empty. Under assistant-only loss this row "
                "is a no-op.",
            ),
        )
        return
    if _normalise(content) != content:
        add(
            rep.warnings,
            Issue(
                line_no,
                "normalised",
                "Content contained control characters or non-NFC sequences; it "
                "will be normalised.",
            ),
            is_error=False,
        )
    rep.usable_rows += 1


def _finish_thinking(
    rep: Report, row_errors: list[Issue], add: _AddIssue
) -> None:
    """Decide `enable_thinking`, blocking on a mixed dataset.

    The verdict and the `mixed_thinking` wording come from
    `temper_core.thinking`, the module the trainer image also runs -- one
    source of truth for what the two halves say about the same dataset -- but
    the sample lines are capped as they are collected here, which is the
    streaming part. The counts are exact either way.
    """
    total = rep.turns_with_think + rep.turns_without_think
    if total == 0:
        rep.enable_thinking = False
        return
    if rep.turns_with_think and rep.turns_without_think:
        add(
            row_errors,
            Issue(
                None,
                "mixed_thinking",
                mixed_thinking_message(
                    rep.turns_with_think,
                    rep.turns_without_think,
                    rep.thinking_lines_with,
                    rep.thinking_lines_without,
                    MAX_SAMPLE,
                ),
            ),
        )
        return
    rep.enable_thinking = bool(rep.turns_with_think)


def _check_size_gates(
    rep: Report, row_errors: list[Issue], add: _AddIssue
) -> None:
    if rep.usable_rows < MIN_ROWS:
        add(
            row_errors,
            Issue(
                None,
                "too_few_rows",
                f"{rep.usable_rows} usable row(s); the minimum is {MIN_ROWS}. "
                f"Below that a run cannot produce a meaningful adapter, so it is "
                f"blocked rather than allowed to waste GPU time.",
            ),
        )
    elif rep.usable_rows < RECOMMENDED_ROWS:
        add(
            rep.warnings,
            Issue(
                None,
                "few_rows",
                f"{rep.usable_rows} usable rows. Training will run, but results "
                f"are typically weak below ~{RECOMMENDED_ROWS} examples.",
            ),
            is_error=False,
        )
