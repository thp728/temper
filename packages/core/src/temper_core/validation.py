"""Dataset validation. Line-numbered errors, or it is not validation.

Report C names dataset validation as where users churn first, and the reason is
always the same: a rejection that does not say *which line* leaves the user
guessing at a file they cannot see. Every error here carries a line number.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from temper_core.thinking import MixedThinkingDataset
from temper_core.thinking import detect as detect_thinking

MIN_ROWS = 10  # hard floor: block
RECOMMENDED_ROWS = 50  # below this: warn
MAX_PREVIEW = 3


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
        }


def _normalise(text: str) -> str:
    """NFC, and strip control characters except tab and newline.

    Not cosmetic: the same visual string existing as two different token
    sequences distorts length statistics and defeats deduplication.
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )


def validate(path: Path, messages_field: str = "messages") -> Report:
    """Validate the dataset at `path`. A thin read over `validate_bytes`."""
    return validate_bytes(path.read_bytes(), messages_field)


def validate_bytes(raw: bytes, messages_field: str = "messages") -> Report:
    """Validate dataset bytes already held. The pure entry point.

    An upload has its bytes before anything is stored, so validation runs on
    them directly rather than reading back what storage just wrote -- one
    read fewer on the hottest path a user waits for. `validate` is the
    path-shaped front door over this.
    """
    rep = Report()
    rows: list[dict[str, Any]] = []

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        rep.errors.append(
            Issue(
                None,
                "encoding",
                f"File is not valid UTF-8 ({e.reason} at byte {e.start}).",
            )
        )
        return rep

    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        rep.row_count += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            rep.errors.append(
                Issue(
                    line_no,
                    "invalid_json",
                    f"Not valid JSON: {e.msg} at column {e.colno}.",
                )
            )
            continue
        if not isinstance(obj, dict):
            rep.errors.append(
                Issue(
                    line_no,
                    "not_an_object",
                    f"Expected a JSON object, got {type(obj).__name__}.",
                )
            )
            continue
        rows.append({"_line": line_no, **obj})

    if rep.row_count == 0:
        rep.errors.append(Issue(None, "empty", "File contains no rows."))
        return rep

    # --- schema ------------------------------------------------------------
    # Only the chat/messages schema is supported. The detection is deliberate
    # rather than assumed, because the schema determines the loss mask -- get
    # it wrong and the model trains on a different objective than intended.
    with_msgs = [r for r in rows if isinstance(r.get(messages_field), list)]
    if not with_msgs:
        keys = sorted(
            {k for r in rows[:20] for k in r if not k.startswith("_")}
        )
        rep.errors.append(
            Issue(
                None,
                "unrecognised_schema",
                f"No row has a '{messages_field}' list. Temper accepts chat-format "
                f'JSONL: {{"{messages_field}": [{{"role": "user", "content": ...}}, '
                f'{{"role": "assistant", "content": ...}}]}}. '
                f"Keys found instead: {keys}.",
            )
        )
        return rep
    rep.schema_type = "chat"

    for r in rows:
        line = r["_line"]
        msgs = r.get(messages_field)
        if not isinstance(msgs, list):
            rep.errors.append(
                Issue(
                    line,
                    "missing_messages",
                    f"Row has no '{messages_field}' list.",
                )
            )
            continue
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        if "assistant" not in roles:
            rep.errors.append(
                Issue(
                    line,
                    "no_assistant_turn",
                    "No assistant turn. The assistant turn is the "
                    "training target, so this row teaches nothing.",
                )
            )
            continue
        if "user" not in roles:
            rep.warnings.append(
                Issue(
                    line,
                    "no_user_turn",
                    "No user turn; the model sees a response with no prompt.",
                )
            )
        last_assistant = [
            m
            for m in msgs
            if isinstance(m, dict) and m.get("role") == "assistant"
        ][-1]
        content = last_assistant.get("content")
        if not isinstance(content, str) or not content.strip():
            rep.errors.append(
                Issue(
                    line,
                    "empty_target",
                    "Final assistant turn is empty. Under "
                    "assistant-only loss this row is a no-op.",
                )
            )
            continue
        if _normalise(content) != content:
            rep.warnings.append(
                Issue(
                    line,
                    "normalised",
                    "Content contained control characters or "
                    "non-NFC sequences; it will be normalised.",
                )
            )
        rep.usable_rows += 1

    # --- thinking mode -----------------------------------------------------
    try:
        think = detect_thinking(rows, messages_field)
        rep.enable_thinking = think.enable_thinking
    except MixedThinkingDataset as e:
        rep.errors.append(Issue(None, "mixed_thinking", str(e)))

    # --- size gates --------------------------------------------------------
    if rep.usable_rows < MIN_ROWS:
        rep.errors.append(
            Issue(
                None,
                "too_few_rows",
                f"{rep.usable_rows} usable row(s); the minimum is {MIN_ROWS}. "
                f"Below that a run cannot produce a meaningful adapter, so it is "
                f"blocked rather than allowed to waste GPU time.",
            )
        )
    elif rep.usable_rows < RECOMMENDED_ROWS:
        rep.warnings.append(
            Issue(
                None,
                "few_rows",
                f"{rep.usable_rows} usable rows. Training will run, but results "
                f"are typically weak below ~{RECOMMENDED_ROWS} examples.",
            )
        )

    rep.preview = [
        {k: v for k, v in r.items() if k != "_line"}
        for r in rows[:MAX_PREVIEW]
    ]
    rep.valid = not rep.errors
    return rep
