"""Reasoning-trace detection for Qwen3-family chat templates.

Qwen3 dense models emit `<think>...</think>` blocks by default, gated by an
`enable_thinking` flag threaded through the chat template. Get this wrong and
training and inference see different templates -- the modal silent failure in
this category: the loss curve looks fine and the model is subtly wrong.

The platform therefore does not guess and does not ask. It reads the dataset:

    all assistant turns have <think>   -> enable_thinking = True
    no assistant turn has <think>      -> enable_thinking = False
    some but not all                   -> BLOCK the job

The third case is the important one. A mixed dataset has no correct answer --
either choice trains half the rows against the wrong template -- so it becomes
a loud, line-numbered validation error rather than a silent coin flip. That is
the whole reason detection is safe enough to prefer over hard-coding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Deliberately permissive about whitespace and attributes, and case-insensitive:
# a near-miss like `< think >` should still count as "the author meant a
# reasoning trace", because treating it as absent is the failure we are trying
# to prevent.
THINK_OPEN = re.compile(r"<\s*think(\s[^>]*)?>", re.IGNORECASE)


class MixedThinkingDataset(ValueError):
    """Raised when only some assistant turns carry reasoning traces."""


@dataclass
class ThinkingReport:
    enable_thinking: bool
    assistant_turns: int
    with_think: int
    without_think: int
    # 1-indexed dataset line numbers, capped -- enough to fix the data without
    # printing a 50,000-line error.
    sample_lines_with: list[int] = field(default_factory=list)
    sample_lines_without: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "enable_thinking": self.enable_thinking,
            "assistant_turns": self.assistant_turns,
            "turns_with_think": self.with_think,
            "turns_without_think": self.without_think,
        }


def _assistant_contents(row: dict, messages_field: str):
    for msg in row.get(messages_field) or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            yield msg.get("content") or ""


def detect(rows, messages_field: str = "messages", sample_cap: int = 5) -> ThinkingReport:
    """Inspect assistant turns and decide `enable_thinking`.

    `rows` is an iterable of parsed JSON objects, in dataset order.
    Raises MixedThinkingDataset when the dataset is ambiguous.
    """
    with_t, without_t = [], []

    for line_no, row in enumerate(rows, start=1):
        for content in _assistant_contents(row, messages_field):
            (with_t if THINK_OPEN.search(content) else without_t).append(line_no)

    total = len(with_t) + len(without_t)
    if total == 0:
        # No assistant turns at all is a dataset problem, but it is the
        # validator's problem, not ours. Default to off so this function stays
        # total.
        return ThinkingReport(False, 0, 0, 0)

    if with_t and without_t:
        raise MixedThinkingDataset(
            f"Dataset mixes reasoning traces with plain responses: "
            f"{len(with_t)} assistant turn(s) contain <think> blocks and "
            f"{len(without_t)} do not. Every row must be consistent, because "
            f"the chat template is applied to the whole dataset -- a mixed set "
            f"trains half the rows against the wrong template. "
            f"With <think>, e.g. lines {with_t[:sample_cap]}; "
            f"without, e.g. lines {without_t[:sample_cap]}."
        )

    enabled = bool(with_t)
    return ThinkingReport(
        enable_thinking=enabled,
        assistant_turns=total,
        with_think=len(with_t),
        without_think=len(without_t),
        sample_lines_with=with_t[:sample_cap],
        sample_lines_without=without_t[:sample_cap],
    )
