"""Tests for reasoning-trace detection.

The decision to detect rather than hard-code `enable_thinking` put new logic on
the highest-frequency silent-failure surface in the system. These tests are the
price of that choice -- all three branches, plus the near-misses that would
otherwise fail silently.

Run:  python -m pytest trainer/test_thinking.py -q
"""

import pytest

from thinking import MixedThinkingDataset, detect


def row(*assistant_contents, user="q"):
    msgs = [{"role": "user", "content": user}]
    for c in assistant_contents:
        msgs.append({"role": "assistant", "content": c})
    return {"messages": msgs}


def test_no_think_blocks_disables():
    rows = [row("42"), row("the answer is 42")]
    r = detect(rows)
    assert r.enable_thinking is False
    assert (r.assistant_turns, r.with_think, r.without_think) == (2, 0, 2)


def test_all_think_blocks_enables():
    rows = [row("<think>reasoning</think>42"), row("<think>more</think>43")]
    r = detect(rows)
    assert r.enable_thinking is True
    assert (r.assistant_turns, r.with_think, r.without_think) == (2, 2, 0)


def test_mixed_dataset_blocks_and_names_lines():
    rows = [row("<think>x</think>1"), row("2"), row("3")]
    with pytest.raises(MixedThinkingDataset) as e:
        detect(rows)
    msg = str(e.value)
    # The error has to be actionable: which lines, and why it matters.
    assert "[1]" in msg and "[2, 3]" in msg
    assert "wrong template" in msg


def test_user_turns_are_ignored():
    """A <think> block in a USER turn is not a reasoning trace we train on."""
    rows = [row("plain", user="<think>user thinking</think>ask")]
    assert detect(rows).enable_thinking is False


def test_multi_turn_rows_count_every_assistant_turn():
    rows = [row("<think>a</think>1", "<think>b</think>2")]
    r = detect(rows)
    assert r.assistant_turns == 2 and r.enable_thinking is True


@pytest.mark.parametrize("variant", [
    "<think>x</think>ans",
    "<THINK>x</THINK>ans",       # case
    "< think >x</think>ans",     # stray whitespace
    "<think id='1'>x</think>a",  # attributes
])
def test_near_misses_still_count_as_thinking(variant):
    """Permissive on purpose. Reading a near-miss as 'no trace' is the
    failure mode this whole module exists to prevent, so lean toward
    detecting."""
    assert detect([row(variant)]).enable_thinking is True


def test_empty_dataset_is_total_not_crashing():
    """Validation owns 'this dataset is empty'. This function stays total."""
    r = detect([])
    assert r.enable_thinking is False and r.assistant_turns == 0


def test_custom_messages_field():
    rows = [{"conversations": [{"role": "assistant", "content": "<think>x</think>y"}]}]
    assert detect(rows, messages_field="conversations").enable_thinking is True
