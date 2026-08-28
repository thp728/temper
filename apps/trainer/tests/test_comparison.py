"""The pure half of the side-by-side comparison (issue #69).

The comparison's decisions live in ``comparison.py`` and are tested here
without hardware: prompt selection is deterministic, decoding settings are
fixed and **recorded** (Spec 011's "recorded means stored on the result and
shown" -- that needs a test, not a docstring), and evaluation failure never
fails the run. The generators are a seam, so the suite injects doubles the
same way template_probe's tests inject a tokenizer.
"""

from __future__ import annotations

import comparison
import pytest


class StubGenerator:
    """A generator double: returns canned text, or raises on demand.

    ``fail_after`` lets a test succeed for the first N calls and then raise,
    which is how a comparison that dies part-way is exercised.
    """

    def __init__(
        self,
        text: str = "generated",
        exc: Exception | None = None,
        fail_after: int | None = None,
    ):
        self._text = text
        self._exc = exc
        self._fail_after = fail_after
        self.calls = 0

    def generate(self, conversation) -> str:
        self.calls += 1
        if self._exc is not None and (
            self._fail_after is None or self.calls > self._fail_after
        ):
            raise self._exc
        return self._text


def held_out_rows(n: int) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(n)
    ]


def conversations(n: int) -> list[list[dict]]:
    return [
        [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
        for i in range(n)
    ]


# --- prompt selection ---------------------------------------------------------


def test_prompt_selection_is_deterministic_under_a_seed():
    rows = held_out_rows(20)
    assert comparison.select_prompts(rows) == comparison.select_prompts(rows)
    assert comparison.select_prompts(
        rows, seed=1
    ) != comparison.select_prompts(rows, seed=2)


def test_prompt_selection_takes_the_fixed_slice_in_dataset_order():
    rows = held_out_rows(10)
    selected = comparison.select_prompts(rows)
    assert len(selected) == comparison.COMPARISON_PROMPT_COUNT
    # Dataset order, not shuffle order: what the page shows is the order the
    # rows were uploaded in.
    assert selected == sorted(
        selected, key=lambda r: r["messages"][0]["content"]
    )


def test_prompt_selection_selects_everything_when_fewer_than_the_slice():
    rows = held_out_rows(2)
    assert len(comparison.select_prompts(rows)) == 2


def test_prompt_selection_never_duplicates_a_row():
    rows = held_out_rows(5)
    picked = comparison.select_prompts(rows, count=10)
    keys = [json_of(r) for r in picked]
    assert len(keys) == len(set(keys))


def json_of(row: dict) -> str:
    import json

    return json.dumps(row["messages"], sort_keys=True)


def test_the_prompt_is_the_conversation_without_the_held_out_answer():
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "the held-out answer"},
    ]
    assert comparison.prompt_messages(messages) == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "q"},
    ]


# --- the record ---------------------------------------------------------------


def test_the_result_carries_both_generations_and_the_prompt():
    outcome = comparison.run_comparison(
        conversations=conversations(1),
        base=StubGenerator("base"),
        tuned=StubGenerator("tuned"),
    )
    assert outcome.ok
    row = outcome.rows[0]
    assert row["prompt"] == [{"role": "user", "content": "q0"}]
    assert row["base"] == "base"
    assert row["tuned"] == "tuned"


def test_the_decoding_settings_are_recorded_on_the_result():
    """Spec 011: recorded means stored on the result, so a reader can tell
    whether two outputs are comparable. This is the test for it, not a
    docstring."""
    outcome = comparison.run_comparison(
        conversations=conversations(1),
        base=StubGenerator(),
        tuned=StubGenerator(),
    )
    record = outcome.to_dict()
    assert record["decoding"] == comparison.COMPARISON_DECODING
    assert record["decoding"]["temperature"] == 0.7
    assert record["decoding"]["max_new_tokens"] == 128
    assert record["decoding"]["do_sample"] is True


def test_the_chosen_checkpoint_is_recorded_with_the_result():
    selection = {"step": 20, "basis": "best_held_out_loss", "reason": "lowest"}
    outcome = comparison.run_comparison(
        conversations=conversations(1),
        base=StubGenerator(),
        tuned=StubGenerator(),
        selection=selection,
    )
    assert outcome.to_dict()["selection"] == selection


# --- evaluation failure never fails the run -----------------------------------


def test_a_failing_generator_records_the_reason_and_never_raises():
    """The negative test that protects a paid run: make the comparison throw,
    and the run's result still records ok: false with the reason, rather than
    the whole run failing."""
    outcome = comparison.run_comparison(
        conversations=conversations(3),
        base=StubGenerator(),
        tuned=StubGenerator(exc=RuntimeError("the tuned model blew up")),
    )
    assert not outcome.ok
    assert "RuntimeError" in (outcome.reason or "")
    assert "blew up" in (outcome.reason or "")
    assert outcome.rows == []
    # The decoding settings are still recorded: "fixed and recorded" holds
    # even when the generation could not run.
    assert outcome.decoding == comparison.COMPARISON_DECODING


def test_a_failing_base_generator_records_the_reason_too():
    outcome = comparison.run_comparison(
        conversations=conversations(1),
        base=StubGenerator(exc=ValueError("no base model")),
        tuned=StubGenerator(),
    )
    assert not outcome.ok
    assert "ValueError" in (outcome.reason or "")


def test_a_failing_generator_still_reports_which_checkpoint_was_compared():
    selection = {"step": 30, "basis": "fallback_last", "reason": "no loss"}
    outcome = comparison.run_comparison(
        conversations=conversations(1),
        base=StubGenerator(exc=RuntimeError("oom")),
        tuned=StubGenerator(),
        selection=selection,
    )
    assert not outcome.ok
    assert outcome.selection == selection


def test_partial_rows_are_dropped_on_failure():
    """A comparison that fails mid-way does not leave half its rows behind: a
    reader cannot tell which rows of a broken run are trustworthy."""
    outcome = comparison.run_comparison(
        conversations=conversations(3),
        base=StubGenerator(),
        tuned=StubGenerator(
            exc=RuntimeError("died on prompt 3"), fail_after=2
        ),
    )
    assert outcome.rows == []


# --- the record's shape -------------------------------------------------------


def test_the_record_omits_the_reason_when_it_went_well():
    outcome = comparison.run_comparison(
        conversations=conversations(1),
        base=StubGenerator("base"),
        tuned=StubGenerator("tuned"),
    )
    assert outcome.to_dict() == {
        "ok": True,
        "decoding": comparison.COMPARISON_DECODING,
        "rows": [
            {
                "prompt": [{"role": "user", "content": "q0"}],
                "base": "base",
                "tuned": "tuned",
            }
        ],
    }


# --- real generations, on hardware ---------------------------------------------
# The comparison's rendering is exercised on a real finished record by the
# browser journeys, whose simulated machine carries a canned comparison. That
# proves the renderer and nothing about the generator. What proves the
# generator is this: the real machine path (render through the chat template,
# generate with the fixed settings, decode) run against a real model, which
# needs torch + transformers + a downloaded model and is therefore deselected
# from the host gate (`just test` runs `-m "not hardware"`). It runs in the
# trainer image, where all three exist.


@pytest.mark.hardware
def test_the_generator_produces_real_text_from_a_real_model():
    """The comparison's generation path -- template render, fixed decoding
    settings, decode -- run against a real tiny model. The output is real
    model text (garbage, from a random untrained model, but real), not a
    fixture, so the criterion "renders with real generations" is proven by
    the code that produces it."""
    from entrypoint import _ModelGenerator
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = "hf-internal-testing/tiny-random-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(repo)
    # tiny-random-gpt2 ships no chat template; the comparison renders through
    # the tokenizer's own template exactly as the entrypoint does, so one is
    # set here the same way the trainer's rows are formatted.
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: "
        "{{ message['content'] }}\n{% endfor %}"
    )
    kwargs = {"enable_thinking": False}
    base_model = AutoModelForCausalLM.from_pretrained(repo)
    tuned_model = AutoModelForCausalLM.from_pretrained(repo)

    base = _ModelGenerator(
        base_model,
        tokenizer,
        "tokenizer_default",
        kwargs,
        comparison.COMPARISON_DECODING,
    )
    tuned = _ModelGenerator(
        tuned_model,
        tokenizer,
        "tokenizer_default",
        kwargs,
        comparison.COMPARISON_DECODING,
    )

    outcome = comparison.run_comparison(
        conversations=conversations(2),
        base=base,
        tuned=tuned,
        decoding=comparison.COMPARISON_DECODING,
        selection={"step": 20, "basis": "best_held_out_loss", "reason": "x"},
    )
    assert outcome.ok
    assert len(outcome.rows) == 2
    for row in outcome.rows:
        # Real generated text, decoded from the model's own tokens.
        assert row["base"].strip()
        assert row["tuned"].strip()
