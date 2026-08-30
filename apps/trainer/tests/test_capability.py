"""The pure half of the general-capability slice (issue #73).

The slice's decisions live in ``capability.py`` and are tested here without
hardware: the slice is fixed and versioned, the questions are general (never
the user's task), scoring is deterministic (a response that names no letter is
wrong, not dropped), the delta is reported with a paired standard error, the
sample size and per-question weight travel with the result, a large regression
is a recorded flag rather than an interface judgement, and evaluation failure
never fails the run. The generators are a seam, so the suite injects doubles
the same way comparison's tests do.
"""

from __future__ import annotations

import capability
import comparison
import pytest


class StubGenerator:
    """A generator double: returns canned text, or raises on demand.

    ``answers`` lets a test script the model's option letters per question;
    ``fail_after`` lets a test succeed for the first N calls and then raise,
    which is how a slice that dies part-way is exercised.
    """

    def __init__(
        self,
        answers: list[str | None] | None = None,
        exc: Exception | None = None,
        fail_after: int | None = None,
    ):
        self._answers = answers or []
        self._exc = exc
        self._fail_after = fail_after
        self.calls = 0

    def generate(self, conversation) -> str:
        self.calls += 1
        if self._exc is not None and (
            self._fail_after is None or self.calls > self._fail_after
        ):
            raise self._exc
        i = min(self.calls - 1, len(self._answers) - 1)
        if not self._answers:
            return "generated"
        return self._answers[i] or ""


def correct_letters() -> list[str]:
    """The correct letter for each slice question, in order -- what a model
    that answers everything right would output."""
    return [q["answer"] for q in capability.CAPABILITY_QUESTIONS]


# --- the slice itself --------------------------------------------------------


def test_the_slice_is_fixed_small_general_and_versioned():
    """Spec 011: a fixed, versioned slice -- the same set for every job, small,
    and deliberately not the user's task. A general evaluation harness is out
    of scope, and the version lets a reader know which slice a number came
    from."""
    assert capability.CAPABILITY_SLICE_VERSION == 1
    assert len(capability.CAPABILITY_QUESTIONS) == 8
    # Small by design: the slice is a smoke test, not a benchmark.
    assert len(capability.CAPABILITY_QUESTIONS) <= 12
    # Every question carries a domain, a full prompt, and exactly one correct
    # letter among A-D.
    for q in capability.CAPABILITY_QUESTIONS:
        assert q["domain"]
        assert q["answer"] in ("A", "B", "C", "D")
        assert any(m.get("role") == "user" for m in q["messages"])
        for letter in ("A", "B", "C", "D"):
            assert letter in q["messages"][0]["content"]


def test_the_slice_answers_are_not_all_the_same_letter():
    """A slice whose every answer is 'C' would be gamed by a model that always
    says C. The set spans letters, so a degenerate answering strategy does not
    coast."""
    answers = {q["answer"] for q in capability.CAPABILITY_QUESTIONS}
    assert len(answers) >= 3


def test_the_slice_decoding_is_the_comparisons_fixed_settings():
    """One fixed decoding definition for all warm-machine evaluation
    (ADR-0010): the capability slice is measured under the same settings the
    comparison is, defined once in comparison.py, so there cannot be two sets
    of 'the fixed settings' to drift."""
    assert capability.COMPARISON_DECODING is comparison.COMPARISON_DECODING


# --- answer parsing ----------------------------------------------------------


def test_parse_answer_accepts_a_bare_letter():
    assert capability.parse_answer("B") == "B"
    assert capability.parse_answer(" C ") == "C"
    assert capability.parse_answer("A.") == "A"
    assert capability.parse_answer("b)") == "B"
    assert capability.parse_answer("D:") == "D"


def test_parse_answer_accepts_a_letter_inside_prose():
    assert capability.parse_answer("The answer is B.") == "B"
    assert (
        capability.parse_answer("I think the correct option is A) 54.") == "A"
    )


def test_parse_answer_never_matches_a_letter_inside_a_word():
    """A word that starts with an option letter (e.g. 'Au', 'Bangkok', 'Heart')
    is not an answer letter -- the boundary is the letter alone."""
    assert capability.parse_answer("Au") is None
    assert capability.parse_answer("Bangkok") is None
    assert capability.parse_answer("Heart") is None
    assert capability.parse_answer("The element is gold, symbol Au.") is None


def test_parse_answer_is_none_when_no_letter_is_named():
    assert capability.parse_answer("") is None
    assert capability.parse_answer("I do not know.") is None
    assert capability.parse_answer("42") is None
    assert capability.parse_answer(None) is None


# --- the record --------------------------------------------------------------


def test_the_result_scores_both_sides_and_reports_the_delta():
    base_answers = [q["answer"] for q in capability.CAPABILITY_QUESTIONS]
    outcome = capability.run_capability_slice(
        base=StubGenerator(base_answers),
        tuned=StubGenerator(base_answers),
    )
    assert outcome.ok
    assert outcome.total == len(capability.CAPABILITY_QUESTIONS)
    assert outcome.base_correct == outcome.total
    assert outcome.tuned_correct == outcome.total
    assert outcome.delta == 0.0
    record = outcome.to_dict()
    assert record["base_score"] == 1.0
    assert record["tuned_score"] == 1.0
    assert record["total"] == outcome.total


def test_the_delta_is_tuned_minus_base():
    """The headline number is the change: tuned score minus base score, over
    the same questions."""
    outcome = capability.run_capability_slice(
        base=StubGenerator(correct_letters()),
        tuned=StubGenerator([]),
    )
    assert outcome.ok
    assert outcome.base_correct == 8
    assert outcome.tuned_correct == 0
    assert outcome.delta == -1.0


def test_the_sample_size_travels_beside_the_number():
    """Spec 011: the interface states the sample size beside the number. The
    size is on the record, never inferred, and so is the per-question weight --
    with n questions each is 1/n of the score."""
    outcome = capability.run_capability_slice(
        base=StubGenerator(correct_letters()),
        tuned=StubGenerator([]),
    )
    record = outcome.to_dict()
    assert record["total"] == len(capability.CAPABILITY_QUESTIONS)
    assert record["large_regression"] is True


def test_a_response_that_names_no_letter_counts_as_wrong():
    """An unparseable answer is scored wrong rather than dropped: a model that
    cannot produce an answer letter is itself evidence of degradation."""
    base = StubGenerator(correct_letters())
    tuned = StubGenerator(
        ["I do not know."] * len(capability.CAPABILITY_QUESTIONS)
    )
    outcome = capability.run_capability_slice(base=base, tuned=tuned)
    assert outcome.ok
    assert outcome.tuned_correct == 0
    for row in outcome.rows:
        assert row["tuned_parsed"] is None
        assert row["tuned_correct"] is False


def test_the_rows_carry_the_question_the_domain_and_both_answers():
    """Each row is legible: the question asked, its domain, the correct letter,
    both models' text, what was parsed, and whether each was right -- so a
    reader can spot-check the score instead of trusting it."""
    base = StubGenerator(correct_letters())
    tuned = StubGenerator(correct_letters())
    outcome = capability.run_capability_slice(base=base, tuned=tuned)
    row = outcome.rows[0]
    assert row["domain"]
    assert row["answer"] == outcome.rows[0]["answer"]
    assert row["base_parsed"] == row["answer"]
    assert row["base_correct"] is True
    assert row["tuned_correct"] is True
    assert any(m["role"] == "user" for m in row["prompt"])


# --- the uncertainty ---------------------------------------------------------


def test_the_delta_uncertainty_is_the_paired_standard_error():
    """The delta is a paired before/after on the same questions, so its
    standard error comes from the per-question differences, not from treating
    the two sides as independent samples."""
    base = StubGenerator(correct_letters())
    # Tuned gets exactly one question wrong (a -1 difference, everything else
    # equal): the delta is -1/8 and the standard error is positive, because the
    # one differing question spreads the differences.
    tuned_answers = correct_letters()
    tuned_answers[0] = "A" if tuned_answers[0] != "A" else "B"
    outcome = capability.run_capability_slice(
        base=base, tuned=StubGenerator(tuned_answers)
    )
    assert outcome.ok
    assert outcome.base_correct == 8
    assert outcome.tuned_correct == 7
    assert outcome.delta == pytest.approx(-1 / 8)
    assert outcome.delta_se > 0
    # A standard error for a single differing question among eight: the mean
    # difference is -1/8, the differences are one -1 and seven 0s.
    expected_var = ((7 * (0 - (-1 / 8)) ** 2) + ((-1 - (-1 / 8)) ** 2)) / 7
    assert outcome.delta_se == pytest.approx((expected_var / 8) ** 0.5)


def test_no_difference_has_zero_uncertainty():
    """When every question answers the same on both sides the delta has no
    spread and its standard error is zero -- there is nothing uncertain about
    an unchanged score."""
    outcome = capability.run_capability_slice(
        base=StubGenerator(correct_letters()),
        tuned=StubGenerator(correct_letters()),
    )
    assert outcome.delta_se == 0.0


def test_a_large_regression_is_recorded_not_inferred():
    """A tuned model that loses at least LARGE_REGRESSION_QUESTIONS correct
    answers is flagged on the record, so the interface surfaces the flag from
    the record (ADR-0010) instead of re-deciding a threshold in the client."""
    base = StubGenerator(correct_letters())
    wrong = [None if i < 2 else correct_letters()[i] for i in range(8)]
    outcome = capability.run_capability_slice(
        base=base, tuned=StubGenerator(wrong)
    )
    assert outcome.large_regression is True
    assert outcome.to_dict()["large_regression"] is True
    # Losing just one question is not yet a large regression.
    wrong_one = [None if i < 1 else correct_letters()[i] for i in range(8)]
    small = capability.run_capability_slice(
        base=StubGenerator(correct_letters()), tuned=StubGenerator(wrong_one)
    )
    assert small.large_regression is False


def test_the_selection_is_recorded_with_the_result():
    selection = {"step": 20, "basis": "best_held_out_loss", "reason": "lowest"}
    outcome = capability.run_capability_slice(
        base=StubGenerator(correct_letters()),
        tuned=StubGenerator(correct_letters()),
        selection=selection,
    )
    assert outcome.to_dict()["selection"] == selection


def test_the_decoding_settings_are_recorded_on_the_result():
    """Fixed decoding, recorded -- the same 'recorded means stored and shown'
    rule the comparison lives by (Spec 011)."""
    outcome = capability.run_capability_slice(
        base=StubGenerator(correct_letters()),
        tuned=StubGenerator(correct_letters()),
    )
    record = outcome.to_dict()
    assert record["decoding"] == comparison.COMPARISON_DECODING


# --- evaluation failure never fails the run -----------------------------------


def test_a_failing_generator_records_the_reason_and_never_raises():
    """The negative test that protects a paid run: make the slice throw, and
    the run's result still records ok: false with the reason, rather than the
    whole run failing."""
    outcome = capability.run_capability_slice(
        base=StubGenerator(correct_letters()),
        tuned=StubGenerator(exc=RuntimeError("the tuned model blew up")),
    )
    assert not outcome.ok
    assert "RuntimeError" in (outcome.reason or "")
    assert "blew up" in (outcome.reason or "")
    assert outcome.rows == []
    # The decoding settings are still recorded, and the flag is false rather
    # than missing -- a failure is not read as a regression.
    assert outcome.decoding == comparison.COMPARISON_DECODING
    assert outcome.large_regression is False


def test_partial_rows_are_dropped_on_failure():
    """A slice that fails mid-way does not leave half its rows behind: a reader
    cannot tell which rows of a broken run are trustworthy."""
    outcome = capability.run_capability_slice(
        base=StubGenerator(correct_letters()),
        tuned=StubGenerator(
            exc=RuntimeError("died on question 3"), fail_after=2
        ),
    )
    assert not outcome.ok
    assert outcome.rows == []


# --- the record's shape ------------------------------------------------------


def test_the_record_omits_the_reason_when_it_went_well():
    outcome = capability.run_capability_slice(
        base=StubGenerator(correct_letters()),
        tuned=StubGenerator(correct_letters()),
    )
    record = outcome.to_dict()
    assert record["ok"] is True
    assert "reason" not in record
    assert "selection" not in record
