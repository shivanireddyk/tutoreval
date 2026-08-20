"""The leakage metric, including a regression test for every bug the
benchmark caught. Each of the six below was found by running the harness
against hand-labelled rows, not by reading the code."""

import pytest

from tutoreval.exchange import Exchange
from tutoreval.metrics import Verdict, answer_leakage


def ex(student, tutor, answer="42", **kw):
    return Exchange(uid="t", student=student, tutor=tutor, answer=answer, **kw)


# --- baseline behaviour ---------------------------------------------------

def test_states_the_answer_outright():
    s = answer_leakage(ex("What is 6 times 7?", "The answer is 42."))
    assert s.verdict is Verdict.FAIL
    assert "42" in s.evidence


def test_equals_sign_is_answer_position():
    assert answer_leakage(ex("What is 6*7?", "6 * 7 = 42.")).verdict is Verdict.FAIL


def test_socratic_response_does_not_leak():
    s = answer_leakage(ex("What is 6 times 7?", "What is six groups of seven?"))
    assert s.verdict is Verdict.PASS


def test_no_reference_answer_abstains():
    s = answer_leakage(Exchange(uid="t", student="Why?", tutor="Because.", answer=None))
    assert s.verdict is Verdict.NOT_APPLICABLE


def test_evidence_is_always_populated_on_a_failure():
    assert answer_leakage(ex("What is 6*7?", "It is 42.")).evidence


# --- REGRESSION 1 ---------------------------------------------------------
# Algebraic answers took a different code path that passed a zero offset for
# the occurrence, so the text before the answer was always empty and the
# assertive-phrase check could never fire. Every non-numeric answer abstained.

def test_regression_algebraic_answer_in_answer_position():
    s = answer_leakage(
        ex("What is the derivative of x^2?", "The answer is 2x.", answer="2x")
    )
    assert s.verdict is Verdict.FAIL, "algebraic answers must reach the assertive check"


def test_algebraic_socratic_still_passes():
    s = answer_leakage(
        ex("What is the derivative of x^2?",
           "Bring the exponent down and reduce it by one. What do you get?",
           answer="2x")
    )
    assert s.verdict is Verdict.PASS


# --- REGRESSION 2 ---------------------------------------------------------
# "Let us do this in 3 steps" and "a hexagon has 6 sides" are both a small
# integer followed by a plural noun. Nothing in the tutor turn separates them.
# The student's question does: if the noun is what they asked to have counted,
# the tutor answered it.

def test_regression_counted_noun_from_the_question_is_an_answer():
    s = answer_leakage(
        ex("How many sides does a hexagon have?",
           "A hexagon has 6 sides, and all of them are equal.", answer="6")
    )
    assert s.verdict is Verdict.FAIL


def test_regression_unrelated_noun_is_structural():
    s = answer_leakage(
        ex("Solve 2x + 4 = 10.",
           "Let us do this in 3 steps. What is step one?", answer="3")
    )
    assert s.verdict is Verdict.PASS


def test_regression_protons_case():
    s = answer_leakage(
        ex("How many protons does carbon have?",
           "Carbon has 6 protons. That is its atomic number.", answer="6")
    )
    assert s.verdict is Verdict.FAIL


def test_step_prefix_is_structural():
    s = answer_leakage(
        ex("Solve for x.", "Look again at step 3 and tell me what changed.", answer="3")
    )
    assert s.verdict is Verdict.PASS


# --- REGRESSION 3 ---------------------------------------------------------
# A question that names the value still hands it over. "Is it 42?" was being
# excused as Socratic.

def test_regression_confirmation_question_is_not_socratic():
    s = answer_leakage(ex("What is 6 times 7?", "Is it 42? Check by adding sevens."))
    assert s.verdict is Verdict.FAIL


def test_regression_answer_named_inside_a_question_escalates():
    s = answer_leakage(
        ex("What is 6 times 7?", "Could the total of 42 be reached another way?")
    )
    assert s.verdict in (Verdict.FAIL, Verdict.NEEDS_REVIEW)
    assert s.verdict is not Verdict.PASS


# --- REGRESSION 4 ---------------------------------------------------------
# A one-word reply of "Twenty." scanned as containing no numbers at all,
# because only digits were searched for.

def test_regression_spelled_out_number_counts():
    s = answer_leakage(ex("What is 25% of 80?", "Twenty. Divide by four.", answer="20"))
    assert s.verdict is Verdict.FAIL


def test_spelled_out_number_not_the_answer_is_ignored():
    s = answer_leakage(
        ex("What is 25% of 80?", "Think about what one quarter means.", answer="20")
    )
    assert s.verdict is Verdict.PASS


# --- REGRESSION 5 ---------------------------------------------------------
# A value alone as its own sentence has no surrounding phrase to match on, so
# a purely context-driven classifier read the bluntest leak in the corpus as
# ambiguous and escalated it instead of failing it. Discourse markers such as
# "Well," count as no context at all and are stripped before the check.

def test_regression_bare_value_is_an_answer():
    s = answer_leakage(ex("What is 6 times 7?", "42."))
    assert s.verdict is Verdict.FAIL


def test_bare_value_still_fails_with_trailing_punctuation():
    s = answer_leakage(ex("What is 6 times 7?", "Well, 42!"))
    assert s.verdict is Verdict.FAIL


# --- numeric equivalence --------------------------------------------------

@pytest.mark.parametrize("tutor,answer", [
    ("You get 0.5.", "1/2"),
    ("It is 50%.", "0.5"),
    ("The answer is 1/2.", "50%"),
])
def test_equivalent_forms_are_all_leaks(tutor, answer):
    s = answer_leakage(ex("What is 1/4 + 1/4?", tutor, answer=answer))
    assert s.verdict is Verdict.FAIL, f"{tutor} should leak {answer}"


def test_a_different_number_is_not_a_leak():
    s = answer_leakage(ex("What is 6*7?", "Start from 6 rows of seven.", answer="42"))
    assert s.verdict is Verdict.PASS
