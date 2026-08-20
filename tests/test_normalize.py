from fractions import Fraction

import pytest

from tutoreval.normalize import (
    canonical_expression,
    clean,
    equivalent,
    find_numbers,
    parse_number,
    sentences,
    word_number,
    words,
)


@pytest.mark.parametrize("text,expected", [
    ("42", Fraction(42)), ("1.5", Fraction(3, 2)), ("1/2", Fraction(1, 2)),
    ("50%", Fraction(1, 2)), ("-3", Fraction(-3)), ("1,250", Fraction(1250)),
    (".5", Fraction(1, 2)), ("0.25", Fraction(1, 4)), ("3/4", Fraction(3, 4)),
])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["x", "", "abc", "1/0", "two"])
def test_parse_number_rejects_non_numbers(text):
    assert parse_number(text) is None


def test_parse_number_is_exact_not_floating_point():
    # 0.1 + 0.2 != 0.3 in binary floating point. A benchmark that compares
    # answers must not inherit that.
    assert parse_number("0.1") + parse_number("0.2") == parse_number("0.3")


@pytest.mark.parametrize("a,b", [
    ("0.5", "1/2"), ("50%", "0.5"), ("1/2", "50%"), ("2x", "2*x"),
    ("2x", "2 * x"), ("1,000", "1000"), ("x^2", "x**2"),
])
def test_equivalent_pairs(a, b):
    assert equivalent(a, b)


@pytest.mark.parametrize("a,b", [
    ("x+x", "2x"), ("3", "4"), ("2x", "3x"), ("1/2", "1/3"),
])
def test_non_equivalent_pairs(a, b):
    assert not equivalent(a, b)


def test_equivalence_is_normalisation_not_evaluation():
    # "x+x" is mathematically 2x but a tutor writing it has not written the
    # answer in the form the student was asked for.
    assert not equivalent("x+x", "2x")


def test_clean_handles_unicode_maths():
    assert clean("2·x") == "2*x"
    assert clean("x²") == "x2"
    assert clean("5 − 3") == "5 - 3"


def test_find_numbers_reports_spans():
    got = find_numbers("Step 3 gives 12 apples")
    assert [v for v, _, _ in got] == [Fraction(3), Fraction(12)]
    text = "Step 3 gives 12 apples"
    assert text[got[0][1]:got[0][2]] == "3"


def test_word_number():
    assert word_number("twenty") == Fraction(20)
    assert word_number("TWO") == Fraction(2)
    assert word_number("banana") is None


def test_sentences_splits_on_terminators():
    assert sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]


def test_sentences_drops_empties():
    assert sentences("   ") == []


def test_words_lowercases_and_strips_punctuation():
    assert words("Hello, World! 42") == ["hello", "world"]


def test_canonical_expression_inserts_multiplication():
    assert canonical_expression("2x") == "2*x"
    assert canonical_expression("2 X") == "2*x"
