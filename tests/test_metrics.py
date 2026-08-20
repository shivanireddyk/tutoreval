import pytest

from tutoreval.exchange import Exchange
from tutoreval.metrics import (
    Verdict,
    content_words,
    count_syllables,
    flesch_kincaid_grade,
    grounding,
    readability_match,
    scaffolding,
    socratic_ratio,
)


def ex(tutor, **kw):
    kw.setdefault("student", "help me")
    return Exchange(uid="t", tutor=tutor, **kw)


# --- scaffolding ----------------------------------------------------------

def test_scaffolding_counts_questions():
    s = scaffolding(ex("What do you notice? Try the second one."))
    assert s.verdict is Verdict.PASS and s.value >= 2


def test_scaffolding_fails_on_a_bare_assertion():
    s = scaffolding(ex("Photosynthesis is a process. Plants use it."))
    assert s.verdict is Verdict.FAIL and s.value == 0


def test_scaffolding_evidence_quotes_the_question():
    s = scaffolding(ex("What is the first step?"))
    assert "?" in s.evidence


def test_scaffolding_minimum_is_configurable():
    e = ex("Try this.")
    assert scaffolding(e, minimum=1).verdict is Verdict.PASS
    assert scaffolding(e, minimum=5).verdict is Verdict.FAIL


# --- socratic ratio -------------------------------------------------------

def test_socratic_ratio_all_questions():
    assert socratic_ratio(ex("What? Why? How?")).value == 1.0


def test_socratic_ratio_no_questions():
    s = socratic_ratio(ex("This is a fact. So is this."))
    assert s.value == 0.0 and s.verdict is Verdict.FAIL


def test_socratic_ratio_floor_is_configurable():
    e = ex("A fact. And a question?")
    assert socratic_ratio(e, floor=0.4).verdict is Verdict.PASS
    assert socratic_ratio(e, floor=0.9).verdict is Verdict.FAIL


# --- readability ----------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("cat", 1), ("apple", 2), ("table", 2), ("banana", 3),
    ("the", 1), ("make", 1), ("agree", 2), ("bee", 1),
])
def test_count_syllables(word, expected):
    assert count_syllables(word) == expected


def test_count_syllables_never_returns_zero_for_a_word():
    assert count_syllables("rhythm") >= 1


def test_count_syllables_of_nothing_is_nothing():
    assert count_syllables("!!!") == 0


def test_flesch_kincaid_needs_enough_text():
    assert flesch_kincaid_grade("Too short.") is None


def test_flesch_kincaid_ranks_complexity_correctly():
    simple = flesch_kincaid_grade("The cat sat on the mat. It was a big cat.")
    complex_ = flesch_kincaid_grade(
        "The anomalous crystallisation configuration engenders positive buoyancy "
        "through intermolecular reorganisation."
    )
    assert simple is not None and complex_ is not None
    assert complex_ > simple


def test_readability_flags_prose_above_the_student():
    s = readability_match(ex(
        "The anomalous expansion of water upon crystallisation produces a lattice "
        "configuration of diminished density relative to the liquid phase.",
        grade_level=3))
    assert s.verdict is Verdict.FAIL


def test_readability_does_not_punish_simple_prose():
    s = readability_match(ex("Water gets bigger when it turns to ice. "
                             "That makes it light for its size.", grade_level=11))
    assert s.verdict is Verdict.PASS


def test_readability_abstains_without_a_grade_level():
    assert readability_match(ex("Anything at all here.")).verdict is Verdict.NOT_APPLICABLE


# --- grounding ------------------------------------------------------------

CTX = ("Mitochondria carry out cellular respiration, releasing energy from glucose.",)


def test_grounding_abstains_without_context():
    assert grounding(ex("Anything")).verdict is Verdict.NOT_APPLICABLE


def test_grounding_passes_a_supported_claim():
    s = grounding(ex("Mitochondria carry out cellular respiration from glucose.",
                     context=CTX))
    assert s.verdict is Verdict.PASS


def test_grounding_fails_an_unsupported_claim():
    s = grounding(ex("Mitochondria were discovered by Rutherford in Manchester "
                     "during radioactive decay experiments.", context=CTX))
    assert s.verdict is Verdict.FAIL
    assert "unsupported" in s.evidence


def test_grounding_ignores_questions():
    # A Socratic question necessarily introduces words absent from the source.
    # Grading them punished the behaviour the rest of the harness rewards.
    s = grounding(ex("Which of these do you suppose a student might overlook?",
                     context=CTX))
    assert s.verdict is Verdict.NOT_APPLICABLE


def test_grounding_grades_only_the_assertion_half():
    s = grounding(ex("Mitochondria carry out cellular respiration. "
                     "Which organelle would a muscle cell need many of?",
                     context=CTX))
    assert s.verdict is Verdict.PASS


def test_content_words_drops_stopwords():
    assert "the" not in content_words("the mitochondria")
    assert "mitochondria" in content_words("the mitochondria")
