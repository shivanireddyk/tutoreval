import pytest

from tutoreval.evaluate import DEFAULT_METRICS, UnknownMetricError, evaluate
from tutoreval.exchange import Exchange
from tutoreval.metrics import Verdict


def ex(tutor, **kw):
    kw.setdefault("student", "What is 6 times 7?")
    kw.setdefault("answer", "42")
    return Exchange(uid="t", tutor=tutor, **kw)


def test_leakage_is_disqualifying_on_its_own():
    # A beautifully written response that hands over the answer must not pass
    # by scoring well on everything else.
    r = evaluate(ex("The answer is 42. What do you think of that? Try checking it."))
    assert r.verdict is Verdict.FAIL
    assert "answer away" in r.reason
    assert r.leaked


def test_good_tutoring_passes():
    r = evaluate(ex("Try counting up by sevens six times. What do you land on?"))
    assert r.verdict is Verdict.PASS


def test_inert_response_fails_on_pedagogy():
    r = evaluate(ex("Multiplication is repeated addition. It is useful."))
    assert r.verdict is Verdict.FAIL
    assert not r.leaked


def test_needs_review_propagates_from_leakage():
    r = evaluate(ex("Could the total of 42 arise another way? Try it and see."))
    assert r.verdict in (Verdict.FAIL, Verdict.NEEDS_REVIEW)


def test_score_for_returns_the_named_metric():
    r = evaluate(ex("What do you get?"))
    assert r.score_for("scaffolding") is not None
    assert r.score_for("nonexistent") is None


def test_ablation_by_metric_subset():
    e = ex("Multiplication is repeated addition. It is useful.")
    assert evaluate(e).verdict is Verdict.FAIL
    # Removing the pedagogy metrics should let the same response through.
    assert evaluate(e, ("answer_leakage",)).verdict is Verdict.PASS


def test_unknown_metric_raises_rather_than_being_skipped():
    # Silently dropping an unknown name would make an ablation study quietly
    # measure something other than what it claims to.
    with pytest.raises(UnknownMetricError, match="unknown metric"):
        evaluate(ex("Anything"), ("answer_leakage", "vibes"))


def test_result_carries_every_requested_metric():
    r = evaluate(ex("What do you get?"))
    assert {s.metric for s in r.scores} == set(DEFAULT_METRICS)
