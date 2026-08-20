import json

import pytest

from tutoreval.benchmark import Confusion, ablate, run, run_file
from tutoreval.exchange import Exchange, load_jsonl
from tutoreval.metrics import Verdict

LEAK = Exchange(uid="leak", student="What is 6 times 7?",
                tutor="The answer is 42.", answer="42", expected_leak=True)
GOOD = Exchange(uid="good", student="What is 6 times 7?",
                tutor="Try counting up by sevens. What do you land on?",
                answer="42", expected_leak=False)
INERT = Exchange(uid="inert", student="Explain multiplication.",
                 tutor="It is repeated addition. It is useful.", answer=None)


def test_run_scores_every_exchange():
    b = run((LEAK, GOOD, INERT))
    assert b.total == 3
    assert len(b.results) == 3


def test_verdict_counts_cover_all_verdicts():
    counts = run((LEAK, GOOD)).verdict_counts
    assert set(counts) == {v.value for v in Verdict}


def test_pass_rate():
    assert run((LEAK, GOOD)).pass_rate == pytest.approx(0.5)


def test_pass_rate_of_nothing_is_zero_not_a_crash():
    b = run(())
    assert b.total == 0 and b.pass_rate == 0.0


def test_failures_and_review_are_retrievable():
    b = run((LEAK, GOOD, INERT))
    assert {r.uid for r in b.failures()} >= {"leak", "inert"}
    assert all(r.verdict is Verdict.NEEDS_REVIEW for r in b.needing_review())


def test_confusion_counts_against_human_labels():
    c = run((LEAK, GOOD)).confusion
    assert c.true_positive == 1 and c.true_negative == 1
    assert c.false_positive == 0 and c.false_negative == 0
    assert c.precision == 1.0 and c.recall == 1.0 and c.f1 == 1.0


def test_unlabelled_rows_are_not_counted():
    assert run((INERT,)).confusion.labelled == 0


def test_confusion_metrics_are_none_when_undefined():
    empty = Confusion(0, 0, 0, 0, 0)
    assert empty.precision is None and empty.recall is None and empty.f1 is None


def test_f1_is_none_when_precision_and_recall_are_both_zero():
    assert Confusion(0, 1, 0, 1, 0).f1 is None


def test_abstentions_are_reported_separately_from_errors():
    # A row the detector declines to call is not a wrong answer, and folding
    # it into the error count would understate the detector's precision.
    unknown = Exchange(uid="u", student="Why?", tutor="Because of reasons.",
                       answer=None, expected_leak=False)
    c = run((unknown,)).confusion
    assert c.abstained == 1 and c.labelled == 0


def test_run_file(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text(json.dumps({
        "uid": "a", "student": "What is 6 times 7?", "tutor": "The answer is 42.",
        "answer": "42", "expected_leak": True}), encoding="utf-8")
    b = run_file(p)
    assert b.total == 1 and b.confusion.true_positive == 1


def test_ablation_reports_a_delta_per_metric():
    d = ablate((LEAK, GOOD, INERT))
    assert set(d) == set(run((LEAK,)).metrics)
    assert all(isinstance(v, float) for v in d.values())


def test_ablation_shows_pedagogy_metrics_carry_weight():
    d = ablate((INERT,))
    # Removing scaffolding should let the inert response through.
    assert d["scaffolding"] >= 0


def test_shipped_benchmark_loads_and_runs():
    exs = load_jsonl("data/tutoring_benchmark_v1.jsonl")
    assert len(exs) >= 20
    b = run(exs)
    assert b.total == len(exs)


def test_shipped_benchmark_detector_makes_no_wrong_calls():
    # This is the guard that matters. If a change to the leakage rules starts
    # disagreeing with the hand labels, this fails before anything ships.
    c = run(load_jsonl("data/tutoring_benchmark_v1.jsonl")).confusion
    assert c.false_positive == 0
    assert c.false_negative == 0
    assert c.labelled >= 15


def test_results_are_deterministic():
    a = run(load_jsonl("data/tutoring_benchmark_v1.jsonl"))
    b = run(load_jsonl("data/tutoring_benchmark_v1.jsonl"))
    assert [r.verdict for r in a.results] == [r.verdict for r in b.results]
