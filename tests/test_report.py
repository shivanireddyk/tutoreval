from tutoreval.benchmark import ablate, run
from tutoreval.evaluate import evaluate
from tutoreval.exchange import Exchange, load_jsonl
from tutoreval.report import render, render_ablation, render_result

LEAK = Exchange(uid="leak", student="What is 6 times 7?",
                tutor="The answer is 42.", answer="42", expected_leak=True)
GOOD = Exchange(uid="good", student="What is 6 times 7?",
                tutor="Try counting up by sevens. What do you land on?",
                answer="42", expected_leak=False)


def test_render_result_names_the_row_and_the_verdict():
    out = render_result(evaluate(LEAK))
    assert "leak" in out and "FAIL" in out


def test_render_result_shows_evidence():
    assert "evidence:" in render_result(evaluate(LEAK))


def test_render_result_hides_inapplicable_metrics():
    assert "NOT_APPLICABLE" not in render_result(evaluate(GOOD))


def test_render_includes_headline_numbers():
    out = render(run((LEAK, GOOD)))
    assert "exchanges evaluated : 2" in out
    assert "PASS" in out and "FAIL" in out


def test_render_includes_detector_quality():
    out = render(run((LEAK, GOOD)))
    assert "precision" in out and "recall" in out


def test_render_says_when_the_detector_is_unmeasured():
    plain = Exchange(uid="x", student="Why?", tutor="Because. Why do you think?")
    assert "unmeasured" in render(run((plain,)))


def test_render_states_the_limitation_of_grounding():
    # The report must not let a lexical overlap number be read as a truth check.
    assert "not a check on whether the response is true" in render(run((LEAK, GOOD)))


def test_render_handles_an_empty_run():
    assert "exchanges evaluated : 0" in render(run(()))


def test_render_ablation_lists_every_metric():
    out = render_ablation(ablate((LEAK, GOOD)))
    assert "answer_leakage" in out and "scaffolding" in out


def test_render_ablation_flags_metrics_that_did_nothing():
    assert "no effect" in render_ablation({"scaffolding": 0.0})


def test_render_ablation_of_nothing():
    assert "nothing to ablate" in render_ablation({})


def test_render_full_benchmark_does_not_crash():
    out = render(run(load_jsonl("data/tutoring_benchmark_v1.jsonl")), show=3)
    assert "Tutoring quality benchmark" in out
