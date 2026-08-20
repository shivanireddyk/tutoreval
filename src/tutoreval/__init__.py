"""tutoreval: measuring whether an AI tutor teaches, not whether it answers.

A tutoring response can be factually perfect and pedagogically useless. Ask a
model "what is the derivative of x squared" and a reply of "2x" is correct and
has taught nobody anything. Most LLM evaluation scores the first property and
is silent on the second.

This library scores the second. It is deterministic, it makes no network
calls, and every score it produces carries the span of text that caused it.

    from tutoreval import Exchange, evaluate

    ex = Exchange(
        uid="d1",
        student="What is the derivative of x^2?",
        tutor="The answer is 2x.",
        answer="2x",
        grade_level=11,
    )
    evaluate(ex).verdict     # Verdict.FAIL
"""

from .benchmark import BenchmarkResult, Confusion, ablate, run, run_file
from .evaluate import DEFAULT_METRICS, Result, UnknownMetricError, evaluate
from .exchange import Exchange, ExchangeError, exchange_from_dict, load_jsonl
from .metrics import (
    Score,
    Verdict,
    answer_leakage,
    count_syllables,
    flesch_kincaid_grade,
    grounding,
    readability_match,
    scaffolding,
    socratic_ratio,
)
from .normalize import canonical_expression, equivalent, parse_number
from .report import render, render_ablation, render_result

__version__ = "0.1.0"

__all__ = [
    "Exchange", "ExchangeError", "exchange_from_dict", "load_jsonl",
    "Score", "Verdict", "answer_leakage", "scaffolding", "socratic_ratio",
    "readability_match", "grounding", "count_syllables", "flesch_kincaid_grade",
    "evaluate", "Result", "DEFAULT_METRICS", "UnknownMetricError",
    "run", "run_file", "ablate", "BenchmarkResult", "Confusion",
    "render", "render_result", "render_ablation",
    "parse_number", "equivalent", "canonical_expression",
    "__version__",
]
