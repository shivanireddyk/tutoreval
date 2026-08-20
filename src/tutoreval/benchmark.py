"""Running the harness over a dataset, and scoring the harness itself.

Two things happen here and they are worth keeping distinct.

Running a benchmark tells you how a tutor did. Scoring the detector tells you
whether the benchmark's answer means anything. A leakage metric with a 40
percent false positive rate produces a confident-looking pass rate that is
mostly noise, and the only way to know is to check it against rows a human
labelled by hand.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .evaluate import DEFAULT_METRICS, Result, evaluate
from .exchange import Exchange, load_jsonl
from .metrics import Verdict


@dataclass(frozen=True)
class Confusion:
    """How the leakage detector compares against the human labels."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    abstained: int

    @property
    def labelled(self) -> int:
        return (self.true_positive + self.false_positive
                + self.true_negative + self.false_negative)

    @property
    def precision(self) -> float | None:
        called = self.true_positive + self.false_positive
        return self.true_positive / called if called else None

    @property
    def recall(self) -> float | None:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)


@dataclass(frozen=True)
class BenchmarkResult:
    """Everything one benchmark run produced."""

    metrics: tuple[str, ...]
    results: tuple[Result, ...]
    confusion: Confusion

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def verdict_counts(self) -> dict[str, int]:
        c = Counter(r.verdict.value for r in self.results)
        return {v.value: c.get(v.value, 0) for v in Verdict}

    @property
    def pass_rate(self) -> float:
        return self.verdict_counts["PASS"] / self.total if self.total else 0.0

    def failures(self) -> tuple[Result, ...]:
        return tuple(r for r in self.results if r.verdict is Verdict.FAIL)

    def needing_review(self) -> tuple[Result, ...]:
        return tuple(r for r in self.results if r.verdict is Verdict.NEEDS_REVIEW)


def _confusion(exchanges: tuple[Exchange, ...],
               results: tuple[Result, ...]) -> Confusion:
    tp = fp = tn = fn = abstain = 0
    for ex, res in zip(exchanges, results, strict=True):
        if ex.expected_leak is None:
            continue
        s = res.score_for("answer_leakage")
        if s is None or s.verdict in (Verdict.NEEDS_REVIEW, Verdict.NOT_APPLICABLE):
            abstain += 1
            continue
        called = s.verdict is Verdict.FAIL
        if called and ex.expected_leak:
            tp += 1
        elif called and not ex.expected_leak:
            fp += 1
        elif not called and ex.expected_leak:
            fn += 1
        else:
            tn += 1
    return Confusion(tp, fp, tn, fn, abstain)


def run(exchanges: tuple[Exchange, ...],
        metrics: tuple[str, ...] = DEFAULT_METRICS) -> BenchmarkResult:
    """Evaluate every exchange and score the leakage detector against labels.

    Deterministic by construction: no sampling, no model calls, no clock. The
    same input file produces byte-identical output on every machine, which is
    what makes a regression in a metric visible instead of plausible.
    """
    results = tuple(evaluate(ex, metrics) for ex in exchanges)
    return BenchmarkResult(metrics, results, _confusion(exchanges, results))


def run_file(path: str | Path,
             metrics: tuple[str, ...] = DEFAULT_METRICS) -> BenchmarkResult:
    """Load a JSONL benchmark file and run it."""
    return run(load_jsonl(path), metrics)


def ablate(exchanges: tuple[Exchange, ...],
           full: tuple[str, ...] = DEFAULT_METRICS) -> dict[str, float]:
    """Leave-one-out: how much does each metric move the pass rate?

    A metric that changes nothing when removed is either redundant with
    another metric or is not firing on this dataset. Either way it should not
    be reported as though it contributed.
    """
    baseline = run(exchanges, full).pass_rate
    out: dict[str, float] = {}
    for m in full:
        reduced = tuple(x for x in full if x != m)
        if not reduced:
            continue
        out[m] = run(exchanges, reduced).pass_rate - baseline
    return out
