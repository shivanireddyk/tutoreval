"""Turning a set of metric scores into one decision about one exchange.

The aggregation rule is deliberately not a weighted average. Averaging lets
a response that hands over the answer pass by being well written, which is
exactly the failure this harness exists to catch. Leakage is treated as
disqualifying on its own; the pedagogical metrics can only fail a response
that did not already fail.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exchange import Exchange
from .metrics import (
    Score,
    Verdict,
    answer_leakage,
    grounding,
    readability_match,
    scaffolding,
    socratic_ratio,
)

DEFAULT_METRICS = (
    "answer_leakage",
    "scaffolding",
    "socratic_ratio",
    "readability_match",
    "grounding",
)

_REGISTRY = {
    "answer_leakage": answer_leakage,
    "scaffolding": scaffolding,
    "socratic_ratio": socratic_ratio,
    "readability_match": readability_match,
    "grounding": grounding,
}


class UnknownMetricError(ValueError):
    """A metric was requested that does not exist. Silently skipping it would
    make an ablation study quietly measure the wrong thing."""


@dataclass(frozen=True)
class Result:
    """Every metric's score for one exchange, plus the overall call."""

    uid: str
    verdict: Verdict
    scores: tuple[Score, ...]
    reason: str

    def score_for(self, metric: str) -> Score | None:
        for s in self.scores:
            if s.metric == metric:
                return s
        return None

    @property
    def leaked(self) -> bool:
        """True when the leakage metric judged that the answer was given away."""
        s = self.score_for("answer_leakage")
        return s is not None and s.verdict is Verdict.FAIL


def evaluate(ex: Exchange, metrics: tuple[str, ...] = DEFAULT_METRICS) -> Result:
    """Score one exchange under the named metrics.

    Passing a subset of metrics is how ablation studies are run: disable one,
    rerun the benchmark, and read off what it was contributing.
    """
    unknown = [m for m in metrics if m not in _REGISTRY]
    if unknown:
        raise UnknownMetricError(
            f"unknown metric(s) {unknown}. Known: {sorted(_REGISTRY)}"
        )

    scores = tuple(_REGISTRY[m](ex) for m in metrics)

    leak = next((s for s in scores if s.metric == "answer_leakage"), None)
    if leak is not None and leak.verdict is Verdict.FAIL:
        return Result(ex.uid, Verdict.FAIL, scores,
                      "gives the answer away, so nothing else can rescue it")
    if leak is not None and leak.verdict is Verdict.NEEDS_REVIEW:
        return Result(ex.uid, Verdict.NEEDS_REVIEW, scores,
                      "cannot decide whether the answer was given away")

    failed = [s for s in scores if s.verdict is Verdict.FAIL]
    if failed:
        return Result(ex.uid, Verdict.FAIL, scores,
                      "fails " + ", ".join(s.metric for s in failed))

    review = [s for s in scores if s.verdict is Verdict.NEEDS_REVIEW]
    if review:
        return Result(ex.uid, Verdict.NEEDS_REVIEW, scores,
                      "needs review on " + ", ".join(s.metric for s in review))

    return Result(ex.uid, Verdict.PASS, scores, "meets every metric applied")
