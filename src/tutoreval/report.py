"""Rendering results for a person, not for a dashboard.

Every number here is followed by the reason for it. A benchmark report that
says "pass rate 0.62" and stops is a report that will be quoted in a slide
deck by someone who never saw which rows failed or why.
"""

from __future__ import annotations

from .benchmark import BenchmarkResult
from .evaluate import Result
from .metrics import Verdict


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render_result(r: Result, indent: str = "") -> str:
    """One exchange, every metric, with evidence."""
    lines = [f"{indent}{r.uid}: {r.verdict.value} ({r.reason})"]
    for s in r.scores:
        if s.verdict is Verdict.NOT_APPLICABLE:
            continue
        value = "n/a" if s.value is None else f"{s.value:.2f}"
        lines.append(f"{indent}  {s.metric:<18} {s.verdict.value:<13} {value:>6}  {s.detail}")
        if s.evidence:
            lines.append(f"{indent}      evidence: {s.evidence}")
    return "\n".join(lines)


def render(b: BenchmarkResult, show: int = 5) -> str:
    """The full run: headline numbers, detector quality, then examples."""
    counts = b.verdict_counts
    out: list[str] = []
    out.append("Tutoring quality benchmark")
    out.append("=" * 64)
    out.append(f"exchanges evaluated : {b.total}")
    out.append(f"metrics applied     : {', '.join(b.metrics)}")
    out.append("")
    share = _pct(counts["PASS"] / b.total) if b.total else "n/a"
    out.append(f"  PASS          {counts['PASS']:>4}   {share}")
    out.append(f"  FAIL          {counts['FAIL']:>4}")
    out.append(f"  NEEDS_REVIEW  {counts['NEEDS_REVIEW']:>4}")
    out.append("")

    c = b.confusion
    out.append("Leakage detector against human labels")
    out.append("-" * 64)
    if c.labelled == 0:
        out.append("  no labelled rows, so the detector itself is unmeasured")
    else:
        out.append(f"  labelled rows : {c.labelled}   (abstained on {c.abstained})")
        out.append(f"  true positive : {c.true_positive:<4} false positive: {c.false_positive}")
        out.append(f"  true negative : {c.true_negative:<4} false negative: {c.false_negative}")
        out.append(f"  precision     : {_pct(c.precision)}")
        out.append(f"  recall        : {_pct(c.recall)}")
        out.append(f"  F1            : {_pct(c.f1)}")
    out.append("")

    fails = b.failures()
    if fails:
        out.append(f"Failures ({len(fails)}), first {min(show, len(fails))} shown")
        out.append("-" * 64)
        for r in fails[:show]:
            out.append(render_result(r, "  "))
            out.append("")

    review = b.needing_review()
    if review:
        out.append(f"Needing human review ({len(review)})")
        out.append("-" * 64)
        for r in review[:show]:
            out.append(render_result(r, "  "))
            out.append("")

    out.append("Lexical grounding measures word overlap with the retrieved passages.")
    out.append("It is not a check on whether the response is true.")
    return "\n".join(out)


def render_ablation(deltas: dict[str, float]) -> str:
    """Leave-one-out effects, largest first."""
    out = ["Ablation: change in pass rate when each metric is removed",
           "-" * 64]
    if not deltas:
        out.append("  nothing to ablate")
        return "\n".join(out)
    for name, d in sorted(deltas.items(), key=lambda kv: -abs(kv[1])):
        arrow = "+" if d > 0 else ""
        note = "" if abs(d) > 1e-9 else "   (no effect on this dataset)"
        out.append(f"  without {name:<18} {arrow}{d * 100:.1f} points{note}")
    return "\n".join(out)
