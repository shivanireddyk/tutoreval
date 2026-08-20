"""The unit of evaluation: one tutoring exchange.

A tutoring exchange is a student turn followed by a tutor turn, plus the
context a grader needs in order to judge the tutor turn: what the correct
answer actually is, what grade level the student is at, and, when the tutor
is retrieval-augmented, what source material the tutor was given.

Everything here is frozen. An exchange that can be mutated between metrics
is an exchange whose score cannot be reproduced, and reproducibility is the
entire point of a benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExchangeError(ValueError):
    """The exchange is not usable. Never swallowed, never defaulted around."""


@dataclass(frozen=True)
class Exchange:
    """One student turn, one tutor turn, and the ground truth to judge it by.

    Attributes:
        uid: stable identifier, so a result can be traced back to its row.
        student: what the student said or asked.
        tutor: the tutor response under evaluation.
        answer: the correct final answer, as a student would write it. May be
            None for open-ended exchanges where no single answer exists; the
            leakage metric abstains rather than guessing in that case.
        grade_level: the US grade level of the student, used by the
            readability metric. None means "do not judge readability".
        context: passages the tutor was given, for grounding. Empty means the
            tutor was not retrieval-augmented and grounding does not apply.
        tags: free-form labels for slicing results, e.g. "algebra", "hint".
        expected_leak: the human label. True when a person judged that this
            response gives the answer away. None when unlabelled. This is what
            lets the benchmark score the leakage detector itself rather than
            only scoring tutors, which is the difference between a metric and
            a metric you have reason to trust.
    """

    uid: str
    student: str
    tutor: str
    answer: str | None = None
    grade_level: int | None = None
    context: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    expected_leak: bool | None = None

    def __post_init__(self) -> None:
        if not self.uid or not self.uid.strip():
            raise ExchangeError("uid is required; a result with no id cannot be traced")
        if not self.tutor or not self.tutor.strip():
            raise ExchangeError(f"{self.uid}: tutor turn is empty, nothing to evaluate")
        if self.grade_level is not None and not (1 <= self.grade_level <= 16):
            raise ExchangeError(
                f"{self.uid}: grade_level {self.grade_level} is outside 1 to 16. "
                f"Use 13 to 16 for undergraduate years."
            )

    @property
    def is_grounded(self) -> bool:
        """True when the tutor was given source material to work from."""
        return len(self.context) > 0


def _require(row: dict[str, Any], key: str, where: str) -> Any:
    if key not in row:
        raise ExchangeError(f"{where}: missing required field {key!r}")
    return row[key]


def exchange_from_dict(row: dict[str, Any], where: str = "row") -> Exchange:
    """Build an Exchange from a decoded JSON object, validating as we go."""
    if not isinstance(row, dict):
        raise ExchangeError(f"{where}: expected an object, got {type(row).__name__}")

    unknown = set(row) - {
        "uid", "student", "tutor", "answer", "grade_level", "context", "tags",
        "expected_leak",
    }
    if unknown:
        raise ExchangeError(
            f"{where}: unknown field(s) {sorted(unknown)}. A misspelled field would "
            f"otherwise be ignored and the row would not mean what its author intended."
        )

    ctx = row.get("context") or []
    if not isinstance(ctx, list) or not all(isinstance(c, str) for c in ctx):
        raise ExchangeError(f"{where}: 'context' must be a list of strings")
    tags = row.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ExchangeError(f"{where}: 'tags' must be a list of strings")

    grade = row.get("grade_level")
    if grade is not None and not isinstance(grade, int):
        raise ExchangeError(f"{where}: 'grade_level' must be an integer or absent")

    expected = row.get("expected_leak")
    if expected is not None and not isinstance(expected, bool):
        raise ExchangeError(f"{where}: 'expected_leak' must be true, false or absent")

    return Exchange(
        uid=str(_require(row, "uid", where)),
        student=str(_require(row, "student", where)),
        tutor=str(_require(row, "tutor", where)),
        answer=row.get("answer"),
        grade_level=grade,
        context=tuple(ctx),
        tags=tuple(tags),
        expected_leak=expected,
    )


def load_jsonl(path: str | Path) -> tuple[Exchange, ...]:
    """Load a benchmark file: one JSON object per line, blank lines ignored."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExchangeError(f"cannot read benchmark file {p}: {exc}") from None

    out: list[Exchange] = []
    seen: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{p.name}:{lineno}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExchangeError(f"{where}: not valid JSON: {exc.msg}") from None
        ex = exchange_from_dict(row, where)
        if ex.uid in seen:
            raise ExchangeError(
                f"{where}: duplicate uid {ex.uid!r}. Duplicate ids make per-row "
                f"results ambiguous and silently double-count in aggregates."
            )
        seen.add(ex.uid)
        out.append(ex)

    if not out:
        raise ExchangeError(f"{p}: contains no exchanges")
    return tuple(out)
