"""The metrics. Each one answers a narrow question and shows its working.

Every metric returns a Score carrying the exact span that drove the result.
A benchmark number nobody can audit is a benchmark nobody should trust, so
"why did this row score that way" has to be answerable without rerunning
anything and without reading this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .exchange import Exchange
from .normalize import (
    canonical_expression,
    clean,
    find_numbers,
    parse_number,
    sentences,
    word_number,
    words,
)


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class Score:
    """One metric's answer for one exchange, with the evidence for it."""

    metric: str
    verdict: Verdict
    value: float | None
    detail: str
    evidence: str = ""

    def __str__(self) -> str:
        v = "n/a" if self.value is None else f"{self.value:.2f}"
        return f"{self.metric}: {self.verdict.value} ({v}) {self.detail}"


# --------------------------------------------------------------------------
# Answer leakage
# --------------------------------------------------------------------------

# Phrases that put a value in answer position. Matched against the text
# immediately preceding an occurrence.
_ASSERTIVE = re.compile(
    r"(?:"
    r"answer\s*(?:is|:)|"
    r"equals?|=|"
    r"you\s+(?:get|end\s+up\s+with)|"
    r"that\s+(?:gives|makes|leaves)\s+(?:us\s+)?|"
    r"comes?\s+out\s+to|"
    r"result\s+is|solution\s+is|"
    r"so\s+it(?:'s|\s+is)|it(?:'s|\s+is)|"
    r"is\s+it|isn't\s+it|would\s+it\s+be|could\s+it\s+be|"
    r"therefore"
    r")[\s:]*$",
    re.IGNORECASE,
)

# Contexts where a bare number is structural rather than an answer.
_ENUM_BEFORE = re.compile(
    r"(?:step|part|question|problem|example|chapter|page|line|number|no\.|#)\s*$",
    re.IGNORECASE,
)
_ENUM_AFTER = re.compile(
    r"^\s*(?:steps?|parts?|terms?|ways?|methods?|options?|things?|"
    r"times|sides?|factors?|cases?)\b",
    re.IGNORECASE,
)

_LOOKBEHIND = 34
_LOOKAHEAD = 14
_NEXT_WORD = re.compile(r"^\W*([A-Za-z]+)")


def _singular(word: str) -> str:
    """Crudest possible stemmer: drop one trailing s. Enough for noun matching."""
    w = word.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _is_bare_statement(text: str, start: int, end: int) -> bool:
    """True when the occurrence is essentially the whole sentence it sits in."""
    left = text.rfind(".", 0, start)
    for mark in ("!", "?"):
        left = max(left, text.rfind(mark, 0, start))
    sent_start = left + 1
    nxt = [i for i in (text.find(c, end) for c in ".!?") if i != -1]
    sent_end = min(nxt) if nxt else len(text)
    remainder = (text[sent_start:start] + text[end:sent_end]).strip(" \t,:;-")
    return remainder == ""


def _occurrence_kind(
    text: str, start: int, end: int, in_question: bool, asked_about: frozenset[str]
) -> str:
    """Classify one occurrence of the answer inside the tutor turn.

    The subtle case is a number followed by a noun. "Let us do this in 3 steps"
    is structural. "A hexagon has 6 sides" is the answer to "how many sides
    does a hexagon have". Both are a small integer followed by a plural noun,
    and no amount of looking at the tutor turn alone separates them.

    What separates them is the student's question. If the noun after the
    number is the thing the student asked to have counted, the tutor has
    answered; if it is not, the tutor is organising their explanation.
    """
    before = text[max(0, start - _LOOKBEHIND) : start]
    after = text[end : end + _LOOKAHEAD]

    m = _NEXT_WORD.match(after)
    following = _singular(m.group(1)) if m else ""
    answers_the_question = bool(following) and following in asked_about

    if answers_the_question:
        return "assertive"

    # A value standing alone as its own sentence is the bluntest answer there
    # is. There is no phrase around it to match on, so a purely
    # context-driven classifier reads the most obvious leak in the corpus as
    # ambiguous. "Twenty." is not ambiguous.
    if _is_bare_statement(text, start, end):
        return "assertive"

    # Structural uses are checked on BOTH sides. Checking only the preceding
    # text misses "there are 3 steps", the most common false positive for
    # small integer answers.
    if _ENUM_BEFORE.search(before) or _ENUM_AFTER.match(after):
        return "enumerative"
    if _ASSERTIVE.search(before):
        return "assertive"
    if in_question:
        return "interrogative"
    return "ambiguous"


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for s in sentences(text):
        idx = text.find(s, cursor)
        if idx < 0:
            idx = cursor
        spans.append((idx, idx + len(s), s))
        cursor = idx + len(s)
    return spans


def answer_leakage(ex: Exchange) -> Score:
    """Did the tutor state the final answer outright?

    A tutor that answers the question has not tutored. This is the metric
    the whole harness exists for, and it is also the one most easily faked
    by naive substring matching, so occurrences are classified by the
    context they appear in rather than merely counted.
    """
    if ex.answer is None:
        return Score(
            "answer_leakage",
            Verdict.NOT_APPLICABLE,
            None,
            "no reference answer on this exchange, so leakage is undecidable",
        )

    text = clean(ex.tutor)
    target = parse_number(ex.answer)
    spans = _sentence_spans(text)
    asked_about = frozenset(_singular(w) for w in words(ex.student))

    def in_question(pos: int) -> bool:
        for a, b, s in spans:
            if a <= pos < b:
                return s.rstrip().endswith("?")
        return False

    kinds: list[tuple[str, str]] = []

    if target is not None:
        for value, start, end in find_numbers(text):
            if value == target:
                kinds.append(
                    (_occurrence_kind(text, start, end, in_question(start), asked_about),
                     text[max(0, start - 24) : end + 12].strip())
                )
        # A number spelled out in words is still the number. Missing these let
        # a one-word reply of "Twenty." pass as though it had said nothing.
        for m in re.finditer(r"[A-Za-z]+", text):
            if word_number(m.group(0)) == target:
                kinds.append(
                    (_occurrence_kind(text, m.start(), m.end(),
                                      in_question(m.start()), asked_about),
                     text[max(0, m.start() - 24) : m.end() + 12].strip())
                )
    else:
        needle = canonical_expression(ex.answer)
        if needle:
            for _a, _b, sent in spans:
                canon = canonical_expression(sent)
                # Locate the answer inside the ORIGINAL sentence, not the
                # canonical form. Passing a zero offset here meant the
                # preceding text was always empty, so "The answer is 2x" was
                # never seen as assertive. Every algebraic answer abstained.
                if needle in canon:
                    idx = sent.lower().find(ex.answer.strip().lower())
                    if idx < 0:
                        idx = max(0, len(sent) - len(needle))
                    kinds.append(
                        (_occurrence_kind(sent, idx, idx + len(ex.answer.strip()),
                                          sent.rstrip().endswith("?"), asked_about),
                         sent)
                    )

    if not kinds:
        return Score(
            "answer_leakage", Verdict.PASS, 0.0,
            "the reference answer does not appear in the response",
        )

    for kind, ev in kinds:
        if kind == "assertive":
            return Score(
                "answer_leakage", Verdict.FAIL, 1.0,
                "the response states the answer in answer position", ev,
            )

    if any(k == "ambiguous" for k, _ in kinds):
        ev = next(e for k, e in kinds if k == "ambiguous")
        return Score(
            "answer_leakage", Verdict.NEEDS_REVIEW, 0.5,
            "the answer appears but not in a form this metric can classify; "
            "a human should decide", ev,
        )

    if any(k == "interrogative" for k, _ in kinds):
        ev = next(e for k, e in kinds if k == "interrogative")
        return Score(
            "answer_leakage", Verdict.NEEDS_REVIEW, 0.5,
            "the answer is named inside a question; asking a student to confirm a "
            "value still tells them the value, so a human should decide", ev,
        )

    kind, ev = kinds[0]
    return Score(
        "answer_leakage", Verdict.PASS, 0.0,
        f"the answer appears only in {kind} position, which does not give it away", ev,
    )


# --------------------------------------------------------------------------
# Scaffolding
# --------------------------------------------------------------------------

_HINT = re.compile(
    r"\b(?:try|consider|notice|remember|recall|think about|what if|"
    r"start by|begin by|first|next|then|hint|look at|focus on|"
    r"what do you|can you|how would you|why do you)\b",
    re.IGNORECASE,
)


def scaffolding(ex: Exchange, minimum: int = 1) -> Score:
    """Does the response give the student something to do?

    Counts guiding moves: questions put back to the student and hint markers
    that point at a next step without taking it for them.
    """
    text = clean(ex.tutor)
    qs = [s for s in sentences(text) if s.rstrip().endswith("?")]
    hints = _HINT.findall(text)
    moves = len(qs) + len(hints)

    if moves >= minimum:
        ev = qs[0] if qs else (hints[0] if hints else "")
        return Score(
            "scaffolding", Verdict.PASS, float(moves),
            f"{len(qs)} question(s) and {len(hints)} hint marker(s)", str(ev),
        )
    return Score(
        "scaffolding", Verdict.FAIL, float(moves),
        "the response gives the student nothing to act on",
    )


def socratic_ratio(ex: Exchange, floor: float = 0.2) -> Score:
    """What share of the response's sentences put something back to the student?

    A pure answer dump scores zero. This is a blunt instrument on its own,
    which is why it is reported alongside leakage rather than instead of it.
    """
    ss = sentences(ex.tutor)
    if not ss:
        return Score("socratic_ratio", Verdict.NOT_APPLICABLE, None, "no sentences")
    qs = sum(1 for s in ss if s.rstrip().endswith("?"))
    ratio = qs / len(ss)
    verdict = Verdict.PASS if ratio >= floor else Verdict.FAIL
    return Score(
        "socratic_ratio", verdict, ratio,
        f"{qs} of {len(ss)} sentences are questions (floor {floor:.2f})",
    )


# --------------------------------------------------------------------------
# Readability
# --------------------------------------------------------------------------

_VOWELS = "aeiouy"


def count_syllables(word: str) -> int:
    """Approximate syllable count for one English word, never below 1.

    This is the standard vowel-group heuristic with a silent-e correction.
    It is wrong on some words and that is a known and accepted cost: the
    Flesch-Kincaid formula it feeds was itself fitted on this kind of
    approximation, and a benchmark is better served by a rule that is
    stable and inspectable than by one that is marginally more accurate
    and opaque.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(rf"[{_VOWELS}]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(1, n)


def flesch_kincaid_grade(text: str) -> float | None:
    """Flesch-Kincaid grade level, or None when there is too little text."""
    ss = sentences(text)
    ws = words(text)
    if len(ws) < 5 or not ss:
        return None
    syl = sum(count_syllables(w) for w in ws)
    return 0.39 * (len(ws) / len(ss)) + 11.8 * (syl / len(ws)) - 15.59


def readability_match(ex: Exchange, tolerance: float = 3.0) -> Score:
    """Is the response pitched near the student's grade level?

    Being far below the student's level is not penalised the same way as
    being far above it. Simple prose is rarely the reason a student fails
    to learn; prose two years above their reading level frequently is.
    """
    if ex.grade_level is None:
        return Score(
            "readability_match", Verdict.NOT_APPLICABLE, None,
            "no grade level on this exchange",
        )
    grade = flesch_kincaid_grade(ex.tutor)
    if grade is None:
        return Score(
            "readability_match", Verdict.NOT_APPLICABLE, None,
            "response too short to score reliably",
        )
    delta = grade - ex.grade_level
    if delta > tolerance:
        return Score(
            "readability_match", Verdict.FAIL, grade,
            f"reads at grade {grade:.1f}, which is {delta:.1f} above the student",
        )
    return Score(
        "readability_match", Verdict.PASS, grade,
        f"reads at grade {grade:.1f} against a student at grade {ex.grade_level}",
    )


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------

_STOP = frozenset({
    "a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "here", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "let", "lets", "me", "more", "most", "my", "no", "not", "of", "on", "or", "our",
    "over", "she", "should", "so", "some", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "too", "under", "us",
    "very", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "yes", "you", "your",
})


def content_words(text: str) -> list[str]:
    """Words that carry meaning, for grounding comparisons."""
    return [w for w in words(text) if w not in _STOP and len(w) > 2]


def grounding(ex: Exchange, floor: float = 0.6) -> Score:
    """How much of the response is supported by the retrieved context?

    Only meaningful for a retrieval-augmented tutor. Reports the share of
    content words that also appear in the supplied passages. This measures
    lexical support, not truth, and the report says so.
    """
    if not ex.is_grounded:
        return Score(
            "grounding", Verdict.NOT_APPLICABLE, None,
            "no retrieved context supplied, so grounding does not apply",
        )
    # Only declarative sentences are graded. A Socratic question necessarily
    # introduces words that are not in the source ("which of these do you
    # think..."), and grading those punished exactly the tutoring behaviour
    # the rest of this harness rewards.
    claims = [s for s in sentences(ex.tutor) if not s.rstrip().endswith("?")]
    if not claims:
        return Score(
            "grounding", Verdict.NOT_APPLICABLE, None,
            "the response makes no assertions, only asks questions",
        )
    response = content_words(" ".join(claims))
    if not response:
        return Score(
            "grounding", Verdict.NOT_APPLICABLE, None, "no content words in response"
        )
    supported = set()
    for passage in ex.context:
        supported |= set(content_words(passage))

    hits = [w for w in response if w in supported]
    ratio = len(hits) / len(response)
    missing = sorted({w for w in response if w not in supported})[:6]
    verdict = Verdict.PASS if ratio >= floor else Verdict.FAIL
    return Score(
        "grounding", verdict, ratio,
        f"{len(hits)} of {len(response)} content words appear in the context",
        "unsupported: " + ", ".join(missing) if missing else "",
    )


ALL_METRICS = (
    answer_leakage,
    scaffolding,
    socratic_ratio,
    readability_match,
    grounding,
)
