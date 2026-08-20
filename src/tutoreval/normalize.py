"""Turning written answers into things that can be compared.

The leakage metric has to decide whether a tutor said the answer. That sounds
like string matching and is not. A student answer of "0.5" is the same answer
as "1/2" and as "50%". An answer of "2x" is the same as "2 * x" and, in most
school contexts, as "2·x". Comparing the surface strings gets all three wrong.

So values are parsed into a canonical form and compared there. Numbers become
exact rational numbers, never floats, because 0.1 + 0.2 is a bad thing to
build a benchmark on. Algebraic expressions are normalised to a token
sequence rather than evaluated, because the point is whether the tutor wrote
the answer down, not whether the expression is mathematically equivalent to
it under some substitution.
"""

from __future__ import annotations

import re
import unicodedata
from fractions import Fraction

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_MULT_SIGNS = {"·", "×", "*", "⋅"}

_NUMBER = re.compile(
    r"""
    (?:(?P<sign>[-+])\s*)?
    (?:
        (?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s*/\s*(?P<den>\d+(?:\.\d+)?)   # 1/2, 3/4
      | (?P<dec>\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)                          # 12, 1.5, .5
    )
    (?:\s*(?P<pct>%))?
    """,
    re.VERBOSE,
)

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "twenty": 20, "thirty": 30, "fifty": 50, "hundred": 100,
}


def clean(text: str) -> str:
    """Normalise unicode, superscripts and multiplication signs to plain ASCII."""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_SUPERSCRIPT)
    for sign in _MULT_SIGNS:
        t = t.replace(sign, "*")
    t = t.replace("−", "-")  # unicode minus
    return t


def parse_number(token: str) -> Fraction | None:
    """Parse a single written number into an exact Fraction, or None.

    Handles integers, decimals, thousands separators, fractions and
    percentages. Returns None when the token is not a number, which the
    caller must treat as "this is not comparable as a number" rather than
    as zero.
    """
    m = _NUMBER.fullmatch(clean(token).strip())
    if m is None:
        return None

    sign = -1 if m.group("sign") == "-" else 1
    if m.group("num") is not None:
        num = Fraction(m.group("num").replace(",", ""))
        den = Fraction(m.group("den"))
        if den == 0:
            return None
        value = num / den
    else:
        value = Fraction(m.group("dec").replace(",", ""))

    if m.group("pct"):
        value = value / 100
    return sign * value


def find_numbers(text: str) -> list[tuple[Fraction, int, int]]:
    """Every number in the text, with the span it occupies.

    Spans are offsets into the cleaned text, not the original, so callers
    that need to quote evidence should quote from clean(text). The span covers
    the number and nothing else: no leading or trailing whitespace, because
    every span here ends up quoted back to a human as evidence.
    """
    cleaned = clean(text)
    out: list[tuple[Fraction, int, int]] = []
    for m in _NUMBER.finditer(cleaned):
        value = parse_number(m.group(0))
        if value is not None:
            out.append((value, m.start(), m.end()))
    return out


def canonical_expression(text: str) -> str:
    """A canonical token string for an algebraic answer.

    "2x", "2 * x" and "2·x" all become "2*x". This is deliberately a
    normalisation and not an evaluation: "x+x" does not become "2*x",
    because a tutor writing "x+x" has not written the answer down in the
    form the student was asked for.
    """
    t = clean(text).lower().strip()
    t = t.replace("^", "**")
    t = re.sub(r"\s+", "", t)
    # insert explicit multiplication between a digit and a following letter
    t = re.sub(r"(?<=\d)(?=[a-z])", "*", t)
    # and between a closing bracket and a letter or digit
    t = re.sub(r"(?<=\))(?=[a-z0-9])", "*", t)
    t = t.rstrip(".")
    return t


def word_number(token: str) -> Fraction | None:
    """Map a spelled-out small number to a Fraction, or None."""
    return (
        Fraction(_WORD_NUMBERS[token.lower()])
        if token.lower() in _WORD_NUMBERS
        else None
    )


def equivalent(a: str, b: str) -> bool:
    """Do these two written answers denote the same thing?

    Numbers are compared exactly as rationals, so "0.5", "1/2" and "50%"
    agree. Anything that is not a number falls back to canonical expression
    comparison. This never raises; an unparseable input simply fails to
    match, which is the safe direction for a leakage test.
    """
    na, nb = parse_number(a), parse_number(b)
    if na is not None and nb is not None:
        return na == nb
    wa = word_number(a.strip()) if na is None else na
    wb = word_number(b.strip()) if nb is None else nb
    if wa is not None and wb is not None:
        return wa == wb
    return canonical_expression(a) == canonical_expression(b)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    """Split into sentences, keeping it simple and predictable.

    A real sentence splitter would handle abbreviations and decimals. This
    one deliberately does not, because every metric downstream reports the
    sentence it fired on, and a splitter whose behaviour is obvious is
    easier to argue with than one that is usually right.
    """
    parts = [s.strip() for s in _SENTENCE_END.split(clean(text).strip())]
    return [p for p in parts if p]


_WORD = re.compile(r"[A-Za-z']+")


def words(text: str) -> list[str]:
    """Alphabetic word tokens, lowercased."""
    return [w.lower() for w in _WORD.findall(clean(text))]
