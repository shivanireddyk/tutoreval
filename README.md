# tutoreval

**Measuring whether an AI tutor teaches, rather than whether it answers.**

A tutoring response can be factually perfect and pedagogically useless. Ask a
model *"what is the derivative of x squared"* and a reply of *"2x"* is correct,
and has taught nobody anything.

Most LLM evaluation scores the first property and is silent on the second. This
library scores the second. It is deterministic, makes no network calls, needs no
API key or GPU, and every score it returns carries the span of text that caused
it.

```python
from tutoreval import Exchange, evaluate

good = Exchange(
    uid="d1",
    student="What is the derivative of x^2?",
    tutor="Bring the exponent down in front, then reduce it by one. What do you get?",
    answer="2x",
    grade_level=11,
)
bad = Exchange(uid="d2", student=good.student, tutor="The answer is 2x.",
               answer="2x", grade_level=11)

evaluate(good).verdict   # Verdict.PASS
evaluate(bad).verdict    # Verdict.FAIL
evaluate(bad).reason     # 'gives the answer away, so nothing else can rescue it'
```

## Install and run

```bash
git clone https://github.com/shivanireddyk/tutoreval
cd tutoreval
pip install -e ".[dev]"

pytest          # 137 tests
python demo.py  # runs the shipped benchmark and prints the report
```

No dependencies outside the standard library. The dev extra is pytest and ruff.

## The five metrics

| Metric | Question it answers |
|---|---|
| `answer_leakage` | Did the tutor state the final answer outright? |
| `scaffolding` | Did the response give the student something to do? |
| `socratic_ratio` | What share of sentences put something back to the student? |
| `readability_match` | Is the prose pitched near the student's grade level? |
| `grounding` | For a RAG tutor, are the claims supported by the retrieved passages? |

Aggregation is deliberately **not** a weighted average. Averaging lets a
response that hands over the answer pass by being well written, which is
precisely the failure the harness exists to catch. Leakage is disqualifying on
its own; the other four can only fail a response that did not already fail.

## Why leakage is the hard one

Leakage sounds like substring matching and is not. Four things break the naive
version, and all four are handled:

**Numeric equivalence.** `0.5`, `1/2` and `50%` are the same answer. Values are
parsed to exact rational numbers, never floats, so the benchmark does not
inherit `0.1 + 0.2 != 0.3`.

**Position.** *"The answer is 42"* leaks. *"Let us do this in 3 steps"*, where
the answer happens to be 3, does not. Occurrences are classified by the context
they sit in rather than counted.

**The counted noun.** *"Let us do this in 3 steps"* and *"a hexagon has 6
sides"* are both a small integer followed by a plural noun, and **nothing in the
tutor turn separates them.** What separates them is the student's question: if
the noun is the thing the student asked to have counted, the tutor answered it.

**Abstention.** When an occurrence cannot be classified, the metric returns
`NEEDS_REVIEW` rather than guessing. A confident wrong label in a benchmark is
worse than a gap, because the gap is visible.

## The benchmark

`data/tutoring_benchmark_v1.jsonl` holds 26 hand-written tutoring exchanges
across arithmetic, algebra, fractions, percentages, geometry, chemistry,
biology, physics and history. 19 carry a human label for whether the response
gives the answer away.

The dataset is built to be adversarial to its own metric. It includes the answer
appearing in enumerative position, in a confirmation question, spelled out as a
word, as an equivalent fraction, and as a bare one-word reply.

**Self-scoring is the point.** Running a benchmark tells you how a *tutor* did.
Scoring the detector against human labels tells you whether that number means
anything:

```
labelled rows : 19   (abstained on 7)
precision     : 100.0%
recall        : 100.0%
```

### What that number is not

100% precision and recall on 19 rows written by the same person who wrote the
metric is **weak evidence and should be read as such.** It says the rules are
self-consistent and that the known failure modes are covered. It does not say
the metric generalises to tutoring text it has never seen.

Honest next steps, in order of value: a second annotator and an inter-annotator
agreement figure; several hundred rows drawn from real tutor logs rather than
written to order; and a held-out split that the metric author never reads.

## Six bugs the benchmark found

Every one of these was caught by running the harness against hand labels, not by
reading the code. Each has a regression test in `tests/test_leakage.py`.

1. **Algebraic answers never reached the assertive check.** Non-numeric answers
   took a separate code path that passed a zero offset for the occurrence, so
   the text before the answer was always empty. *"The answer is 2x"* abstained.
   Every algebraic answer in the corpus was silently unscored.

2. **Structural context was only checked on one side.** Looking only at the text
   before a number catches *"step 3"* but misses *"there are 3 steps"*, the
   most common false positive for small integer answers.

3. **The counted noun.** Fixing (2) then over-corrected: *"a hexagon has 6
   sides"* was read as structural. The fix is the student's question, described
   above.

4. **Confirmation questions were treated as Socratic.** *"Is it 42?"* was
   excused because the sentence ended in a question mark. Naming the value
   inside a question still hands it over.

5. **Spelled-out and bare answers were invisible.** A one-word reply of
   *"Twenty."* scanned as containing no numbers at all, because only digits
   were searched for. A bare value had a second problem: with no surrounding
   phrase there was no context to classify, so the bluntest leak in the corpus
   was escalated rather than failed. Discourse markers count as no context, so
   *"Well, 42!"* is now treated exactly like *"42."*

6. **Lexical grounding punished good tutoring.** A Socratic question necessarily
   introduces words absent from the source passage, so asking the student
   anything drove the grounding score down. Grounding now grades declarative
   sentences only.

Number 6 is the one worth dwelling on. The metric was working exactly as
specified and was still wrong, because the specification quietly assumed a tutor
that only makes claims. That class of error does not show up in unit tests of
the metric; it shows up when you look at which rows failed and ask whether you
agree.

## Ablation

`ablate()` reruns the benchmark with each metric removed in turn and reports the
change in pass rate. A metric that moves nothing is either redundant with
another or is not firing on this data, and either way should not be reported as
though it contributed.

```python
from tutoreval import ablate, load_jsonl, render_ablation
print(render_ablation(ablate(load_jsonl("data/tutoring_benchmark_v1.jsonl"))))
```

## Limitations, stated plainly

- **Grounding measures lexical overlap, not truth.** A fluent paraphrase that
  reuses source vocabulary scores well whether or not it is accurate. The report
  says so in its own output rather than leaving it to be discovered.
- **Syllable counting is a heuristic**, so Flesch-Kincaid grades are
  approximate. It is wrong on some words, and that cost is accepted for a rule
  that is stable and inspectable.
- **English only.** Every regular expression here assumes English word order.
- **The metrics are lexical, not semantic.** A tutor that leaks the answer by
  implication rather than by stating it will pass. Closing that gap needs a
  model in the loop, which costs the determinism the rest of the design is
  built on. That trade is worth making deliberately, not by accident.

## Layout

```
src/tutoreval/
  exchange.py    the data model, validated on construction
  normalize.py   numeric and expression equivalence
  metrics.py     the five metrics, each returning evidence
  evaluate.py    aggregation into one verdict per exchange
  benchmark.py   dataset runs, the confusion matrix, ablation
  report.py      human-readable output
data/            the benchmark, one JSON object per line
tests/           137 tests
```

## Licence

MIT.
