import json

import pytest

from tutoreval.exchange import (
    Exchange,
    ExchangeError,
    exchange_from_dict,
    load_jsonl,
)


def test_minimal_exchange():
    ex = Exchange(uid="a", student="q", tutor="t")
    assert ex.answer is None
    assert not ex.is_grounded


def test_empty_uid_rejected():
    with pytest.raises(ExchangeError, match="uid is required"):
        Exchange(uid="  ", student="q", tutor="t")


def test_empty_tutor_turn_rejected():
    with pytest.raises(ExchangeError, match="nothing to evaluate"):
        Exchange(uid="a", student="q", tutor="   ")


@pytest.mark.parametrize("grade", [0, 17, -1])
def test_grade_level_out_of_range(grade):
    with pytest.raises(ExchangeError, match="outside 1 to 16"):
        Exchange(uid="a", student="q", tutor="t", grade_level=grade)


def test_exchange_is_frozen():
    # An exchange that can be mutated between metrics is an exchange whose
    # score cannot be reproduced.
    ex = Exchange(uid="a", student="q", tutor="t")
    with pytest.raises(AttributeError):
        ex.tutor = "changed"


def test_is_grounded_reflects_context():
    assert Exchange(uid="a", student="q", tutor="t", context=("p",)).is_grounded


def test_unknown_field_is_an_error_not_a_shrug():
    with pytest.raises(ExchangeError, match="unknown field"):
        exchange_from_dict({"uid": "a", "student": "q", "tutor": "t", "grade": 5})


def test_missing_required_field():
    with pytest.raises(ExchangeError, match="missing required field"):
        exchange_from_dict({"uid": "a", "student": "q"})


def test_context_must_be_list_of_strings():
    with pytest.raises(ExchangeError, match="'context' must be"):
        exchange_from_dict({"uid": "a", "student": "q", "tutor": "t", "context": "p"})


def test_expected_leak_must_be_boolean():
    with pytest.raises(ExchangeError, match="must be true, false or absent"):
        exchange_from_dict(
            {"uid": "a", "student": "q", "tutor": "t", "expected_leak": "yes"}
        )


def _write(tmp_path, rows):
    p = tmp_path / "b.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_load_jsonl_round_trip(tmp_path):
    p = _write(tmp_path, [{"uid": "a", "student": "q", "tutor": "t", "answer": "1"}])
    (exs,) = load_jsonl(p)
    assert exs.uid == "a" and exs.answer == "1"


def test_load_jsonl_rejects_duplicate_uids(tmp_path):
    p = _write(tmp_path, [
        {"uid": "a", "student": "q", "tutor": "t"},
        {"uid": "a", "student": "q2", "tutor": "t2"},
    ])
    with pytest.raises(ExchangeError, match="duplicate uid"):
        load_jsonl(p)


def test_load_jsonl_reports_line_number(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text('{"uid":"a","student":"q","tutor":"t"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ExchangeError, match="b.jsonl:2"):
        load_jsonl(p)


def test_load_jsonl_ignores_blank_lines(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text('\n{"uid":"a","student":"q","tutor":"t"}\n\n', encoding="utf-8")
    assert len(load_jsonl(p)) == 1


def test_load_jsonl_empty_file_is_an_error(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ExchangeError, match="contains no exchanges"):
        load_jsonl(p)


def test_missing_file():
    with pytest.raises(ExchangeError, match="cannot read"):
        load_jsonl("/nonexistent/nope.jsonl")
