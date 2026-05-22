"""Tests for dora_mini.answer_parsing — pure stdlib, runs on Mac."""
from dora_mini.answer_parsing import parse_true_false, parse_yes_no


def test_yes_no_basic():
    assert parse_yes_no("Yes") == "yes"
    assert parse_yes_no("No.") == "no"
    assert parse_yes_no("The answer is yes, definitely.") == "yes"


def test_yes_no_first_match_wins():
    assert parse_yes_no("no, actually yes") == "no"


def test_yes_no_none_when_absent():
    assert parse_yes_no("maybe") is None
    assert parse_yes_no("") is None


def test_yes_no_word_boundary():
    # 'yesterday' must not match 'yes'
    assert parse_yes_no("yesterday it rained") is None


def test_true_false_basic():
    assert parse_true_false("TRUE") == "true"
    assert parse_true_false("the correct answer is false") == "false"
    assert parse_true_false("nonsense") is None
