"""Tests for dora_mini.answer_parsing — pure stdlib, runs on Mac."""
from dora_mini.answer_parsing import (
    parse_true_false,
    parse_yes_no,
    parse_yes_no_or_true_false,
)


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


def test_yes_no_or_true_false_maps_yes_no_to_true_false():
    assert parse_yes_no_or_true_false("yes") == "true"
    assert parse_yes_no_or_true_false("no") == "false"


def test_yes_no_or_true_false_passes_true_false_through():
    assert parse_yes_no_or_true_false("true") == "true"
    assert parse_yes_no_or_true_false("false") == "false"


def test_yes_no_or_true_false_handles_realistic_completions():
    # The low-rank cs170k runs emit exactly these — see the Step 8 parser-check diagnostic.
    assert parse_yes_no_or_true_false("the correct answer is yes") == "true"
    assert parse_yes_no_or_true_false("the correct answer is no") == "false"
    assert parse_yes_no_or_true_false("the correct answer is true") == "true"


def test_yes_no_or_true_false_case_insensitive():
    assert parse_yes_no_or_true_false("YES") == "true"
    assert parse_yes_no_or_true_false("False.") == "false"


def test_yes_no_or_true_false_first_match_wins():
    assert parse_yes_no_or_true_false("no, actually yes") == "false"


def test_yes_no_or_true_false_none_when_absent():
    assert parse_yes_no_or_true_false("maybe later") is None
    assert parse_yes_no_or_true_false("") is None
