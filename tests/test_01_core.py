"""Correctness tests for ZLF Index."""

import pytest

from zlfi import score


def test_identical_sequences():
    """Identical sequences should return 1.0."""
    assert score("ABC", "ABC") == 1.0
    assert score("Hello World", "Hello World") == 1.0


def test_disjunctive_sequences():
    """Disjunctive sequences should return 0.0."""
    assert score("ABC", "XYZ") == 0.0
    assert score("Foo", "Bar") == 0.0


def test_high_structural_similarity():
    """High similarity test."""
    assert score("AAAA", "AAAB") == 0.8


def test_partial_subsequence_match():
    """Partial match test."""
    s = score("context", "This is a contextual test")
    assert pytest.approx(s, 0.0001) == 1.0


def test_empty_query_sequence():
    """Empty query sequence should return 0.0."""
    assert score("", "ABC") == 0.0


def test_empty_reference_sequence():
    """Empty reference sequence should return 0.0."""
    assert score("ABC", "") == 0.0


def test_both_empty_sequences():
    """Both empty sequences should return 0.0."""
    assert score("", "") == 0.0


def test_subset_sequence():
    """Query is a perfect contiguous subset of Reference."""
    # N ="is a" (len = 4)
    # M ="This is a test"
    # The max score max contiguous is "is a", len 4. Wait, the formula evaluates N.
    # N is perfectly in M, so score should be 1.0
    assert score("is a", "This is a test") == 1.0


def test_specific_similarity_scores():
    """Test specific similarity scores confirmed by naive implementation."""
    test_cases = [
        ("hello", "hello", 1.0000),
        ("xhello", "hello", 0.6250),
        ("hxello", "hello", 0.3750),
        ("hexllo", "hello", 0.2500),
        ("helxlo", "hello", 0.2500),
        ("hellxo", "hello", 0.3750),
        ("hellox", "hello", 0.6250),
    ]
    for seq_n, seq_m, expected_score in test_cases:
        s = score(seq_n, seq_m)
        assert pytest.approx(s, 0.0001) == expected_score
