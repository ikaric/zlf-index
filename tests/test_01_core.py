"""Tests for the Suffix Automaton structural similarity (pure SA, no BM25)."""

import pytest

from suffix25 import score


class TestIdenticalAndDisjoint:
    def test_identical_returns_one(self):
        assert score("ABC", "ABC") == 1.0
        assert score("Hello World", "Hello World") == 1.0

    def test_disjoint_returns_zero(self):
        assert score("ABC", "XYZ") == 0.0
        assert score("Foo", "Bar") == 0.0

    def test_empty_query(self):
        assert score("", "ABC") == 0.0

    def test_empty_reference(self):
        assert score("ABC", "") == 0.0

    def test_both_empty(self):
        assert score("", "") == 0.0


class TestStructuralScoring:
    def test_high_overlap(self):
        assert score("AAAA", "AAAB") == 0.8

    def test_query_is_substring_of_reference(self):
        assert score("is a", "This is a test") == 1.0
        assert score("context", "This is a contextual test") == pytest.approx(1.0, abs=1e-4)

    def test_single_char_inserted_at_each_position(self):
        """Insert one foreign character at each position of 'hello' and verify
        known scores against the naive implementation."""
        cases = [
            ("hello", "hello", 1.0000),
            ("xhello", "hello", 0.6250),
            ("hxello", "hello", 0.3750),
            ("hexllo", "hello", 0.2500),
            ("helxlo", "hello", 0.2500),
            ("hellxo", "hello", 0.3750),
            ("hellox", "hello", 0.6250),
        ]
        for query, ref, expected in cases:
            assert score(query, ref) == pytest.approx(expected, abs=1e-4), (
                f"score({query!r}, {ref!r}) expected {expected}"
            )

    def test_score_is_asymmetric(self):
        a = score("fox", "The quick brown fox jumps")
        b = score("The quick brown fox jumps", "fox")
        assert a == 1.0
        assert b < a

    def test_more_overlap_scores_higher(self):
        ref = "The quick brown fox jumps over the lazy dog"
        less_overlap = score("quick xxx xxx xxx xxx", ref)
        more_overlap = score("quick brown fox xxx xxx", ref)
        assert more_overlap > less_overlap

    def test_order_matters(self):
        ref = "ABCDEF"
        forward = score("ABCD", ref)
        backward = score("DCBA", ref)
        assert forward > backward
