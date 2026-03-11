"""Tests for the high-level Python API: Document, Corpus, score(), and alpha modes."""

import math

import pytest

from suffix25 import Corpus, Document, score


# ── Document ─────────────────────────────────────────────────────────────────

class TestDocument:
    def test_score_exact_substring(self):
        doc = Document("The quick brown fox jumps over the lazy dog")
        assert doc.score("lazy dog") == 1.0

    def test_score_no_overlap(self):
        doc = Document("The quick brown fox jumps over the lazy dog")
        assert doc.score("xyz123") < 0.1

    def test_alpha_is_ignored(self):
        doc = Document("The quick brown fox")
        s1 = doc.score("brown fox", alpha=0.0)
        s2 = doc.score("brown fox", alpha=0.5)
        s3 = doc.score("brown fox", alpha=1.0)
        assert s1 == s2 == s3

    def test_serialization_roundtrip(self):
        original = Document("The quick brown fox jumps over the lazy dog")
        try:
            data = original.dumps()
        except NotImplementedError:
            pytest.skip("Serialization requires Cython extension")

        assert isinstance(data, bytes)
        assert len(data) > 0

        restored = Document.loads(data)
        assert original.score("lazy dog") == restored.score("lazy dog")
        assert original.score("quick fox") == restored.score("quick fox")


# ── Corpus: pure suffix automaton (alpha=1.0) ────────────────────────────────

class TestCorpusSuffixAutomaton:
    """Tests with alpha=1.0 to isolate the suffix automaton scoring path."""

    DOCS = [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn canine leaps above a sleepy hound",
        "Python programming language is awesome for search engines",
        "The quick quick fox",
    ]

    def test_alpha1_matches_standalone_document(self):
        corpus = Corpus(self.DOCS)
        query = "quick quick fox dog"
        corpus_scores = corpus.score_all(query, alpha=1.0)
        standalone_scores = [Document(d).score(query) for d in self.DOCS]

        for i, (cs, ss) in enumerate(zip(corpus_scores, standalone_scores, strict=True)):
            assert math.isclose(cs, ss, rel_tol=1e-9), (
                f"doc {i}: corpus={cs}, standalone={ss}"
            )

    def test_scores_independent_of_corpus_order(self):
        query = "quick brown fox"
        corpus_fwd = Corpus(self.DOCS)
        corpus_rev = Corpus(list(reversed(self.DOCS)))

        scores_fwd = corpus_fwd.score_all(query, alpha=1.0)
        scores_rev = corpus_rev.score_all(query, alpha=1.0)

        for i in range(len(self.DOCS)):
            ri = len(self.DOCS) - 1 - i
            assert math.isclose(scores_fwd[i], scores_rev[ri], rel_tol=1e-9)

    def test_search_alpha1_returns_correct_top(self):
        docs = ["apples and oranges", "just oranges", "completely unrelated"]
        corpus = Corpus(docs)
        top = corpus.search("apples and oranges", k=2, alpha=1.0)
        assert top[0] == 0

    def test_exact_substring_scores_one(self):
        corpus = Corpus(["The quick brown fox", "lazy dog", "hello world"])
        scores = corpus.score_all("brown fox", alpha=1.0)
        assert scores[0] == 1.0


# ── Corpus: pure BM25 (alpha=0.0) ────────────────────────────────────────────

class TestCorpusBM25:
    """Tests with alpha=0.0 to isolate the BM25 scoring path."""

    def test_bm25_prefers_rare_term(self):
        docs = [
            "the the the the the",
            "the cat sat on the mat",
            "a rare unique special token here",
        ]
        corpus = Corpus(docs)
        scores = corpus.score_all("rare unique", alpha=0.0)
        assert scores[2] > scores[0]
        assert scores[2] > scores[1]

    def test_bm25_zero_for_absent_term(self):
        docs = ["apple banana cherry", "dog cat bird"]
        corpus = Corpus(docs)
        scores = corpus.score_all("xylophone", alpha=0.0)
        assert all(s == 0.0 for s in scores)

    def test_bm25_term_in_fewer_docs_scores_higher(self):
        docs = [
            "apple banana cherry",
            "apple banana date",
            "elderberry fig grape",
            "apple cherry elderberry",
        ]
        corpus = Corpus(docs)
        # "fig" appears in only 1 doc, "apple" in 3 docs -- fig has higher IDF
        fig_scores = corpus.score_all("fig", alpha=0.0)
        apple_scores = corpus.score_all("apple", alpha=0.0)
        fig_best = max(fig_scores)
        apple_best = max(apple_scores)
        assert fig_best > apple_best

    def test_bm25_multi_term_query(self):
        docs = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "a rare unique special token here",
            "another document with words",
        ]
        corpus = Corpus(docs)
        scores = corpus.score_all("rare unique special", alpha=0.0)
        assert scores[2] > scores[0]
        assert scores[2] > scores[1]
        assert scores[2] > scores[3]

    def test_search_alpha0_ranks_by_bm25(self):
        docs = [
            "nothing relevant here",
            "the quick brown fox jumps over the lazy dog",
            "fox fox fox fox fox",
        ]
        corpus = Corpus(docs)
        top = corpus.search("fox", k=3, alpha=0.0)
        assert top[0] == 2


# ── Corpus: hybrid (alpha=0.5) ───────────────────────────────────────────────

class TestCorpusHybrid:
    """Tests with the default alpha=0.5 hybrid fusion."""

    def test_hybrid_is_between_sa_and_bm25(self):
        docs = [
            "The quick brown fox jumps over the lazy dog",
            "A completely different sentence about nothing",
            "brown fox and lazy dog together",
        ]
        corpus = Corpus(docs)
        query = "brown fox lazy dog"

        sa_scores = corpus.score_all(query, alpha=1.0)
        bm25_scores = corpus.score_all(query, alpha=0.0)
        hybrid_scores = corpus.score_all(query, alpha=0.5)

        for i in range(len(docs)):
            lo = min(sa_scores[i], bm25_scores[i])
            hi = max(sa_scores[i], bm25_scores[i])
            if lo == hi == 0.0:
                assert hybrid_scores[i] == 0.0
            else:
                assert hybrid_scores[i] >= lo - 1e-9, (
                    f"doc {i}: hybrid={hybrid_scores[i]} < min(sa={sa_scores[i]}, bm25={bm25_scores[i]})"
                )

    def test_hybrid_default_alpha(self):
        corpus = Corpus(["hello world", "foo bar"])
        default_scores = corpus.score_all("hello")
        explicit_scores = corpus.score_all("hello", alpha=0.5)
        assert default_scores == explicit_scores

    def test_bm25_normalization_produces_zero_to_one_range(self):
        docs = [
            "alpha beta gamma delta",
            "epsilon zeta eta theta",
            "alpha alpha alpha alpha",
        ]
        corpus = Corpus(docs)
        scores = corpus.score_all("alpha", alpha=0.5)
        for s in scores:
            assert 0.0 <= s <= 1.0 + 1e-9


# ── Corpus: general operations ────────────────────────────────────────────────

class TestCorpusOperations:
    def test_add_and_len(self):
        corpus = Corpus()
        assert len(corpus) == 0
        corpus.add("doc one")
        corpus.add("doc two")
        assert len(corpus) == 2

    def test_add_document_object(self):
        doc = Document("pre-compiled document")
        corpus = Corpus()
        corpus.add(doc)
        assert len(corpus) == 1
        scores = corpus.score_all("pre-compiled", alpha=1.0)
        assert scores[0] == 1.0

    def test_empty_corpus_returns_empty(self):
        corpus = Corpus()
        assert corpus.score_all("anything") == []
        assert corpus.search("anything") == []

    def test_serialization_roundtrip(self):
        docs = [
            "First document here",
            "Second document for testing",
            "Third document is the longest one of them all",
        ]
        original = Corpus(docs)

        try:
            data = original.dumps()
        except NotImplementedError:
            pytest.skip("Serialization requires Cython extension")

        assert isinstance(data, bytes)
        restored = Corpus.loads(data)
        assert len(restored) == 3

        query = "document"
        for alpha in (0.0, 0.5, 1.0):
            orig_scores = original.score_all(query, alpha=alpha)
            rest_scores = restored.score_all(query, alpha=alpha)
            for i, (o, r) in enumerate(zip(orig_scores, rest_scores)):
                assert math.isclose(o, r, rel_tol=1e-9), (
                    f"alpha={alpha}, doc {i}: original={o}, restored={r}"
                )


# ── Unified score() function ─────────────────────────────────────────────────

class TestUnifiedScore:
    def test_str_vs_str(self):
        assert score("test", "This is a test document") == 1.0

    def test_str_vs_document(self):
        doc = Document("This is a test document")
        assert score("test", doc) == 1.0

    def test_str_vs_list_str(self):
        results = score("fox", ["fox", "bar", "fox hunt"])
        assert len(results) == 3
        assert results[0] == 1.0
        assert results[1] == 0.0

    def test_str_vs_list_document(self):
        docs = [Document("fox"), Document("bar"), Document("fox hunt")]
        results = score("fox", docs)
        assert len(results) == 3
        assert results[0] == 1.0
        assert results[1] == 0.0

    def test_str_vs_corpus(self):
        corpus = Corpus(["fox", "bar", "fox hunt"])
        results = score("fox", corpus, alpha=1.0)
        assert len(results) == 3
        assert results[0] == 1.0
        assert results[1] == 0.0

    def test_list_str_and_list_document_agree(self):
        """Both list paths use pure SA (no BM25), so they should match exactly."""
        texts = ["The quick brown fox", "lazy dog", "hello world"]
        query = "quick brown"

        str_scores = score(query, texts)
        doc_scores = score(query, [Document(t) for t in texts])

        for i, (ss, ds) in enumerate(zip(str_scores, doc_scores)):
            assert math.isclose(ss, ds, rel_tol=1e-9), (
                f"doc {i}: list[str]={ss}, list[Document]={ds}"
            )

    def test_empty_list(self):
        assert score("query", []) == []

    def test_invalid_target_raises(self):
        with pytest.raises(TypeError):
            score("query", 42)
