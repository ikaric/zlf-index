"""Tests for the high-level Python API."""

import pytest

from zlfi import Corpus, Document, score


def test_document_score():
    """Test 1-to-1 scoring with Document."""
    doc = Document("The quick brown fox jumps over the lazy dog")
    assert doc.score("lazy dog") == 1.0
    assert doc.score("xyz123") < 0.1  # very little overlap


def test_document_serialization():
    """Test serializing and deserializing a Document."""
    original = Document("The quick brown fox jumps over the lazy dog")

    # Try serialization
    try:
        data = original.dumps()
    except NotImplementedError:
        pytest.skip("Serialization requires Cython extension")

    assert isinstance(data, bytes)
    assert len(data) > 0

    # Try deserialization
    restored = Document.loads(data)

    # Verify exact same scoring behavior
    assert original.score("lazy dog") == restored.score("lazy dog")
    assert original.score("quick fox") == restored.score("quick fox")


def test_corpus_basics():
    """Test building and scoring a Corpus."""
    docs = ["The quick brown fox", "jumps over the", "lazy dog"]
    corpus = Corpus(docs)

    assert len(corpus) == 3

    # Add one more
    corpus.add("hello world")
    assert len(corpus) == 4

    # Score all
    scores = corpus.score_all("fox")
    assert len(scores) == 4
    assert scores[0] == 1.0  # exact match for "fox" in first doc
    assert scores[1] < 0.2  # "fox" shares very little with "jumps over the"

    # Search top k
    top_indices = corpus.search("over the", k=2)
    assert len(top_indices) == 2
    assert top_indices[0] == 1  # 2nd document is best match


def test_corpus_alpha_1_matches_pure_zlfi():
    """alpha=1.0 must produce identical scores to standalone Document scoring."""
    import math

    docs = [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn canine leaps above a sleepy hound",
        "Python programming language is awesome for search engines",
        "The quick quick fox",
    ]
    corpus = Corpus(docs)
    query = "quick quick fox dog"

    hybrid_sa = corpus.score_all(query, alpha=1.0)
    pure_sa = [Document(d).score(query) for d in docs]

    for i, (h, p) in enumerate(zip(hybrid_sa, pure_sa, strict=True)):
        assert math.isclose(h, p, rel_tol=1e-9), (
            f"doc {i}: alpha=1.0 gave {h}, standalone ZLFI gave {p}"
        )


def test_corpus_alpha_0_matches_rank_bm25():
    """alpha=0.0 must produce identical scores to rank_bm25.BM25Okapi."""
    import math
    import re

    from rank_bm25 import BM25Okapi

    docs = [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn canine leaps above a sleepy hound",
        "Python programming language is awesome for search engines",
        "The quick quick fox",
        "brown brown brown brown",
        "fox fox fox fox fox fox fox fox",
    ]
    queries = [
        "quick fox",
        "brown dog lazy",
        "python search engines",
        "canine hound sleepy",
        "quick quick fox dog",
        "nonexistent terms xyz",
    ]

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    tokenized_docs = [tokenize(d) for d in docs]
    bm25_ref = BM25Okapi(tokenized_docs)

    corpus = Corpus(docs)

    for query in queries:
        our_scores = corpus.score_all(query, alpha=0.0)
        ref_scores = bm25_ref.get_scores(tokenize(query)).tolist()

        assert len(our_scores) == len(ref_scores), (
            f"length mismatch for query '{query}'"
        )
        for i, (ours, ref) in enumerate(
            zip(our_scores, ref_scores, strict=True)
        ):
            assert math.isclose(ours, ref, rel_tol=1e-9, abs_tol=1e-12), (
                f"query='{query}', doc {i}: ours={ours}, "
                f"rank_bm25={ref}"
            )


def test_sa_document_independence():
    """Each document's SA score must be independent of other corpus members."""
    import math

    docs = [
        "The quick brown fox",
        "jumps over the lazy dog",
        "completely unrelated text",
    ]
    query = "quick brown fox"

    corpus_full = Corpus(docs)
    full_scores = corpus_full.score_all(query, alpha=1.0)

    for i, doc_text in enumerate(docs):
        standalone = Document(doc_text).score(query)
        assert math.isclose(full_scores[i], standalone, rel_tol=1e-9), (
            f"doc {i}: corpus gave {full_scores[i]}, "
            f"standalone gave {standalone}"
        )

    corpus_reversed = Corpus(list(reversed(docs)))
    rev_scores = corpus_reversed.score_all(query, alpha=1.0)
    for i in range(len(docs)):
        ri = len(docs) - 1 - i
        assert math.isclose(full_scores[i], rev_scores[ri], rel_tol=1e-9), (
            f"doc {i}: order-dependent scores "
            f"{full_scores[i]} vs {rev_scores[ri]}"
        )


def test_corpus_serialization():
    """Test serializing and deserializing an entire Corpus."""
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

    # Verify scoring is identical
    query = "document"
    orig_scores = original.score_all(query)
    rest_scores = restored.score_all(query)

    assert orig_scores == rest_scores


def test_unified_score_function():
    """Test the overloaded score() function."""
    doc_str = "This is a test document"
    query = "test"

    # 1. str vs str
    assert score(query, doc_str) == 1.0

    # 2. str vs Document
    doc_obj = Document(doc_str)
    assert score(query, doc_obj) == 1.0

    # 3. str vs list[str]
    str_list = ["foo", doc_str, "bar"]
    scores1 = score(query, str_list)
    assert len(scores1) == 3
    assert scores1[1] == 1.0

    # 4. str vs list[Document]
    doc_list = [Document("foo"), doc_obj, Document("bar")]
    scores2 = score(query, doc_list)
    assert scores1 == scores2

    # 5. str vs Corpus
    corpus = Corpus(str_list)
    scores3 = score(query, corpus)
    assert scores1 == scores3
