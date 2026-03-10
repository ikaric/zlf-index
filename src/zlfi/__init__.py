"""Zipf Latent Fusion Index (ZLFI)

A sequential similarity metric that evaluates the structural, ordered, and contiguous
overlap between a query sequence and a reference sequence.
"""

# ruff: noqa: I001

from __future__ import annotations

import array
import re

try:
    # Use the Cython-accelerated implementation when available
    # fmt: off
    from zlfi._core import (  # type: ignore[import-untyped, unused-ignore, import-not-found]
        BM25Index,
        SuffixAutomatonWrapper as State,
        batch_scores as _batch_scores,
        batch_top_k as _batch_top_k,
        build_suffix_automaton as _build_suffix_automaton,
        zlfi as _zlfi,
    )
    # fmt: on
except ImportError:
    # Fall back to pure-Python implementation
    # fmt: off
    from zlfi.core import State, BM25Index  # type: ignore[assignment, unused-ignore]
    from zlfi.core import (
        build_suffix_automaton as _build_suffix_automaton,
        zlfi as _zlfi
    )
    # fmt: on

    def _batch_scores(query: str, automatons: list[State]) -> list[float]:
        return [_zlfi(query, a) for a in automatons]

    def _batch_top_k(query: str, automatons: list[State], k: int = 10) -> list[int]:
        scores = _batch_scores(query, automatons)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def _tokenize(text: str) -> list[str]:
    """Extract standard lowercase alphanumeric tokens for BM25."""
    return re.findall(r"\w+", text.lower())


class Document:
    """A compiled Suffix Automaton for a single document."""

    def __init__(self, text: str | None = None, _state: State | None = None):
        if _state is not None:
            self._state = _state
        elif text is not None:
            self._state = _build_suffix_automaton(text)
        else:
            raise ValueError("Must provide either text or _state")

    def score(self, query: str, alpha: float = 0.5) -> float:
        """Compute similarity against a query string.

        Note: Standalone Document objects do not support BM25 fusion.
        Alpha is ignored.
        """
        return float(_zlfi(query, self._state))

    def dumps(self) -> bytes:
        """Serialize the Document into portable raw bytes."""
        if not hasattr(self._state, "dumps"):
            raise NotImplementedError("Serialization requires the Cython extension.")
        return bytes(self._state.dumps())

    @classmethod
    def loads(cls, data: bytes) -> Document:
        """Load a Document from raw serialized bytes."""
        if not hasattr(State, "loads"):
            raise NotImplementedError("Serialization requires the Cython extension.")
        return cls(_state=State.loads(data))


class Corpus:
    """A vectorized collection of compiled Documents."""

    def __init__(self, docs: list[str] | list[Document] | None = None):
        self._states: list[State] = []
        self._vocab: dict[str, int] = {}
        self._bm25 = BM25Index()
        if docs:
            for doc in docs:
                self.add(doc)

    def add(self, doc: str | Document) -> None:
        """Add a new document to the corpus."""
        if isinstance(doc, str):
            self._states.append(_build_suffix_automaton(doc))
            tokens = _tokenize(doc)
            token_ids = []
            for t in tokens:
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)
                token_ids.append(self._vocab[t])
            self._bm25.add_doc(array.array("i", token_ids))
        elif isinstance(doc, Document):
            self._states.append(doc._state)
            self._bm25.add_doc(array.array("i"))
        else:
            raise TypeError("doc must be a str or Document")

    def _get_bm25_scores(self, query: str) -> list[float]:
        if not self._states:
            return []
        tokens = _tokenize(query)
        token_ids = [self._vocab[t] for t in tokens if t in self._vocab]
        return list(self._bm25.score_all(array.array("i", token_ids)))

    def score_all(self, query: str, alpha: float = 0.5) -> list[float]:
        """Score all documents against the query using late fusion."""
        if not self._states:
            return []

        if alpha == 1.0:
            return list(_batch_scores(query, self._states))

        bm25_scores = self._get_bm25_scores(query)
        if alpha == 0.0:
            return bm25_scores

        sa_scores = _batch_scores(query, self._states)

        if not bm25_scores:
            return list(sa_scores)

        min_b = min(bm25_scores)
        max_b = max(bm25_scores)
        range_b = max_b - min_b

        result = [0.0] * len(sa_scores)
        if range_b == 0:
            for i in range(len(sa_scores)):
                result[i] = alpha * sa_scores[i]
        else:
            for i in range(len(sa_scores)):
                norm_b = (bm25_scores[i] - min_b) / range_b
                result[i] = (alpha * sa_scores[i]) + ((1.0 - alpha) * norm_b)

        return result

    def search(self, query: str, k: int = 10, alpha: float = 0.5) -> list[int]:
        """Return the indices of the top K highest scoring documents."""
        if not self._states:
            return []

        if alpha == 1.0:
            return _batch_top_k(query, self._states, k=k)  # type: ignore[no-any-return]

        scores = self.score_all(query, alpha=alpha)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    def __len__(self) -> int:
        return len(self._states)

    def dumps(self) -> bytes:
        """Serialize the entire corpus into a single byte stream."""
        import json

        if not hasattr(State, "dumps"):
            raise NotImplementedError("Serialization requires the Cython extension.")

        # 1. State serialization
        num_docs = len(self._states)
        payloads = [state.dumps() for state in self._states]

        result = bytearray()
        result.extend(num_docs.to_bytes(4, "little"))
        for payload in payloads:
            result.extend(len(payload).to_bytes(4, "little"))
            result.extend(payload)

        # 2. Vocab serialization
        vocab_bytes = json.dumps(self._vocab).encode("utf-8")
        result.extend(len(vocab_bytes).to_bytes(4, "little"))
        result.extend(vocab_bytes)

        # 3. BM25 serialization
        bm25_bytes = self._bm25.dumps()
        result.extend(len(bm25_bytes).to_bytes(4, "little"))
        result.extend(bm25_bytes)

        return bytes(result)

    @classmethod
    def loads(cls, data: bytes) -> Corpus:
        """Load a Corpus from a byte stream."""
        import json

        if not hasattr(State, "loads"):
            raise NotImplementedError("Serialization requires the Cython extension.")

        if len(data) < 4:
            raise ValueError("Invalid serialized data: too short.")

        num_docs = int.from_bytes(data[0:4], "little")
        corpus = cls()

        offset = 4
        for _ in range(num_docs):
            if offset + 4 > len(data):
                raise ValueError("Invalid serialized data: unexpected EOF.")
            payload_len = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4

            if offset + payload_len > len(data):
                raise ValueError("Invalid serialized data: unexpected EOF payload.")

            state = State.loads(data[offset : offset + payload_len])
            corpus._states.append(state)
            offset += payload_len

        if offset + 4 <= len(data):
            vocab_len = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4
            if offset + vocab_len <= len(data):
                raw = data[offset : offset + vocab_len]
                corpus._vocab = json.loads(raw.decode("utf-8"))
                offset += vocab_len

        if offset + 4 <= len(data):
            bm25_len = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4
            if offset + bm25_len <= len(data):
                corpus._bm25 = BM25Index.loads(data[offset : offset + bm25_len])
                offset += bm25_len

        return corpus


def score(  # noqa: C901
    query: str,
    target: str | Document | Corpus | list[str] | list[Document],
    alpha: float = 0.5,
) -> float | list[float]:
    """
    Overloaded similarity scoring function.

    If target is a single string or Document, returns a single float score.
    If target is a list of strings, list of Documents, or a Corpus,
    returns a list of scores.
    """
    if isinstance(target, str):
        return float(_zlfi(query, target))

    elif isinstance(target, Document):
        return target.score(query, alpha=alpha)

    elif isinstance(target, Corpus):
        return target.score_all(query, alpha=alpha)

    elif isinstance(target, list):
        if not target:
            return []

        # Fast path 1: Homogeneous list of Documents
        if all(isinstance(x, Document) for x in target):
            states1 = [x._state for x in target if isinstance(x, Document)]
            return list(_batch_scores(query, states1))

        # Fast path 2: Homogeneous list of strings
        if all(isinstance(x, str) for x in target):
            states2 = [_build_suffix_automaton(x) for x in target if isinstance(x, str)]
            return list(_batch_scores(query, states2))

        # Slow path: Mixed lists of str and Document
        states = []
        for item in target:
            if isinstance(item, str):
                states.append(_build_suffix_automaton(item))
            elif isinstance(item, Document):
                states.append(item._state)
            else:
                raise TypeError("List elements must be str or Document")
        return list(_batch_scores(query, states))

    else:
        raise TypeError(f"Unsupported target type: {type(target)}")


__all__ = ["Document", "Corpus", "score", "State"]
