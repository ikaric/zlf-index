"""Suffix25 (Sequence Similarity).

Implements the linear-time Sequence Similarity algorithm based on a Suffix Automaton,
evaluating the structural, ordered, and contiguous overlap between a query sequence N
and a reference sequence M.
"""

from __future__ import annotations

import math


class State:
    """A Suffix Automaton state."""

    def __init__(self, length: int = 0, link: int = -1) -> None:
        self.length = length
        self.link = link
        self.transitions: dict[str, int] = {}


def build_suffix_automaton(m: str) -> list[State]:
    """Builds a Suffix Automaton for string M in O(|M|) time."""
    st = [State(0, -1)]
    last = 0
    for c in m:
        cur = len(st)
        st.append(State(st[last].length + 1, 0))
        p = last
        # Follow suffix links and add transitions
        while p != -1 and c not in st[p].transitions:
            st[p].transitions[c] = cur
            p = st[p].link

        if p == -1:
            st[cur].link = 0
        else:
            q = st[p].transitions[c]
            if st[p].length + 1 == st[q].length:
                st[cur].link = q
            else:
                # Split state q into clone
                clone = len(st)
                st.append(State(st[p].length + 1, st[q].link))
                st[clone].transitions = st[q].transitions.copy()
                while p != -1 and st[p].transitions.get(c) == q:
                    st[p].transitions[c] = clone
                    p = st[p].link
                st[q].link = clone
                st[cur].link = clone
        last = cur
    return st


def suffix25(n: str, m: str | list[State]) -> float:
    """Computes the normalized contextual similarity exactly in O(|N| + |M|) time.

    Args:
        n: The query sequence.
        m: The reference sequence OR a pre-built Suffix Automaton (list of States).

    Returns:
        The normalized contextual similarity score [0.0, 1.0].
    """
    n_len = len(n)
    if n_len == 0:
        return 0.0

    # 1. Build Suffix Automaton on the reference string M
    # O(|M|) if string, O(1) if cached
    st = m if isinstance(m, list) else build_suffix_automaton(m)

    delta_score = 0
    v = 0  # Current state in automaton
    length = 0  # Length of current longest match

    # 2. Traverse automaton with query string N: O(|N|)
    for c in n:
        # If no transition for 'c', fall back using suffix links
        while v != 0 and c not in st[v].transitions:
            v = st[v].link
            length = st[v].length

        # If there is a transition, advance the state and increment length
        if c in st[v].transitions:
            v = st[v].transitions[c]
            length += 1
        else:
            v = 0
            length = 0

        # Add the sum of lengths of all valid substrings ending at this character
        # Sum of 1 + 2 + ... + l = (l * (l + 1)) // 2
        delta_score += (length * (length + 1)) // 2

    # 3. Normalize exactly as before
    t_delta = (n_len * (n_len + 1) * (n_len + 2)) // 6

    return delta_score / t_delta


class BM25Index:
    """BM25 inverted index using the ATIRE IDF variant (matches rank_bm25.BM25Okapi).

    Operates on integer token IDs to match the Cython hybrid architecture.
    """

    def __init__(
        self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25
    ) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.doc_freqs: dict[int, int] = {}
        self.doc_lengths: list[int] = []
        self.avg_doc_len: float = 0.0
        self.num_docs: int = 0
        self.postings: dict[int, dict[int, int]] = {}
        self._idf_cache: dict[int, float] = {}
        self._idf_dirty: bool = True

    def add_doc(self, token_ids: list[int]) -> None:
        """Adds a document to the index, represented as a list of integer token IDs."""
        doc_id = self.num_docs
        self.num_docs += 1

        doc_len = len(token_ids)
        self.doc_lengths.append(doc_len)

        delta = (doc_len - self.avg_doc_len) / self.num_docs
        self.avg_doc_len = self.avg_doc_len + delta

        freqs: dict[int, int] = {}
        for term_id in token_ids:
            freqs[term_id] = freqs.get(term_id, 0) + 1

        for term_id, tf in freqs.items():
            if term_id not in self.postings:
                self.postings[term_id] = {}
            self.postings[term_id][doc_id] = tf
            self.doc_freqs[term_id] = self.doc_freqs.get(term_id, 0) + 1

        self._idf_dirty = True

    def _rebuild_idf(self) -> None:
        """Recompute IDF cache using the ATIRE BM25 variant (matches rank_bm25)."""
        self._idf_cache.clear()
        idf_sum = 0.0
        n = self.num_docs
        for term_id, df in self.doc_freqs.items():
            val = math.log(n - df + 0.5) - math.log(df + 0.5)
            self._idf_cache[term_id] = val
            idf_sum += val

        avg_idf = idf_sum / len(self._idf_cache) if self._idf_cache else 0.0
        eps = self.epsilon * avg_idf
        for term_id, val in self._idf_cache.items():
            if val < 0:
                self._idf_cache[term_id] = eps

        self._idf_dirty = False

    def score_all(self, query_token_ids: list[int]) -> list[float]:
        """Scores all documents against the query (ATIRE BM25 variant)."""
        scores = [0.0] * self.num_docs
        if self.num_docs == 0 or not query_token_ids:
            return scores

        if self._idf_dirty:
            self._rebuild_idf()

        q_freqs: dict[int, int] = {}
        for tid in query_token_ids:
            q_freqs[tid] = q_freqs.get(tid, 0) + 1

        for term_id, q_tf in q_freqs.items():
            if term_id not in self.postings:
                continue

            idf = self._idf_cache.get(term_id, 0.0)

            for doc_id, tf in self.postings[term_id].items():
                doc_len = self.doc_lengths[doc_id]
                norm = (
                    1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
                )
                term_score = (
                    idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
                )
                scores[doc_id] += term_score * q_tf

        return scores

    def dumps(self) -> bytes:
        import json

        data = {
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "doc_freqs": self.doc_freqs,
            "doc_lengths": self.doc_lengths,
            "avg_doc_len": self.avg_doc_len,
            "num_docs": self.num_docs,
            "postings": self.postings,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def loads(cls, data: bytes) -> BM25Index:
        import json

        data_dict = json.loads(data.decode("utf-8"))
        obj = cls(
            k1=data_dict["k1"],
            b=data_dict["b"],
            epsilon=data_dict.get("epsilon", 0.25),
        )
        # Convert string keys back to int
        obj.doc_freqs = {int(k): v for k, v in data_dict["doc_freqs"].items()}
        obj.doc_lengths = data_dict["doc_lengths"]
        obj.avg_doc_len = data_dict["avg_doc_len"]
        obj.num_docs = data_dict["num_docs"]
        obj.postings = {
            int(k): {int(dk): dv for dk, dv in v.items()}
            for k, v in data_dict["postings"].items()
        }
        return obj
