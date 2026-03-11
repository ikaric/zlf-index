<a id="top"></a>
<div align="center">

# Suffix25

### Suffix Automaton-Based Text Similarity and Retrieval

[Quick Start](#quick-start) • [API](#api) • [Algorithm](#algorithm) • [Implementation](#implementation) • [Benchmarks](#benchmarks) • [Use Cases](#use-cases)

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyPI version](https://badge.fury.io/py/suffix25.svg)](https://pypi.org/project/suffix25/)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

Suffix25 is the evolved implementation of the **Contextual Similarity** metric originally presented at IEEE ICAT 2017. It computes the **contiguous, ordered structural overlap** between two text sequences using a Suffix Automaton (DAWG), producing a normalized similarity score in [0, 1] in linear time $O(|N| + |M|)$ with no language-specific tokenizers or stemmers required. This version adds a high-performance Cython/C core, corpus-level retrieval with a built-in BM25 inverted index, and a tunable `alpha` parameter for fusing structural and frequency-based scores.

> **Original paper:** *[Contextual Similarity: Quasilinear-Time Search and Comparison for Sequential Data](https://ieeexplore.ieee.org/document/8252957/)* — Ilhan Karic, IEEE ICAT 2017.
>
> **Cited by:** *[Improving Sentence Retrieval Using Sequence Similarity](https://doi.org/10.3390/app10124316)* — Boban, Doko & Gotovac, Applied Sciences 2020. Used the contextual similarity formula as a drop-in replacement for exact term matching inside TF-ISF, BM25, and Language Modeling for sentence retrieval. Achieved statistically significant improvements across all three TREC novelty track collections (2002-2004), with BM25 P@10 improving from 0.142 to 0.33 on TREC 2002.

---

## Quick Start

**Prerequisites:** Python 3.12+

```bash
pip install suffix25
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add suffix25
```

### Basic Usage

```python
from suffix25 import score, Document, Corpus

query = "brown fox"
document_text = "The quick brown fox jumps over the lazy dog."

# Raw string-to-string comparison
similarity = score(query, document_text)

# Pre-compiled Document (builds Suffix Automaton once, reuse across queries)
doc = Document(document_text)
fast_score = score(query, doc)

# Corpus search with tunable alpha
corpus = Corpus([
    "The quick brown fox jumps over the lazy dog.",
    "A completely different sentence.",
    "Foxes are red and brown.",
])

scores = score(query, corpus, alpha=0.5)  # 50% structural + 50% BM25
scores = score(query, corpus, alpha=1.0)  # Pure structural matching
scores = score(query, corpus, alpha=0.0)  # Pure BM25

top_results = corpus.search(query, k=10)

# Zero-copy serialization
raw_bytes = corpus.dumps()
loaded_corpus = Corpus.loads(raw_bytes)
```

---

## API

### `score(query, target, alpha=0.5)`

Universal entry point. Returns a single `float` for scalar targets, or `list[float]` for collection targets.

| `target` type | Behavior |
|---|---|
| `str` | Builds a temporary automaton, scores, returns `float` |
| `Document` | Scores against the pre-compiled automaton, returns `float` |
| `list[str]` | Builds automata for each string, returns `list[float]` |
| `list[Document]` | Batch scores across pre-compiled automata, returns `list[float]` |
| `Corpus` | Scores all documents with alpha fusion, returns `list[float]` |

### `Document(text)`

A compiled Suffix Automaton for a single text string. Build once, query many times.

```python
doc = Document("The quick brown fox jumps over the lazy dog.")
doc.score("lazy dog")  # -> 1.0 (exact substring)
doc.score("xyz")       # -> 0.0 (no overlap)

raw = doc.dumps()              # serialize to bytes (Cython only)
doc = Document.loads(raw)      # deserialize
```

`Document.score()` always uses pure suffix automaton matching. The `alpha` parameter has no effect on standalone documents since there is no BM25 index for a single document.

### `Corpus(docs)`

An indexed collection of documents with both suffix automata and a BM25 inverted index.

```python
corpus = Corpus(["doc one", "doc two", "doc three"])
corpus.add("doc four")

scores = corpus.score_all("query", alpha=0.5)   # list[float], all documents
top_k  = corpus.search("query", k=3, alpha=0.5) # list[int], top-k indices

len(corpus)  # -> 4

raw = corpus.dumps()              # serialize to bytes (Cython only)
corpus = Corpus.loads(raw)        # deserialize
```

### The `alpha` parameter

When scoring against a `Corpus`, the final score is a weighted blend:

$$\text{score} = \alpha \cdot \text{SA} + (1 - \alpha) \cdot \text{BM25}_{\text{norm}}$$

| `alpha` | Behavior |
|:-------:|----------|
| `1.0` | Pure suffix automaton — contiguous structural matching only |
| `0.0` | Pure BM25 — term frequency / IDF only |
| `0.5` | Hybrid blend (default) |

Set `alpha` closer to `1.0` for structural or multilingual workloads where tokenization is unreliable, closer to `0.0` for keyword-heavy queries where term rarity matters, or leave it at `0.5` for a balanced default.

---

## Algorithm

### Contextual Similarity (Suffix Automaton)

Given a query sequence $N$ and a reference document $M$, the unnormalized contextual similarity $\delta(N, M)$ is the sum of lengths of all contiguous substrings of $N$ that also appear in $M$:

$$\delta(N, M) = \sum_{i=1}^{|N|} \sum_{j=i}^{|N|} \left| N_{i \ldots j} \cap M \right|$$

where $N_{i \ldots j}$ is the substring of $N$ from position $i$ to $j$, and the intersection yields the length of the match (or 0 if absent).

The maximum possible score occurs when $N$ is a substring of $M$ (or $N = M$), giving:

$$T_\delta = \binom{|N|+2}{3} = \frac{|N|(|N|+1)(|N|+2)}{6}$$

The normalized similarity is:

$$\bar{\delta}(N, M) = \frac{\delta(N, M)}{T_\delta} \in [0, 1]$$

**Naive evaluation** requires $O(|N|^2 \times |M|)$ time. The key insight is to replace this with a **Suffix Automaton** (DAWG) that compresses all substrings of $M$ into a finite-state machine:

1. **Build** — Compile document $M$ into a Suffix Automaton in $O(|M|)$ time and $O(|M|)$ space. The automaton accepts exactly the set of all substrings of $M$.
2. **Query** — Traverse the automaton with query $N$ character-by-character. At each position $i$, maintain the current state $v$ and the length $l_i$ of the longest suffix of $N_{1 \ldots i}$ that appears in $M$. On a mismatch, follow suffix links until a valid transition exists or the root is reached.
3. **Score** — At each position $i$, all substrings of $N$ ending at $i$ that appear in $M$ have lengths $1, 2, \ldots, l_i$. Their total contribution is:

$$\sum_{k=1}^{l_i} k = \frac{l_i(l_i + 1)}{2}$$

4. **Accumulate and normalize** — The full score is:

$$\bar{\delta}(N, M) = T_\delta^{-1} \sum_{i=1}^{|N|} \frac{l_i(l_i + 1)}{2}$$

Total complexity: $O(|N| + |M|)$.

### BM25 (Term Frequency)

For corpus-level retrieval, each document is also indexed with an ATIRE-variant BM25. Given a query $Q$ tokenized into terms $\{t_1, \ldots, t_n\}$, the BM25 score for document $d$ is:

$$\text{BM25}(Q, d) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{tf(t, d) \cdot (k_1 + 1)}{tf(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)} \cdot c(t, Q)$$

where:
- $tf(t, d)$ is the term frequency of $t$ in document $d$
- $|d|$ is the document length in tokens
- $\text{avgdl}$ is the average document length across the corpus
- $c(t, Q)$ is the count of term $t$ in the query
- $k_1 = 1.5$, $b = 0.75$

The IDF uses the ATIRE variant:

$$\text{IDF}(t) = \ln\!\left(N - df(t) + 0.5\right) - \ln\!\left(df(t) + 0.5\right)$$

where $N$ is the total number of documents and $df(t)$ is the number of documents containing term $t$. Negative IDF values (terms in more than half the corpus) are floored to $\epsilon \cdot \overline{\text{IDF}}$ with $\epsilon = 0.25$.

### Late Fusion (Hybrid Score)

The BM25 raw scores are min-max normalized to $[0, 1]$ before fusion:

$$\text{BM25}_{\text{norm}}(Q, d) = \frac{\text{BM25}(Q, d) - \min_d}{\max_d - \min_d}$$

The final hybrid score for each document $d$ is:

$$S(Q, d) = \alpha \cdot \bar{\delta}(Q, d) + (1 - \alpha) \cdot \text{BM25}_{\text{norm}}(Q, d)$$

where $\alpha \in [0, 1]$ controls the blend. Since $\bar{\delta} \in [0, 1]$ and $\text{BM25}_{\text{norm}} \in [0, 1]$, the hybrid score $S \in [0, 1]$.

---

## Implementation

The core is a Cython/C extension (`_core.pyx`) compiled with `-O3` and optional `-march=native -ffast-math` (set `SUFFIX25_NATIVE=1`). All safety checks are disabled at the Cython level (`boundscheck=False`, `wraparound=False`, `cdivision=True`). A pure-Python fallback (`core.py`) is used when the extension is unavailable.

### Two-Phase Automaton Construction

The suffix automaton is built in two phases to separate algorithmic correctness from memory layout optimization:

**Phase 1 — Build (`BuildSA`).** A temporary structure with full 256-entry transition tables per state (one `int` per possible byte value). This is the textbook suffix automaton construction: extend one character at a time, create new states, follow suffix links, and clone states when needed. The 256-wide tables make transition lookups trivial during construction but waste memory (1 KB per state).

**Phase 2 — Compact (`CompactSA`).** The build-phase automaton is compacted into a flat array of **64-byte records** — exactly one CPU L1 cache line per state. Records are allocated via `posix_memalign` with 64-byte alignment to prevent any state from straddling a cache line boundary. The build-phase tables are freed after compaction.

### 64-Byte Record Layout

Each state occupies exactly 64 bytes:

```
Offset  Size  Field
──────  ────  ─────────────────────────────
 0      4B    length        (unsigned int)
 4      4B    link          (unsigned int, suffix link target)
 8      1B    total_count   (total number of transitions)
 9      1B    inline_count  (transitions stored inline, <= 8)
12      4B    overflow_start(index into overflow arrays)
16      8B    chars[8]      (inline transition characters)
24      32B   targets[8]    (inline transition targets, unsigned int each)
──────  ────
        64B   total
```

Most states in natural language text have fewer than 8 outgoing transitions. These fit entirely inline — the state's full transition map lives in the same cache line as its metadata. States with more than 8 transitions store the first 8 inline and the rest in separate overflow arrays (`ovf_chars`, `ovf_targets`), indexed by `overflow_start`.

### SWAR Transition Lookup

Character lookups within a state use SIMD Within A Register (SWAR) bit manipulation on the 8-byte `chars` array. The target character is broadcast to all 8 byte lanes of a 64-bit word, XORed against the packed chars, and then a standard null-byte detection trick identifies the matching lane:

```c
V = *(uint64_t *)chars;
mask = c | (c << 8) | (c << 16) | (c << 32);  // broadcast
diff = V ^ mask;
has_zero = (diff - 0x0101010101010101) & ~diff & 0x8080808080808080;
j = __builtin_ctzll(has_zero) >> 3;  // byte index of match
```

This finds a matching character among up to 8 candidates in constant time without branching. If the character is not found inline and the state has overflow transitions, those are searched with a short linear scan.

### Prefetching

During query traversal, after resolving a transition to the next state, the record for that state is prefetched with `__builtin_prefetch(addr, 0, 1)` (read, low temporal locality). Since each record is cache-line-aligned, this prefetch loads exactly the data needed for the next iteration, hiding memory latency when the automaton is too large to fit in L1/L2.

### Query Scoring Loop

The query loop runs entirely under `nogil` with no Python object interaction. At each byte position $i$, it follows suffix links until a valid transition exists or the root is reached, then accumulates $l_i(l_i + 1)/2$ into a running `long long` sum. After the final byte, the sum is divided by $T_\delta$ to produce the normalized score. The entire traversal is branch-light: the inner suffix-link loop typically executes 0-1 iterations for natural language text.

### OpenMP Parallelization

`batch_scores` and `batch_top_k` pre-extract raw C pointers to each document's `CompactSA` struct into a flat `CompactSA**` array, then score all documents in parallel using `prange(n, schedule='static')` under a `nogil` block. Each document's automaton is independent, so there is no synchronization overhead.

`batch_top_k` additionally extracts the top-k results using a C-level min-heap ($O(n \log k)$), avoiding a full sort.

### BM25 Inverted Index

The BM25 component is implemented entirely in C-level structs within Cython. Token IDs are integers mapped from a Python-side vocabulary dict. Each term has a `PostingList` struct (dynamically resized arrays of `doc_ids` and `tfs`). The IDF cache is lazily rebuilt only when the corpus changes (`idf_dirty` flag). Scoring iterates directly over posting lists, accumulating per-document BM25 contributions without materializing a full score matrix.

### Zero-Copy Serialization

`dumps()` and `loads()` on both `SuffixAutomatonWrapper` and `BM25Index` use direct `memcpy` of the underlying C arrays — the record buffer, overflow arrays, posting lists, and doc lengths are written and read as contiguous byte blocks. No field-by-field iteration, no format parsing. The serialized format is:

| Segment | Size |
|---|---|
| `num_states` | 4 bytes |
| `ovf_size` | 4 bytes |
| Records | `num_states * 64` bytes |
| Overflow chars | `ovf_size` bytes |
| Overflow targets | `ovf_size * 4` bytes |

### Compiler Flags

| Flag | Purpose |
|---|---|
| `-O3` | Full optimization (always on) |
| `-march=native` | Use host CPU instruction set (when `SUFFIX25_NATIVE=1`) |
| `-ffast-math` | Relax IEEE float for faster division (when `SUFFIX25_NATIVE=1`) |
| `-fopenmp` | Enable OpenMP threading for `prange` |

---

## Benchmarks

Measured on a synthetic dataset: **5,000 documents** (200 words each), **100 queries** (5 words each).

### Single Document Inference

Raw time to score one query against one pre-compiled `Document`.

| Method | Time |
|--------|------|
| Native BM25 | 0.20 us/query |
| Suffix Automaton | 0.20 us/query |
| Hybrid (alpha=0.5) | 0.20 us/query |

### Corpus Throughput

Time to score one query against all 5,000 documents and extract top-k results.

| Method | Time |
|--------|------|
| Native BM25 | 0.57 ms/query |
| Suffix Automaton | 1.54 ms/query |
| Hybrid (alpha=0.5) | 2.24 ms/query |

### Indexing

| Corpus Size | Time | Components |
|-------------|------|------------|
| 5,000 docs | 2.43 s | Suffix Automata + BM25 Index |

Reproduce with:

```bash
make benchmark-speed
```

---

## Use Cases

### Retrieval-Augmented Generation (RAG)

Use Suffix25 as a retrieval layer to find the most relevant context chunks before passing them to an LLM. The structural matching catches contiguous phrase overlaps that bag-of-words methods miss, while the BM25 fusion handles keyword relevance.

```python
from suffix25 import Corpus

chunks = [
    "The mitochondria is the powerhouse of the cell. It produces ATP through oxidative phosphorylation.",
    "Cell membranes are composed of a lipid bilayer with embedded proteins for transport.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight in chloroplasts.",
    "DNA replication occurs during the S phase of the cell cycle, producing two identical copies.",
    "Ribosomes translate messenger RNA into proteins through a process called translation.",
]

corpus = Corpus(chunks)

query = "how does the cell produce energy"
top_indices = corpus.search(query, k=2, alpha=0.5)
context = "\n".join(chunks[i] for i in top_indices)

# Pass `context` to your LLM as grounding material
```

### LLM Hallucination Detection

Score a model-generated claim against the source context. If the structural overlap is near zero, the claim has no grounding in the source and is likely hallucinated -- provable mathematically without an evaluator LLM.

```python
from suffix25 import score

source = "The Eiffel Tower was completed in 1889 and stands 330 meters tall."

claim_grounded = "The Eiffel Tower is 330 meters tall."
claim_hallucinated = "The Eiffel Tower was designed by Gustave Courbet in 1910."

score(claim_grounded, source)      # high overlap — grounded
score(claim_hallucinated, source)  # low overlap — hallucinated
```

### Deterministic Intent Routing

Map noisy natural-language user inputs to predetermined function signatures without an LLM in the loop.

```python
from suffix25 import score, Document

intents = {
    "cancel_order": Document("cancel my order stop the shipment"),
    "track_order": Document("where is my order track package shipping status"),
    "return_item": Document("return refund send back exchange item"),
}

user_input = "i want to send this thing back for a refund"
matches = {name: doc.score(user_input) for name, doc in intents.items()}
best = max(matches, key=matches.get)  # -> "return_item"
```

### Fuzzy Autocompletion and Spell Correction

Score misspelled or partial queries against a dictionary, leveraging the suffix automaton to handle partial overlaps naturally.

```python
from suffix25 import score

dictionary = ["authentication", "authorization", "automobile", "automatic", "autonomous"]
typo = "authentcaton"

scores = score(typo, dictionary)
best = dictionary[scores.index(max(scores))]  # -> "authentication"
```

### Plagiarism Detection

Generate structural overlap scores between documents instantly. Contiguous match tracking makes this resilient against localized paraphrasing.

```python
from suffix25 import Document

original = Document(
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn."
)
suspect = "A quick brown fox jumped over a lazy dog by the riverbank at dawn."

overlap = original.score(suspect)  # high score indicates substantial copied structure
```

### Language and Information Detection

Identify whether a byte sequence contains structured language or resembles encrypted/random noise by comparing structural overlap against a known language sample.

```python
from suffix25 import score

english_ref = "the of and to a in is that it was for on are with"

score("this is a normal sentence", english_ref)  # high — structured language
score("xk2!qz@9v#m&8fw", english_ref)            # near zero — noise/encrypted
```

---

## Development

To contribute to Suffix25 or build it from source:

```bash
git clone https://github.com/ikaric/suffix25.git
cd suffix25
make install-dev
```

This will set up a development environment with `uv`, install dependencies, and compile the Cython extensions.

---

## Citation

```bibtex
@inproceedings{karic2017contextual,
  title={Contextual Similarity: Quasilinear-Time Search and Comparison for Sequential Data},
  author={Karić, Ilhan},
  booktitle={2017 XXVI International Conference on Information, Communication and Automation Technologies (ICAT)},
  year={2017},
  organization={IEEE},
  doi={10.1109/ICAT.2017.8171609}
}
```

---

<div align="center">

[Back to top](#top)

</div>
