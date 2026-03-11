<a id="top"></a>
<div align="center">

# Zipf Latent Fusion Index (ZLFI)

### A Linear-Time, Suffix Automaton-Based Sequence Similarity Metric for Zero-Shot Text Retrieval

[Why Not Just BM25?](#why-not-just-bm25) • [Key Results](#key-results) • [Quick Start](#quick-start) • [The Alpha Parameter](#the-alpha-parameter) • [Algorithm](#algorithm) • [Benchmarks](#benchmarks) • [Native Use Cases](#native-use-cases)

[![Python 3.14+](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9)](https://docs.astral.sh/uv/)
[![Pytest](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Why Not Just BM25?

BM25 is the standard lexical retrieval algorithm. It works well for English keyword search. However, it has fundamental architectural constraints:

- It **destroys word order** — "the dog bit the man" and "the man bit the dog" produce identical scores.
- It **requires global corpus statistics** — the IDF component cannot be computed without observing every document, making single-document or streaming scoring impossible.
- It **requires language-specific tokenization** — CJK, Arabic, Thai, and other scripts need dedicated tokenizers and stemmers to function at all.

ZLFI is a fundamentally different algorithm. It evaluates the **contiguous, ordered structural overlap** between two sequences using a Suffix Automaton (DAWG) in linear time O(|N| + |M|). It requires no tokenizer, no stemmer, no corpus-level statistics, and works identically across all languages and scripts — from English to Japanese to Telugu — with zero configuration.

This is not a BM25 replacement for English keyword search. It is a **complementary similarity metric** with its own strengths, and the two can be fused via a tunable `alpha` parameter for hybrid retrieval. *(Note: True streaming and stateless scoring requires `alpha=1.0`; the default hybrid mode `alpha=0.5` incorporates BM25, and therefore reintroduces the need for global corpus statistics).*

> **Research Paper:** *[Contextual Similarity: Quasilinear-Time Search and Comparison for Sequential Data](https://ieeexplore.ieee.org/document/8252957/)* — Ilhan Karić, University of "Džemal Bijedić", Faculty of Information Technologies (IEEE ICAT, 2017).

---

## Key Results

### Speed

Benchmarked against `rank_bm25.BM25Okapi` (the standard Python BM25 library) over a 5,000-document synthetic corpus:

| Scenario | `rank_bm25` | ZLFI | Speedup |
|----------|-------------|------|---------|
| Single document scoring | 16.62 μs | 0.18 μs | **93x** |
| 5,000-doc corpus (hybrid, `alpha=0.5`) | 4.07 ms | 2.33 ms | **1.74x** |

Even while running *both* the Suffix Automaton and a native BM25 engine concurrently, the hybrid mode is nearly twice as fast as the Python BM25 baseline alone.

> **Note on evaluation scope:** All retrieval results below are from **preliminary quick tests** run under hardware memory constraints (24 GB RAM). Quick tests use subsampled corpora and limited query sets. They are directionally indicative but **not directly comparable to published leaderboard numbers**, which are computed over full corpora (often millions of documents). Full evaluations can be reproduced on 64+ GB hardware using `make benchmark-beir-full` and `make benchmark-mrtydi-full`. We present these results transparently as preliminary evidence, not definitive claims.

### Multilingual Retrieval (Mr. TyDi — Preliminary)

The hybrid engine (`alpha=0.5`) matches or outperforms BM25 on **all 11 languages** in our quick test, with substantial gains on morphologically rich and non-Latin scripts:

| Language | BM25 MRR@100 | ZLFI Hybrid MRR@100 | Improvement |
|----------|:------------:|:-------------------:|:-----------:|
| Japanese | 44.3% | 60.6% | **+16.3%** |
| Telugu | 55.3% | 70.6% | **+15.3%** |
| Arabic | 84.6% | 91.2% | **+6.7%** |
| English | 70.9% | 71.0% | +0.0% |
| *All 11 languages* | — | — | *Zero regressions* |

### English Retrieval (BEIR 2.0 — Preliminary)

On English-centric benchmarks (6 of 18 datasets), the hybrid engine matches BM25 within 0.1% average NDCG@10:

| | BM25 | ZLFI Hybrid | Difference |
|-|:----:|:-----------:|:----------:|
| Average NDCG@10 | 40.2% | 40.3% | +0.1% |

Full benchmark tables and methodology are detailed in the [Benchmarks](#benchmarks) section.

---

## Quick Start

**Prerequisites:** Python 3.14+, [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone https://github.com/ikaric/zlfi.git
cd zlfi
make install-dev
```

```python
from zlfi import score, Document, Corpus

query = "brown fox"
document_text = "The quick brown fox jumps over the lazy dog."

# Raw string-to-string comparison
similarity = score(query, document_text)

# Pre-compiled Document (builds Suffix Automaton once, reuse across queries)
doc = Document(document_text)
fast_score = score(query, doc)

# Corpus with tunable alpha
corpus = Corpus([
    "The quick brown fox jumps over the lazy dog.",
    "A completely different sentence.",
    "Foxes are red and brown.",
])

scores = score(query, corpus, alpha=0.5)  # Hybrid: 50% structural + 50% BM25
scores = score(query, corpus, alpha=1.0)  # Pure ZLFI (structural only)
scores = score(query, corpus, alpha=0.0)  # Pure BM25 (frequency only)

top_results = corpus.search(query, k=10)

# Zero-copy serialization for persistence
raw_bytes = corpus.dumps()
loaded_corpus = Corpus.loads(raw_bytes)
```

### Development Commands

```bash
make test                    # Full Pytest suite with coverage
make check                   # Lint + typecheck + unit tests
make benchmark-speed         # Speed comparison vs rank_bm25
make benchmark-beir-quick    # BEIR 2.0 evaluation (6 datasets, ~15 min)
make benchmark-mrtydi-quick  # Mr. TyDi evaluation (11 languages, ~5 min)
```

---

## The Alpha Parameter

The `Corpus` engine integrates both a Suffix Automaton and a native Cython BM25 inverted index. The `alpha` parameter controls the fusion:

$$\text{Final Score} = \alpha \cdot \text{ZLFI}_{sa} + (1 - \alpha) \cdot \text{BM25}_{norm}$$

| `alpha` | Behavior |
|:-------:|----------|
| `1.0` | Pure ZLFI — structural substring matching only |
| `0.0` | Pure BM25 — term frequency / IDF only |
| `0.5` | Hybrid — equal blend (default) |

### Why Fusion?

During preliminary evaluation, we observed complementary failure modes:

- **Pure ZLFI (`alpha=1.0`)** showed strong gains on morphologically rich languages (Japanese +16.3%, Telugu +15.3%, Arabic +6.7% in quick tests) but underperformed on English keyword-heavy benchmarks (BEIR), where term rarity (IDF) is the dominant signal.
- **Pure BM25 (`alpha=0.0`)** performed well on English BEIR benchmarks but showed weaker multilingual retrieval (Mr. TyDi), where tokenization quality degrades and word order matters more.
- **Hybrid (`alpha=0.5`)** eliminated all regressions across both quick-test suites — matching BM25 on BEIR while retaining the multilingual gains.

The alpha parameter is exposed as a tunable knob: set it closer to `1.0` for structural/multilingual workloads, closer to `0.0` for English keyword search, or leave it at `0.5` for balanced performance.

---

## Algorithm

### BM25 (Frequency-Based Baseline)

Our native BM25 implementation uses the **ATIRE variant** — the same IDF formula as `rank_bm25.BM25Okapi`:

$$\text{IDF}(t) = \ln\!\bigl(N - \text{df}(t) + 0.5\bigr) - \ln\!\bigl(\text{df}(t) + 0.5\bigr)$$

with negative IDF values floored to $\varepsilon \cdot \overline{\text{IDF}}$ (where $\varepsilon = 0.25$ and $\overline{\text{IDF}}$ is the corpus-wide average). This ensures **score-identical** results when `alpha=0.0`.

BM25 operates by:

1. **Tokenization** — splitting documents into individual words, discarding word order.
2. **Inverted index construction** — mapping each unique term to the documents containing it.
3. **IDF weighting** — computing inverse document frequency across the *entire* corpus. Rare terms receive high weight; common terms receive low weight.
4. **Scoring** — looking up query terms in the inverted index and computing a weighted frequency score.

**Key constraint:** BM25 requires access to global corpus statistics. The IDF value for any term can only be computed after observing every document in the collection.

### ZLFI (Structure-Based)

ZLFI evaluates the structural overlap between a query N and a reference document M. The [original paper](https://ieeexplore.ieee.org/document/8252957/) defines the unnormalized contextual similarity δ(N, M) as the sum of lengths of all substrings of N that also appear in M:

$$\delta(N, M) = \sum_{i=0}^{|N|-1} \sum_{j=i+1}^{|N|} |N[i:j) \cap M|$$

This is normalized by the maximum possible score (the score of N against itself):

$$T_\delta = \binom{|N|+2}{3} = \frac{|N|(|N|+1)(|N|+2)}{6}$$

The normalized similarity δ̅(N, M) ∈ [0.0, 1.0] is δ(N, M) / T_δ.

The paper proves that for two random sequences drawn from a uniform alphabet, the expected longest common substring grows only logarithmically with sequence length. This means the contextual similarity of unrelated sequences asymptotically approaches zero — any substantial score is statistically unlikely to arise by coincidence.

#### Suffix Automaton Optimization

The naïve evaluation of δ(N, M) requires O(|N|² × |M|) time. The key algorithmic insight is to replace this with a **Suffix Automaton** (DAWG):

1. **Build** — compile document M into a Suffix Automaton in O(|M|) time. This finite-state machine accepts exactly the set of all substrings of M.
2. **Query** — traverse query N character-by-character through the automaton in O(|N|) time. At each position, the automaton maintains the length of the longest matching substring via suffix links.
3. **Score** — exploit the closed-form identity: if the longest match ending at position `i` has length `l`, the contribution of all matching substrings at that position is exactly `l(l+1)/2`.
4. **Normalize** — divide by `|N|(|N|+1)(|N|+2)/6`.

Total complexity: **O(|N| + |M|)** — linear in both query and document length.

---

## Implementation

The algorithm is implemented as a high-performance Cython/C extension with hardware-aware optimizations. Full source: [`_core.pyx`](src/zlfi/_core.pyx).

### Two-Phase Build Architecture

**Phase 1: Full Build.** The standard online Suffix Automaton construction runs over document bytes. Each state maintains a full 256-entry transition table (one slot per byte value) for O(1) lookups during construction.

**Phase 2: Compaction.** After construction, the automaton is compacted into **64-byte records** — exactly one CPU cache line:

```
Record Layout (64 bytes per state):
┌─────────────────────────────────────────────────────┐
│ [0:4]   length         (unsigned int)               │
│ [4:8]   suffix_link    (unsigned int)               │
│ [8]     total_count    (unsigned char)              │
│ [9]     inline_count   (unsigned char)              │
│ [12:16] overflow_start (unsigned int)               │
│ [16:24] chars[8]       (8 × unsigned char)          │
│ [24:56] targets[8]     (8 × unsigned int)           │
└─────────────────────────────────────────────────────┘
```

Up to 8 transitions are stored inline. States with more than 8 spill into a separate overflow array. This layout ensures each state aligns to a single L1 cache line, preventing cross-boundary cache misses. 32-bit indices support strings up to ~2.14 billion characters.

### SWAR Bit-Banging for Transition Lookup

Character lookup uses [SWAR](https://en.wikipedia.org/wiki/SWAR) (SIMD Within A Register) bit manipulation. The 8-byte `chars` array is loaded as a single `unsigned long long`, XOR'd against the broadcast query character, and tested for zero bytes using the Mycroft null-byte detection formula:

```c
diff = chars_as_u64 ^ broadcast_char;
has_zero = (diff - 0x0101010101010101) & ~diff & 0x8080808080808080;
hit_index = __builtin_ctzll(has_zero) >> 3;
```

This finds the matching transition in a single cycle — no branches, no loop.

### Cache-Line Alignment and Prefetching

Records are allocated via `posix_memalign()` with 64-byte alignment. During the query traversal hot loop, the implementation prefetches the next state's record:

```c
__builtin_prefetch(csa.records + v * RECORD_SIZE, 0, 1);
```

This hides memory latency by loading the next state's record into L2 cache while the current state is still being processed.

### OpenMP Parallelization

Batch scoring across multiple documents releases the GIL and distributes work via OpenMP:

```cython
with nogil:
    for i in prange(n, schedule='static'):
        scores[i] = csa_query(csa_ptrs[i], q_buf, q_len)
```

Each document's automaton is fully independent — embarrassingly parallel with zero synchronization.

### Native Cython BM25 Inverted Index (ATIRE Variant)

The hybrid engine includes a zero-dependency C-level BM25 implementation using the ATIRE IDF formula (score-identical to `rank_bm25.BM25Okapi` at `alpha=0.0`):

1. **Vocabulary mapping** — term strings mapped to integer IDs via Python `dict` at the `Corpus` level.
2. **Posting lists** — dynamically allocated C-struct arrays (`int *doc_ids`, `int *tfs`) with geometric-growth `realloc` inside `_core.pyx`.
3. **C-level scoring** — the inner BM25 scoring loop iterates over contiguous C posting list arrays, avoiding Python object overhead per term-document pair.

### Zero-Copy Serialization

`Corpus.dumps()` / `Corpus.loads()` serialize both the Suffix Automata and the BM25 inverted index as raw byte blobs. Deserialization performs direct `memcpy` of binary records into allocated memory — no JSON parsing, no field-by-field reconstruction.

### Summary of Optimizations vs. Original Paper

| Dimension | Original Paper (2017) | v6 Cython Implementation |
|-----------|----------------------|--------------------------|
| Language | Python (dict-based) | Cython → C (raw pointers) |
| Transition table | Python dict per state | 64-byte inline records + overflow |
| Transition lookup | O(1) hash table | O(1) SWAR bit-banging (branchless) |
| Memory per state | ~500+ bytes (Python dict) | 64 bytes (1 cache line) |
| Native indexing | None | Zero-dependency C BM25 (ATIRE variant) |
| String limit | System RAM | ~2.14 billion characters (32-bit uint) |
| Memory alignment | None | `posix_memalign` 64-byte aligned |
| Prefetching | None | `__builtin_prefetch` on state advance |
| Parallelism | Sequential | OpenMP `prange` across documents |
| Serialization | N/A | Binary blob `memcpy` dumps/loads |

---

## Benchmarks

### Speed

Evaluated via `make benchmark-speed` over a 5,000-document synthetic corpus, comparing against `rank_bm25.BM25Okapi`.

#### Single Document Inference (Per Query)

| Method | Time (μs/query) | Speedup |
|--------|:---------------:|:-------:|
| `rank_bm25` (baseline) | 16.62 μs | 1.00x |
| **ZLFI** (`alpha=1.0`) | **0.18 μs** | **93x** |
| **ZLFI Hybrid** (`alpha=0.5`) | **0.18 μs** | **93x** |

#### Multi-Corpus Throughput (1 Query vs 5,000 Documents)

| Method | Time (ms/query) | Speedup |
|--------|:---------------:|:-------:|
| `rank_bm25` (baseline) | 4.07 ms | 1.00x |
| **ZLFI BM25** (`alpha=0.0`) | **0.55 ms** | **7.4x** |
| **ZLFI** (`alpha=1.0`) | **2.19 ms** | **1.9x** |
| **ZLFI Hybrid** (`alpha=0.5`) | **2.33 ms** | **1.7x** |

#### Indexing

| Engine | Time (s) | Components |
|--------|:--------:|------------|
| `rank_bm25` | 0.10s | Dict/list BM25 arrays |
| ZLFI Corpus | 2.47s | Suffix Automata + native BM25 |

Indexing is slower because ZLFI constructs a full Suffix Automaton DAWG per document alongside the BM25 index. This upfront cost enables the substantially faster query-time performance shown above.

**Note on Dynamic Corpora:** For static corpora, this indexing latency is a one-time non-issue. For highly dynamic databases where documents are constantly inserted or updated, this could become a bottleneck. The current `Corpus` implementation is designed for static workloads and does not support dynamic insertions; it must be rebuilt or appended to in batches.

### BEIR 2.0 (Preliminary — 6 of 18 Datasets)

Evaluated via `make benchmark-beir-quick`. Both systems rank the same candidate document pool. Full BEIR evaluation requires 64+ GB RAM.

| Dataset | BM25 NDCG@10 | ZLFI Hybrid NDCG@10 | Difference |
|---------|:------------:|:-------------------:|:----------:|
| SciFact | 66.5% | 66.4% | -0.1% |
| NFCorpus | 31.0% | 31.0% | +0.1% |
| ArguAna | 46.3% | 46.3% | +0.0% |
| SciDocs | 15.0% | 15.3% | +0.2% |
| FiQA | 23.4% | 23.4% | +0.1% |
| TREC-COVID | 59.0% | 59.3% | +0.4% |
| **Average** | **40.2%** | **40.3%** | **+0.1%** |

**Context (BEIR 2.0 Leaderboard, 2026):**

| Rank | Model | Avg NDCG@10 | Type |
|:----:|-------|:-----------:|------|
| 1 | Voyage-Large-2 | 54.8% | Dense |
| 2 | Cohere Embed v4 | 53.7% | Dense |
| 8 | ColBERT-v2 | 49.1% | Late Interaction |
| 9 | BM25 | 41.2% | Sparse |

### Mr. TyDi (Preliminary — 11 Languages, 5 Queries Each)

Evaluated via `make benchmark-mrtydi-quick` (5 queries, ~100 corpus docs per language). Metrics follow the official [castorini/mr.tydi](https://github.com/castorini/mr.tydi) protocol. Full evaluation requires 64+ GB RAM.

| Language | BM25 MRR@100 | ZLFI Hybrid MRR@100 | Improvement |
|----------|:------------:|:-------------------:|:-----------:|
| English | 70.9% | 71.0% | +0.0% |
| Arabic | 84.6% | 91.2% | **+6.7%** |
| Japanese | 44.3% | 60.6% | **+16.3%** |
| Korean | 100.0% | 100.0% | 0.0% |
| Bengali | 81.0% | 81.4% | +0.4% |
| Finnish | 80.3% | 80.3% | 0.0% |
| Indonesian | 100.0% | 100.0% | 0.0% |
| Russian | 83.7% | 83.8% | +0.1% |
| Swahili | 82.5% | 82.5% | 0.0% |
| Telugu | 55.3% | 70.6% | **+15.3%** |
| Thai | 82.5% | 82.9% | +0.4% |

The hybrid engine matches or outperforms BM25 on all 11 languages with zero regressions. The largest gains appear on continuous-character and morphologically rich scripts where BM25's tokenization is weakest.

### Limitations and Methodology

**All BEIR and Mr. TyDi results are preliminary.** They were produced under hardware memory constraints (24 GB RAM) using subsampled corpora and limited query sets. Specifically:

- **BEIR quick test:** 6 of 18 datasets evaluated. Corpus sizes range from thousands to hundreds of thousands of documents (not millions). Recall@1000 is artificially high because BM25 retrieves nearly the entire small candidate pool.
- **Mr. TyDi quick test:** All 11 languages evaluated, but with only 5 queries and ~100 corpus documents per language. Recall@100 is trivially 100% because the candidate pool is smaller than the retrieval cutoff. On full corpora (e.g., 32.9M English Wikipedia articles), BM25 Recall@100 drops to ~54%.
- **Speed benchmarks** use a 5,000-document synthetic corpus and are reproducible without special hardware.

These quick-test results are **directionally indicative** — they demonstrate the complementary behavior of ZLFI and BM25 — but they are not directly comparable to published leaderboard numbers. Full-corpus evaluations require 64+ GB RAM and can be reproduced with `make benchmark-beir-full` and `make benchmark-mrtydi-full`.

Our BM25 quick-test average (40.2% NDCG@10 across 6 datasets) is consistent with the published BEIR leaderboard value of 41.2% across all 18 datasets, providing confidence that the evaluation pipeline is correctly configured.

**Reference (Mr. TyDi Official BM25 — Full Corpus):**

| | Ar | Bn | En | Fi | Id | Ja | Ko | Ru | Sw | Te | Th | Avg |
|-|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|
| BM25 (default) | 36.8% | 41.8% | 14.0% | 28.4% | 37.6% | 21.1% | 28.5% | 31.3% | 38.9% | 34.3% | 40.1% | 32.1% |
| BM25 (tuned) | 36.6% | 41.3% | 15.0% | 28.7% | 38.2% | 21.7% | 28.0% | 32.9% | 39.6% | 42.4% | 41.6% | 33.3% |

*Source: [castorini/mr.tydi](https://github.com/castorini/mr.tydi)*

---

## Native Use Cases

ZLFI is not limited to document reranking. Because it operates on raw character arrays in linear time with no global state, it extends naturally to domains where BM25 is architecturally unsuitable:

**Linear-Time LLM Hallucination Detection.** Score a model-generated claim against the source context. If the structural overlap is statistically indistinguishable from random noise (as formally defined in the [original paper](https://ieeexplore.ieee.org/document/8252957/)), the claim is likely hallucinated. Being able to mathematically prove that an LLM's output has minimal contiguous structural overlap with the source context—in linear time, without needing a second "evaluator" LLM—is a massive standalone capability enabled by ZLFI.

**Information and Language Detection.** The algorithm's expected overlap against uniform random data can be computed analytically. This enables quantitative determination of whether an unknown byte sequence contains structured language or is encrypted/random noise.

**Deterministic Intent Routing.** Map noisy natural-language user inputs to predetermined function signatures without an LLM:

```python
from zlfi import Corpus

tools = {
    "Switch on the living room lights": turn_on_lights,
    "Play some music on the speakers": play_music,
}
router = Corpus(list(tools.keys()))

user_input = "could u switch on the lights pls"
matched = router.search(user_input, k=1)[0]
list(tools.values())[matched]()  # Executes turn_on_lights()
```

**Fuzzy Autocompletion and Spell Correction.** Score misspelled queries (e.g., `"teh qyick"`) against a dictionary of valid terms. The Suffix Automaton gracefully handles partial and reordered matches that BM25's strict lexical matching would miss.

**Log Grouping and Error Deduplication.** Cluster similarly structured application tracebacks by recognizing shared contiguous substrings, even when localized IDs, timestamps, or memory addresses differ.

**Plagiarism Detection.** Generate structural overlap scores between documents in milliseconds, resistant to simple paraphrasing due to the contiguous-substring matching model.

---

## Comparative Analysis

### Memory Architecture

| Property | BM25 | ZLFI |
|----------|------|------|
| **Global state required?** | Yes — full inverted index | No — each document scored independently |
| **Can stream documents?** | No — IDF requires full corpus | Yes — constant memory, one document at a time |
| **Memory for 1M docs (Fully Loaded)**| ~2–3 GB (inverted index) | ~50–100+ GB depending on doc length (DAWGs in RAM) |
| **Memory for 32M docs (Streaming)** | N/A (requires full load) | ~5 MB (one document at a time) |
| **External engines?** | Typically Elasticsearch/Lucene | Pure Python/Cython, no dependencies |

### Strengths and Trade-offs

| Dimension | BM25 | ZLFI |
|-----------|------|------|
| **English keyword search** | Strong — IDF weights rare terms | Weaker — no term rarity awareness |
| **Multilingual (CJK, Arabic, etc.)** | Requires language-specific tokenizers | Strong — character-level, language-agnostic |
| **Exact phrase matching** | Bag-of-words loses word order | Preserves contiguous sequential structure |
| **Typo/noise tolerance** | Strict lexical matching | Suffix Automaton handles partial matches |
| **Memory footprint** | Scales with corpus size | Constant — each document independent |
| **Production maturity** | Decades of tooling | Early-stage, research-grade |
| **Training data** | None (unsupervised) | None (unsupervised) |

BM25 and ZLFI address different failure modes. BM25 excels when term rarity is the primary signal. ZLFI excels when sequential structure, morphological complexity, or memory constraints dominate. The alpha parameter allows practitioners to blend both signals according to their workload.

---

## Citation

If you use ZLFI in your work, please cite the foundational paper:

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
