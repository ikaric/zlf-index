"""Speed Evaluation: Suffix25 Index Engine."""

import os
import sys
import time
import random
import argparse
from typing import Dict, List

# Set sys.path if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from suffix25 import Corpus, Document, score
from tqdm import tqdm

def generate_random_text(words: List[str], length: int) -> str:
    return " ".join(random.choices(words, k=length))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Speed of Suffix25 Index")
    parser.add_argument("--docs", type=int, default=5000, help="Number of documents in corpus")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries to run")
    parser.add_argument("--doc-len", type=int, default=200, help="Average words per document")
    parser.add_argument("--query-len", type=int, default=5, help="Average words per query")
    return parser.parse_args()

def _bench(fn, iterations=5):
    """Return the minimum total time across multiple trials."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)

def evaluate_speed():
    args = parse_args()
    
    print(f"=== Comprehensive Speed Benchmark ===")
    print(f"Corpus size: {args.docs} documents ({args.doc_len} words/doc)")
    print(f"Queries: {args.queries} queries ({args.query_len} words/query)")
    print("-" * 30)

    # Generate synthetic data
    print("Generating synthetic dataset...")
    vocab = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello", "world", 
             "search", "engine", "retrieval", "algorithm", "speed", "test", "benchmark", "fast", 
             "slow", "performance", "metric", "evaluate", "accuracy", "precision", "recall"]
    
    corpus_texts = [generate_random_text(vocab, args.doc_len) for _ in range(args.docs)]
    queries = [generate_random_text(vocab, args.query_len) for _ in range(args.queries)]
    
    single_doc_text = corpus_texts[0]
    single_query = queries[0]

    results = {}

    # =========================================================
    # Phase 1: Indexing & Preparation
    # =========================================================
    print("\n--- Indexing Phase ---")
    
    # Suffix25 Index (Native BM25 + Suffix Automaton)
    start = time.perf_counter()
    suffix25_corpus = Corpus(corpus_texts)
    idx_suffix25 = time.perf_counter() - start
    print(f"[Suffix25 Corpus]     Indexing time: {idx_suffix25:.4f}s")

    # Single Document setup
    suffix25_single_doc = Document(single_doc_text)
    
    results['indexing'] = {
        'suffix25_corpus': idx_suffix25,
    }

    # =========================================================
    # Phase 2: Single Document Inference Speed
    # =========================================================
    print("\n--- Single Document Inference (1 Query vs 1 Document) ---")
    iterations = 10_000

    # Native BM25 alpha=0.0
    single_native_bm25 = _bench(lambda: [suffix25_single_doc.score(single_query, alpha=0.0) for _ in range(iterations)], 3)
    # Native Suffix Automaton alpha=1.0
    single_suffix25_sa = _bench(lambda: [suffix25_single_doc.score(single_query, alpha=1.0) for _ in range(iterations)], 3)
    # Native Hybrid alpha=0.5
    single_suffix25_hybrid = _bench(lambda: [suffix25_single_doc.score(single_query, alpha=0.5) for _ in range(iterations)], 3)

    print(f"[Native BM25]    {single_native_bm25:.4f}s for {iterations:,} calls")
    print(f"[Suffix Auto]    {single_suffix25_sa:.4f}s for {iterations:,} calls")
    print(f"[Hybrid Eng.]    {single_suffix25_hybrid:.4f}s for {iterations:,} calls")

    results['single'] = {
        'native_bm25': (single_native_bm25 / iterations) * 1e6,
        'suffix_automaton': (single_suffix25_sa / iterations) * 1e6,
        'hybrid': (single_suffix25_hybrid / iterations) * 1e6,
    }

    # =========================================================
    # Phase 3: Multi-Corpus Throughput (1 Query vs N Documents)
    # =========================================================
    print(f"\n--- Multi-Corpus Throughput ({args.queries} Queries vs {args.docs} Documents) ---")
    
    # 1. Native BM25 only (alpha = 0.0)
    multi_native_bm25 = 0.0
    for q in tqdm(queries, desc="[Native BM25]    "):
        start = time.perf_counter()
        scores = suffix25_corpus.score_all(q, alpha=0.0)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        multi_native_bm25 += (time.perf_counter() - start)

    # 2. Suffix Automaton only (alpha = 1.0)
    multi_suffix25_sa = 0.0
    for q in tqdm(queries, desc="[Suffix Auto]    "):
        start = time.perf_counter()
        scores = suffix25_corpus.score_all(q, alpha=1.0)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        multi_suffix25_sa += (time.perf_counter() - start)
        
    # 3. Hybrid (alpha = 0.5)
    multi_suffix25_hybrid = 0.0
    for q in tqdm(queries, desc="[Hybrid Eng.]    "):
        start = time.perf_counter()
        scores = suffix25_corpus.score_all(q, alpha=0.5)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        multi_suffix25_hybrid += (time.perf_counter() - start)

    results['multi'] = {
        'native_bm25': (multi_native_bm25 / args.queries) * 1e3,
        'suffix_automaton': (multi_suffix25_sa / args.queries) * 1e3,
        'hybrid': (multi_suffix25_hybrid / args.queries) * 1e3,
    }


    # =========================================================
    # FINAL MARKDOWN TABLES
    # =========================================================
    print("\n" + "="*80)
    print("FINAL AGGREGATED SPEED BENCHMARK RESULTS")
    print("="*80 + "\n")
        
    # 1. Single Document Inference
    print("### Single Document Inference Speed (Per Query)")
    print("Measures the raw time to score a single query string against a single loaded Document.\n")
    print("| Method | Time (\u03bcs/query) |")
    print("|--------|-------------------|")
    s = results['single']
    print(f"| SUFFIX25 Native BM25 | {s['native_bm25']:.2f} \u03bcs |")
    print(f"| SUFFIX25 Suffix Auto | {s['suffix_automaton']:.2f} \u03bcs |")
    print(f"| SUFFIX25 Hybrid (0.5)| {s['hybrid']:.2f} \u03bcs |")
    print("\n")

    # 2. Multi Corpus Throughput
    print(f"### Multi-Corpus Throughput (1 Query vs {args.docs} Documents)")
    print(f"Measures the time to score a single query against the entire loaded {args.docs}-document Corpus and extract top-k.\n")
    print("| Method | Time (ms/query) |")
    print("|--------|-----------------|")
    m = results['multi']
    print(f"| Suffix25 Native BM25 | {m['native_bm25']:.2f} ms |")
    print(f"| Suffix25 Suffix Auto | {m['suffix_automaton']:.2f} ms |")
    print(f"| Suffix25 Hybrid (0.5)| {m['hybrid']:.2f} ms |")
    print("\n")
    
    # 3. Indexing Performance
    print("### Indexing Performance")
    print(f"Measures the time to build the initial index from raw strings for {args.docs} documents.\n")
    print("| Engine | Indexing Time (s) | Components Built |")
    print("|--------|-------------------|------------------|")
    idx = results['indexing']
    print(f"| Suffix25 Corpus | {idx['suffix25_corpus']:.4f}s | Suffix Automata + Native BM25 |")
    print("\n")


if __name__ == "__main__":
    evaluate_speed()
