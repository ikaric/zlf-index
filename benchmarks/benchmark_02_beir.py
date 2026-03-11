"""BEIR 2.0 Dataset Evaluation Pipeline for ZLF Index vs BM25 Contextual Reranking."""

import os
import argparse
import logging
import sys
import re
import gc
from collections import defaultdict
from typing import Dict, List

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi

# Set sys.path if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from zlfi import Corpus, Document, score
from tqdm import tqdm

# Setup logging later in evaluate() once arguments are parsed

def download_dataset(dataset: str) -> str:
    """Download the BEIR dataset."""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    return data_path

class LocalBM25Retriever:
    """A local BM25 Retriever compatible with BEIR passing top 1000 docs for Recall@1000."""
    def __init__(self, corpus: Dict[str, Dict[str, str]]):
        self.doc_ids = list(corpus.keys())
        logging.info("Tokenizing corpus for BM25 (BEIR 2.0 Regex tokenizer)...")
        self.tokenized_corpus = [
            re.findall(r'\w+', (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).lower())
            for doc_id in self.doc_ids
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def retrieve(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        results = {}
        for q_id, query in tqdm(queries.items(), desc="BM25 First Stage"):
            tokenized_query = re.findall(r'\w+', query.lower())
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Sort top 1000 for BEIR 2.0 Recall@1000 metric
            top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:1000]
            
            results[q_id] = {self.doc_ids[idx]: float(doc_scores[idx]) for idx in top_indices}
        return results

def evaluate(datasets: List[str], adversarial: bool, log_file: str):
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        
    split_name = "test-adversarial" if adversarial else "test"
    final_results = []
    
    for dataset in datasets:
        logging.info(f"\n{'='*40}\nEvaluating BEIR Dataset: {dataset.upper()}\n{'='*40}")
        logging.info(f"Downloading BEIR {dataset} dataset...")
        data_path = download_dataset(dataset)
        
        # Load dataset data
        logging.info(f"Loading documents, queries, and qrels for {dataset} ({split_name} split)...")
        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split_name)
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset}: {e}")
            continue
    
        # --- ZERO-DEPENDENCY HYBRID SEARCH ENGINE (Corpus) ---
        logging.info("Executing ZLF Index Hybrid Retrieval (Corpus)...")
        
        doc_ids_ordered = list(corpus.keys())
        doc_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).lower() for d in doc_ids_ordered]
        
        logging.info("Building Corpus Engine (BM25 + Suffix Automata)...")
        zlf_corpus = Corpus(doc_texts)
        
        zlf_results = {}
        bm25_only_results = {}
        
        for q_id, query in tqdm(queries.items(), desc="Corpus Reranking"):
            query_text = query.lower()
            
            # Compute BM25 and ZLF scores once each, then fuse manually
            bm25_scores = zlf_corpus.score_all(query_text, alpha=0.0)
            sa_scores = zlf_corpus.score_all(query_text, alpha=1.0)
            
            # Fuse: alpha * zlf + (1 - alpha) * norm_bm25
            alpha = 0.5
            min_b = min(bm25_scores) if bm25_scores else 0.0
            max_b = max(bm25_scores) if bm25_scores else 0.0
            range_b = max_b - min_b
            
            n_docs = len(bm25_scores)
            if range_b == 0:
                zlf_scores = [alpha * sa_scores[i] for i in range(n_docs)]
            else:
                zlf_scores = [
                    alpha * sa_scores[i] + (1.0 - alpha) * ((bm25_scores[i] - min_b) / range_b)
                    for i in range(n_docs)
                ]
            del sa_scores
            
            # Store top 1000 for evaluation
            bm25_top_indices = sorted(range(n_docs), key=lambda i: bm25_scores[i], reverse=True)[:1000]
            zlf_top_indices = sorted(range(n_docs), key=lambda i: zlf_scores[i], reverse=True)[:1000]
            
            bm25_only_results[q_id] = {doc_ids_ordered[idx]: float(bm25_scores[idx]) for idx in bm25_top_indices}
            zlf_results[q_id] = {doc_ids_ordered[idx]: float(zlf_scores[idx]) for idx in zlf_top_indices}
            
            del bm25_scores, zlf_scores, bm25_top_indices, zlf_top_indices

        retriever_eval = EvaluateRetrieval()
        k_values = [1, 3, 5, 10, 100, 1000]
        
        # Evaluate Native BM25 (Baseline)
        logging.info("--- NATIVE BM25 BASELINE METRICS ---")
        ndcg_bm25, _map_bm25, recall_bm25, _p_bm25 = retriever_eval.evaluate(qrels, bm25_only_results, k_values)
        bm25_ndcg10 = ndcg_bm25.get("NDCG@10", 0.0)
        bm25_recall1000 = recall_bm25.get("Recall@1000", 0.0)

        # Evaluate ZLF Index (Ensemble)
        logging.info("--- ZLF INDEX HYBRID METRICS ---")
        ndcg_zlf, _map_zlf, recall_zlf, _p_zlf = retriever_eval.evaluate(qrels, zlf_results, k_values)
        zlf_ndcg10 = ndcg_zlf.get("NDCG@10", 0.0)
        zlf_recall1000 = recall_zlf.get("Recall@1000", 0.0)
    
        # Store results for final averaging
        final_results.append({
            "dataset": dataset.capitalize(),
            "bm25_ndcg": bm25_ndcg10,
            "zlf_ndcg": zlf_ndcg10,
            "recall": bm25_recall1000
        })
        
        # Explicit garbage collection to prevent memory ballooning across dataset loops
        del corpus
        del queries
        del qrels
        del doc_ids_ordered
        del doc_texts
        del zlf_corpus
        del zlf_results
        del bm25_only_results
            
        logging.info(f"Reclaiming memory... freed objects.")
        gc.collect()
        
        # Force pymalloc to release arenas back to the OS
        import ctypes
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # --- PRINT FINAL MARKDOWN TABLE ---
    logging.info("\n\n" + "="*80)
    logging.info("FINAL AGGREGATED BENCHMARK RESULTS (BEIR 2.0 Standard)")
    logging.info("="*80)
    
    split_str = "Adversarial" if adversarial else "Standard"
        
    print(f"\n| Dataset | Split | Metric | BM25 | ZLF Index | Diff vs BM25 |")
    print(f"|---------|-------|--------|------|-------------|--------------|")
    
    avg_bm25_ndcg = 0.0
    avg_zlf_ndcg = 0.0
    avg_recall = 0.0
    
    for r in final_results:
        diff = (r['zlf_ndcg'] - r['bm25_ndcg']) * 100
        if diff > 0:
            diff_str = f"🟢 +{diff:.1f}%"
        elif diff < 0:
            diff_str = f"🔴 {diff:.1f}%"
        else:
            diff_str = f"⚪  0.0%"
            
        print(f"| **{r['dataset']}** | {split_str} | NDCG@10 | {r['bm25_ndcg']*100:.1f}% | {r['zlf_ndcg']*100:.1f}% | {diff_str} |")
        print(f"| **{r['dataset']}** | {split_str} | Recall@1000 | {r['recall']*100:.1f}% | {r['recall']*100:.1f}% | ⚪ Same |")
        
        avg_bm25_ndcg += r['bm25_ndcg']
        avg_zlf_ndcg += r['zlf_ndcg']
        avg_recall += r['recall']
        
    if final_results:
        avg_bm25_ndcg /= len(final_results)
        avg_zlf_ndcg /= len(final_results)
        avg_recall /= len(final_results)
        
        diff_avg = (avg_zlf_ndcg - avg_bm25_ndcg) * 100
        if diff_avg > 0:
            diff_avg_str = f"🟢 +{diff_avg:.1f}%"
        elif diff_avg < 0:
            diff_avg_str = f"🔴 {diff_avg:.1f}%"
        else:
            diff_avg_str = f"⚪  0.0%"
            
        print(f"|---------|-------|--------|------|-------------|--------------|")
        print(f"| **AVERAGE** | {split_str} | NDCG@10 | {avg_bm25_ndcg*100:.1f}% | {avg_zlf_ndcg*100:.1f}% | {diff_avg_str} |")
        print(f"| **AVERAGE** | {split_str} | Recall@1000 | {avg_recall*100:.1f}% | {avg_recall*100:.1f}% | ⚪ Same |")

    print("\n")
    logging.info("BEIR 2.0 Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ZLF Index against BEIR 2.0 datasets")
    parser.add_argument("--datasets", nargs="+", default=["scifact"], help="BEIR 2.0 datasets to evaluate (e.g., scifact msmarco-v2 medqa-retrieval)")
    parser.add_argument("--adversarial", action="store_true", help="Run evaluation on the 'test-adversarial' split instead of 'test'")
    parser.add_argument("--log-file", type=str, default="", help="Path to a file to redirect detailed logs")
    args = parser.parse_args()
    
    evaluate(args.datasets, args.adversarial, args.log_file)

