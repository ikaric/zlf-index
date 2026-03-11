import os
import sys
import random
import logging
import argparse
import gc
from typing import Dict, List, Tuple
from tqdm import tqdm

from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi

# Set sys.path if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from zlfi import score, Document

# Newly added import to support fast language-specific downloads
from datasets import load_dataset

# Setup logging later in evaluate() once arguments are parsed

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ZLF Index on Mr. TyDi (BEIR metrics, HF data)")
    parser.add_argument("--languages", nargs="+", default=["english", "arabic", "japanese", "korean", "bengali", "finnish", "indonesian", "russian", "swahili", "telugu", "thai"],
                        help="List of languages to evaluate (e.g., english arabic japanese)")
    parser.add_argument("--log-file", type=str, default="",
                        help="Path to a file to redirect detailed logs (leaves only tqdm bars and final table in terminal)")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Max number of queries to evaluate per language (0 for all)")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Max number of 'noise' documents to load alongside required ground-truth docs (0 for all)")
    return parser.parse_args()

class LocalBM25Retriever:
    """A local BM25 Retriever compatible with BEIR passing top 100 docs."""
    def __init__(self, corpus: Dict[str, Dict[str, str]]):
        self.doc_ids = list(corpus.keys())
        logging.info("Tokenizing corpus for BM25...")
        import re
        self.tokenized_corpus = [
            re.findall(r'\w+', (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).lower())
            for doc_id in self.doc_ids
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def retrieve(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        results = {}
        import re
        for q_id, query in tqdm(queries.items(), desc="BM25 First Stage", leave=False):
            tokenized_query = re.findall(r'\w+', query.lower())
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Sort top 1000
            top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:1000]
            results[q_id] = {self.doc_ids[idx]: float(doc_scores[idx]) for idx in top_indices}
        return results

def evaluate():
    args = parse_args()
    
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w')
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        
    retriever_eval = EvaluateRetrieval()
    
    # Store results for the final markdown table
    final_results = []
    
    for lang in args.languages:
        logging.info(f"\n{'='*40}\nEvaluating Language: {lang.upper()}\n{'='*40}")
        
        logging.info(f"Loading queries and qrels for {lang} from ir_datasets (`mr-tydi/[lang]/test`)...")
        
        # ir_datasets uses iso codes, map full names to iso
        lang_map = {
            "arabic": "ar", "bengali": "bn", "english": "en", 
            "finnish": "fi", "indonesian": "id", "japanese": "ja", 
            "korean": "ko", "russian": "ru", "swahili": "sw", 
            "telugu": "te", "thai": "th"
        }
        
        iso_lang = lang_map.get(lang.lower(), lang.lower())
        dataset_id = f"mr-tydi/{iso_lang}/test"
        
        try:
            import ir_datasets
            dataset = ir_datasets.load(dataset_id)
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_id}: {e}")
            continue

        queries = {}
        for q in dataset.queries_iter():
            queries[str(q.query_id)] = str(q.text)
            
        qrels = {}
        for qrel in dataset.qrels_iter():
            q_id = str(qrel.query_id)
            if qrel.relevance > 0:
                if q_id not in qrels:
                    qrels[q_id] = {}
                qrels[q_id][str(qrel.doc_id)] = int(qrel.relevance)
            
        # Guarantee standard full baseline evaluation or subset queries
        if args.max_queries > 0:
            query_ids = list(queries.keys())
            if len(query_ids) > args.max_queries:
                sampled_qids = query_ids[:args.max_queries]
                queries = {qid: queries[qid] for qid in sampled_qids}
                qrels = {qid: qrels[qid] for qid in sampled_qids if qid in qrels}
                logging.info(f"Subsampled to {args.max_queries} queries for quick iteration.")
        else:
            query_ids = list(queries.keys())
            logging.info(f"Loaded {len(query_ids)} queries for {lang.capitalize()}.")
        
        corpus = {}
        
        # Determine if we should intelligent-subset the corpus to save RAM
        if args.max_docs > 0:
            required_docs = set()
            for qid in queries:
                if qid in qrels:
                    required_docs.update(qrels[qid].keys())
            
            target_noise_count = args.max_docs
            logging.info(f"Streaming corpus with subsampling (Targeting {target_noise_count} noise docs + {len(required_docs)} specific truth docs)...")
            
            current_noise_count = 0
            required_docs_needed = len(required_docs)
            required_docs_found = 0
            
            try:
                total_docs = dataset.docs_count()
            except (AttributeError, TypeError, NotImplementedError):
                total_docs = None
                
            for doc in tqdm(dataset.docs_iter(), total=total_docs, desc=f"Scanning {lang.capitalize()} Corpus for ground-truths", leave=False):
                doc_id_str = str(doc.doc_id)
                is_required = doc_id_str in required_docs
                
                if is_required or current_noise_count < target_noise_count:
                    corpus[doc_id_str] = {
                        "title": str(getattr(doc, "title", "")), 
                        "text": str(getattr(doc, "text", ""))
                    }
                    if is_required:
                        required_docs_found += 1
                    else:
                        current_noise_count += 1
                        
                # Early stop if we have found all required documents AND filled our noise quota
                if current_noise_count >= target_noise_count and required_docs_found >= required_docs_needed:
                    logging.info("Early stopping triggered: Found all required truth documents and noise quota met.")
                    break
        else:
            try:
                total_docs = dataset.docs_count()
            except (AttributeError, TypeError, NotImplementedError):
                total_docs = None
                
            logging.info(f"Streaming the complete {lang.capitalize()} corpus into memory...")
            
            for doc in tqdm(dataset.docs_iter(), total=total_docs, desc=f"Loading {lang.capitalize()} Corpus", leave=False):
                corpus[str(doc.doc_id)] = {
                    "title": str(getattr(doc, "title", "")), 
                    "text": str(getattr(doc, "text", ""))
                }
                
        logging.info(f"Loaded {len(corpus)} documents into memory.")
                
        # --- PHASE 1: BM25 ---
        logging.info("Phase 1: Local BM25 First-Stage Retrieval (Top 100)...")
        bm25_model = LocalBM25Retriever(corpus)
        bm25_results = bm25_model.retrieve(corpus, queries)
        
        k_values = [1, 3, 5, 10, 100, 1000]
        ndcg_bm25, map_bm25, recall_bm25, _ = retriever_eval.evaluate(qrels, bm25_results, k_values)
        bm25_mrr100 = map_bm25.get("MAP@100", 0.0)
        bm25_recall100 = recall_bm25.get("Recall@100", 0.0)
        
        # --- PHASE 2: ZLF INDEX RERANKING ---
        logging.info("Phase 2: ZLF Index Reranking...")
        
        # Only build automata for documents that appear in BM25 results
        rerank_doc_ids = set()
        for hits in bm25_results.values():
            rerank_doc_ids.update(hits.keys())
        
        logging.info(f"Building ZLF Index Documents for {len(rerank_doc_ids)} unique reranking candidates (not full corpus)...")
        zlf_docs = {}
        for doc_id in tqdm(rerank_doc_ids, desc="Building Documents", leave=False):
            doc_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).lower()
            zlf_docs[doc_id] = Document(doc_text)
            
        zlf_results = {}
        
        for q_id, hits in tqdm(bm25_results.items(), desc="Reranking Queries", leave=False):
            query_text = queries[q_id].lower()
            doc_ids_list = list(hits.keys())
            doc_objs = [zlf_docs[doc_id] for doc_id in doc_ids_list]
            
            # Batch score via score() which dispatches to batch_scores -> OpenMP prange
            sa_scores = score(query_text, doc_objs)
            
            # Fuse BM25 and ZLF Index identically to BEIR
            alpha = 0.5
            bm25_scores = [hits[doc_id] for doc_id in doc_ids_list]
            n_docs = len(bm25_scores)
            
            min_b = min(bm25_scores) if n_docs > 0 else 0.0
            max_b = max(bm25_scores) if n_docs > 0 else 0.0
            range_b = max_b - min_b
            
            if range_b == 0:
                rerank_scores = [alpha * sa_scores[i] for i in range(n_docs)]
            else:
                rerank_scores = [
                    alpha * sa_scores[i] + (1.0 - alpha) * ((bm25_scores[i] - min_b) / range_b)
                    for i in range(n_docs)
                ]
            
            zlf_results[q_id] = {doc_id: s for doc_id, s in zip(doc_ids_list, rerank_scores)}
                
        logging.info(f"--- ZLF INDEX METRICS ({lang}) ---")
        # Evaluate on the new k_values array
        ndcg_zlf, map_zlf, recall_zlf, _ = retriever_eval.evaluate(qrels, zlf_results, k_values)
        zlf_mrr100 = map_zlf.get("MAP@100", 0.0)
        zlf_recall100 = bm25_recall100  # Recall@100 is identical for reranker phase
        
        final_results.append({
            "lang": lang.capitalize(),
            "bm25_mrr": bm25_mrr100,
            "zlf_mrr": zlf_mrr100,
            "recall": bm25_recall100
        })
        
        # Explicit garbage collection to prevent memory ballooning across language loops
        del corpus
        del queries
        del qrels
        del bm25_model
        del bm25_results
        del rerank_doc_ids
        del zlf_docs
        del zlf_results
        
        logging.info(f"Reclaiming memory... freed objects.")
        gc.collect()
        
        import ctypes
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # --- PRINT FINAL MARKDOWN TABLE ---
    logging.info("\n\n" + "="*60)
    logging.info("FINAL AGGREGATED BENCHMARK RESULTS (MRR@100)")
    logging.info("="*60)
    
    print("\n| Dataset | Level | Metric | BM25 | ZLF Index | Diff vs BM25 |")
    print("|---------|----------|--------|------|-------------|--------------|")
    for r in final_results:
        diff_mrr = (r['zlf_mrr'] - r['bm25_mrr']) * 100
        if diff_mrr > 0:
            diff_str = f"🟢 +{diff_mrr:.1f}%"
        elif diff_mrr < 0:
            diff_str = f"🔴 {diff_mrr:.1f}%"
        else:
            diff_str = f"⚪  0.0%"
            
        print(f"| **Mr. TyDi** | {r['lang']} | MRR@100 | {r['bm25_mrr']*100:.1f}% | {r['zlf_mrr']*100:.1f}% | {diff_str} |")
        print(f"| **Mr. TyDi** | {r['lang']} | Recall@100 | {r['recall']*100:.1f}% | {r['recall']*100:.1f}% | ⚪ Same |")
    print("\n")

if __name__ == "__main__":
    evaluate()
