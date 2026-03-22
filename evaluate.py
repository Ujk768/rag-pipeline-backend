from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os, json

# your existing pruning functions imported from main
from main import prune_cosine, prune_cosine_whitened, prune_kmeans, prune_mmr, prune_docpruner

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("HOST"),
    "port": os.getenv("PORT"),
}

STRATEGIES = ["none", "cosine", "cosine_whitened", "kmeans", "mmr", "docpruner"]

def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn

def reset_table():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks;")
    conn.commit()
    cur.close(); conn.close()

def index_corpus(corpus, embeddings_map, strategy):
    """Prune and insert corpus embeddings into pgvector."""
    reset_table()
    
    doc_ids = list(corpus.keys())
    all_embeddings = np.array([embeddings_map[doc_id] for doc_id in doc_ids])
    # Mock chunks for the pruning functions
    chunks = [{"page_number": 0, "sentence_chunk": ""} for _ in doc_ids]

    # --- FIX STARTS HERE ---
    if strategy == "none" or len(doc_ids) <= 1:
        kept_indices = list(range(len(doc_ids)))
    
    elif strategy == "docpruner":
        # Using k_factor=0.0 keeps roughly 50% based on the mean
        kept_indices, _ = prune_docpruner(all_embeddings, chunks, k_factor=0.0)

    elif strategy == "kmeans":
        # Force a 50% budget for the test instead of sqrt(n)
        kept_indices, _ = prune_kmeans(all_embeddings, chunks, n_clusters=int(len(doc_ids) * 0.5))

    elif strategy == "mmr":
        # Force a 50% budget for the test
        kept_indices, _ = prune_mmr(all_embeddings, chunks, target_k=int(len(doc_ids) * 0.5))
    
    elif strategy in ["cosine", "cosine_whitened"]:
        # Ensure these are using the "flipped" logic (keeping high similarity)
        # if you updated them in main.py. 
        # For now, let's use your existing imports:
        if strategy == "cosine":
            kept_indices, _ = prune_cosine(all_embeddings, chunks)
        else:
            kept_indices, _ = prune_cosine_whitened(all_embeddings, chunks)

    conn = get_conn()
    cur = conn.cursor()
    from psycopg2.extras import execute_values
    rows = [
        (doc_ids[i], all_embeddings[i].tolist())
        for i in kept_indices
    ]
    # store doc_id in content column for retrieval mapping
    execute_values(cur,
        "INSERT INTO document_chunks (page_number, content, embedding) VALUES %s",
        [(0, doc_id, emb) for doc_id, emb in rows], template="(%s, %s, %s::vector)"
    )
    conn.commit()
    cur.close(); conn.close()
    return len(kept_indices), len(doc_ids)

def retrieve(queries, query_embeddings, top_k=10):
    results = {}
    conn = get_conn()
    cur = conn.cursor()
    
    # query_ids allows us to map the loop correctly
    query_ids = list(queries.keys())
    
    for qid, qemb in zip(query_ids, query_embeddings):
        qemb_list = qemb.tolist()
        # We use cosine similarity (1 - distance)
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (qemb_list, qemb_list, top_k))
        
        rows = cur.fetchall()
        # BEIR expects {query_id: {doc_id: score}}
        results[qid] = {row[0]: float(row[1]) for row in rows}
        
    cur.close()
    conn.close()
    return results

def main():
    # Download scifact (small — 5k docs, 300 queries)
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
    data_path = util.download_and_unzip(url, "beir_datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)}")

    # Embed everything once — reused across all strategy runs
    model = SentenceTransformer("all-mpnet-base-v2")
    
    print("Embedding corpus...")
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[d]["title"] + " " + corpus[d]["text"] for d in doc_ids]
    corpus_embeddings = model.encode(doc_texts, batch_size=64, show_progress_bar=True)
    embeddings_map = {doc_ids[i]: corpus_embeddings[i] for i in range(len(doc_ids))}

    print("Embedding queries...")
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]
    query_embeddings = model.encode(query_texts, batch_size=64, show_progress_bar=True)

    results_summary = {}

    for strategy in STRATEGIES:
        print(f"\n--- Strategy: {strategy} ---")
        kept, total = index_corpus(corpus, embeddings_map, strategy)
        print(f"Indexed {kept}/{total} docs ({round(100*kept/total, 1)}% kept)")

        retrieved = retrieve(queries, query_embeddings, top_k=10)

        evaluator = EvaluateRetrieval()
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, retrieved, [10])

        results_summary[strategy] = {
            "ndcg@10": round(ndcg["NDCG@10"], 4),
            "recall@10": round(recall["Recall@10"], 4),
            "docs_stored": kept,
            "total_docs": total,
            "storage_pct": round(100 * kept / total, 1),
        }
        print(f"nDCG@10: {results_summary[strategy]['ndcg@10']} | Recall@10: {results_summary[strategy]['recall@10']}")

    print("\n\n===== RESULTS =====")
    print(f"{'Strategy':<20} {'nDCG@10':<12} {'Recall@10':<12} {'Stored':<10} {'Storage%'}")
    print("-" * 65)
    for s, r in results_summary.items():
        print(f"{s:<20} {r['ndcg@10']:<12} {r['recall@10']:<12} {r['docs_stored']:<10} {r['storage_pct']}%")

    with open("beir_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\nSaved to beir_results.json")

if __name__ == "__main__":
    main()