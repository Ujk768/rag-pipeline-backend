import os
import math
import re
import fitz  # PyMuPDF
import torch
import numpy as np
import psycopg2
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from contextlib import asynccontextmanager
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Remove bitsandbytes when running on a non-GPU environment.
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from typing import Literal, Optional
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",   
    "*",
]

load_dotenv()

# CONFIGURATION
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")

# Leaves headroom for prompt template + generated answer.
# We can raise this if our model has a larger context window (e.g. 28000 for 32K models).
# .!! See github issues - one person has to research the limits of our model
FULL_CONTEXT_TOKEN_LIMIT = 6000

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"), # Must be DB_USER in .env — USER is a reserved Linux variable
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("HOST"),
    "port": os.getenv("PORT"),
}

# Valid pruning strategy literals — used for type checking and API docs
PruningStrategy = Literal["none", "cosine", "cosine_whitened", "kmeans", "mmr"]


# DATABASE
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn

def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()  # commit before registering so the type physically exists
    
    register_vector(conn)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id           SERIAL PRIMARY KEY,
            page_number  INTEGER,
            content      TEXT,
            embedding    vector(768)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[INFO] Database initialized and ready.")

# GLOBAL STATE
processing_status = {"status": "idle", "chunks": 0, "error": None, "mode": None}
full_context_pages: list[dict] = []

# Persistent pruning report exposed to frontend
# Stores detailed stats from the most recent pruning run.
# Reset on each new upload. Frontend polls /pruning-report to visualize results.
pruning_report: dict = {}

embedding_model = None
llm_model = None
tokenizer = None
nlp = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use this block instead if you're on a Mac with MPS support - Comment out the line above
# if torch.cuda.is_available():
#     DEVICE = "cuda"
# elif torch.backends.mps.is_available():
#     DEVICE = "mps"
# else:
#     DEVICE = "cpu"


# LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Init DB...")
    init_db()

    global embedding_model, llm_model, tokenizer, nlp

    print("[INFO] Loading spaCy...")
    nlp = English()
    nlp.add_pipe("sentencizer")

    print("[INFO] Loading embedding model...")
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

    print("[INFO] Loading LLM and tokenizer...")

    if DEVICE == "cpu":
        print("[WARNING] CUDA not found. Loading LLM in full precision on CPU (Slow).")
        quantization_config = None
        current_device_map = None
    else:
        print("[INFO] CUDA found. Loading LLM with 4-bit quantization for faster inference.")
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
        )
        current_device_map = "auto"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        # device_map="auto",
        device_map=current_device_map,
    )

    if DEVICE == "cuda":
        print("[INFO] Compiling model for faster inference (this may take a few minutes on first run)...")
        llm_model = torch.compile(llm_model)
    
    # Uncomment this and comment out the two ifs above if having issues with bits and bytes
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    # llm_model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_ID,
    #     token=HF_TOKEN,
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    #     device_map="auto",
    #     #device_map="auto", 
    # )
    print("[INFO] All models loaded successfully!")
    yield
    print("[INFO] Shutting down...")


app = FastAPI(lifespan=lifespan, title="Local RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],              # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],              # Allows all headers
)


# REQUEST SCHEMAS
# CHANGED: added use_maxsim flag — MaxSim is now a retrieval-side option, not an indexing strategy
class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.7
    max_new_tokens: int = 256
    use_maxsim: bool = False  # if True, re-ranks retrieved chunks using MaxSim before answering


# HELPER FUNCTIONS
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

def chunk_sentences(sentences: list[str], chunk_size: int = 20, overlap: int = 5) -> list[str]:
    """
    Splits a flat list of sentences into overlapping chunks.
    
    - Divides total sentences evenly across chunks to avoid a small trailing chunk
    - Each chunk extends 'overlap' sentences before and after its boundaries
    """
    total = len(sentences)
    
    if total == 0:
        return []
    
    # If the whole document is smaller than chunk_size, just return it as one chunk
    if total <= chunk_size + 2 * overlap:
        return [" ".join(sentences)]
    
    # Calculate number of chunks and distribute sentences evenly
    num_chunks = math.ceil(total / chunk_size)
    base_size = total // num_chunks
    remainder = total % num_chunks

    # Build chunk boundary indices
    chunks = []
    start = 0
    for i in range(num_chunks):
        # Distribute remainder sentences across the first few chunks
        end = start + base_size + (1 if i < remainder else 0)
        
        # Apply overlap: read 'overlap' sentences before and after boundaries
        overlap_start = max(0, start - overlap)
        overlap_end = min(total, end + overlap)
        
        chunk_text = " ".join(sentences[overlap_start:overlap_end])
        chunks.append(chunk_text)
        
        start = end
    
    return chunks



# ADDED: PRUNING STRATEGY IMPLEMENTATIONS
#
# Three strategies, each returns:
#   kept_indices  : list[int]  — which chunks survive pruning
#   pruning_stats : dict       — detailed metrics for frontend visualization
#
# All three share the same signature so process_pdf() can call them uniformly.

def compute_whitening_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Whitening transform
    Computes a whitening matrix W from the embedding matrix using PCA.

    Whitening solves the *anisotropy* problem: raw Sentence Transformer
    embeddings cluster in a narrow cone, making cosine similarities
    artificially inflated for unrelated chunks. Whitening:
      1. Decorrelates embedding dimensions (removes covariance)
      2. Normalises each dimension to unit variance

    After whitening, cosine similarity becomes a much stronger signal —
    0.85 post-whitening means genuinely similar content, not just shared
    embedding geometry.

    Returns W such that: e_whitened = embeddings @ W.T
    """
    # Centre the embeddings
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # Covariance matrix: (d x d)
    cov = np.cov(centered, rowvar=False)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Clip tiny/negative eigenvalues for numerical stability
    eigenvalues = np.clip(eigenvalues, a_min=1e-8, a_max=None)

    # Whitening matrix: W = diag(1/sqrt(λ)) @ V.T
    W = (eigenvectors / np.sqrt(eigenvalues)).T  # shape: (d, d)

    return W, mean


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Vectorised cosine similarity
    Computes cosine similarity between every row in A and every row in B.
    Returns a matrix of shape (len(A), len(B)).
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    
    return A_norm @ B_norm.T


# Strategy 1 - Cosine Similarity Pruning
def prune_cosine(
    embeddings: np.ndarray,
    chunks: list[dict],
    threshold_multiplier: float = 0.85,
) -> tuple[list[int], dict]:
    """
    Prunes chunks whose embeddings are too similar to the document centroid.

    Logic:
      - Compute the centroid (mean) of all embeddings.
      - Measure cosine similarity of each embedding vs the centroid.
      - Chunks *above* the threshold are near-average — redundant. Prune them.
      - Chunks *below* the threshold carry distinct information. Keep them.

    The threshold is adaptive: mean(scores) * threshold_multiplier.
    Higher multiplier → more aggressive pruning.

    Limitation: raw cosine similarity is noisy due to anisotropy in
    Sentence Transformer embeddings. Use cosine_whitened for cleaner signal.
    """
    n = len(embeddings)
    centroid = embeddings.mean(axis=0, keepdims=True)  # shape: (1, d)

    # Cosine similarity of each chunk vs centroid
    scores = cosine_similarity_matrix(embeddings, centroid).flatten()  # shape: (n,)

    # Adaptive threshold: chunks *above* this are too average → prune
    threshold = float(scores.mean() * threshold_multiplier)

    kept_indices = [i for i, s in enumerate(scores) if s <= threshold]

    # Edge case: if everything got pruned, keep the most distinct chunk
    if not kept_indices:
        kept_indices = [int(np.argmin(scores))]

    pruned_indices = [i for i in range(n) if i not in set(kept_indices)]

    stats = _build_pruning_stats(
        strategy="cosine",
        n_total=n,
        kept_indices=kept_indices,
        pruned_indices=pruned_indices,
        scores=scores.tolist(),
        threshold=threshold,
        chunks=chunks,
        extra={
            "threshold_multiplier": threshold_multiplier,
            "score_meaning": "cosine similarity vs centroid — higher = more redundant",
        },
    )

    return kept_indices, stats


# Strategy 2 - Cosine + Whitening Pruning
def prune_cosine_whitened(
    embeddings: np.ndarray,
    chunks: list[dict],
    threshold_multiplier: float = 0.85,
) -> tuple[list[int], dict]:
    """
    Same logic as prune_cosine() but applies whitening first.

    Whitening corrects the anisotropy problem in Sentence Transformer embeddings:
    embeddings that appear similar under raw cosine are often just occupying
    the same narrow region of vector space - not genuinely semantically close.

    After whitening, the vector space is normalised so that cosine similarity
    scores are more discriminative and meaningful. Pruning decisions made on
    whitened embeddings are more trustworthy.

    Steps:
      1. Compute and apply whitening matrix W to all embeddings
      2. Run cosine similarity vs centroid on whitened embeddings
      3. Prune chunks above adaptive threshold
    """
    n = len(embeddings)

    # Compute whitening transform
    W, mean = compute_whitening_matrix(embeddings)

    # Apply whitening: project each embedding into the whitened space
    whitened = (embeddings - mean) @ W.T  # shape: (n, d)

    # Centroid in whitened space
    centroid = whitened.mean(axis=0, keepdims=True)

    # Cosine similarity in whitened space
    scores = cosine_similarity_matrix(whitened, centroid).flatten()

    # Adaptive threshold
    threshold = float(scores.mean() * threshold_multiplier)

    kept_indices = [i for i, s in enumerate(scores) if s <= threshold]

    if not kept_indices:
        kept_indices = [int(np.argmin(scores))]

    pruned_indices = [i for i in range(n) if i not in set(kept_indices)]

    stats = _build_pruning_stats(
        strategy="cosine_whitened",
        n_total=n,
        kept_indices=kept_indices,
        pruned_indices=pruned_indices,
        scores=scores.tolist(),
        threshold=threshold,
        chunks=chunks,
        extra={
            "threshold_multiplier": threshold_multiplier,
            "score_meaning": "cosine similarity vs centroid in whitened space — more discriminative than raw cosine",
            "whitening_applied": True,
        },
    )

    return kept_indices, stats


# ADDED: Strategy 3 - K-Means Clustering Based Selection
def prune_kmeans(
    embeddings: np.ndarray,
    chunks: list[dict],
    n_clusters: int = None,
) -> tuple[list[int], dict]:
    """
    K-Means clustering based representative selection.

    Instead of scoring individual vectors against a centroid or each other,
    this strategy clusters all document embeddings into K groups and keeps
    the one chunk per cluster that is closest to its cluster centroid.

    Why this is more principled than cosine pruning:
      - Cosine pruning compares everything to one global average point,
        which ignores the actual topical structure of the document.
      - K-Means respects the real distribution — if a document covers
        3 distinct topics, 3 clusters naturally emerge, and we keep one
        representative per topic.
      - K maps directly onto the project's cost-based framing:
        K is the storage budget. You decide how many vectors you can afford,
        and the algorithm finds the best K representatives for that budget.

    n_clusters defaults to sqrt(n) if not specified — a common heuristic
    that scales the budget with document size.
    """
    from sklearn.cluster import KMeans

    n = len(embeddings)

    # Default: sqrt(n) clusters, capped at n (can't have more clusters than chunks)
    if n_clusters is None:
        n_clusters = max(1, min(int(np.sqrt(n)), n))

    # Edge case: if n_clusters >= n, just keep everything
    if n_clusters >= n:
        kept_indices = list(range(n))
        pruned_indices = []
        scores = [0.0] * n
        threshold = 0.0
    else:
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=5, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_  # shape: (n_clusters, d)

        # For each cluster, find the chunk closest to its centroid
        # Score = cosine distance to assigned centroid (lower = better representative)
        scores = []
        for i, emb in enumerate(embeddings):
            centroid = centroids[labels[i]]
            cos_sim = float(
                np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-8)
            )
            scores.append(1.0 - cos_sim)  # convert to distance: lower = closer to centroid

        # Keep the chunk with the lowest distance (best representative) per cluster
        kept_indices = []
        for cluster_id in range(n_clusters):
            members = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            if members:
                best = min(members, key=lambda i: scores[i])
                kept_indices.append(best)

        kept_indices = sorted(kept_indices)
        pruned_indices = [i for i in range(n) if i not in set(kept_indices)]
        threshold = float(np.mean(scores))

    stats = _build_pruning_stats(
        strategy="kmeans",
        n_total=n,
        kept_indices=kept_indices,
        pruned_indices=pruned_indices,
        scores=scores,
        threshold=threshold,
        chunks=chunks,
        extra={
            "n_clusters": n_clusters,
            "score_meaning": "cosine distance to assigned cluster centroid — lower = better representative",
            "selection_rule": "one chunk per cluster — the closest to its centroid is kept",
        },
    )

    return kept_indices, stats


# ADDED: Strategy 4 - Maximal Marginal Relevance (MMR)
def prune_mmr(
    embeddings: np.ndarray,
    chunks: list[dict],
    target_k: int = None,
    lambda_param: float = 0.5,
) -> tuple[list[int], dict]:
    """
    Maximal Marginal Relevance (MMR) representative selection.

    MMR selects vectors iteratively. At each step, the next vector chosen
    must maximise a trade-off between:
      - Relevance: similarity to the document centroid (coverage)
      - Diversity: dissimilarity to already-selected vectors (novelty)

    Score = lambda * sim_to_centroid - (1 - lambda) * max_sim_to_selected

    lambda=1.0 → pure relevance (greedy, similar to cosine pruning)
    lambda=0.0 → pure diversity (maximally spread out)
    lambda=0.5 → balanced coverage + diversity (default)

    Why this fits the project:
      - It explicitly balances what a good representative sketch needs:
        you want vectors that cover the document's content AND don't
        duplicate each other. No other strategy does both simultaneously.
      - target_k is your cost parameter — you directly control how many
        vectors get stored, just like K in K-Means.
      - This is probably the most directly applicable strategy to the
        cost-based representative selection framing in the project brief.
    """
    n = len(embeddings)

    # Default target_k: same heuristic as kmeans
    if target_k is None:
        target_k = max(1, min(int(np.sqrt(n)), n))

    if target_k >= n:
        kept_indices = list(range(n))
        pruned_indices = []
        scores = [1.0] * n
        threshold = 0.0
    else:
        # Normalise embeddings once for efficient cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normed = embeddings / norms

        # Document centroid (relevance anchor)
        centroid = normed.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Relevance scores: cosine similarity of each chunk to centroid
        relevance = normed @ centroid  # shape: (n,)

        selected = []
        remaining = list(range(n))

        # MMR iterative selection
        while len(selected) < target_k and remaining:
            if not selected:
                # First pick: most relevant to centroid
                best = max(remaining, key=lambda i: relevance[i])
            else:
                # Subsequent picks: balance relevance vs similarity to already-selected
                selected_embs = normed[selected]  # shape: (len(selected), d)

                best_score = -np.inf
                best = remaining[0]
                for i in remaining:
                    sim_to_centroid = relevance[i]
                    sim_to_selected = float((normed[i] @ selected_embs.T).max())
                    mmr_score = lambda_param * sim_to_centroid - (1 - lambda_param) * sim_to_selected
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = i

            selected.append(best)
            remaining.remove(best)

        kept_indices = sorted(selected)
        pruned_indices = [i for i in range(n) if i not in set(kept_indices)]

        # Scores: relevance to centroid (for reporting — higher = more representative)
        scores = relevance.tolist()
        threshold = float(np.mean(scores))

    stats = _build_pruning_stats(
        strategy="mmr",
        n_total=n,
        kept_indices=kept_indices,
        pruned_indices=pruned_indices,
        scores=scores,
        threshold=threshold,
        chunks=chunks,
        extra={
            "target_k": target_k,
            "lambda_param": lambda_param,
            "score_meaning": "cosine similarity to document centroid — used as relevance signal in MMR",
            "selection_rule": f"iterative MMR with lambda={lambda_param} — balances coverage and diversity",
        },
    )

    return kept_indices, stats


# ADDED: MaxSim re-ranking for retrieval side

# After pgvector returns the top-N candidates by cosine distance, MaxSim re-ranks them
# by selecting the subset whose collective coverage of the query is maximised.
# This keeps MaxSim's diversity benefit without touching what gets stored in the DB.
def maxsim_rerank(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int = 5,
) -> list[int]:
    """
    MaxSim re-ranking at retrieval time.

    Selects top_k candidates from a pool using the same iterative
    diversity logic that was previously applied at index time, but now
    applied to the small retrieval candidate set (e.g. top-20 from pgvector).

    For each selection step, we pick the candidate that has the highest
    maximum similarity to any query token — i.e. the one most likely to
    contain a direct answer to some part of the query.

    In practice with sentence-level embeddings (not token-level like ColBERT),
    this reduces to: pick candidates that are diverse relative to each other
    while still being relevant to the query.
    """
    n = len(candidate_embeddings)
    if top_k >= n:
        return list(range(n))

    # Normalise
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    c_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)

    # Relevance to query
    relevance = c_norms @ q_norm  # shape: (n,)

    selected = []
    remaining = list(range(n))

    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            sel_embs = c_norms[selected]
            best_score = -np.inf
            best = remaining[0]
            for i in remaining:
                rel = relevance[i]
                redundancy = float((c_norms[i] @ sel_embs.T).max())
                score = rel - 0.5 * redundancy  # fixed lambda=0.5 for retrieval
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(best)
        remaining.remove(best)

    return selected


# Shared stats builder — produces the full pruning report
def _build_pruning_stats(
    strategy: str,
    n_total: int,
    kept_indices: list[int],
    pruned_indices: list[int],
    scores: list[float],
    threshold: float,
    chunks: list[dict],
    extra: dict,
) -> dict:
    """
    Builds the structured pruning report that gets exposed via /pruning-report.

    Every field here is intentionally included so the frontend can visualize:
      - summary counts and rates
      - per-chunk breakdown (score, kept/pruned, page, content preview)
      - threshold used and its derivation
      - strategy-specific metadata
    """
    n_kept = len(kept_indices)
    n_pruned = len(pruned_indices)
    kept_set = set(kept_indices)

    per_chunk = []
    for i, chunk in enumerate(chunks):
        per_chunk.append({
            "index": i,
            "page_number": chunk["page_number"],
            "content_preview": chunk["sentence_chunk"][:120] + ("..." if len(chunk["sentence_chunk"]) > 120 else ""),
            "score": round(scores[i], 6),
            "kept": i in kept_set,
            "pruned": i not in kept_set,
        })

    return {
        "strategy": strategy,
        "summary": {
            "total_chunks": n_total,
            "chunks_kept": n_kept,
            "chunks_pruned": n_pruned,
            "retention_rate_pct": round(100 * n_kept / n_total, 2),
            "pruning_rate_pct": round(100 * n_pruned / n_total, 2),
            "storage_vectors_saved": n_pruned,
            "estimated_storage_saved_pct": round(100 * n_pruned / n_total, 2),
        },
        "threshold": {
            "value": round(threshold, 6),
            "description": "adaptive - derived from mean(scores) × multiplier",
        },
        "score_stats": {
            "min": round(min(scores), 6),
            "max": round(max(scores), 6),
            "mean": round(float(np.mean(scores)), 6),
            "std": round(float(np.std(scores)), 6),
        },
        "per_chunk_detail": per_chunk,
        "strategy_metadata": extra,
    }


# PDF PROCESSING
def process_pdf(file_path: str, filename: str, pruning_strategy: str = "none"):
    global processing_status, full_context_pages, pruning_report
    
    try:
        processing_status = {"status": "processing", "chunks": 0, "error": None, "mode": None}
        full_context_pages = []
        pruning_report = {}  # ADDED: reset report on each new upload

        print(f"[INFO] Opening PDF: {filename}")
        document = fitz.open(file_path)
        total_pages = len(document)
        print(f"[INFO] PDF has {total_pages} pages")

        # Extract all text up front — needed for token counting regardless of mode
        full_text_by_page = []
        for page_num, page in enumerate(document):
            text_blocks = page.get_text("blocks")
            # Join blocks with newlines to keep names and affiliations separate
            text = "\n".join([block[4] for block in text_blocks])
            if text.strip():
                full_text_by_page.append({"page_number": page_num + 1, "text": text})

        document.close()
        os.remove(file_path)

        if not full_text_by_page:
            processing_status.update({"status": "failed", "error": "No text found in PDF"})
            return
        
        # Decide mode based on actual token count, not page count
        full_text = "\n".join(p["text"] for p in full_text_by_page)
        token_count = len(tokenizer.encode(full_text))
        print(f"[INFO] Document token count: {token_count} (limit: {FULL_CONTEXT_TOKEN_LIMIT})")

        if token_count <= FULL_CONTEXT_TOKEN_LIMIT:
            # FULL-CONTEXT MODE
            # Document fits in the LLM's context window.
            # No chunking, no embeddings, no DB — just keep pages in memory.
            print("[INFO] Using full-context mode.")
            full_context_pages = full_text_by_page
            processing_status.update({
                "status": "done",
                "mode": "full_context",
                "chunks": 0,
            })
            print(f"[INFO] Done! {len(full_text_by_page)} pages held in memory.")

        else:
            # RAG MODE
            # Document is too large for the context window.
            # Chunk, embed, and store in pgvector for similarity search at query time.
            print("[INFO] Using RAG mode.")
            all_sentences = []
            for page_data in full_text_by_page:
                sentences = [str(s) for s in nlp(page_data["text"]).sents]
                all_sentences.extend(sentences)

            print(f"[INFO] Total sentences collected: {len(all_sentences)}")

            chunked_texts = chunk_sentences(all_sentences, chunk_size=10, overlap=2)
            raw_pages_and_text = [
                {"page_number": None, "sentence_chunk": chunk}
                for chunk in chunked_texts
            ]


            total_chunks = len(raw_pages_and_text)
            print(f"[INFO] Total chunks created: {total_chunks}")

            if total_chunks == 0:
                processing_status.update({"status": "failed", "error": "No valid chunks after splitting"})
                return

            print(f"[INFO] Generating embeddings for {total_chunks} chunks...")
            text_chunks = [item["sentence_chunk"] for item in raw_pages_and_text]
            new_embeddings = embedding_model.encode(
                text_chunks,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

            # ADDED: Apply pruning strategy before inserting into DB
            # Pruning happens here, after embeddings are computed but before
            # any vectors are written to pgvector. This is intentional:
            # we want to measure and report on the full set, then store only
            # the survivors. raw_pages_and_text and new_embeddings stay intact
            # for reporting; kept_indices filters what actually gets stored.

            kept_indices = list(range(total_chunks))  # default: keep everything

            if pruning_strategy != "none" and total_chunks > 1:
                print(f"[INFO] Applying pruning strategy: {pruning_strategy}")

                # CHANGED: dispatch block updated — maxsim removed, kmeans and mmr added
                if pruning_strategy == "cosine":
                    kept_indices, report = prune_cosine(new_embeddings, raw_pages_and_text)

                elif pruning_strategy == "cosine_whitened":
                    kept_indices, report = prune_cosine_whitened(new_embeddings, raw_pages_and_text)

                elif pruning_strategy == "kmeans":
                    # ADDED: K-Means dispatch
                    kept_indices, report = prune_kmeans(new_embeddings, raw_pages_and_text)

                elif pruning_strategy == "mmr":
                    # ADDED: MMR dispatch
                    kept_indices, report = prune_mmr(new_embeddings, raw_pages_and_text)

                else:
                    report = {"strategy": "none", "summary": {"total_chunks": total_chunks}}

                pruning_report = report  # persist for /pruning-report endpoint

                n_kept = len(kept_indices)
                n_pruned = total_chunks - n_kept
                print(
                    f"[INFO] Pruning complete — kept {n_kept}/{total_chunks} chunks "
                    f"({report['summary']['pruning_rate_pct']}% pruned)"
                )
            else:
                pruning_report = {
                    "strategy": "none",
                    "summary": {
                        "total_chunks": total_chunks,
                        "chunks_kept": total_chunks,
                        "chunks_pruned": 0,
                        "retention_rate_pct": 100.0,
                        "pruning_rate_pct": 0.0,
                        "storage_vectors_saved": 0,
                        "estimated_storage_saved_pct": 0.0,
                    },
                }

            # Insert only the surviving chunks into pgvector
            print("[INFO] Inserting into Postgres...")
            conn = get_db_connection()
            cur = conn.cursor()

            kept_set = set(kept_indices)
            rows = [
                (raw_pages_and_text[i]["page_number"], raw_pages_and_text[i]["sentence_chunk"], new_embeddings[i].tolist())
                for i in range(total_chunks)
                if i in kept_set
            ]

            execute_values(
                cur,
                "INSERT INTO document_chunks (page_number, content, embedding) VALUES %s",
                rows,
            )
            conn.commit()
            cur.close()
            conn.close()

            chunks_stored = len(rows)
            processing_status.update({
                "status": "done",
                "mode": "rag",
                "chunks": chunks_stored,
                # ADDED: expose summary stats directly in /status response ──
                "pruning_strategy": pruning_strategy,
                "pruning_summary": pruning_report.get("summary", {}),
            })
            print(f"[INFO] Done! {chunks_stored} chunks stored (strategy: {pruning_strategy}).")

    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {e}")
        processing_status.update({"status": "failed", "error": str(e)})
        if os.path.exists(file_path):
            os.remove(file_path)


# ENDPOINTS

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # CHANGED: pruning_strategy literal updated — maxsim removed, kmeans and mmr added
    pruning_strategy: PruningStrategy = Query(
        default="none",
        description=(
            "Pruning strategy to apply before storing embeddings. "
            "'none': store all chunks. "
            "'cosine': prune chunks too close to centroid. "
            "'cosine_whitened': cosine pruning on whitened embedding space (most discriminative). "
            "'kmeans': cluster embeddings into K groups, keep one representative per cluster. "
            "'mmr': iterative selection balancing relevance and diversity (Maximal Marginal Relevance)."
        ),
    ),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    if processing_status.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Already processing a file. Poll /status.")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    # ADDED: pass pruning_strategy into the background task
    background_tasks.add_task(process_pdf, temp_path, file.filename, pruning_strategy)
    return {
        "message": "Upload received, processing in background. Poll /status to check.",
        "pruning_strategy": pruning_strategy,
    }


@app.post("/query")
async def query_document(request: QueryRequest):
    mode = processing_status.get("mode")

    if mode is None or processing_status.get("status") != "done":
        raise HTTPException(
            status_code=400,
            detail=f"No document ready. Current status: {processing_status.get('status')}",
        )

    if mode == "full_context":
        # Pages are already in memory — no DB call needed
        if not full_context_pages:
            raise HTTPException(status_code=404, detail="Full-context data missing from memory.")
        
        context_text = "\n\n".join(f"[Page {p['page_number']}]\n{p['text']}" for p in full_context_pages)
        rows = [(p["page_number"], p["text"]) for p in full_context_pages]
    
    else:
        # RAG: embed the query and retrieve the top candidates from pgvector
        query_embedding = embedding_model.encode(request.query, convert_to_numpy=True)
        conn = get_db_connection()
        cur = conn.cursor()

        # CHANGED: fetch a larger candidate pool when use_maxsim=True so re-ranking has room to work
        # With maxsim=False we fetch 5 directly. With maxsim=True we fetch 20 then re-rank to 5.
        fetch_limit = 20 if request.use_maxsim else 5
        cur.execute("""
            SELECT page_number, content, embedding
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (query_embedding, fetch_limit))
        raw_rows = cur.fetchall()
        cur.close()
        conn.close()

        if not raw_rows:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        # ADDED: MaxSim re-ranking at retrieval time (only when use_maxsim=True)
        # This is where MaxSim belongs — it re-ranks the candidate pool to maximise
        # query coverage and diversity, without affecting what's stored in the index.
        if request.use_maxsim and len(raw_rows) > 5:
            candidate_embeddings = np.array([np.array(r[2]) for r in raw_rows])
            top_indices = maxsim_rerank(query_embedding, candidate_embeddings, top_k=5)
            rows = [(raw_rows[i][0], raw_rows[i][1]) for i in top_indices]
        else:
            rows = [(r[0], r[1]) for r in raw_rows[:5]]

        context_text = ""
        for i, row in enumerate(rows):
            context_text += f"SOURCE {i + 1} (Page {row[0]}):\n{row[1]}\n\n"

    base_prompt = f"""Think of yourself as an assistant that has read the document and is now answering questions about it.
Using your knowledge and the context, answer the question as best you can. If you don't know the answer, say you don't know.
Return only the answer, not the thought process.

Context:
{context_text}

User query: {request.query}
Answer:"""

    dialogue_template = [{"role": "user", "content": base_prompt}]
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("[INFO] Generating answer...")
    input_ids = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(
        **input_ids,
        temperature=request.temperature,
        do_sample=True,
        max_new_tokens=request.max_new_tokens,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id, # stop as soon as answer is complete
        pad_token_id=tokenizer.eos_token_id, # prevents padding warning that slows things
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_answer = output_text.split("model")[-1].strip() if "model" in output_text else output_text.strip()

    return {
        "query": request.query,
        "answer": clean_answer,
        "mode": mode,
        # CHANGED: added maxsim_applied flag to response so frontend can show which retrieval method was used
        "maxsim_applied": request.use_maxsim and mode == "rag",
        "sources": [{"page": row[0], "text": row[1][:100] + "..."} for row in rows],
    }


@app.get("/status")
async def get_status():
    return processing_status


# /pruning-report endpoint
# Returns the full pruning report from the most recent upload.
# Includes per-chunk scores, kept/pruned flags, thresholds, and summary stats.
# This is the primary endpoint the frontend should poll to visualize results.
@app.get("/pruning-report")
async def get_pruning_report():
    if not pruning_report:
        raise HTTPException(
            status_code=404,
            detail="No pruning report available. Upload a document with a pruning strategy first.",
        )
    return pruning_report


@app.get("/chunks")
async def get_stored_chunks(limit: int = Query(default=20, ge=1, le=200)):
    """
    Returns the chunks currently stored in pgvector, including their full
    embedding vectors. Used by the dev-mode frontend to render real vector
    heat-maps and confirm what actually survived pruning.
 
    Query params:
      limit (int, 1-200, default 20) — max chunks to return.
 
    Returns:
      mode        : "rag" | "full_context" — current processing mode
      total       : total rows currently in document_chunks
      chunks      : list of {id, page_number, content, embedding}
    """
    mode = processing_status.get("mode")
 
    if mode == "full_context":
        # No vectors stored — return pages as pseudo-chunks without embeddings
        return {
            "mode": "full_context",
            "total": len(full_context_pages),
            "chunks": [
                {
                    "id": i + 1,
                    "page_number": p["page_number"],
                    "content": p["text"][:500],
                    "embedding": [],   # no embedding in full-context mode
                }
                for i, p in enumerate(full_context_pages[:limit])
            ],
        }
 
    # RAG mode — fetch from pgvector
    conn = get_db_connection()
    cur = conn.cursor()
 
    # Total count
    cur.execute("SELECT COUNT(*) FROM document_chunks;")
    total = cur.fetchone()[0]
 
    # Fetch rows including the raw embedding vector
    cur.execute(
        "SELECT id, page_number, content, embedding FROM document_chunks ORDER BY id LIMIT %s;",
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
 
    chunks = [
        {
            "id": int(row[0]),
            "page_number": int(row[1]) if row[1] is not None else None,
            "content": row[2],
            # pgvector returns a list of numpy.float32 — FastAPI can't serialize them.
            # Explicitly cast each element to Python float.
            "embedding": [float(v) for v in row[3]],
        }
        for row in rows
    ]
 
    return {
        "mode": "rag",
        "total": total,
        "chunks": chunks,
    }


@app.get("/reset")
async def reset_data():
    global processing_status, full_context_pages, pruning_report

    # Clear pgvector chunks (RAG mode)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks;")
    conn.commit()
    cur.close()
    conn.close()

    # Clear in-memory pages (full-context mode)
    full_context_pages = []
    pruning_report = {}  # clear report on reset
    processing_status = {"status": "idle", "chunks": 0, "error": None, "mode": None}
    return {"message": "Data reset successfully."}