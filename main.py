import os
import re
import fitz  # PyMuPDF
import numpy as np
import psycopg2
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from contextlib import asynccontextmanager
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from typing import Literal, Optional
from fastapi.middleware.cors import CORSMiddleware
import httpx

origins = [
    "https://adaptive-rag.vercel.app/",
    "*",
    "http://localhost:3000",
]

load_dotenv()

# CONFIGURATION
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
LLM_API_KEY = os.getenv("LLM_API_KEY")
DATA_BASE_URL = os.getenv("DATA_BASE_URL")

FULL_CONTEXT_TOKEN_LIMIT = 6000

# Encode batch size: small enough to keep RAM flat on 1GB machines.
# Each batch of 16 chunks at 384-dim float32 ≈ 24KB — negligible.
# Increasing this does not meaningfully speed up encoding; the bottleneck
# is the transformer forward pass, not data loading.
ENCODE_BATCH_SIZE = 16

# DB write batch: number of rows passed to execute_values per commit.
# Keeps transaction size small and avoids building a giant in-memory list.
DB_WRITE_BATCH = 100

# Embedding dimensionality — must match the model loaded below.
EMBEDDING_DIM = 384

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("HOST"),
    "port": os.getenv("PORT"),
}

PruningStrategy = Literal["none", "cosine", "cosine_whitened", "kmeans", "mmr"]


# DATABASE
def get_db_connection():
    conn = psycopg2.connect(DATA_BASE_URL, sslmode="require", connect_timeout=5)
    register_vector(conn)
    return conn

def init_db():
    conn = psycopg2.connect(DATA_BASE_URL, sslmode="require", connect_timeout=5)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_status (
            id SERIAL PRIMARY KEY,
            key TEXT UNIQUE,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        INSERT INTO app_status (key, value)
        VALUES ('processing_status', '{"status": "idle", "chunks": 0, "error": None, "mode": None}')
        ON CONFLICT (key) DO NOTHING;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id           SERIAL PRIMARY KEY,
            page_number  INTEGER,
            content      TEXT,
            embedding    vector(384)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[INFO] Database initialized, status tracking ready.")


# GLOBAL STATE
processing_status = {"status": "idle", "chunks": 0, "error": None, "mode": None}
full_context_pages: list[dict] = []
pruning_report: dict = {}

llm_model = None
tokenizer = None
DEVICE = "cpu"


# LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Init DB...")
    init_db()

    global embedding_model, nlp

    print("[INFO] Loading spaCy...")
    nlp = English()
    nlp.add_pipe("sentencizer")

    print("[INFO] Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    print("[INFO] Startup complete.")
    yield
    print("[INFO] Shutting down...")


app = FastAPI(lifespan=lifespan, title="Local RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REQUEST SCHEMAS
class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.7
    max_new_tokens: int = 256
    use_maxsim: bool = False


# HELPER FUNCTIONS
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()


def split_list(input_list: list[str], slice_size: int = 10) -> list[list[str]]:
    return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]


def iter_chunks(full_text_by_page: list[dict], nlp, slice_size: int = 10):
    """
    Generator that yields sentence chunks one at a time.
    Avoids materialising the full chunk list when pruning is not needed,
    keeping peak RAM proportional to one spaCy batch rather than the whole doc.
    When pruning IS needed, callers materialise this into a list — that's fine
    because text is orders of magnitude smaller than float embeddings.
    """
    count =0 
    page_texts = [p["text"] for p in full_text_by_page]
    for page_data, doc in zip(full_text_by_page, nlp.pipe(page_texts, batch_size=50)):
        sentences = [str(s) for s in doc.sents]
        for chunk in split_list(sentences, slice_size):
            joined = "".join(chunk).replace("  ", " ").strip()
            joined = re.sub(r"\.([A-Z])", r". \1", joined)
            if joined:
                count += 1

                if count % 50 == 0:
                    print(f"[CHUNK GEN] Yielded {count} chunks...")
                yield {"page_number": page_data["page_number"], "sentence_chunk": joined}


# PRUNING STRATEGY IMPLEMENTATIONS

def compute_whitening_matrix(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes a whitening matrix W from the embedding matrix using PCA.
    Returns W and the mean vector so callers can apply: e_w = (e - mean) @ W.T
    """
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, a_min=1e-8, a_max=None)
    W = (eigenvectors / np.sqrt(eigenvalues)).T
    return W, mean


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T


def prune_cosine(
    embeddings: np.ndarray,
    chunks: list[dict],
    threshold_multiplier: float = 0.85,
) -> tuple[list[int], dict]:
    n = len(embeddings)
    centroid = embeddings.mean(axis=0, keepdims=True)
    scores = cosine_similarity_matrix(embeddings, centroid).flatten()
    threshold = float(scores.mean() * threshold_multiplier)
    kept_indices = [i for i, s in enumerate(scores) if s <= threshold]
    if not kept_indices:
        kept_indices = [int(np.argmin(scores))]
    pruned_indices = [i for i in range(n) if i not in set(kept_indices)]
    stats = _build_pruning_stats(
        strategy="cosine", n_total=n, kept_indices=kept_indices,
        pruned_indices=pruned_indices, scores=scores.tolist(),
        threshold=threshold, chunks=chunks,
        extra={"threshold_multiplier": threshold_multiplier,
               "score_meaning": "cosine similarity vs centroid — higher = more redundant"},
    )
    return kept_indices, stats


def prune_cosine_whitened(
    embeddings: np.ndarray,
    chunks: list[dict],
    threshold_multiplier: float = 0.85,
) -> tuple[list[int], dict]:
    n = len(embeddings)
    W, mean = compute_whitening_matrix(embeddings)
    whitened = (embeddings - mean) @ W.T
    centroid = whitened.mean(axis=0, keepdims=True)
    scores = cosine_similarity_matrix(whitened, centroid).flatten()
    threshold = float(scores.mean() * threshold_multiplier)
    kept_indices = [i for i, s in enumerate(scores) if s <= threshold]
    if not kept_indices:
        kept_indices = [int(np.argmin(scores))]
    pruned_indices = [i for i in range(n) if i not in set(kept_indices)]
    stats = _build_pruning_stats(
        strategy="cosine_whitened", n_total=n, kept_indices=kept_indices,
        pruned_indices=pruned_indices, scores=scores.tolist(),
        threshold=threshold, chunks=chunks,
        extra={"threshold_multiplier": threshold_multiplier,
               "score_meaning": "cosine similarity vs centroid in whitened space",
               "whitening_applied": True},
    )
    return kept_indices, stats


def prune_kmeans(
    embeddings: np.ndarray,
    chunks: list[dict],
    n_clusters: int = None,
) -> tuple[list[int], dict]:
    from sklearn.cluster import KMeans
    n = len(embeddings)
    if n_clusters is None:
        n_clusters = max(1, min(int(np.sqrt(n)), n))
    if n_clusters >= n:
        kept_indices = list(range(n))
        pruned_indices = []
        scores = [0.0] * n
        threshold = 0.0
    else:
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=5, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        scores = []
        for i, emb in enumerate(embeddings):
            centroid = centroids[labels[i]]
            cos_sim = float(
                np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-8)
            )
            scores.append(1.0 - cos_sim)
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
        strategy="kmeans", n_total=n, kept_indices=kept_indices,
        pruned_indices=pruned_indices, scores=scores, threshold=threshold,
        chunks=chunks,
        extra={"n_clusters": n_clusters,
               "score_meaning": "cosine distance to assigned cluster centroid",
               "selection_rule": "one chunk per cluster — closest to centroid kept"},
    )
    return kept_indices, stats


def prune_mmr(
    embeddings: np.ndarray,
    chunks: list[dict],
    target_k: int = None,
    lambda_param: float = 0.5,
) -> tuple[list[int], dict]:
    n = len(embeddings)
    if target_k is None:
        target_k = max(1, min(int(np.sqrt(n)), n))
    if target_k >= n:
        kept_indices = list(range(n))
        pruned_indices = []
        scores = [1.0] * n
        threshold = 0.0
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normed = embeddings / norms
        centroid = normed.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        relevance = normed @ centroid
        selected = []
        remaining = list(range(n))
        while len(selected) < target_k and remaining:
            if not selected:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                selected_embs = normed[selected]
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
        scores = relevance.tolist()
        threshold = float(np.mean(scores))
    stats = _build_pruning_stats(
        strategy="mmr", n_total=n, kept_indices=kept_indices,
        pruned_indices=pruned_indices, scores=scores, threshold=threshold,
        chunks=chunks,
        extra={"target_k": target_k, "lambda_param": lambda_param,
               "score_meaning": "cosine similarity to document centroid",
               "selection_rule": f"iterative MMR with lambda={lambda_param}"},
    )
    return kept_indices, stats


def maxsim_rerank(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int = 5,
) -> list[int]:
    n = len(candidate_embeddings)
    if top_k >= n:
        return list(range(n))
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    c_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
    relevance = c_norms @ q_norm
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
                score = rel - 0.5 * redundancy
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(best)
        remaining.remove(best)
    return selected


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
def _insert_rows_batched(cur, rows: list[tuple], batch_size: int = DB_WRITE_BATCH):
    """
    Writes rows to document_chunks in fixed-size batches so the caller's
    connection never holds a transaction larger than `batch_size` rows.
    Each batch is committed immediately and the list is discarded.
    """
    for i in range(0, len(rows), batch_size):
        execute_values(
            cur,
            "INSERT INTO document_chunks (page_number, content, embedding) VALUES %s",
            rows[i:i + batch_size],
        )
        cur.connection.commit()


def process_pdf(file_path: str, filename: str, pruning_strategy: str = "none"):
    global processing_status, full_context_pages, pruning_report

    try:
        processing_status = {"status": "processing", "chunks": 0, "error": None, "mode": None}
        full_context_pages = []
        pruning_report = {}

        print(f"[INFO] Opening PDF: {filename}")
        document = fitz.open(file_path)
        total_pages = len(document)
        print(f"[INFO] PDF has {total_pages} pages")

        full_text_by_page = []
        for page_num, page in enumerate(document):
            text_blocks = page.get_text("blocks")
            text = "\n".join([block[4] for block in text_blocks])
            if text.strip():
                full_text_by_page.append({"page_number": page_num + 1, "text": text})

        document.close()
        os.remove(file_path)

        if not full_text_by_page:
            processing_status.update({"status": "failed", "error": "No text found in PDF"})
            return

        full_text = "\n".join(p["text"] for p in full_text_by_page)
        token_count = len(full_text) // 4
        print(f"[INFO] Approx token count: {token_count} (limit: {FULL_CONTEXT_TOKEN_LIMIT})")

        if token_count <= FULL_CONTEXT_TOKEN_LIMIT:
            # FULL-CONTEXT MODE — no embeddings needed
            print("[INFO] Using full-context mode.")
            full_context_pages = full_text_by_page
            processing_status.update({"status": "done", "mode": "full_context", "chunks": 0})
            print(f"[INFO] Done! {len(full_text_by_page)} pages held in memory.")

        else:
            # RAG MODE
            print("[INFO] Using RAG mode.")

            if pruning_strategy == "none":
                # ----------------------------------------------------------------
                # STREAMING PATH (no pruning)
                # Encode and write one small batch at a time.
                # Peak RAM = one batch of embeddings (~24KB for ENCODE_BATCH_SIZE=16)
                # rather than the entire document's worth.
                # ----------------------------------------------------------------
                print("[INFO] Streaming encode + insert (no pruning).")
                conn = get_db_connection()
                cur = conn.cursor()
                chunks_stored = 0
                batch_chunks: list[dict] = []
                for chunk in iter_chunks(full_text_by_page, nlp):
                    batch_chunks.append(chunk)
                    if len(batch_chunks) >= ENCODE_BATCH_SIZE:
                        texts = [c["sentence_chunk"] for c in batch_chunks]
                        embs = embedding_model.encode(
                            texts, batch_size=ENCODE_BATCH_SIZE,
                            convert_to_numpy=True, show_progress_bar=True,
                        )
                        rows = [
                            (c["page_number"], c["sentence_chunk"], embs[j].tolist())
                            for j, c in enumerate(batch_chunks)
                        ]
                        _insert_rows_batched(cur, rows)
                        chunks_stored += len(rows)
                        del embs, rows
                        batch_chunks = []

                # Flush remaining chunks that didn't fill a full batch
                if batch_chunks:
                    texts = [c["sentence_chunk"] for c in batch_chunks]
                    embs = embedding_model.encode(
                        texts, batch_size=ENCODE_BATCH_SIZE,
                        convert_to_numpy=True, show_progress_bar=False,
                    )
                    rows = [
                        (c["page_number"], c["sentence_chunk"], embs[j].tolist())
                        for j, c in enumerate(batch_chunks)
                    ]
                    _insert_rows_batched(cur, rows)
                    chunks_stored += len(rows)
                    del embs, rows

                cur.close()
                conn.close()

                pruning_report = {
                    "strategy": "none",
                    "summary": {
                        "total_chunks": chunks_stored,
                        "chunks_kept": chunks_stored,
                        "chunks_pruned": 0,
                        "retention_rate_pct": 100.0,
                        "pruning_rate_pct": 0.0,
                        "storage_vectors_saved": 0,
                        "estimated_storage_saved_pct": 0.0,
                    },
                }
                processing_status.update({
                    "status": "done", "mode": "rag", "chunks": chunks_stored,
                    "pruning_strategy": "none",
                    "pruning_summary": pruning_report["summary"],
                })
                print(f"[INFO] Done! {chunks_stored} chunks stored (streaming, no pruning).")

            else:
                # ----------------------------------------------------------------
                # PRUNING PATH
                # Pruning strategies (cosine, kmeans, mmr) need the full embedding
                # matrix to make decisions. We avoid keeping it in RAM by writing
                # it to a memory-mapped temp file on disk. numpy.memmap lets the
                # pruning functions address the array normally while the OS pages
                # in only the rows that are actually touched.
                #
                # Pass 1: materialise chunks as text (small), encode to mmap file.
                # Pass 2: run pruning on mmap array → kept_indices list.
                # Pass 3: re-read survivors from mmap, write to DB in batches.
                # Cleanup: delete temp file.
                # ----------------------------------------------------------------
                print(f"[INFO] Pruning path — materialising chunks for strategy: {pruning_strategy}")

                # Materialise chunk metadata (text only — much smaller than floats)
                raw_pages_and_text = list(iter_chunks(full_text_by_page, nlp))
                total_chunks = len(raw_pages_and_text)
                print(f"[INFO] Total chunks: {total_chunks}")

                if total_chunks == 0:
                    processing_status.update({"status": "failed", "error": "No valid chunks after splitting"})
                    return

                # --- Pass 1: encode to memmap ---
                mmap_fd, mmap_path = tempfile.mkstemp(suffix=".npy")
                os.close(mmap_fd)  # numpy opens it itself

                try:
                    mmap_emb = np.memmap(
                        mmap_path, dtype="float32", mode="w+",
                        shape=(total_chunks, EMBEDDING_DIM),
                    )
                    print(f"[INFO] Encoding {total_chunks} chunks to memmap ({mmap_path})...")
                    for i in range(0, total_chunks, ENCODE_BATCH_SIZE):
                        batch_texts = [
                            raw_pages_and_text[j]["sentence_chunk"]
                            for j in range(i, min(i + ENCODE_BATCH_SIZE, total_chunks))
                        ]
                        batch_embs = embedding_model.encode(
                            batch_texts, batch_size=ENCODE_BATCH_SIZE,
                            convert_to_numpy=True, show_progress_bar=False,
                        )
                        mmap_emb[i:i + len(batch_texts)] = batch_embs
                        del batch_embs
                    mmap_emb.flush()
                    print("[INFO] Encoding complete, flushed to disk.")

                    # Re-open as read-only for pruning
                    mmap_ro = np.memmap(
                        mmap_path, dtype="float32", mode="r",
                        shape=(total_chunks, EMBEDDING_DIM),
                    )

                    # --- Pass 2: prune ---
                    print(f"[INFO] Applying pruning strategy: {pruning_strategy}")
                    if pruning_strategy == "cosine":
                        kept_indices, report = prune_cosine(mmap_ro, raw_pages_and_text)
                    elif pruning_strategy == "cosine_whitened":
                        kept_indices, report = prune_cosine_whitened(mmap_ro, raw_pages_and_text)
                    elif pruning_strategy == "kmeans":
                        kept_indices, report = prune_kmeans(mmap_ro, raw_pages_and_text)
                    elif pruning_strategy == "mmr":
                        kept_indices, report = prune_mmr(mmap_ro, raw_pages_and_text)
                    else:
                        kept_indices = list(range(total_chunks))
                        report = {"strategy": "none", "summary": {"total_chunks": total_chunks}}

                    pruning_report = report
                    n_kept = len(kept_indices)
                    n_pruned = total_chunks - n_kept
                    print(
                        f"[INFO] Pruning complete — kept {n_kept}/{total_chunks} chunks "
                        f"({report['summary'].get('pruning_rate_pct', 0)}% pruned)"
                    )

                    # --- Pass 3: insert survivors in batches ---
                    print("[INFO] Inserting survivors into Postgres...")
                    conn = get_db_connection()
                    cur = conn.cursor()
                    kept_set = set(kept_indices)

                    batch_rows: list[tuple] = []
                    chunks_stored = 0
                    for i in range(total_chunks):
                        if i not in kept_set:
                            continue
                        batch_rows.append((
                            raw_pages_and_text[i]["page_number"],
                            raw_pages_and_text[i]["sentence_chunk"],
                            mmap_ro[i].tolist(),  # pages in from disk only when accessed
                        ))
                        if len(batch_rows) >= DB_WRITE_BATCH:
                            _insert_rows_batched(cur, batch_rows)
                            chunks_stored += len(batch_rows)
                            batch_rows = []

                    # Flush remaining
                    if batch_rows:
                        _insert_rows_batched(cur, batch_rows)
                        chunks_stored += len(batch_rows)

                    cur.close()
                    conn.close()

                finally:
                    # Always clean up the temp file, even if something raised
                    try:
                        os.unlink(mmap_path)
                    except OSError:
                        pass

                processing_status.update({
                    "status": "done", "mode": "rag", "chunks": chunks_stored,
                    "pruning_strategy": pruning_strategy,
                    "pruning_summary": pruning_report.get("summary", {}),
                })
                print(f"[INFO] Done! {chunks_stored} chunks stored (strategy: {pruning_strategy}).")

    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {e}")
        processing_status.update({"status": "failed", "error": str(e)})
        if os.path.exists(file_path):
            os.remove(file_path)


# LLM CALL
async def call_openrouter(prompt: str, temperature: float, max_new_tokens: int) -> str:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                },
                timeout=60.0,
            )
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown LLM Error")
                print(f"[ERROR] LLM API Failed: {error_msg}")
                return f"LLM Error: {error_msg}"
            data = response.json()
            print(data)
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[CRITICAL] LLM Call Crashed: {e}")
            return "The AI is currently unavailable."


# ENDPOINTS

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pruning_strategy: PruningStrategy = Query(default="none"),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if processing_status.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Already processing a file. Poll /status.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
        )

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)
    del contents

    background_tasks.add_task(process_pdf, temp_path, file.filename, pruning_strategy)
    return {
        "message": "Upload received, processing in background. Poll /status to check.",
        "pruning_strategy": pruning_strategy,
    }
def has_stored_data() -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT EXISTS (SELECT 1 FROM document_chunks LIMIT 1);")
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        print(f"[ERROR] Database check failed: {e}")
        return False


@app.post("/query")
async def query_document(request: QueryRequest):
    has_rag_data = has_stored_data()
    has_mem_data = len(full_context_pages) > 0

    if not has_rag_data and not has_mem_data:
        raise HTTPException(
            status_code=400,
            detail="No document data found. Please upload a PDF first."
        )

    mode = "full_context" if has_mem_data else "rag"

    if mode == "full_context":
        if not full_context_pages:
            raise HTTPException(status_code=404, detail="Full-context data missing from memory.")
        context_text = "\n\n".join(f"[Page {p['page_number']}]\n{p['text']}" for p in full_context_pages)
        rows = [(p["page_number"], p["text"]) for p in full_context_pages]

    else:
        query_embedding = embedding_model.encode(request.query, convert_to_numpy=True)
        conn = get_db_connection()
        cur = conn.cursor()
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

        if request.use_maxsim and len(raw_rows) > 5:
            candidate_embeddings = np.array([np.array(r[2]) for r in raw_rows])
            top_indices = maxsim_rerank(query_embedding, candidate_embeddings, top_k=5)
            rows = [(raw_rows[i][0], raw_rows[i][1]) for i in top_indices]
        else:
            rows = [(r[0], r[1]) for r in raw_rows[:5]]

        context_text = ""
        for i, row in enumerate(rows):
            context_text += f"SOURCE {i + 1} (Page {row[0]}):\n{row[1]}\n\n"

    base_prompt = f"""You are an assistant that has read a document and answers questions about it.
Using the context below, answer the question. If you don't have the context and don't know the answer, say so.
Return only the answer.

Context:
{context_text}

User query: {request.query}
Answer:"""

    clean_answer = await call_openrouter(base_prompt, request.temperature, request.max_new_tokens)

    return {
        "query": request.query,
        "answer": clean_answer,
        "mode": mode,
        "maxsim_applied": request.use_maxsim and mode == "rag",
        "sources": [{"page": row[0], "text": row[1][:100] + "..."} for row in rows],
    }


@app.get("/status")
async def get_status():
    if processing_status.get("status") == "processing":
        return processing_status
    if has_stored_data():
        return {"status": "done", "mode": "rag", "message": "Existing data found in database. Ready for queries."}
    if len(full_context_pages) > 0:
        return {"status": "done", "mode": "full_context"}
    return {"status": "idle", "message": "No data found."}


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
    mode = processing_status.get("mode")

    if mode == "full_context":
        return {
            "mode": "full_context",
            "total": len(full_context_pages),
            "chunks": [
                {
                    "id": i + 1,
                    "page_number": p["page_number"],
                    "content": p["text"][:500],
                    "embedding": [],
                }
                for i, p in enumerate(full_context_pages[:limit])
            ],
        }

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM document_chunks;")
    total = cur.fetchone()[0]
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
            "page_number": int(row[1]),
            "content": row[2],
            "embedding": [float(v) for v in row[3]],
        }
        for row in rows
    ]
    return {"mode": "rag", "total": total, "chunks": chunks}


@app.get("/reset")
async def reset_data():
    global processing_status, full_context_pages, pruning_report

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks;")
    conn.commit()
    cur.close()
    conn.close()

    full_context_pages = []
    pruning_report = {}
    processing_status = {"status": "idle", "chunks": 0, "error": None, "mode": None}
    return {"message": "Data reset successfully."}