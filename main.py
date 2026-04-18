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
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
from prune import prune_cosine, prune_cosine_whitened, prune_kmeans, prune_mmr, compute_whitening_matrix, _build_pruning_stats, maxsim_rerank

origins = [
    "https://adaptive-rag.vercel.app/",
    "*",
    "http://localhost:3000",
]

load_dotenv()

# CONFIGURATION
LLM_API_KEY = os.getenv("LLM_API_KEY")
DATA_BASE_URL = os.getenv("DATA_BASE_URL")
# EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL")
EMBEDDING_SERVICE_URL = "http://localhost:7000"

FULL_CONTEXT_TOKEN_LIMIT = 6000
MAX_TOKEN_COUNT = 80000

# With embedding offloaded, encode batch size only affects
# how many texts we send per HTTP request to the embedding service.
# 64 is a good balance — not too large to timeout, not too small to be chatty.
ENCODE_BATCH_SIZE = 64
DB_WRITE_BATCH = 100
EMBEDDING_DIM = 384

MAX_FILE_SIZE_MB = 4
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_CHUNKS = 300

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
        VALUES ('processing_status', '{"status": "idle", "chunks": 0, "error": null, "mode": null}')
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
    print("[INFO] Database initialized.")


# GLOBAL STATE
processing_status = {"status": "idle", "chunks": 0, "error": None, "mode": None}
full_context_pages: list[dict] = []
pruning_report: dict = {}



# LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Init DB...")
    init_db()

    global nlp
    print("[INFO] Loading spaCy...")
    nlp = English()
    nlp.add_pipe("sentencizer")

    print("[INFO] Startup complete.")
    yield
    print("[INFO] Shutting down...")


app = FastAPI(lifespan=lifespan, title="RAG API")

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


# EMBEDDING CLIENT
# Calls the dedicated embedding service instead of running a local model.
# Retries on transient failures with a short backoff.
def embed_texts(texts: list[str]) -> np.ndarray:
    print(f"[INFO] Requesting embeddings for {len(texts)} texts from embedding service...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[INFO] Embedding service request (attempt {attempt + 1}/{max_retries})...")
            with httpx.Client() as client:
                response = client.post(
                    f"{EMBEDDING_SERVICE_URL}/embed",
                    json={"texts": texts},
                    timeout=60.0,
                )
                response.raise_for_status()
                return np.array(response.json()["embeddings"])
        except httpx.HTTPError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Embedding service failed after {max_retries} attempts: {e}")
            print(f"[WARN] Embedding service error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s


# HELPER FUNCTIONS
def split_list(input_list: list[str], slice_size: int = 10) -> list[list[str]]:
    return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]


def iter_chunks(full_text_by_page: list[dict], nlp, slice_size: int = 10):
    count = 0
    page_texts = [p["text"] for p in full_text_by_page]
    for page_data, doc in zip(full_text_by_page, nlp.pipe(page_texts, batch_size=16)):
        sentences = [str(s) for s in doc.sents]
        for chunk in split_list(sentences, slice_size):
            joined = "".join(chunk).replace("  ", " ").strip()
            joined = re.sub(r"\.([A-Z])", r". \1", joined)
            if joined:
                count += 1
                if count % 50 == 0:
                    print(f"[CHUNK GEN] Yielded {count} chunks...")
                yield {"page_number": page_data["page_number"], "sentence_chunk": joined}

# PDF PROCESSING
def _insert_rows_batched(cur, rows: list[tuple], batch_size: int = DB_WRITE_BATCH):
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

        # Clear existing data
        print("[INFO] Clearing existing document data...")
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM document_chunks;")
        conn.commit()
        cur.close()
        conn.close()
        print("[INFO] Database cleared.")

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
            print("[INFO] Using full-context mode.")
            full_context_pages = full_text_by_page
            processing_status.update({"status": "done", "mode": "full_context", "chunks": 0})
            print(f"[INFO] Done! {len(full_text_by_page)} pages held in memory.")
            return

        # RAG MODE
        print("[INFO] Using RAG mode.")

        if pruning_strategy == "none":
            print("[INFO] Streaming encode + insert (no pruning).")
            conn = get_db_connection()
            cur = conn.cursor()
            chunks_stored = 0
            batch_chunks: list[dict] = []

            for chunk in iter_chunks(full_text_by_page, nlp):
                batch_chunks.append(chunk)
                if len(batch_chunks) >= ENCODE_BATCH_SIZE:
                    texts = [c["sentence_chunk"] for c in batch_chunks]
                    embs = embed_texts(texts)
                    rows = [(c["page_number"], c["sentence_chunk"], embs[j].tolist())
                            for j, c in enumerate(batch_chunks)]
                    _insert_rows_batched(cur, rows)
                    chunks_stored += len(rows)
                    del embs, rows
                    batch_chunks = []

            if batch_chunks:
                texts = [c["sentence_chunk"] for c in batch_chunks]
                embs = embed_texts(texts)
                rows = [(c["page_number"], c["sentence_chunk"], embs[j].tolist())
                        for j, c in enumerate(batch_chunks)]
                _insert_rows_batched(cur, rows)
                chunks_stored += len(rows)
                del embs, rows

            cur.close()
            conn.close()

            pruning_report = {
                "strategy": "none",
                "summary": {
                    "total_chunks": chunks_stored, "chunks_kept": chunks_stored,
                    "chunks_pruned": 0, "retention_rate_pct": 100.0,
                    "pruning_rate_pct": 0.0, "storage_vectors_saved": 0,
                    "estimated_storage_saved_pct": 0.0,
                },
            }
            processing_status.update({
                "status": "done", "mode": "rag", "chunks": chunks_stored,
                "pruning_strategy": "none", "pruning_summary": pruning_report["summary"],
            })
            print(f"[INFO] Done! {chunks_stored} chunks stored (streaming).")

        else:
            print(f"[INFO] Pruning path — strategy: {pruning_strategy}")
            raw_pages_and_text = list(iter_chunks(full_text_by_page, nlp))
            total_chunks = len(raw_pages_and_text)
            print(f"[INFO] Total chunks: {total_chunks}")

            if total_chunks == 0:
                processing_status.update({"status": "failed", "error": "No valid chunks after splitting"})
                return

            # Fall back to streaming if chunk count is too high for pruning
            if total_chunks > MAX_CHUNKS:
                print(f"[WARN] {total_chunks} chunks exceeds MAX_CHUNKS={MAX_CHUNKS}, falling back to streaming.")
                conn = get_db_connection()
                cur = conn.cursor()
                chunks_stored = 0
                batch_chunks = []
                for chunk in raw_pages_and_text:
                    batch_chunks.append(chunk)
                    if len(batch_chunks) >= ENCODE_BATCH_SIZE:
                        texts = [c["sentence_chunk"] for c in batch_chunks]
                        embs = embed_texts(texts)
                        rows = [(c["page_number"], c["sentence_chunk"], embs[j].tolist())
                                for j, c in enumerate(batch_chunks)]
                        _insert_rows_batched(cur, rows)
                        chunks_stored += len(rows)
                        del embs, rows
                        batch_chunks = []
                if batch_chunks:
                    texts = [c["sentence_chunk"] for c in batch_chunks]
                    embs = embed_texts(texts)
                    rows = [(c["page_number"], c["sentence_chunk"], embs[j].tolist())
                            for j, c in enumerate(batch_chunks)]
                    _insert_rows_batched(cur, rows)
                    chunks_stored += len(rows)
                cur.close()
                conn.close()
                pruning_report = {
                    "strategy": "none (fallback — chunk limit exceeded)",
                    "summary": {"total_chunks": chunks_stored, "chunks_kept": chunks_stored,
                                "chunks_pruned": 0, "retention_rate_pct": 100.0,
                                "pruning_rate_pct": 0.0, "storage_vectors_saved": 0,
                                "estimated_storage_saved_pct": 0.0},
                }
                processing_status.update({
                    "status": "done", "mode": "rag", "chunks": chunks_stored,
                    "pruning_strategy": "none", "pruning_summary": pruning_report["summary"],
                })
                print(f"[INFO] Done! {chunks_stored} chunks stored (fallback streaming).")
                return

            # Encode all chunks to memmap via embedding service
            mmap_fd, mmap_path = tempfile.mkstemp(suffix=".npy")
            os.close(mmap_fd)

            try:
                mmap_emb = np.memmap(mmap_path, dtype="float32", mode="w+",
                                     shape=(total_chunks, EMBEDDING_DIM))
                print(f"[INFO] Encoding {total_chunks} chunks via embedding service...")
                for i in range(0, total_chunks, ENCODE_BATCH_SIZE):
                    batch_texts = [raw_pages_and_text[j]["sentence_chunk"]
                                   for j in range(i, min(i + ENCODE_BATCH_SIZE, total_chunks))]
                    batch_embs = embed_texts(batch_texts)
                    mmap_emb[i:i + len(batch_texts)] = batch_embs
                    del batch_embs
                mmap_emb.flush()
                print("[INFO] Encoding complete.")

                mmap_ro = np.memmap(mmap_path, dtype="float32", mode="r",
                                    shape=(total_chunks, EMBEDDING_DIM))

                print(f"[INFO] Applying pruning: {pruning_strategy}")
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
                print(f"[INFO] Pruning done — kept {len(kept_indices)}/{total_chunks} chunks")

                conn = get_db_connection()
                cur = conn.cursor()
                kept_set = set(kept_indices)
                batch_rows, chunks_stored = [], 0

                for i in range(total_chunks):
                    if i not in kept_set:
                        continue
                    batch_rows.append((
                        raw_pages_and_text[i]["page_number"],
                        raw_pages_and_text[i]["sentence_chunk"],
                        mmap_ro[i].tolist(),
                    ))
                    if len(batch_rows) >= DB_WRITE_BATCH:
                        _insert_rows_batched(cur, batch_rows)
                        chunks_stored += len(batch_rows)
                        batch_rows = []

                if batch_rows:
                    _insert_rows_batched(cur, batch_rows)
                    chunks_stored += len(batch_rows)

                cur.close()
                conn.close()

            finally:
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
                headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                },
                timeout=60.0,
            )
            if response.status_code != 200:
                error_msg = response.json().get("error", {}).get("message", "Unknown LLM Error")
                print(f"[ERROR] LLM API Failed: {error_msg}")
                return f"LLM Error: {error_msg}"
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[CRITICAL] LLM Call Crashed: {e}")
            return "The AI is currently unavailable."


# ENDPOINTS
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
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)
    del contents

    background_tasks.add_task(process_pdf, temp_path, file.filename, pruning_strategy)
    return {"message": "Upload received, processing in background. Poll /status.", "pruning_strategy": pruning_strategy}


@app.post("/query")
async def query_document(request: QueryRequest):
    has_rag_data = has_stored_data()
    has_mem_data = len(full_context_pages) > 0

    if not has_rag_data and not has_mem_data:
        raise HTTPException(status_code=400, detail="No document data found. Please upload a PDF first.")

    mode = "full_context" if has_mem_data else "rag"

    if mode == "full_context":
        context_text = "\n\n".join(f"[Page {p['page_number']}]\n{p['text']}" for p in full_context_pages)
        rows = [(p["page_number"], p["text"]) for p in full_context_pages]
    else:
        # embed_texts is sync — fine here since query is a single short string
        query_embedding = embed_texts([request.query])[0]
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

        context_text = "".join(f"SOURCE {i+1} (Page {row[0]}):\n{row[1]}\n\n" for i, row in enumerate(rows))

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
    current = processing_status.get("status")
    if current in ("processing", "failed", "done"):
        return processing_status
    if has_stored_data():
        return {"status": "done", "mode": "rag", "message": "Existing data found in database."}
    if len(full_context_pages) > 0:
        return {"status": "done", "mode": "full_context"}
    return {"status": "idle", "message": "No data found."}


@app.get("/pruning-report")
async def get_pruning_report():
    if not pruning_report:
        raise HTTPException(status_code=404, detail="No pruning report available.")
    return pruning_report


@app.get("/chunks")
async def get_stored_chunks(limit: int = Query(default=20, ge=1, le=200)):
    mode = processing_status.get("mode")
    if mode == "full_context":
        return {
            "mode": "full_context",
            "total": len(full_context_pages),
            "chunks": [{"id": i+1, "page_number": p["page_number"], "content": p["text"][:500], "embedding": []}
                       for i, p in enumerate(full_context_pages[:limit])],
        }
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM document_chunks;")
    total = cur.fetchone()[0]
    cur.execute("SELECT id, page_number, content, embedding FROM document_chunks ORDER BY id LIMIT %s;", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {
        "mode": "rag", "total": total,
        "chunks": [{"id": int(r[0]), "page_number": int(r[1]), "content": r[2],
                    "embedding": [float(v) for v in r[3]]} for r in rows],
    }


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