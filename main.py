import asyncio
from functools import partial
import os
import re
import fitz  # PyMuPDF
import numpy as np
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from contextlib import asynccontextmanager
from spacy.lang.en import English
from dotenv import load_dotenv
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
import httpx
from prune import prune_cosine, prune_cosine_whitened, prune_kmeans, prune_mmr, maxsim_rerank
from embedding import embedding_text, warm_embedding_service
from db import clear_existing_data, clear_pruned_chunks, count_active_chunks_from_pruned, get_db_connection, get_document_info, has_pruned_data, has_stored_data, init_db, _insert_rows_batched, insert_pruned_rows_batched, upsert_document_info


origins = [
    "https://adaptive-rag.vercel.app/",
    "*",
    "http://localhost:3000",
]

load_dotenv()

# CONFIGURATION
LLM_API_KEY = os.getenv("LLM_API_KEY")

FULL_CONTEXT_TOKEN_LIMIT = 6000
# MAX_TOKEN_COUNT = 80000

# With embedding offloaded, encode batch size only affects
# how many texts we send per HTTP request to the embedding service.
# 64 is a good balance — not too large to timeout, not too small to be chatty.
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", 64))
DB_WRITE_BATCH = int(os.getenv("DB_WRITE_BATCH", 100))

EMBEDDING_DIM = 1024

MAX_FILE_SIZE_MB = 4
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_CHUNKS = 300

PruningStrategy = Literal["none", "cosine", "cosine_whitened", "kmeans", "mmr"]

# GLOBAL STATE
processing_status = {"status": "idle", "chunks": 0, "error": None, "mode": None}
full_context_pages: list[dict] = []
pruning_report: dict = {}




# LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Init DB...")
    init_db()

    print("[INFO] Warming up embedding service...")
    warm_embedding_service()
    
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




# HELPER FUNCTIONS
def split_list(input_list: list[str], slice_size: int = 10) -> list[list[str]]:
    return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]


def iter_chunks(full_text_by_page: list[dict], nlp, slice_size: int = 10):
    count = 0
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



def process_pdf(file_path: str, filename: str):
    global processing_status, full_context_pages, pruning_report

    try:
        processing_status = {"status": "processing", "chunks": 0, "error": None, "mode": None}
        full_context_pages = []
        # Clear existing data
        clear_existing_data()

        print(f"[INFO] Opening PDF: {filename}")
        document = fitz.open(file_path)
        total_pages = len(document)
        upsert_document_info(filename, total_pages) 
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

        
        print("[INFO] Streaming encode + insert (no pruning).")
        conn = get_db_connection()
        chunks_stored = 0
        batch_chunks: list[dict] = []

        for chunk in iter_chunks(full_text_by_page, nlp):
            batch_chunks.append(chunk)
            if len(batch_chunks) >= ENCODE_BATCH_SIZE:
                texts = [c["sentence_chunk"] for c in batch_chunks]
                embs = embedding_text(texts)
                rows = [(c["page_number"], c["sentence_chunk"], embs[j].tolist())
                    for j, c in enumerate(batch_chunks)]
                _insert_rows_batched(conn, rows)   # pass conn, not cur
                chunks_stored += len(rows)
                del embs, rows
                batch_chunks = []

        if batch_chunks:
            texts = [c["sentence_chunk"] for c in batch_chunks]
            embs = embedding_text(texts)
            rows = [(c["page_number"], c["sentence_chunk"], embs[j].tolist())
                for j, c in enumerate(batch_chunks)]
            _insert_rows_batched(conn, rows)       # pass conn, not cur
            chunks_stored += len(rows)
            del embs, rows
        
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
@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
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

    background_tasks.add_task(process_pdf, temp_path, file.filename)
    return {"message": "Upload received, processing in background. Poll /status."}


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
        loop = asyncio.get_event_loop()
        query_embedding = (await loop.run_in_executor(None, partial(embedding_text, [request.query])))[0]

        conn = get_db_connection()
        cur = conn.cursor()
        fetch_limit = 20 if request.use_maxsim else 5

        # Route to pruned table if a pruning run exists, otherwise use main table
        table = "document_chunks_pruned" if has_pruned_data() else "document_chunks"
        cur.execute(f"""
            SELECT page_number, content, embedding
            FROM {table}
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
        "serving_from": table if mode == "rag" else "full_context",
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

async def _apply_pruning_to_existing(strategy: str) -> dict:
    """
    Always reads from document_chunks (full set), applies pruning,
    then replaces document_chunks_pruned entirely.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    # Always pull from the immutable source table
    cur.execute("SELECT id, page_number, content, embedding FROM document_chunks ORDER BY id;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=400, detail="No stored chunks found. Upload a document first.")

    ids          = [r[0] for r in rows]
    raw_chunks   = [{"page_number": r[1], "sentence_chunk": r[2]} for r in rows]
    total_chunks = len(rows)

    mmap_fd, mmap_path = tempfile.mkstemp(suffix=".npy")
    os.close(mmap_fd)
    try:
        mmap_emb = np.memmap(mmap_path, dtype="float32", mode="w+",
                             shape=(total_chunks, EMBEDDING_DIM))
        for i, r in enumerate(rows):
            mmap_emb[i] = np.array(r[3], dtype="float32")
        mmap_emb.flush()
        mmap_ro = np.memmap(mmap_path, dtype="float32", mode="r",
                            shape=(total_chunks, EMBEDDING_DIM))

        if strategy == "cosine":
            kept_indices, report = prune_cosine(mmap_ro, raw_chunks)
        elif strategy == "cosine_whitened":
            kept_indices, report = prune_cosine_whitened(mmap_ro, raw_chunks)
        elif strategy == "kmeans":
            kept_indices, report = prune_kmeans(mmap_ro, raw_chunks)
        elif strategy == "mmr":
            kept_indices, report = prune_mmr(mmap_ro, raw_chunks)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy '{strategy}'.")

        # Replace pruned table entirely
        clear_pruned_chunks()

        conn = get_db_connection()
        batch = [
            (
                ids[i],                        # source_chunk_id
                raw_chunks[i]["page_number"],
                raw_chunks[i]["sentence_chunk"],
                mmap_ro[i].tolist(),
                strategy,
            )
            for i in kept_indices
        ]
        insert_pruned_rows_batched(conn, batch)
        conn.close()

    finally:
        try:
            os.unlink(mmap_path)
        except OSError:
            pass

    global pruning_report
    pruning_report = report
    return report


@app.post("/prune")
async def prune_existing(
    pruning_strategy: PruningStrategy = Query(default="cosine"),
):
    """Re-prune the vectors already stored in the DB — no re-embedding needed."""
    if pruning_strategy == "none":
        raise HTTPException(status_code=400, detail="Specify a real pruning strategy (cosine, cosine_whitened, kmeans, mmr).")
    if processing_status.get("status") == "processing":
        raise HTTPException(status_code=409, detail="A document is currently being processed. Try again after /status returns 'done'.")
    if not has_stored_data():
        raise HTTPException(status_code=400, detail="No stored chunks found. Upload a document first.")

    report = await _apply_pruning_to_existing(pruning_strategy)
    return {"message": f"Pruning complete using strategy '{pruning_strategy}'.", "report": report}

@app.get("/document")
async def get_document_info_route():
    info = get_document_info()
    has_full_ctx = len(full_context_pages) > 0
    has_rag = has_stored_data()

    if not info and not has_full_ctx and not has_rag:
        return {"has_document": False}

    return {
        "has_document":  True,
        "document_name": info["document_name"] if info else None,
        "total_pages":   info["total_pages"]   if info else None,
        "uploaded_at":   info["uploaded_at"]   if info else None,
        "mode":          "full_context" if has_full_ctx else "rag",
        "active_chunks": len(full_context_pages) if has_full_ctx else count_active_chunks_from_pruned(),
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

@app.get("/health")
async def health_check():
    return {"status": "ok"}