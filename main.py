import os
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
from transformers import AutoTokenizer, AutoModelForCausalLM # UJK - , BitsAndBytesConfig
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from typing import Literal, Optional

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
PruningStrategy = Literal["none", "cosine", "maxsim", "cosine_whitened"]


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

# Added: persistent pruning report exposed to frontend
# Stores detailed stats from the most recent pruning run.
# Reset on each new upload. Frontend polls /pruning-report to visualize results.
pruning_report: dict = {}

embedding_model = None
llm_model = None
tokenizer = None
nlp = None

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu" -> UJK: Uncomment and comment out the next six lines

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


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
    
    # quantization_config = BitsAndBytesConfig( -> UJK: Uncomment
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    # )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        # quantization_config=quantization_config, -> UJK Uncomment
        low_cpu_mem_usage=True,
        device_map={"": DEVICE}, #UJK comment this out and uncomment the next line
        #device_map="auto", 
    )
    print("[INFO] All models loaded successfully!")
    yield
    print("[INFO] Shutting down...")


app = FastAPI(lifespan=lifespan, title="Local RAG API")


# REQUEST SCHEMAS
class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.7
    max_new_tokens: int = 256


# HELPER FUNCTIONS
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()


def split_list(input_list: list[str], slice_size: int = 10) -> list[list[str]]:
    return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]


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


# Strategy 1 - Cosine Similarity Pruning─
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

# Strategy 2 - MaxSim Pruning
def prune_maxsim(
    embeddings: np.ndarray,
    chunks: list[dict],
    threshold_multiplier: float = 0.95,
) -> tuple[list[int], dict]:
    """
    Prunes chunks that are highly similar to *at least one other* chunk.

    Logic:
      - For each chunk, compute its maximum cosine similarity to any other chunk.
      - High MaxSim score → this chunk has a near-duplicate. Prune it.
      - Low MaxSim score → this chunk is unlike everything else. Keep it.

    This is the DocPruner-aligned strategy. It mirrors how MaxSim retrieval
    works at query time: you only need one good matching vector per concept,
    so duplicates are safe to drop without hurting recall.

    O(n²) pairwise comparison — acceptable for typical document sizes.
    """
    n = len(embeddings)

    # Full pairwise cosine similarity matrix
    sim_matrix = cosine_similarity_matrix(embeddings, embeddings)

    # Zero out diagonal (self-similarity = 1.0, not useful)
    np.fill_diagonal(sim_matrix, 0.0)

    # MaxSim score per chunk: highest similarity to any neighbour
    scores = sim_matrix.max(axis=1)  # shape: (n,)

    # Adaptive threshold: chunks above this have near-duplicates → prune
    threshold = float(scores.mean() * threshold_multiplier)

    kept_indices = [i for i, s in enumerate(scores) if s <= threshold]

    if not kept_indices:
        kept_indices = [int(np.argmin(scores))]

    pruned_indices = [i for i in range(n) if i not in set(kept_indices)]

    stats = _build_pruning_stats(
        strategy="maxsim",
        n_total=n,
        kept_indices=kept_indices,
        pruned_indices=pruned_indices,
        scores=scores.tolist(),
        threshold=threshold,
        chunks=chunks,
        extra={
            "threshold_multiplier": threshold_multiplier,
            "score_meaning": "max cosine similarity to any other chunk — higher = more redundant",
        },
    )

    return kept_indices, stats

# Strategy 3 - Cosine + Whitening Pruning
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
            raw_pages_and_text = []
            for page_data in full_text_by_page:
                sentences = [str(s) for s in nlp(page_data["text"]).sents]
                for chunk in split_list(sentences, 10):
                    joined = "".join(chunk).replace("  ", " ").strip()
                    joined = re.sub(r"\.([A-Z])", r". \1", joined)
                    if joined:
                        raw_pages_and_text.append({
                            "page_number": page_data["page_number"],
                            "sentence_chunk": joined,
                        })

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

                if pruning_strategy == "cosine":
                    kept_indices, report = prune_cosine(new_embeddings, raw_pages_and_text)

                elif pruning_strategy == "maxsim":
                    kept_indices, report = prune_maxsim(new_embeddings, raw_pages_and_text)

                elif pruning_strategy == "cosine_whitened":
                    kept_indices, report = prune_cosine_whitened(new_embeddings, raw_pages_and_text)

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
    # ADDED: pruning_strategy query param — default "none" preserves existing behaviour
    pruning_strategy: PruningStrategy = Query(
        default="none",
        description=(
            "Pruning strategy to apply before storing embeddings. "
            "'none': store all chunks. "
            "'cosine': prune chunks too close to centroid. "
            "'maxsim': prune chunks that have near-duplicates. "
            "'cosine_whitened': cosine pruning on whitened embedding space (most discriminative)."
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
        # RAG: embed the query and retrieve the top-5 most similar chunks
        query_embedding = embedding_model.encode(request.query, convert_to_numpy=True)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT page_number, content
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT 5
        """, (query_embedding,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No relevant context found.")
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