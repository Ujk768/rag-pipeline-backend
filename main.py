import os
import re
import fitz  # PyMuPDF
import torch
import numpy as np
import psycopg2
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv


load_dotenv()

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    # FIX 1: Renamed from USER -> DB_USER to avoid collision with Linux $USER env var.
    # Update your .env file: DB_USER=your_postgres_username
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("HOST"),
    "port": os.getenv("PORT"),
}


def get_db_connection(register=True):
    conn = psycopg2.connect(**DB_CONFIG)
    if register:
        register_vector(conn)
    return conn


def init_db():
    # FIX 2: Use a single connection for the whole init sequence.
    # Previously the code opened one connection without registering the vector type,
    # called register_vector, but then the CREATE TABLE still used that same cursor —
    # which worked, but if any step failed the connection was left open/dirty.
    # Now we do it cleanly: enable extension → commit → register → create table → commit.
    conn = psycopg2.connect(**DB_CONFIG)  # raw connection, no vector type yet
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()  # commit BEFORE registering so the type physically exists

    register_vector(conn)  # now safe to register

    # 768 dimensions for "all-mpnet-base-v2"
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            page_number INTEGER,
            content TEXT,
            embedding vector(768)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[INFO] Database initialized and ready.")


# --- GLOBAL STATE ---
processing_status = {"status": "idle", "chunks": 0, "error": None}

embedding_model = None
llm_model = None
tokenizer = None
nlp = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Init DB...")
    init_db()

    global embedding_model, llm_model, tokenizer, nlp

    print("[INFO] Loading Spacy...")
    nlp = English()
    nlp.add_pipe("sentencizer")

    print("[INFO] Loading Embedding Model...")
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

    print("[INFO] Loading LLM and Tokenizer...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    print("[INFO] All models loaded successfully!")
    yield
    print("[INFO] Shutting down...")


app = FastAPI(lifespan=lifespan, title="Local RAG API")


# --- REQUEST SCHEMAS ---
class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.7
    max_new_tokens: int = 256


# --- HELPER FUNCTIONS ---
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()


def split_list(input_list: list[str], slice_size: int = 10) -> list[list[str]]:
    return [input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)]


def process_pdf(file_path: str, filename: str):
    global processing_status

    try:
        processing_status = {"status": "processing", "chunks": 0, "error": None}

        print(f"[INFO] Opening PDF: {filename}")
        document = fitz.open(file_path)
        total_pages = len(document)
        print(f"[INFO] PDF has {total_pages} pages")

        raw_pages_and_text = []

        for page_num, page in enumerate(document):
            text = text_formatter(page.get_text())
            if not text.strip():
                print(f"[WARN] Page {page_num + 1} is empty, skipping...")
                continue

            sentences = [str(s) for s in nlp(text).sents]
            for chunk in split_list(sentences, 10):
                joined = "".join(chunk).replace("  ", " ").strip()
                joined = re.sub(r"\.([A-Z])", r". \1", joined)
                if len(joined) / 4 > 30:
                    raw_pages_and_text.append(
                        {"page_number": page_num + 1, "sentence_chunk": joined}
                    )

        document.close()
        os.remove(file_path)

        total_chunks = len(raw_pages_and_text)
        print(f"[INFO] Total chunks created: {total_chunks}")

        if total_chunks == 0:
            processing_status["status"] = "failed"
            processing_status["error"] = "No valid text chunks found in PDF"
            return

        print(f"[INFO] Generating embeddings for {total_chunks} chunks...")
        text_chunks = [item["sentence_chunk"] for item in raw_pages_and_text]

        # FIX 3: encode in one batched call with show_progress_bar so you can
        # see it isn't frozen. convert_to_numpy=True is correct for pgvector.
        new_embeddings = embedding_model.encode(
            text_chunks,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        print("[INFO] Inserting into Postgres...")
        conn = get_db_connection()
        cur = conn.cursor()

        # FIX 4: Use executemany-style batching with psycopg2's execute_values
        # for dramatically faster inserts (single round-trip instead of N round-trips).
        from psycopg2.extras import execute_values

        rows = [
            (item["page_number"], item["sentence_chunk"], emb.tolist())
            for item, emb in zip(raw_pages_and_text, new_embeddings)
        ]
        execute_values(
            cur,
            "INSERT INTO document_chunks (page_number, content, embedding) VALUES %s",
            rows,
        )

        conn.commit()
        cur.close()
        conn.close()

        processing_status["status"] = "done"
        processing_status["chunks"] = total_chunks
        print(f"[INFO] Done! {total_chunks} chunks embedded and stored.")

    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {e}")
        processing_status["status"] = "failed"
        processing_status["error"] = str(e)
        if os.path.exists(file_path):
            os.remove(file_path)


# --- ENDPOINTS ---

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if processing_status.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Already processing a file. Poll /status.")

    temp_path = f"temp_{file.filename}"
    
    # No aiofiles needed — read in chunks using UploadFile's built-in async read
    with open(temp_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1 MB at a time
            buffer.write(chunk)

    background_tasks.add_task(process_pdf, temp_path, file.filename)
    return {"message": "Upload received, processing in background. Poll /status to check."}

@app.post("/query")
async def query_document(request: QueryRequest):
    if processing_status.get("status") != "done":
        raise HTTPException(
            status_code=400,
            detail=f"No document ready. Current status: {processing_status.get('status')}",
        )

    print(f"[INFO] Searching for: {request.query}")
    query_embedding = embedding_model.encode(request.query, convert_to_numpy=True)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT page_number, content
        FROM document_chunks
        ORDER BY embedding <=> %s
        LIMIT 5
        """,
        (query_embedding,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    context_text = "- " + "\n- ".join(row[1] for row in rows)

    base_prompt = f"""Use the following context snippets to respond to the query.
Take a moment to pull out the key information before answering.
Return only the answer, not the thought process.

Context items:
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
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = llm_model.generate(
        **input_ids,
        temperature=request.temperature,
        do_sample=True,
        max_new_tokens=request.max_new_tokens,
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_answer = output_text.split("model")[-1].strip() if "model" in output_text else output_text.strip()

    return {
        "query": request.query,
        "answer": clean_answer,
        "sources": [{"page": row[0], "text": row[1][:100] + "..."} for row in rows],
    }


@app.get("/status")
async def get_status():
    return processing_status


@app.get("/reset")
async def reset_data():
    global processing_status

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks;")
    conn.commit()
    cur.close()
    conn.close()

    processing_status = {"status": "idle", "chunks": 0, "error": None}

    return {"message": "Data reset successfully."}
