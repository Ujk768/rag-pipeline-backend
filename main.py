import os
import re
import fitz  # PyMuPDF
import torch
import psycopg2
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pgvector.psycopg2 import register_vector  
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",   
    "*",
]

load_dotenv()

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")

# Leaves headroom for prompt template + generated answer.
# Raise this if your model has a larger context window (e.g. 28000 for 32K models).
FULL_CONTEXT_TOKEN_LIMIT = 6000

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),   # Must be DB_USER in .env — USER is a reserved Linux variable
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("HOST"),
    "port": os.getenv("PORT"),
}


# --- DATABASE ---

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()           # commit before registering so the type physically exists
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


# --- GLOBAL STATE ---
# full_context_pages: holds pages in memory when the document is small enough
#   to pass directly to the LLM — no DB involved.
# For RAG mode this is always empty; chunks live in pgvector instead.
processing_status = {"status": "idle", "chunks": 0, "error": None, "mode": None}
full_context_pages: list[dict] = []

embedding_model = None
llm_model = None
tokenizer = None
nlp = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- LIFESPAN ---

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
        # device_map="auto",
        device_map=current_device_map,
    )
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

# --- REQUEST SCHEMAS ---

class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.7
    max_new_tokens: int = 256


# --- HELPER FUNCTIONS ---

def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()


def split_list(input_list: list[str], slice_size: int = 10) -> list[list[str]]:
    return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]


# --- PDF PROCESSING ---

def process_pdf(file_path: str, filename: str):
    global processing_status, full_context_pages

    try:
        processing_status = {"status": "processing", "chunks": 0, "error": None, "mode": None}
        full_context_pages = []

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
            # ── FULL-CONTEXT MODE ──
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
            # ── RAG MODE ──
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

            print("[INFO] Inserting into Postgres...")
            conn = get_db_connection()
            cur = conn.cursor()
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

            processing_status.update({
                "status": "done",
                "mode": "rag",
                "chunks": total_chunks,
            })
            print(f"[INFO] Done! {total_chunks} chunks embedded and stored.")

    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {e}")
        processing_status.update({"status": "failed", "error": str(e)})
        if os.path.exists(file_path):
            os.remove(file_path)


# --- ENDPOINTS ---

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if processing_status.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Already processing a file. Poll /status.")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # stream in 1 MB chunks
            buffer.write(chunk)

    background_tasks.add_task(process_pdf, temp_path, file.filename)
    return {"message": "Upload received, processing in background. Poll /status to check."}


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
        eos_token_id=tokenizer.eos_token_id,  # stop as soon as answer is complete
        pad_token_id=tokenizer.eos_token_id,  # prevents padding warning that slows things
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


@app.get("/reset")
async def reset_data():
    global processing_status, full_context_pages

    # Clear pgvector chunks (RAG mode)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks;")
    conn.commit()
    cur.close()
    conn.close()

    # Clear in-memory pages (full-context mode)
    full_context_pages = []
    processing_status = {"status": "idle", "chunks": 0, "error": None, "mode": None}
    return {"message": "Data reset successfully."}