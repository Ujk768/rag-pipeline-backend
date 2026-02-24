import os
import re
import fitz  # PyMuPDF
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
import os

load_dotenv()


# --- GLOBAL STATE ---
# In a production app, you'd use a Vector DB (like Chroma or FAISS) instead of global variables.
pages_and_chunks = []
embeddings_tensor = None
processing_status = {"status": "idle", "chunks": 0}
# Models
embedding_model = None
llm_model = None
tokenizer = None
nlp = None

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads models into memory when the server starts."""
    global embedding_model, llm_model, tokenizer, nlp
    
    print("[INFO] Loading Spacy...")
    nlp = English()
    nlp.add_pipe("sentencizer")

    print("[INFO] Loading Embedding Model...")
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

    print("[INFO] Loading LLM and Tokenizer...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="auto" # <--- THIS IS THE CRITICAL FIX
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

# --- HELPER FUNCTIONS (From your notebook) ---
def text_formatter(text: str) -> str:
    return text.replace('\n', ' ').strip()

def split_list(input_list: list[str], slice_size: int = 10) -> list[list[str]]:
    return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]

# def process_pdf(file_path: str, filename: str):
#     global pages_and_chunks, embeddings_tensor, processing_status
#     processing_status["status"] = "processing"
#     if not file.filename.endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported.")

#     # Save temp file
#     temp_path = f"temp_{file.filename}"
#     with open(temp_path, "wb") as buffer:
#         buffer.write(await file.read())

#     print("[INFO] Processing PDF...")
#     document = fitz.open(temp_path)
    
#     raw_pages_and_text = []
#     for page_num, page in enumerate(document):
#         text = text_formatter(page.get_text())
#         sentences = [str(sentence) for sentence in nlp(text).sents]
#         sentence_chunks = split_list(sentences, 10)
        
#         for chunk in sentence_chunks:
#             joined_chunk = "".join(chunk).replace("  ", " ").strip()
#             joined_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk)
            
#             # Filter out tiny chunks
#             if len(joined_chunk) / 4 > 30: 
#                 raw_pages_and_text.append({
#                     "page_number": page_num + 1,
#                     "sentence_chunk": joined_chunk
#                 })
    
#     document.close()
#     os.remove(temp_path)

#     print("[INFO] Creating Embeddings...")
#     text_chunks = [item["sentence_chunk"] for item in raw_pages_and_text]
    
#     # Generate embeddings in batches
#     new_embeddings = embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True)
    
#     # Update global state
#     pages_and_chunks = raw_pages_and_text
#     embeddings_tensor = new_embeddings


def process_pdf(file_path: str, filename: str):
    global pages_and_chunks, embeddings_tensor, processing_status

    try:
        processing_status["status"] = "processing"
        processing_status["chunks"] = 0
        processing_status["error"] = None

        print(f"[INFO] Opening PDF: {filename}")
        document = fitz.open(file_path)
        total_pages = len(document)
        print(f"[INFO] PDF has {total_pages} pages")

        raw_pages_and_text = []

        for page_num, page in enumerate(document):
            print(f"[INFO] Processing page {page_num + 1}/{total_pages}")
            text = text_formatter(page.get_text())

            if not text.strip():
                print(f"[WARN] Page {page_num + 1} is empty, skipping...")
                continue

            sentences = [str(sentence) for sentence in nlp(text).sents]
            sentence_chunks = split_list(sentences, 10)

            for chunk in sentence_chunks:
                joined_chunk = "".join(chunk).replace("  ", " ").strip()
                joined_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk)

                if len(joined_chunk) / 4 > 30:
                    raw_pages_and_text.append({
                        "page_number": page_num + 1,
                        "sentence_chunk": joined_chunk
                    })

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

        new_embeddings = embedding_model.encode(
            text_chunks,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        # Update global state
        pages_and_chunks = raw_pages_and_text
        embeddings_tensor = new_embeddings

        processing_status["status"] = "done"
        processing_status["chunks"] = total_chunks
        print(f"[INFO] Done! {total_chunks} chunks embedded and ready.")

    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {e}")
        processing_status["status"] = "failed"
        processing_status["error"] = str(e)

        # Cleanup temp file if it still exists
        if os.path.exists(file_path):
            os.remove(file_path)


# --- ENDPOINTS ---

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    background_tasks.add_task(process_pdf, temp_path, file.filename)
    
    return {"message": "Upload received, processing in background. Poll /status to check."}
    


@app.post("/query")
async def query_document(request: QueryRequest):
    """Queries the processed document and generates an answer."""
    global pages_and_chunks, embeddings_tensor

    if embeddings_tensor is None or len(pages_and_chunks) == 0:
        raise HTTPException(status_code=400, detail="No document uploaded or processed yet.")

    print(f"[INFO] Searching for: {request.query}")
    query_embedding = embedding_model.encode(request.query, convert_to_tensor=True)
    
    # Similarity Search
    dot_scores = util.dot_score(query_embedding, embeddings_tensor)[0]
    scores, indices = torch.topk(dot_scores, k=5)
    
    context_items = [pages_and_chunks[i] for i in indices]
    context_text = "- " + "\n- ".join(item["sentence_chunk"] for item in context_items)

    # Build Prompt
    base_prompt = f"""Based on the following context items, please answer the query.
    Note: interpret the query broadly, for example "macros" refers to "macronutrients".
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.

    Context items:
    {context_text}

    User query: {request.query}
    Answer:"""

    dialogue_template = [{"role": "user", "content": base_prompt}]
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template,
        tokenize=False,
        add_generation_prompt=True
    )

    print("[INFO] Generating Answer...")
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = llm_model.generate(
        **input_ids,
        temperature=request.temperature,
        do_sample=True,
        max_new_tokens=request.max_new_tokens
    )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up output to only return the generated answer
    # clean_answer = output_text.replace(prompt, "").strip()
    # Instead of replacing the prompt, split on "model" which is Gemma's response tag
    if "model" in output_text:
        clean_answer = output_text.split("model")[-1].strip()
    else:
        clean_answer = output_text.strip()

    return {
        "query": request.query,
        "answer": clean_answer,
        "sources": [{"page": item["page_number"], "text": item["sentence_chunk"][:100] + "..."} for item in context_items]
    }

@app.get("/status")
async def get_status():
    return processing_status