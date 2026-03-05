# 🔍 Local RAG Pipeline API

> Retrieval-Augmented Generation over your own PDFs — fully local, no external AI APIs.

Built with **FastAPI**, **pgvector**, and **Hugging Face Transformers**. Upload a PDF, ask questions, get grounded answers with page-level source citations — all running on your own hardware.

---

## How It Works

```
PDF Upload → Text Extraction → Sentence Chunking → Embeddings → PostgreSQL (pgvector)
                                                                        ↓
User Query → Query Embedding → Cosine Similarity Search → Top-K Chunks → LLM → Answer
```

**Three stages:**
1. **Ingestion** — Parse PDF, split into sentence chunks, generate embeddings, store in PostgreSQL
2. **Retrieval** — Embed the query, find the most semantically similar chunks via vector search
3. **Generation** — Pass retrieved context + query to a local LLM to produce a grounded answer

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | FastAPI |
| PDF Parsing | PyMuPDF (`fitz`) |
| Sentence Splitting | spaCy sentencizer |
| Embedding Model | `all-mpnet-base-v2` |
| Vector Store | PostgreSQL + pgvector |
| LLM | Configurable via `MODEL_ID` env var |
| Quantization | BitsAndBytes (4-bit) |
| DB Driver | psycopg2 + pgvector-python |

---

## Embedding Model: `all-mpnet-base-v2`

The embedding model is the most critical component in a RAG pipeline — if it can't accurately capture semantic meaning, the LLM will receive irrelevant context regardless of its capability.

**`sentence-transformers/all-mpnet-base-v2`** was chosen because:

- 🏆 **Best-in-class accuracy** — ranked #1 on the sentence-transformers semantic similarity benchmark, outperforming `all-MiniLM-L6-v2` and `all-distilroberta-v1`
- 🧠 **768-dimensional embeddings** — richer representation than smaller 384-dim models, matching the `vector(768)` column in PostgreSQL
- 🔄 **Bidirectional attention (MPNet)** — attends to both preceding and following tokens simultaneously, giving deeper contextual understanding than unidirectional models
- 🔒 **Fully local** — no API keys, no data leaves your machine, no per-call cost
- ⚖️ **Good size/quality trade-off** — ~420 MB, small enough to sit alongside a quantized LLM on a single consumer GPU

### Why 10-Sentence Chunks?

Chunk size directly affects embedding quality:

| Chunk Size | Problem |
|------------|---------|
| Too small (1–2 sentences) | Lacks topic context; embedding is too narrow |
| Too large (full page) | Averages over many topics; embedding is too diffuse |
| **10 sentences** ✅ | Captures a complete thought; focused enough for precise retrieval |

---

## LLM: Configurable via `MODEL_ID`

The LLM is not hardcoded — set `MODEL_ID` in your `.env` to any Hugging Face causal LM. Tested with:
- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `google/gemma-2-9b-it`

### 4-Bit Quantization (BitsAndBytes)

Without quantization, a 7B model in float16 needs ~14 GB VRAM. With 4-bit quantization it fits in **4–6 GB**, making it accessible on consumer GPUs (RTX 3060, RTX 4070, etc.).

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  # dequantize to fp16 for compute
)
```

### Chat Templates

`tokenizer.apply_chat_template()` is used instead of hardcoding prompt tokens. Each model family uses different special tokens (`[INST]` for Llama, `<start_of_turn>user` for Gemma, etc.) — this handles it automatically based on the loaded tokenizer.

---

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with the `pgvector` extension
- CUDA-capable GPU (recommended) or CPU fallback
- Hugging Face account with access to your chosen model

### 1. Install Python dependencies

```bash
pip install fastapi uvicorn pymupdf spacy sentence-transformers transformers
pip install bitsandbytes accelerate psycopg2-binary pgvector python-dotenv
python -m spacy download en_core_web_sm
```

### 2. Enable pgvector in PostgreSQL

```sql
-- Run as superuser in psql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3

DB_NAME=your_database_name
DB_USER=your_postgres_username     # ⚠️ Must be DB_USER, not USER (USER is a reserved Linux variable)
DB_PASSWORD=your_password
HOST=localhost
PORT=5432
```

### 4. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will initialize the database, load spaCy, the embedding model, and the LLM on startup. This may take a few minutes on first run.

---

## API Reference

### `POST /upload`
Upload a PDF for processing. Runs in the background — returns immediately.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

```json
{ "message": "Upload received, processing in background. Poll /status to check." }
```

---

### `GET /status`
Poll processing progress.

```bash
curl http://localhost:8000/status
```

```json
{ "status": "done", "chunks": 142, "error": null }
```

Possible status values: `idle` · `processing` · `done` · `failed`

---

### `POST /query`
Ask a question about the uploaded document.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main argument of chapter 3?", "temperature": 0.7, "max_new_tokens": 256}'
```

```json
{
  "query": "What is the main argument of chapter 3?",
  "answer": "The main argument of chapter 3 is...",
  "sources": [
    { "page": 12, "text": "In this chapter we argue that..." }
  ]
}
```

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Natural language question |
| `temperature` | float | `0.7` | LLM sampling temperature |
| `max_new_tokens` | int | `256` | Max tokens to generate |

---

### `GET /reset`
Delete all stored chunks and reset status to idle.

```bash
curl http://localhost:8000/reset
```

---

## Performance Notes

- **Batch inserts** — Uses `psycopg2.extras.execute_values` to insert all embeddings in a single query (10–50× faster than a per-row loop)
- **Chunked upload streaming** — File is read in 1 MB chunks to avoid blocking the async event loop
- **Background processing** — PDF parsing and embedding run in a `BackgroundTask`; the upload endpoint returns immediately
- **Progress visibility** — `show_progress_bar=True` on `embedding_model.encode()` so you can confirm the server isn't frozen during large documents

---

## Known Limitations

- **Single document at a time** — A `document_id` column would be needed for true multi-document support
- **In-process status tracking** — `processing_status` is a global dict; inconsistent across multiple uvicorn workers. Use Redis or a DB row for production multi-worker deployments
- **No authentication** — Add OAuth2 or API key middleware before any public deployment
- **CPU is slow** — Embedding and LLM inference on CPU can take several minutes. A CUDA GPU is strongly recommended

---

## Project Structure

```
.
├── main.py          # FastAPI app, endpoints, processing logic
├── .env             # Environment variables (never commit this)
├── .env.example     # Template for environment variables
└── README.md
```

---

## License

MIT
