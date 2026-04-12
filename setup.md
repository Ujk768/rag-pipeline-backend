# Adaptive RAG — Backend Setup

## Requirements

- Python 3.10+
- Docker Desktop (running)
- A `.env` file (see below)

---

## 1. Clone and set up the environment

```bash
git clone https://github.com/Ujk768/rag-pipeline-backend.git
cd rag-pipeline-backend

python3 -m venv ragenv
source ragenv/bin/activate        # Windows: ragenv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 2. Set up pgvector with Docker

```bash
# Pull the image (one time only)
docker pull pgvector/pgvector:pg16

# Start the container (one time only)
docker run --name pgvector-rag \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=ragdb \
  -e POSTGRES_USER=postgres \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

Every time you restart your machine, just run:
```bash
docker start pgvector-rag
```

---

## 3. Create your `.env` file

Create a file called `.env` in the backend folder:

```
HOST=localhost
PORT=5432
DB_NAME=ragdb
DB_USER=postgres
DB_PASSWORD=password
HF_TOKEN=your_huggingface_token_here
MODEL_ID=google/gemma-2-2b-it
```

> **HF_TOKEN**: get yours at https://huggingface.co/settings/tokens  
> You need to accept Gemma's licence at https://huggingface.co/google/gemma-2-2b-it before it will download.

---

## 4. Start the server

```bash
source ragenv/bin/activate
uvicorn main:app --reload --port 8000
```

Wait for:
```
[INFO] All models loaded successfully!
```

Then open http://localhost:8000/docs to see all endpoints.

---

## 5. GPU users (NVIDIA)

If you have a CUDA GPU, in `main.py`:

- Uncomment the `BitsAndBytesConfig` block (marked `UJK`)
- Uncomment `quantization_config=quantization_config`
- Change `device_map={"": DEVICE}` to `device_map="auto"`
- Uncomment `# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` and remove the six lines below it

Also make sure `bitsandbytes` is installed:
```bash
pip install bitsandbytes
```

---

## Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| `POST` | `/upload` | Upload a PDF. Optional `?pruning_strategy=` param |
| `GET` | `/status` | Check indexing progress |
| `POST` | `/query` | Ask a question. Set `use_maxsim: true` for re-ranking |
| `GET` | `/pruning-report` | Full stats from last pruning run |
| `GET` | `/reset` | Wipe the database and start fresh |

### Pruning strategies (pass as query param on `/upload`)

| Strategy | What it does |
|----------|--------------|
| `none` | Store everything (default) |
| `cosine` | Drop chunks too close to the document centroid |
| `cosine_whitened` | Same as cosine but with whitening applied first (more accurate) |
| `kmeans` | Cluster all chunks into K groups, keep one representative per cluster |
| `mmr` | Iterative selection — balances coverage and diversity |

### MaxSim (retrieval side)

Pass `"use_maxsim": true` in your `/query` JSON body to enable MaxSim re-ranking at query time. This fetches 20 candidates from pgvector then re-ranks to the top 5 based on query coverage and diversity.

---

## curl Examples

### Upload a PDF (no pruning)
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/document.pdf"
```

### Upload with a pruning strategy
```bash
# K-Means
curl -X POST "http://localhost:8000/upload?pruning_strategy=kmeans" \
  -F "file=@/path/to/your/document.pdf"

# MMR
curl -X POST "http://localhost:8000/upload?pruning_strategy=mmr" \
  -F "file=@/path/to/your/document.pdf"

# Cosine
curl -X POST "http://localhost:8000/upload?pruning_strategy=cosine" \
  -F "file=@/path/to/your/document.pdf"

# Cosine + Whitening
curl -X POST "http://localhost:8000/upload?pruning_strategy=cosine_whitened" \
  -F "file=@/path/to/your/document.pdf"
```

### Check indexing status
```bash
curl http://localhost:8000/status
```

Example response:
```json
{
  "status": "done",
  "chunks": 42,
  "error": null,
  "mode": "rag",
  "pruning_strategy": "kmeans",
  "pruning_summary": {
    "total_chunks": 80,
    "chunks_kept": 42,
    "chunks_pruned": 38,
    "pruning_rate_pct": 47.5
  }
}
```

### Ask a question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main contributions of this paper?",
    "temperature": 0.7,
    "max_new_tokens": 100
  }'
```

### Ask a question with MaxSim re-ranking
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main contributions of this paper?",
    "temperature": 0.7,
    "max_new_tokens": 100,
    "use_maxsim": true
  }'
```

### Get the pruning report
```bash
curl http://localhost:8000/pruning-report
```

Returns the full breakdown — per-chunk scores, kept/pruned flags, threshold used, and summary stats. Only available after uploading with a non-`none` strategy.

Example response (truncated):
```json
{
  "strategy": "kmeans",
  "summary": {
    "total_chunks": 80,
    "chunks_kept": 42,
    "chunks_pruned": 38,
    "retention_rate_pct": 52.5,
    "pruning_rate_pct": 47.5,
    "storage_vectors_saved": 38,
    "estimated_storage_saved_pct": 47.5
  },
  "threshold": {
    "value": 0.123456,
    "description": "adaptive - derived from mean(scores) × multiplier"
  },
  "score_stats": {
    "min": 0.001,
    "max": 0.48,
    "mean": 0.21,
    "std": 0.09
  },
  "per_chunk_detail": [
    {
      "index": 0,
      "page_number": 1,
      "content_preview": "This paper presents an adaptive indexing strategy...",
      "score": 0.043,
      "kept": true,
      "pruned": false
    }
  ]
}
```

### Reset the database
```bash
curl http://localhost:8000/reset
```

---

## Troubleshooting
```

**Model takes forever / freezes**  
→ On MacBook, generation is slow (MPS). Normal range is 20–90s per query. If it hangs past 3 minutes, restart uvicorn and lower `max_new_tokens` in the request body.

**`bitsandbytes` won't install on Mac**  
→ Expected. Skip it. It's CUDA-only. The Mac path uses float16 without quantization.