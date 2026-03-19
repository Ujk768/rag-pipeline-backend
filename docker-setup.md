# Local RAG API — Docker Setup

A Retrieval-Augmented Generation (RAG) API that runs entirely locally using FastAPI, pgvector, and a Hugging Face LLM.

---

## Prerequisites

| Tool | Install |
|---|---|
| Docker Desktop | https://docs.docker.com/get-docker/ |
| Docker Compose | Included in Docker Desktop |
| Git | https://git-scm.com/ |

> **No GPU? No problem.** The app falls back to CPU automatically. First startup will be slow while the LLM loads, but it works.

---

## Quick Start

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in:
- `HF_TOKEN` — your Hugging Face access token (https://huggingface.co/settings/tokens)
- `MODEL_ID` — the model you want to use (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)
- `DB_PASSWORD` — any password you like

### 3. Build and start

```bash
docker compose up --build
```

The first build downloads Python packages and the spaCy model (~a few minutes). Subsequent starts are fast.

The API is ready when you see:
```
[INFO] All models loaded successfully!
```

> **Note:** The LLM itself is downloaded from Hugging Face on first use (can be several GB). It is cached in the container's pip cache between restarts.

---

## API Usage

### Upload a PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

### Check processing status

```bash
curl http://localhost:8000/status
```

Wait until `"status": "done"` before querying.

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the document?"}'
```

### Reset (clear document from DB and memory)

```bash
curl http://localhost:8000/reset
```

---

## Updating `main.py` (no rebuild needed)

The `main.py` file is **mounted directly** from your local folder into the container.  
After pulling the latest version from GitHub, just restart the container:

```bash
git pull
docker compose restart api
```

---

## GPU Support (Linux only)

1. Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Uncomment the `deploy:` block in `docker-compose.yml`
3. Rebuild: `docker compose up --build`

---

## File Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .env                  ← created by you (git-ignored)
├── main.py
└── requirements.txt
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on port 8000 | Wait for `All models loaded` — LLM load takes time |
| `No module named 'pgvector'` | Rebuild: `docker compose build --no-cache api` |
| DB auth errors | Check `.env` — `DB_USER` must not be `USER` (reserved on Linux) |
| Out of memory on CPU | Try a smaller model, e.g. `microsoft/phi-2` |