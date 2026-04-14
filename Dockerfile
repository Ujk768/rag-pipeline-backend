# STAGE 1: Builder

FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ libpq-dev
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
# STAGE 2: Final Image

FROM python:3.11-slim
WORKDIR /app
# Only install runtime essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Copy only the installed packages from builder

COPY --from=builder /root/.local /root/.local
COPY main.py .
# Pre-download model (keep this as a layer)

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8080
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080", "--timeout", "120"]
