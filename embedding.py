import asyncio

import httpx
import numpy as np
import time
import os
import voyageai
from dotenv import load_dotenv


load_dotenv()

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")


# Warm Embedding Service on startup
def warm_embedding_service():
    try:
        with httpx.Client() as client:
            response = client.get(f"{EMBEDDING_SERVICE_URL}/health", timeout=10.0)
            if response.status_code == 200:
                print("[INFO] Embedding service is healthy.")
            else:
                print(f"[WARN] Embedding service health check failed with status code: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to embedding service during warm-up: {e}")    



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
                    timeout=120.0,
                )
                response.raise_for_status()
                return np.array(response.json()["embeddings"])
        except httpx.HTTPError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Embedding service failed after {max_retries} attempts: {e}")
            print(f"[WARN] Embedding service error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s

async def _embed_texts_vector_async(texts: list[str]) -> np.ndarray:
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    results = []
    for i in range(0, len(texts), 128):
        batch = texts[i:i+128]
        try:
            result = vo.embed(batch, model="voyage-4-lite", input_type="document")
            results.extend(result.embeddings)
        except Exception as e:
            raise RuntimeError(f"Voyage embedding failed: {e}")
    return np.array(results)

# def embedding_text(texts: list[str]) -> np.ndarray:
#     try:
#         return asyncio.run(_embed_texts_vector_async(texts))
#     except Exception as e:
#         print(f"[ERROR] Failed to get embeddings from Gemini API: {e}")
#         raise RuntimeError("Embedding service is currently unavailable.")

def embedding_text(texts: list[str]) -> np.ndarray:
    """Synchronous Voyage embedding — safe to call from both sync and async contexts."""
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    results = []
    for i in range(0, len(texts), 128):
        batch = texts[i:i + 128]
        try:
            result = vo.embed(batch, model="voyage-4-lite", input_type="document")
            results.extend(result.embeddings)
        except Exception as e:
            raise RuntimeError(f"Voyage embedding failed: {e}")
    return np.array(results)