import numpy as np

# PRUNING IMPLEMENTATIONS
def compute_whitening_matrix(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, a_min=1e-8, a_max=None)
    W = (eigenvectors / np.sqrt(eigenvalues)).T
    return W, mean


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T


def prune_cosine(embeddings, chunks, threshold_multiplier=0.85):
    n = len(embeddings)
    centroid = embeddings.mean(axis=0, keepdims=True)
    scores = cosine_similarity_matrix(embeddings, centroid).flatten()
    threshold = float(scores.mean() * threshold_multiplier)
    kept_indices = [i for i, s in enumerate(scores) if s <= threshold] or [int(np.argmin(scores))]
    pruned_indices = [i for i in range(n) if i not in set(kept_indices)]
    return kept_indices, _build_pruning_stats("cosine", n, kept_indices, pruned_indices, scores.tolist(), threshold, chunks,
        {"threshold_multiplier": threshold_multiplier, "score_meaning": "cosine similarity vs centroid"})


def prune_cosine_whitened(embeddings, chunks, threshold_multiplier=0.85):
    n = len(embeddings)
    W, mean = compute_whitening_matrix(embeddings)
    whitened = (embeddings - mean) @ W.T
    centroid = whitened.mean(axis=0, keepdims=True)
    scores = cosine_similarity_matrix(whitened, centroid).flatten()
    threshold = float(scores.mean() * threshold_multiplier)
    kept_indices = [i for i, s in enumerate(scores) if s <= threshold] or [int(np.argmin(scores))]
    pruned_indices = [i for i in range(n) if i not in set(kept_indices)]
    return kept_indices, _build_pruning_stats("cosine_whitened", n, kept_indices, pruned_indices, scores.tolist(), threshold, chunks,
        {"threshold_multiplier": threshold_multiplier, "whitening_applied": True})


def prune_kmeans(embeddings, chunks, n_clusters=None):
    from sklearn.cluster import KMeans
    n = len(embeddings)
    if n_clusters is None:
        n_clusters = max(1, min(int(np.sqrt(n)), n))
    if n_clusters >= n:
        return list(range(n)), _build_pruning_stats("kmeans", n, list(range(n)), [], [0.0]*n, 0.0, chunks, {"n_clusters": n_clusters})
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=5, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    scores = [1.0 - float(np.dot(embeddings[i], centroids[labels[i]]) /
              (np.linalg.norm(embeddings[i]) * np.linalg.norm(centroids[labels[i]]) + 1e-8))
              for i in range(n)]
    kept_indices = sorted([min([i for i, lbl in enumerate(labels) if lbl == c], key=lambda i: scores[i])
                           for c in range(n_clusters)])
    pruned_indices = [i for i in range(n) if i not in set(kept_indices)]
    return kept_indices, _build_pruning_stats("kmeans", n, kept_indices, pruned_indices, scores, float(np.mean(scores)), chunks,
        {"n_clusters": n_clusters, "selection_rule": "one chunk per cluster"})


def prune_mmr(embeddings, chunks, target_k=None, lambda_param=0.5):
    n = len(embeddings)
    if target_k is None:
        target_k = max(1, min(int(np.sqrt(n)), n))
    if target_k >= n:
        return list(range(n)), _build_pruning_stats("mmr", n, list(range(n)), [], [1.0]*n, 0.0, chunks, {"target_k": target_k})
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms
    centroid = normed.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-8)
    relevance = normed @ centroid
    selected, remaining = [], list(range(n))
    while len(selected) < target_k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            sel_embs = normed[selected]
            best, best_score = remaining[0], -np.inf
            for i in remaining:
                score = lambda_param * relevance[i] - (1 - lambda_param) * float((normed[i] @ sel_embs.T).max())
                if score > best_score:
                    best_score, best = score, i
        selected.append(best)
        remaining.remove(best)
    kept_indices = sorted(selected)
    pruned_indices = [i for i in range(n) if i not in set(kept_indices)]
    return kept_indices, _build_pruning_stats("mmr", n, kept_indices, pruned_indices, relevance.tolist(),
        float(np.mean(relevance)), chunks, {"target_k": target_k, "lambda_param": lambda_param})


def maxsim_rerank(query_embedding, candidate_embeddings, top_k=5):
    n = len(candidate_embeddings)
    if top_k >= n:
        return list(range(n))
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    c_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
    relevance = c_norms @ q_norm
    selected, remaining = [], list(range(n))
    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            sel_embs = c_norms[selected]
            best, best_score = remaining[0], -np.inf
            for i in remaining:
                score = relevance[i] - 0.5 * float((c_norms[i] @ sel_embs.T).max())
                if score > best_score:
                    best_score, best = score, i
        selected.append(best)
        remaining.remove(best)
    return selected


def _build_pruning_stats(strategy, n_total, kept_indices, pruned_indices, scores, threshold, chunks, extra):
    kept_set = set(kept_indices)
    return {
        "strategy": strategy,
        "summary": {
            "total_chunks": n_total,
            "chunks_kept": len(kept_indices),
            "chunks_pruned": len(pruned_indices),
            "retention_rate_pct": round(100 * len(kept_indices) / n_total, 2),
            "pruning_rate_pct": round(100 * len(pruned_indices) / n_total, 2),
            "storage_vectors_saved": len(pruned_indices),
            "estimated_storage_saved_pct": round(100 * len(pruned_indices) / n_total, 2),
        },
        "threshold": {"value": round(threshold, 6), "description": "adaptive - derived from mean(scores) × multiplier"},
        "score_stats": {
            "min": round(min(scores), 6), "max": round(max(scores), 6),
            "mean": round(float(np.mean(scores)), 6), "std": round(float(np.std(scores)), 6),
        },
        "per_chunk_detail": [
            {"index": i, "page_number": c["page_number"],
             "content_preview": c["sentence_chunk"][:120] + ("..." if len(c["sentence_chunk"]) > 120 else ""),
             "score": round(scores[i], 6), "kept": i in kept_set, "pruned": i not in kept_set}
            for i, c in enumerate(chunks)
        ],
        "strategy_metadata": extra,
    }

