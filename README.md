# Pruning Strategies for Vector Search

This document summarizes the pruning strategies used to improve efficiency in vector similarity search systems. These methods reduce the number of candidate vectors that need to be fully evaluated while preserving high retrieval accuracy.

---

## 1. Cosine Similarity Pruning

Cosine Similarity Pruning removes vectors that are unlikely to be similar to the query based on a similarity threshold.

### Idea
Instead of computing similarity for every vector in the dataset, we compute the cosine similarity between the query vector and candidate vectors. If the similarity is below a predefined threshold, the candidate is discarded early.

### Cosine Similarity Formula

\[
\text{cosine\_similarity}(q, v) =
\frac{q \cdot v}{\|q\|\|v\|}
\]

Where:
- `q` = query vector  
- `v` = candidate vector  
- `q · v` = dot product  
- `||q||`, `||v||` = vector magnitudes

### Process
1. Normalize vectors if required.
2. Compute cosine similarity with the query.
3. Discard vectors below a similarity threshold.
4. Rank remaining vectors.

### Benefits
- Reduces unnecessary similarity computations
- Works well with high-dimensional embeddings
- Simple and efficient

---

## 2. MaxSim Pruning

MaxSim Pruning eliminates candidates whose **maximum possible similarity** with the query cannot exceed the current best results.

### Idea
During search, we maintain the top-k results. For any candidate vector, we estimate the **maximum similarity it could achieve**. If that maximum value is lower than the current k-th best score, the candidate can be safely pruned.

### Process
1. Maintain a running list of the current **top-k similarities**.
2. Estimate an **upper bound similarity (MaxSim)** for each candidate.
3. If:
4. The candidate is pruned and skipped.

### Benefits
- Reduces full similarity computations
- Effective in approximate nearest neighbor search
- Improves runtime for large datasets

---

## 3. Cosine + Whitening Pruning

This approach combines **cosine similarity pruning with whitening transformation** to improve similarity accuracy before pruning.

### Idea
Vector embeddings often contain correlated dimensions. **Whitening** decorrelates the dimensions and normalizes variance, producing a more uniform vector space. Cosine similarity pruning is then applied on the transformed vectors.

### Whitening Transformation

Whitening transforms a vector as:


Where:
- `v` = original embedding
- `μ` = mean vector
- `W` = whitening matrix (often derived from PCA covariance)
- `v'` = whitened vector

### Process
1. Compute dataset mean and covariance.
2. Apply whitening transformation to all vectors.
3. Transform the query vector using the same parameters.
4. Apply cosine similarity pruning on the whitened vectors.

### Benefits
- Reduces embedding dimension correlation
- Improves similarity consistency
- Enables more reliable pruning decisions

---

## Summary

| Method | Key Idea | Advantage |
|------|------|------|
| Cosine Similarity Pruning | Remove vectors below similarity threshold | Fast filtering |
| MaxSim Pruning | Remove vectors whose max possible similarity is too low | Reduces expensive comparisons |
| Cosine + Whitening Pruning | Decorrelate embeddings before similarity pruning | Improves similarity quality |

---

These pruning techniques help scale vector search systems by reducing computational overhead while maintaining high retrieval performance.
