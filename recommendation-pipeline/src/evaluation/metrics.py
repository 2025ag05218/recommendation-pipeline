"""Ranking and rating metrics (pure numpy, unit-tested in tests/)."""
from __future__ import annotations

import numpy as np


def precision_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG@k."""
    gains = np.array([1.0 if r in relevant else 0.0 for r in recs[:k]])
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal = 1.0 / np.log2(np.arange(2, min(len(relevant), k) + 2))
    idcg = float(ideal.sum())
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(all_recs: list[list[int]], n_items: int) -> float:
    return len({i for recs in all_recs for i in recs}) / n_items


def intra_list_diversity(recs: list[int], item_vectors: np.ndarray) -> float:
    """Mean pairwise (1 - cosine similarity) of the recommended items' genre vectors."""
    if len(recs) < 2:
        return 0.0
    v = item_vectors[recs]
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    vn = v / np.maximum(norm, 1e-9)
    sim = vn @ vn.T
    n = len(recs)
    off_diag = sim[~np.eye(n, dtype=bool)]
    return float(1.0 - off_diag.mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
