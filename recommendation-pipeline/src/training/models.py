"""Recommender models. All share one interface so the evaluator treats them identically.

    fit(data)                    learn from a `TrainData` bundle
    score_matrix()               (n_users x n_items) relevance scores, higher = better
    predict_rating(u_idx, i_idx) optional explicit-rating prediction (None if unsupported)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common import get_path, read_json
from src.features.feature_store import FeatureStore, genre_matrix


@dataclass
class TrainData:
    R: sp.csr_matrix                 # user x item ratings (train only)
    P: sp.csr_matrix                 # user x item binary "liked" matrix (rating >= threshold)
    user_ids: list[int]
    item_ids: list[int]
    user_pref: np.ndarray            # user x genre preference shares  (from the feature store)
    item_genres: np.ndarray          # item x genre multi-hot          (from the feature store)
    item_pop: np.ndarray             # item training popularity        (from the feature store)
    item_bayes: np.ndarray           # smoothed item mean rating       (from the feature store)
    global_mean: float

    @classmethod
    def load(cls, cfg: dict) -> "TrainData":
        proc = get_path(cfg, "processed_dir", "x.parquet").parent
        thr = cfg["preparation"]["positive_threshold"]
        R = sp.load_npz(proc / "interaction_matrix.npz").tocsr()
        idx = read_json(proc / "matrix_index.json")
        fs = FeatureStore.load(cfg)
        uf = fs.get_user_features(idx["user_ids"])
        itf = fs.get_item_features(idx["item_ids"])
        pcols = [c for c in uf.columns if c.startswith("genre_pref_")]
        gcols = [c for c in itf.columns if c.startswith("genre_")]
        P = R.copy()
        P.data = (P.data >= thr).astype("float32")
        P.eliminate_zeros()
        return cls(R=R, P=P, user_ids=idx["user_ids"], item_ids=idx["item_ids"],
                   user_pref=uf[pcols].fillna(0).to_numpy("float32"),
                   item_genres=itf[gcols].to_numpy("float32"),
                   item_pop=itf["item_popularity"].to_numpy("float32"),
                   item_bayes=itf["item_bayes_rating"].to_numpy("float32"),
                   global_mean=float(R.data.mean()))


def _row_max_scale(m: np.ndarray) -> np.ndarray:
    mx = m.max(axis=1, keepdims=True)
    return np.divide(m, mx, out=np.zeros_like(m), where=mx > 0)


class PopularityRecommender:
    """Baseline: recommend the items that most users liked, identically for everyone."""
    name = "popularity"

    def __init__(self, **_):
        self.params: dict = {}

    def fit(self, d: TrainData):
        self.n_users = d.R.shape[0]
        self.pop_score = np.asarray(d.P.sum(axis=0)).ravel().astype("float32")      # number of likes
        self.item_rating = d.item_bayes
        return self

    def score_matrix(self) -> np.ndarray:
        return np.tile(self.pop_score, (self.n_users, 1))

    def predict_rating(self, u_idx, i_idx):
        return np.clip(self.item_rating[i_idx], 1, 5)


class ItemCFRecommender:
    """Personalised: item-based collaborative filtering (shrunk cosine similarity).

    ranking score(u, i) = sum_{j liked by u} sim(i, j)
    rating  pred (u, i) = mu_i + sum_j sim(i,j) (r_uj - mu_j) / sum_j |sim(i,j)|  over the k most similar rated items
    """
    name = "item_cf"

    def __init__(self, similarity_shrinkage: float = 10, rating_neighbors: int = 30, **_):
        self.lam, self.k = similarity_shrinkage, rating_neighbors
        self.params = {"similarity_shrinkage": similarity_shrinkage, "rating_neighbors": rating_neighbors}

    @staticmethod
    def _shrunk_cosine(M: sp.csr_matrix, lam: float) -> np.ndarray:
        """Item-item cosine of the columns of M with significance shrinkage co / (co + lam)."""
        Mb = M.tocsc()
        dot = (Mb.T @ Mb).toarray().astype("float32")
        norms = np.sqrt(np.diag(dot))
        sim = dot / np.maximum(np.outer(norms, norms), 1e-9)
        co = (Mb.T.astype(bool).astype("float32") @ Mb.astype(bool).astype("float32")).toarray()
        sim = sim * (co / (co + lam))
        np.fill_diagonal(sim, 0.0)
        return sim

    def fit(self, d: TrainData):
        self.P = d.P
        self.S_rank = self._shrunk_cosine(d.P, self.lam)                            # implicit "liked" similarity
        # adjusted cosine for rating prediction: centre each rating on the user's mean
        R = d.R.tocsr().astype("float64")
        counts = np.diff(R.indptr)
        user_mean = np.divide(np.asarray(R.sum(axis=1)).ravel(), counts, out=np.zeros(R.shape[0]), where=counts > 0)
        Rc = R.copy()
        Rc.data = Rc.data - np.repeat(user_mean, counts)
        self.S_rate = self._shrunk_cosine(Rc.astype("float32"), self.lam)
        self.R = R
        self.item_mean = d.item_bayes.astype("float64")
        self.mu_items = d.item_bayes.astype("float64")
        return self

    def score_matrix(self) -> np.ndarray:
        return np.asarray(self.P @ self.S_rank, dtype="float32")

    def predict_rating(self, u_idx, i_idx):
        preds = np.empty(len(u_idx), dtype="float64")
        indptr, indices, data = self.R.indptr, self.R.indices, self.R.data
        for n, (u, i) in enumerate(zip(u_idx, i_idx)):
            cols = indices[indptr[u]:indptr[u + 1]]
            ratings = data[indptr[u]:indptr[u + 1]]
            sims = self.S_rate[i, cols]
            if len(cols) > self.k:
                top = np.argpartition(-np.abs(sims), self.k)[: self.k]
                cols, ratings, sims = cols[top], ratings[top], sims[top]
            denom = np.abs(sims).sum()
            base = self.mu_items[i]
            preds[n] = base if denom <= 1e-9 else base + (sims * (ratings - self.mu_items[cols])).sum() / denom
        return np.clip(preds, 1, 5)


class ContentBasedRecommender:
    """Personalised: cosine between the user's genre-preference profile and the item's genres.

    Uses the *features* (genre_pref_*, genre_*) from the feature store. A tiny popularity term
    breaks ties between items with identical genres.
    """
    name = "content_based"

    def __init__(self, **_):
        self.params: dict = {"tie_break": "popularity * 1e-6"}

    def fit(self, d: TrainData):
        U, G = d.user_pref, d.item_genres
        Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
        Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
        self._scores = (Un @ Gn.T).astype("float32") + 1e-6 * (d.item_pop / max(d.item_pop.max(), 1))
        return self

    def score_matrix(self) -> np.ndarray:
        return self._scores

    def predict_rating(self, u_idx, i_idx):
        return None


class HybridRecommender:
    """Personalised: weighted blend of item-CF and content scores (each scaled to [0, 1] per user)."""
    name = "hybrid"

    def __init__(self, cf_weight: float = 0.7, content_weight: float = 0.3,
                 similarity_shrinkage: float = 10, rating_neighbors: int = 30, **_):
        self.w_cf, self.w_ct = cf_weight, content_weight
        self.cf = ItemCFRecommender(similarity_shrinkage=similarity_shrinkage, rating_neighbors=rating_neighbors)
        self.content = ContentBasedRecommender()
        self.params = {"cf_weight": cf_weight, "content_weight": content_weight,
                       "similarity_shrinkage": similarity_shrinkage}

    def fit(self, d: TrainData):
        self.cf.fit(d)
        self.content.fit(d)
        return self

    def score_matrix(self) -> np.ndarray:
        return self.w_cf * _row_max_scale(self.cf.score_matrix()) + self.w_ct * _row_max_scale(
            np.clip(self.content.score_matrix(), 0, None))

    def predict_rating(self, u_idx, i_idx):
        return self.cf.predict_rating(u_idx, i_idx)           # rating head comes from the CF component


MODEL_REGISTRY = {
    PopularityRecommender.name: PopularityRecommender,
    ItemCFRecommender.name: ItemCFRecommender,
    ContentBasedRecommender.name: ContentBasedRecommender,
    HybridRecommender.name: HybridRecommender,
}
