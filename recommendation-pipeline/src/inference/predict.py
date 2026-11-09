"""Prediction / serving: top-N recommendations for a user, re-using the SAME feature code as training.

    python -m src.inference.predict --user 42 --k 10 --model hybrid
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from src.common import get_path, load_config
from src.features.feature_store import FeatureStore
from src.training.models import TrainData


def recommend_for_user(cfg: dict, user_id: int, k: int = 10, model_name: str = "hybrid") -> pd.DataFrame:
    data = TrainData.load(cfg)
    store = FeatureStore.load(cfg)
    items = pd.read_parquet(get_path(cfg, "processed_dir", "items_clean.parquet"))
    model = joblib.load(get_path(cfg, "models_dir", f"{model_name}.joblib"))

    u = data.user_ids.index(user_id)
    scores = model.score_matrix()[u].copy()
    scores[data.R[u].toarray().ravel() > 0] = -np.inf                       # never re-recommend seen items
    top = np.argsort(-scores)[:k]
    rec = pd.DataFrame({"rank": np.arange(1, k + 1), "item_id": [data.item_ids[i] for i in top],
                        "score": scores[top]})
    rec = rec.merge(items[["item_id", "title", "genres"]], on="item_id")
    rec = rec.merge(store.get_item_features(rec["item_id"])[["item_id", "item_popularity", "item_bayes_rating"]], on="item_id")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default="hybrid", choices=["popularity", "item_cf", "content_based", "hybrid"])
    a = ap.parse_args()
    cfg = load_config()
    uf = FeatureStore.load(cfg).get_user_features([a.user]).iloc[0]
    print(f"User {a.user}: {int(uf['user_activity_count'])} ratings, avg {uf['user_avg_rating']:.2f}, "
          f"preferred genre = {uf['preferred_genre']}")
    with pd.option_context("display.max_colwidth", 45, "display.width", 160):
        print(recommend_for_user(cfg, a.user, a.k, a.model).to_string(index=False))


if __name__ == "__main__":
    main()
