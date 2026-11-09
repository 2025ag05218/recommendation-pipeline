"""Stage 4 - Data transformation.

Builds the derived datasets that the feature and model stages consume. Everything here is
computed from the TRAIN split only, so no information from the test period leaks in.

  * user_activity_summary.parquet - one row per user
  * item_popularity.parquet       - one row per item
  * interaction_matrix.npz        - sparse user x item rating matrix (CSR)
  * matrix_index.json             - row/column -> user_id / item_id mapping
"""
from __future__ import annotations

import pandas as pd
import scipy.sparse as sp

from src.common import get_logger, get_path, logged_stage, write_json

log = get_logger("transformation")


def user_activity_summary(train: pd.DataFrame, thr: float) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = g.agg(
        n_interactions=("item_id", "size"),
        n_positive=("rating", lambda r: int((r >= thr).sum())),
        avg_rating=("rating", "mean"),
        first_interaction=("timestamp", "min"),
        last_interaction=("timestamp", "max"),
    ).reset_index()
    out["active_days"] = (out["last_interaction"] - out["first_interaction"]).dt.days + 1
    return out


def item_popularity(train: pd.DataFrame, thr: float) -> pd.DataFrame:
    g = train.groupby("item_id")
    out = g.agg(
        n_interactions=("user_id", "size"),
        n_positive=("rating", lambda r: int((r >= thr).sum())),
        avg_rating=("rating", "mean"),
        last_interaction=("timestamp", "max"),
    ).reset_index()
    out["popularity_rank"] = out["n_interactions"].rank(method="min", ascending=False).astype(int)
    return out.sort_values("popularity_rank").reset_index(drop=True)


def build_interaction_matrix(train: pd.DataFrame, user_ids: list[int], item_ids: list[int]) -> sp.csr_matrix:
    u_map = {u: i for i, u in enumerate(user_ids)}
    i_map = {i: j for j, i in enumerate(item_ids)}
    rows = train["user_id"].map(u_map).to_numpy()
    cols = train["item_id"].map(i_map).to_numpy()
    return sp.csr_matrix((train["rating"].to_numpy(dtype="float32"), (rows, cols)),
                         shape=(len(user_ids), len(item_ids)))


@logged_stage("transformation")
def run(cfg: dict) -> dict:
    thr = cfg["preparation"]["positive_threshold"]
    proc = get_path(cfg, "processed_dir", "x.parquet").parent
    train = pd.read_parquet(proc / "train.parquet")
    users = pd.read_parquet(proc / "users_clean.parquet")
    items = pd.read_parquet(proc / "items_clean.parquet")

    act = user_activity_summary(train, thr)
    pop = item_popularity(train, thr)
    user_ids, item_ids = users["user_id"].tolist(), items["item_id"].tolist()
    matrix = build_interaction_matrix(train, user_ids, item_ids)

    act.to_parquet(proc / "user_activity_summary.parquet", index=False)
    pop.to_parquet(proc / "item_popularity.parquet", index=False)
    sp.save_npz(proc / "interaction_matrix.npz", matrix)
    write_json({"user_ids": user_ids, "item_ids": item_ids}, proc / "matrix_index.json")

    summary = {"matrix_shape": list(matrix.shape), "nnz": int(matrix.nnz),
               "density_pct": round(100 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]), 3),
               "items_without_train_interactions": int(len(item_ids) - len(pop))}
    write_json(summary, get_path(cfg, "reports_dir", "transformation_summary.json"))
    return summary
