"""Stage 3 - Data preparation.

* repairs the WARN-level issues left by validation (impute / standardise)
* removes duplicate (user, item) interactions (keeps the latest)
* drops users with too little history
* creates the **time-based train/test split**

Outputs (data/processed/): users_clean, items_clean, interactions_clean, train, test (parquet)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import get_logger, get_path, logged_stage, write_json

log = get_logger("preparation")


def clean_users(df: pd.DataFrame, v: dict) -> pd.DataFrame:
    out = pd.DataFrame({"user_id": pd.to_numeric(df["user_id"]).astype("int64")})
    age = pd.to_numeric(df["age"], errors="coerce")
    age = age.where((age >= v["age_min"]) & (age <= v["age_max"]))
    out["age"] = age.fillna(age.median()).astype("float64")                       # impute with median
    gender = df["gender"].str.strip().str.upper()
    out["gender"] = gender.where(gender.isin(v["allowed_genders"]), "U")           # U = unknown
    out["occupation"] = df["occupation"].fillna("unknown").str.strip().str.lower()  # standardise text
    out["zip_code"] = df["zip_code"].fillna("unknown").astype(str).str.strip().str.upper()
    return out.sort_values("user_id").reset_index(drop=True)


def clean_items(df: pd.DataFrame, v: dict) -> pd.DataFrame:
    out = pd.DataFrame({"item_id": pd.to_numeric(df["item_id"]).astype("int64")})
    out["title"] = df["title"].fillna("unknown title").str.strip()
    year = pd.to_datetime(df["release_date"], format="%d-%b-%Y", errors="coerce").dt.year.astype("float64")
    year = year.where((year >= v["release_year_min"]) & (year <= v["release_year_max"]))
    out["release_year"] = year.fillna(year.median()).astype("int64")                # impute with median
    genres = df["genres"].fillna("unknown").str.strip().str.lower()
    out["genres"] = genres.str.replace(r"\s+", "_", regex=True)                     # pipe-separated, lower-case
    return out.sort_values("item_id").reset_index(drop=True)


def clean_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df[["user_id", "item_id", "rating", "timestamp"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    out = out.sort_values(["user_id", "item_id", "timestamp"])
    before = len(out)
    out = out.drop_duplicates(subset=["user_id", "item_id"], keep="last")           # keep latest rating
    return out.reset_index(drop=True), before - len(out)


def time_based_split(df: pd.DataFrame, split_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_cfg["strategy"] == "global":
        cutoff = df["timestamp"].quantile(split_cfg["global_cutoff_quantile"])
        return df[df["timestamp"] <= cutoff].copy(), df[df["timestamp"] > cutoff].copy()

    # per_user: the most recent `test_fraction` of every user's history is the test set
    df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    rank = df.groupby("user_id").cumcount()
    n = df.groupby("user_id")["item_id"].transform("size")
    n_test = np.maximum(1, np.floor(n * split_cfg["test_fraction"])).astype(int)
    is_test = rank >= (n - n_test)
    return df[~is_test].copy(), df[is_test].copy()


@logged_stage("preparation")
def run(cfg: dict) -> dict:
    v, p = cfg["validation"], cfg["preparation"]
    vdir = get_path(cfg, "processed_dir", "validated", "x.parquet").parent
    users = clean_users(pd.read_parquet(vdir / "users.parquet"), v)
    items = clean_items(pd.read_parquet(vdir / "items.parquet"), v)
    inter, n_dupe_pairs = clean_interactions(pd.read_parquet(vdir / "interactions.parquet"))

    counts = inter.groupby("user_id")["item_id"].transform("size")
    n_sparse = int((counts < p["min_user_interactions"]).sum())
    inter = inter[counts >= p["min_user_interactions"]].reset_index(drop=True)
    users = users[users["user_id"].isin(inter["user_id"])].reset_index(drop=True)

    train, test = time_based_split(inter, p["split"])
    # keep only test rows whose user and item were seen in training (standard offline protocol)
    test = test[test["user_id"].isin(train["user_id"])].reset_index(drop=True)
    train = train.reset_index(drop=True)

    out = get_path(cfg, "processed_dir", "x.parquet").parent
    users.to_parquet(out / "users_clean.parquet", index=False)
    items.to_parquet(out / "items_clean.parquet", index=False)
    inter.to_parquet(out / "interactions_clean.parquet", index=False)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)

    summary = {
        "users": len(users), "items": len(items), "interactions": len(inter),
        "duplicate_user_item_pairs_removed": n_dupe_pairs,
        "interactions_dropped_sparse_users": n_sparse,
        "split_strategy": p["split"]["strategy"],
        "train_rows": len(train), "test_rows": len(test),
        "train_time_range": [str(train["timestamp"].min()), str(train["timestamp"].max())],
        "test_time_range": [str(test["timestamp"].min()), str(test["timestamp"].max())],
        "sparsity_pct": round(100 * (1 - len(inter) / (len(users) * len(items))), 2),
    }
    write_json(summary, get_path(cfg, "reports_dir", "preparation_summary.json"))
    log.info("Split: %d train / %d test rows", len(train), len(test))
    return summary
