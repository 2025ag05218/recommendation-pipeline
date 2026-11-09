"""Stage 5 - Feature management (a small, file-based feature layer).

Design
------
* One **registry** (`FEATURES`) documents every feature: name, entity, dtype, definition.
  The feature catalog in reports/feature_catalog.md is generated from it, so docs never drift.
* **One set of compute functions** (`compute_user_features`, `compute_item_features`,
  `compute_interaction_features`) is used by both the offline training pipeline and by the
  prediction / serving code in `src/inference/predict.py` -> no training/serving skew.
* Every feature is computed **point-in-time** relative to `as_of` (default: the last
  training timestamp) and only from training data, so test-period information cannot leak.
* `FeatureStore` persists the tables to data/features/*.parquet and serves lookups by id.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common import get_logger, get_path, logged_stage, write_json

log = get_logger("features")


# --------------------------------------------------------------------------- registry / catalog
@dataclass(frozen=True)
class Feature:
    name: str
    entity: str       # user | item | interaction
    dtype: str
    description: str
    logic: str


FEATURES: list[Feature] = [
    # --- user features
    Feature("user_activity_count", "user", "int", "Number of rated items in the training history", "count(interactions) per user"),
    Feature("user_avg_rating", "user", "float", "Mean rating given by the user", "mean(rating) per user"),
    Feature("user_rating_std", "user", "float", "Spread of the user's ratings (0 if one rating)", "std(rating) per user, NaN -> 0"),
    Feature("user_positive_ratio", "user", "float", "Share of the user's ratings that are likes (rating >= threshold)", "mean(rating >= 4)"),
    Feature("days_since_last_interaction", "user", "float", "Recency: days between the user's last interaction and `as_of`", "(as_of - max(timestamp)) in days"),
    Feature("user_active_span_days", "user", "int", "Days between first and last interaction", "max(ts) - min(ts) + 1"),
    Feature("preferred_genre", "user", "string", "Genre with the highest share in the user's liked items", "argmax(genre_pref_*)"),
    Feature("user_distinct_genres", "user", "int", "Number of genres the user has liked at least once", "count(genre_pref_* > 0)"),
    Feature("genre_pref_<genre>  (19 cols)", "user", "float", "Share of the user's liked items that belong to <genre> (rows sum to 1)", "L1-normalised P @ item_genre matrix"),
    # --- item features
    Feature("item_popularity", "item", "int", "Number of training interactions with the item", "count(interactions) per item"),
    Feature("item_popularity_pct", "item", "float", "Popularity percentile (1.0 = most popular)", "rank(item_popularity) / n_items"),
    Feature("item_avg_rating", "item", "float", "Raw mean rating (global mean if unseen)", "mean(rating) per item"),
    Feature("item_bayes_rating", "item", "float", "Shrunk mean rating, robust for items with few ratings", "(sum + m*mu) / (n + m)"),
    Feature("item_positive_ratio", "item", "float", "Share of the item's ratings that are likes", "mean(rating >= 4)"),
    Feature("item_age_years", "item", "int", "Age of the item at `as_of`", "year(as_of) - release_year"),
    Feature("item_n_genres", "item", "int", "Number of genres attached to the item", "sum(genre_*)"),
    Feature("genre_<genre>  (19 cols)", "item", "int(0/1)", "Multi-hot genre indicator", "split(genres, '|')"),
    # --- interaction features
    Feature("rating_minus_user_mean", "interaction", "float", "How much the rating deviates from the user's own average", "rating - user_avg_rating"),
    Feature("rating_minus_item_mean", "interaction", "float", "How much the rating deviates from the item's average", "rating - item_avg_rating"),
    Feature("genre_affinity", "interaction", "float", "Cosine similarity between user genre profile and item genres", "cos(genre_pref_*, genre_*)"),
    Feature("interaction_age_days", "interaction", "float", "Age of the interaction at `as_of`", "(as_of - timestamp) in days"),
    Feature("time_decay_weight", "interaction", "float", "Exponential recency weight for training", "0.5 ** (age_days / half_life)"),
]


def write_feature_catalog(cfg: dict, genre_cols: list[str]) -> None:
    lines = ["# Feature Catalog", "",
             "Generated from `src/features/feature_store.py::FEATURES`. All features are computed from the "
             "training split only, point-in-time at `as_of` (= last training timestamp).", "",
             f"Genres used for the `genre_*` columns: {', '.join(genre_cols)}.", "",
             "| Feature | Entity | Type | Description | Computation |", "|---|---|---|---|---|"]
    for f in FEATURES:
        lines.append(f"| `{f.name}` | {f.entity} | {f.dtype} | {f.description} | {f.logic} |")
    lines += ["", "## Storage", "",
              "| Table | Path | Key |", "|---|---|---|",
              "| User features | `data/features/user_features.parquet` | `user_id` |",
              "| Item features | `data/features/item_features.parquet` | `item_id` |",
              "| Interaction features | `data/features/interaction_features.parquet` | `(user_id, item_id)` |",
              "", "## Reuse for training and prediction", "",
              "`FeatureStore.load()` returns the stored tables; `compute_user_features()` can be called on a "
              "fresh interaction history to produce identical features at prediction time "
              "(see `src/inference/predict.py`)."]
    get_path(cfg, "reports_dir", "feature_catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- helpers
def genre_matrix(items: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Multi-hot item x genre frame (columns genre_<name>), indexed by item_id."""
    split = items["genres"].str.split("|")
    all_genres = sorted({g for gs in split for g in gs})
    data = np.array([[1 if g in gs else 0 for g in all_genres] for gs in split], dtype="int8")
    cols = [f"genre_{g}" for g in all_genres]
    return pd.DataFrame(data, columns=cols, index=items["item_id"].to_numpy()), cols


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return np.where((na * nb) > 0, (a * b).sum(axis=1) / np.where(na * nb == 0, 1, na * nb), 0.0)


# --------------------------------------------------------------------------- compute functions (shared by train & serve)
def compute_user_features(history: pd.DataFrame, items: pd.DataFrame, as_of: pd.Timestamp, thr: float) -> pd.DataFrame:
    """User features from an interaction history (user_id, item_id, rating, timestamp)."""
    gm, gcols = genre_matrix(items)
    g = history.groupby("user_id")
    feats = g.agg(
        user_activity_count=("item_id", "size"),
        user_avg_rating=("rating", "mean"),
        user_rating_std=("rating", "std"),
        user_positive_ratio=("rating", lambda r: float((r >= thr).mean())),
        last_ts=("timestamp", "max"),
        first_ts=("timestamp", "min"),
    )
    feats["user_rating_std"] = feats["user_rating_std"].fillna(0.0)
    feats["days_since_last_interaction"] = (as_of - feats["last_ts"]).dt.total_seconds() / 86400
    feats["user_active_span_days"] = (feats["last_ts"] - feats["first_ts"]).dt.days + 1

    # genre preference = distribution of genres over the user's LIKED items (fallback: all items)
    liked = history[history["rating"] >= thr]
    use = liked.groupby("user_id").size().reindex(feats.index, fill_value=0) > 0
    src = pd.concat([liked, history[~history["user_id"].isin(use[use].index)]])
    counts = src.join(gm, on="item_id").groupby("user_id")[gm.columns.tolist()].sum()
    counts = counts.reindex(feats.index).fillna(0.0)
    share = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    share.columns = [c.replace("genre_", "genre_pref_") for c in share.columns]

    feats["preferred_genre"] = share.idxmax(axis=1).str.replace("genre_pref_", "", regex=False)
    feats.loc[share.sum(axis=1) == 0, "preferred_genre"] = "none"
    feats["user_distinct_genres"] = (counts > 0).sum(axis=1).astype(int)
    out = feats.drop(columns=["last_ts", "first_ts"]).join(share)
    return out.reset_index()


def compute_item_features(history: pd.DataFrame, items: pd.DataFrame, as_of: pd.Timestamp, thr: float, m: float) -> pd.DataFrame:
    gm, gcols = genre_matrix(items)
    mu = float(history["rating"].mean())
    g = history.groupby("item_id")
    st = g.agg(item_popularity=("user_id", "size"), rating_sum=("rating", "sum"),
               item_avg_rating=("rating", "mean"),
               item_positive_ratio=("rating", lambda r: float((r >= thr).mean())))
    out = items[["item_id", "release_year"]].set_index("item_id").join(st)
    out["item_popularity"] = out["item_popularity"].fillna(0).astype(int)
    out["rating_sum"] = out["rating_sum"].fillna(0.0)
    out["item_bayes_rating"] = (out["rating_sum"] + m * mu) / (out["item_popularity"] + m)
    out["item_avg_rating"] = out["item_avg_rating"].fillna(mu)
    out["item_positive_ratio"] = out["item_positive_ratio"].fillna(0.0)
    out["item_popularity_pct"] = out["item_popularity"].rank(method="average") / len(out)
    out["item_age_years"] = (as_of.year - out["release_year"]).clip(lower=0).astype(int)
    out["item_n_genres"] = gm.sum(axis=1)
    out = out.drop(columns=["release_year", "rating_sum"]).join(gm)
    return out.reset_index(names="item_id")


def compute_interaction_features(history: pd.DataFrame, user_f: pd.DataFrame, item_f: pd.DataFrame,
                                 as_of: pd.Timestamp, half_life: float) -> pd.DataFrame:
    gcols = [c for c in item_f.columns if c.startswith("genre_") and not c.startswith("genre_pref_")]
    pcols = [c.replace("genre_", "genre_pref_") for c in gcols]
    uf = user_f.set_index("user_id")
    itf = item_f.set_index("item_id")
    df = history[["user_id", "item_id", "rating", "timestamp"]].copy()
    df["rating_minus_user_mean"] = df["rating"] - df["user_id"].map(uf["user_avg_rating"])
    df["rating_minus_item_mean"] = df["rating"] - df["item_id"].map(itf["item_avg_rating"])
    df["genre_affinity"] = _cosine_rows(uf.loc[df["user_id"], pcols].to_numpy(float),
                                        itf.loc[df["item_id"], gcols].to_numpy(float))
    df["interaction_age_days"] = (as_of - df["timestamp"]).dt.total_seconds() / 86400
    df["time_decay_weight"] = 0.5 ** (df["interaction_age_days"] / half_life)
    return df.drop(columns=["rating", "timestamp"]).reset_index(drop=True)


# --------------------------------------------------------------------------- store
class FeatureStore:
    """Tiny parquet-backed feature store with lookup by entity id."""

    def __init__(self, user_features: pd.DataFrame, item_features: pd.DataFrame, interaction_features: pd.DataFrame | None = None):
        self.user_features, self.item_features = user_features, item_features
        self.interaction_features = interaction_features

    @classmethod
    def load(cls, cfg: dict) -> "FeatureStore":
        d = get_path(cfg, "features_dir", "x.parquet").parent
        inter = d / "interaction_features.parquet"
        return cls(pd.read_parquet(d / "user_features.parquet"), pd.read_parquet(d / "item_features.parquet"),
                   pd.read_parquet(inter) if inter.exists() else None)

    def save(self, cfg: dict) -> None:
        d = get_path(cfg, "features_dir", "x.parquet").parent
        self.user_features.to_parquet(d / "user_features.parquet", index=False)
        self.item_features.to_parquet(d / "item_features.parquet", index=False)
        if self.interaction_features is not None:
            self.interaction_features.to_parquet(d / "interaction_features.parquet", index=False)

    def get_user_features(self, user_ids) -> pd.DataFrame:
        return self.user_features.set_index("user_id").reindex(list(user_ids)).reset_index()

    def get_item_features(self, item_ids) -> pd.DataFrame:
        return self.item_features.set_index("item_id").reindex(list(item_ids)).reset_index()


@logged_stage("features")
def run(cfg: dict) -> dict:
    fc, thr = cfg["features"], cfg["preparation"]["positive_threshold"]
    proc = get_path(cfg, "processed_dir", "x.parquet").parent
    train = pd.read_parquet(proc / "train.parquet")
    items = pd.read_parquet(proc / "items_clean.parquet")
    users = pd.read_parquet(proc / "users_clean.parquet")

    as_of = pd.Timestamp(fc["as_of"], tz="UTC") if fc["as_of"] else train["timestamp"].max()
    user_f = compute_user_features(train, items, as_of, thr)
    # add users that have no training history (kept in the table with defaults)
    user_f = users[["user_id"]].merge(user_f, on="user_id", how="left")
    item_f = compute_item_features(train, items, as_of, thr, fc["bayes_prior_strength"])
    inter_f = compute_interaction_features(train, user_f.dropna(subset=["user_avg_rating"]), item_f, as_of,
                                           fc["time_decay_half_life_days"])
    FeatureStore(user_f, item_f, inter_f).save(cfg)
    write_feature_catalog(cfg, [c.replace("genre_", "") for c in item_f.columns if c.startswith("genre_")])

    summary = {"as_of": str(as_of), "user_features": user_f.shape, "item_features": item_f.shape,
               "interaction_features": inter_f.shape}
    write_json(summary, get_path(cfg, "reports_dir", "features_summary.json"))
    return summary
