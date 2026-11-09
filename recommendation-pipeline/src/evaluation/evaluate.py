"""Stage 7 - Model evaluation.

Protocol
  * relevant items of a user = test items rated >= positive_threshold
  * candidates = every catalogue item the user did NOT interact with in the training period
  * metrics @K: Precision, Recall, NDCG (binary relevance), catalogue Coverage, intra-list Diversity
  * rating metrics RMSE / MAE over ALL test ratings, for models that predict ratings
Metrics are logged to the MLflow run created during training and written to reports/.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.common import get_logger, get_path, logged_stage, read_json, write_json
from src.evaluation import metrics as M
from src.training.models import MODEL_REGISTRY, TrainData
from src.training.train import setup_mlflow

log = get_logger("evaluation")


def top_k_unseen(scores: np.ndarray, seen: np.ndarray, k: int) -> np.ndarray:
    """Top-k item indices per row, excluding items already seen in training."""
    s = np.where(seen, -np.inf, scores)
    part = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def evaluate_model(model, data: TrainData, test: pd.DataFrame, k: int, thr: float) -> dict:
    u_map = {u: i for i, u in enumerate(data.user_ids)}
    i_map = {i: j for j, i in enumerate(data.item_ids)}
    t = test.assign(u=test["user_id"].map(u_map), i=test["item_id"].map(i_map)).dropna(subset=["u", "i"])
    t = t.astype({"u": int, "i": int})

    relevant = t[t["rating"] >= thr].groupby("u")["i"].apply(set).to_dict()
    eval_users = np.array(sorted(relevant))
    scores = model.score_matrix()[eval_users]
    seen = data.R[eval_users].toarray() > 0
    recs = top_k_unseen(scores, seen, k)

    p, r, n, d = [], [], [], []
    for row, u in zip(recs, eval_users):
        lst, rel = row.tolist(), relevant[u]
        p.append(M.precision_at_k(lst, rel, k))
        r.append(M.recall_at_k(lst, rel, k))
        n.append(M.ndcg_at_k(lst, rel, k))
        d.append(M.intra_list_diversity(lst, data.item_genres))
    out = {
        f"precision_at_{k}": float(np.mean(p)), f"recall_at_{k}": float(np.mean(r)),
        f"ndcg_at_{k}": float(np.mean(n)),
        "coverage": M.catalog_coverage(recs.tolist(), len(data.item_ids)),
        "diversity": float(np.mean(d)), "n_eval_users": int(len(eval_users)),
    }
    pred = model.predict_rating(t["u"].to_numpy(), t["i"].to_numpy())
    if pred is not None:
        out["rmse"] = M.rmse(t["rating"].to_numpy(), pred)
        out["mae"] = M.mae(t["rating"].to_numpy(), pred)
    return out


def _plot(df: pd.DataFrame, k: int, path) -> None:
    cols = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", "coverage", "diversity"]
    ax = df.set_index("model")[cols].plot(kind="bar", figsize=(9, 4.2), width=0.8)
    ax.set_title(f"Model comparison (K={k}) - MovieLens 100K, time-based split")
    ax.set_ylabel("score")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    ax.set_ylim(0, 0.9)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@logged_stage("evaluation")
def run(cfg: dict, model_names: list[str] | None = None) -> dict:
    k, thr = cfg["models"]["top_k"], cfg["preparation"]["positive_threshold"]
    setup_mlflow(cfg)
    data = TrainData.load(cfg)
    test = pd.read_parquet(get_path(cfg, "processed_dir", "test.parquet"))

    rows = []
    for name in model_names or list(MODEL_REGISTRY):
        model = joblib.load(get_path(cfg, "models_dir", f"{name}.joblib"))
        meta = read_json(get_path(cfg, "models_dir", f"{name}.run.json"))
        res = evaluate_model(model, data, test, k, thr)
        with mlflow.start_run(run_id=meta["run_id"]):
            mlflow.log_metrics({m: v for m, v in res.items() if m != "n_eval_users"})
            mlflow.log_param("eval_users", res["n_eval_users"])
        rows.append({"model": name, **res})
        log.info("%s -> %s", name, {m: round(v, 4) for m, v in res.items()})

    df = pd.DataFrame(rows)
    reports = get_path(cfg, "reports_dir", "x.csv").parent
    df.to_csv(reports / "results.csv", index=False)
    cols = [c for c in df.columns if c not in ("model", "n_eval_users")]
    lines = [f"# Model Evaluation Results (K = {k})", "",
             f"Evaluated on {int(df['n_eval_users'].iloc[0])} users with at least one relevant "
             f"(rating >= {thr}) item in the held-out, most-recent part of their history. "
             "Best value per column in **bold** (RMSE / MAE: lower is better, all others: higher is better).", "",
             "| Model | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    best = {c: (df[c].min() if c in ("rmse", "mae") else df[c].max()) for c in cols if df[c].notna().any()}
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            if pd.isna(r[c]):
                cells.append("n/a")
            else:
                s = f"{r[c]:.4f}"
                cells.append(f"**{s}**" if np.isclose(r[c], best[c]) else s)
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    (reports / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot(df, k, reports / "model_comparison.png")
    write_json(rows, reports / "results.json")
    return {"results": rows}
