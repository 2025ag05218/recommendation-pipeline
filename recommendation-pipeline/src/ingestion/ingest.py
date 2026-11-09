"""Stage 1 - Data ingestion.

Loads user, item and interaction data from the MovieLens 100K dataset and writes them as
*raw, untyped* CSV files (every column is kept as text, like data arriving from an external
source). Optionally injects a small, seeded set of faults so the validation stage has real
defects to find - see `ingestion.inject_faults` in configs/config.yaml.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import ROOT, get_logger, get_path, logged_stage, write_json

log = get_logger("ingestion")


# --------------------------------------------------------------------------- loading
def _locate_dataset(cfg: dict) -> Path:
    """Use the bundled copy if present, otherwise download the official zip."""
    local = ROOT / cfg["dataset"]["local_dir"]
    if (local / "u.data").exists():
        log.info("Using local dataset at %s", local)
        return local
    url = cfg["dataset"]["download_url"]
    log.info("Local copy not found - downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted, configured URL)
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(local.parent)
    return local


def load_movielens_100k(src: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    genres = [g.split("|")[0] for g in (src / "u.genre").read_text(encoding="latin-1").splitlines() if g.strip()]

    users = pd.read_csv(
        src / "u.user", sep="|", header=None, dtype=str,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    raw_items = pd.read_csv(src / "u.item", sep="|", header=None, dtype=str, encoding="latin-1")
    flags = raw_items.iloc[:, 5:5 + len(genres)].astype(int).to_numpy()
    genre_str = [
        "|".join(g for g, f in zip(genres, row) if f == 1) or np.nan for row in flags
    ]
    items = pd.DataFrame({
        "item_id": raw_items.iloc[:, 0],
        "title": raw_items.iloc[:, 1],
        "release_date": raw_items.iloc[:, 2],
        "genres": genre_str,
    })
    interactions = pd.read_csv(
        src / "u.data", sep="\t", header=None, dtype=str,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return users, items, interactions


# --------------------------------------------------------------------------- fault injection
def inject_faults(users: pd.DataFrame, items: pd.DataFrame, inter: pd.DataFrame, cfg: dict, seed: int):
    """Corrupt a seeded random subset of rows. Returns corrupted copies and a manifest."""
    rng = np.random.default_rng(seed)
    f = cfg["ingestion"]["faults"]
    users, items, inter = users.copy(), items.copy(), inter.copy()
    manifest: dict[str, dict] = {"interactions": {}, "users": {}, "items": {}}

    def pick(df: pd.DataFrame, n: int, used: set) -> np.ndarray:
        pool = np.array([i for i in df.index if i not in used])
        chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
        used.update(chosen.tolist())
        return chosen

    # ---- interactions (modify distinct rows so each fault type is independent)
    used: set = set()
    ic = f["interactions"]
    idx = pick(inter, ic["invalid_rating_value"], used)
    inter.loc[idx, "rating"] = rng.choice(["0", "6", "7", "-1", "10"], size=len(idx))
    manifest["interactions"]["invalid_rating_value"] = len(idx)

    idx = pick(inter, ic["non_numeric_rating"], used)
    inter.loc[idx, "rating"] = rng.choice(["five", "N/A", "good", "4stars"], size=len(idx))
    manifest["interactions"]["non_numeric_rating"] = len(idx)

    idx = pick(inter, ic["missing_rating"], used)
    inter.loc[idx, "rating"] = np.nan
    manifest["interactions"]["missing_rating"] = len(idx)

    idx = pick(inter, ic["missing_user_id"], used)
    inter.loc[idx, "user_id"] = np.nan
    manifest["interactions"]["missing_user_id"] = len(idx)

    idx = pick(inter, ic["invalid_timestamp"], used)
    inter.loc[idx, "timestamp"] = rng.choice(["0", "4102444800", "not_a_timestamp", "-5"], size=len(idx))
    manifest["interactions"]["invalid_timestamp"] = len(idx)

    idx = pick(inter, ic["orphan_user_id"], used)
    inter.loc[idx, "user_id"] = rng.integers(5000, 9000, size=len(idx)).astype(str)
    manifest["interactions"]["orphan_user_id"] = len(idx)

    idx = pick(inter, ic["orphan_item_id"], used)
    inter.loc[idx, "item_id"] = rng.integers(5000, 9000, size=len(idx)).astype(str)
    manifest["interactions"]["orphan_item_id"] = len(idx)

    dup_idx = rng.choice(inter.index.difference(list(used)), size=ic["duplicate_rows"], replace=False)
    inter = pd.concat([inter, inter.loc[dup_idx]], ignore_index=True)
    manifest["interactions"]["duplicate_rows"] = len(dup_idx)

    # ---- users
    used = set()
    uc = f["users"]
    idx = pick(users, uc["invalid_age"], used)
    users.loc[idx, "age"] = rng.choice(["-3", "250", "0", "999"], size=len(idx))
    manifest["users"]["invalid_age"] = len(idx)
    idx = pick(users, uc["invalid_gender"], used)
    users.loc[idx, "gender"] = rng.choice(["X", "?", "unknown"], size=len(idx))
    manifest["users"]["invalid_gender"] = len(idx)
    idx = pick(users, uc["missing_occupation"], used)
    users.loc[idx, "occupation"] = np.nan
    manifest["users"]["missing_occupation"] = len(idx)
    dup_idx = rng.choice(users.index.difference(list(used)), size=uc["duplicate_rows"], replace=False)
    users = pd.concat([users, users.loc[dup_idx]], ignore_index=True)
    manifest["users"]["duplicate_rows"] = len(dup_idx)

    # ---- items
    used = set()
    tc = f["items"]
    idx = pick(items, tc["missing_title"], used)
    items.loc[idx, "title"] = np.nan
    manifest["items"]["missing_title"] = len(idx)
    idx = pick(items, tc["missing_genres"], used)
    items.loc[idx, "genres"] = np.nan
    manifest["items"]["missing_genres"] = len(idx)
    dup_idx = rng.choice(items.index.difference(list(used)), size=tc["duplicate_rows"], replace=False)
    items = pd.concat([items, items.loc[dup_idx]], ignore_index=True)
    manifest["items"]["duplicate_rows"] = len(dup_idx)

    return users, items, inter, manifest


# --------------------------------------------------------------------------- stage entrypoint
@logged_stage("ingestion")
def run(cfg: dict) -> dict:
    src = _locate_dataset(cfg)
    users, items, inter = load_movielens_100k(src)
    log.info("Loaded %d users, %d items, %d interactions", len(users), len(items), len(inter))

    manifest = {}
    if cfg["ingestion"]["inject_faults"]:
        users, items, inter, manifest = inject_faults(users, items, inter, cfg, cfg["seed"])
        log.info("Injected faults (seed=%s): %s", cfg["seed"], manifest)

    out = get_path(cfg, "raw_dir", "csv", "x.csv").parent
    users.to_csv(out / "users.csv", index=False)
    items.to_csv(out / "items.csv", index=False)
    inter.to_csv(out / "interactions.csv", index=False)
    summary = {
        "dataset": cfg["dataset"]["name"],
        "rows": {"users": len(users), "items": len(items), "interactions": len(inter)},
        "fault_injection": manifest or "disabled",
    }
    write_json(summary, get_path(cfg, "reports_dir", "ingestion_summary.json"))
    return summary
