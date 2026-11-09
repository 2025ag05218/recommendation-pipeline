"""Plain-Python entry point (no orchestrator needed).

    python run_pipeline.py                 # run every stage in dependency order
    python run_pipeline.py --stage features   # run a single stage (used by the Airflow DAG)
    python run_pipeline.py --stage training --model item_cf
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from src.common import load_config, reset_run_log
from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

STAGE_ORDER = ["ingestion", "validation", "preparation", "transformation", "features", "training", "evaluation"]


def run_stage(cfg: dict, stage: str, model: str | None = None):
    if stage == "ingestion":
        return ingest.run(cfg)
    if stage == "validation":
        return validate.run(cfg)
    if stage == "preparation":
        return prepare.run(cfg)
    if stage == "transformation":
        return transform.run(cfg)
    if stage == "features":
        return feature_store.run(cfg)
    if stage == "training":
        names = [model] if model else train.MODEL_NAMES
        with ThreadPoolExecutor(max_workers=1) as pool:      # sequential: keeps the SQLite MLflow store simple
            return list(pool.map(lambda m: train.run(cfg, m), names))
    if stage == "evaluation":
        return evaluate.run(cfg, [model] if model else None)
    raise ValueError(f"unknown stage {stage}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", choices=STAGE_ORDER, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage:
        run_stage(cfg, args.stage, args.model)
        return
    reset_run_log(cfg)
    for stage in STAGE_ORDER:
        run_stage(cfg, stage)


if __name__ == "__main__":
    main()
