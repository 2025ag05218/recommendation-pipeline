"""Prefect orchestration of the recommendation pipeline.

Task graph (arrows = dependencies; the four training tasks run in parallel):

  ingest -> validate -> prepare -> transform -> build_features -+-> train[popularity]    -+
                                                                +-> train[item_cf]       -+-> evaluate
                                                                +-> train[content_based] -+
                                                                +-> train[hybrid]        -+

Run once:        python dags/recommendation_flow.py
Schedule daily:  python dags/recommendation_flow.py --serve      (cron below)
Prefect UI:      prefect server start     (then open http://127.0.0.1:4200 and run the flow)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, task  # noqa: E402

from src.common import load_config, reset_run_log  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.features import feature_store  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.validation import validate  # noqa: E402


@task(name="ingest", retries=1, retry_delay_seconds=5)
def ingest_task(cfg):
    return ingest.run(cfg)


@task(name="validate")
def validate_task(cfg):
    return validate.run(cfg)


@task(name="prepare")
def prepare_task(cfg):
    return prepare.run(cfg)


@task(name="transform")
def transform_task(cfg):
    return transform.run(cfg)


@task(name="build_features")
def features_task(cfg):
    return feature_store.run(cfg)


@task(name="train")
def train_task(cfg, model_name):
    return train.run(cfg, model_name)


@task(name="evaluate")
def evaluate_task(cfg):
    return evaluate.run(cfg)


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None):
    cfg = load_config(config_path)
    reset_run_log(cfg)
    a = ingest_task.submit(cfg)
    b = validate_task.submit(cfg, wait_for=[a])
    c = prepare_task.submit(cfg, wait_for=[b])
    d = transform_task.submit(cfg, wait_for=[c])
    e = features_task.submit(cfg, wait_for=[d])
    trained = [train_task.with_options(name=f"train[{m}]").submit(cfg, m, wait_for=[e]) for m in train.MODEL_NAMES]
    result = evaluate_task.submit(cfg, wait_for=trained).result()
    print("Pipeline finished. Results table: reports/results.md")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="register a daily 02:00 schedule and serve the flow")
    args = ap.parse_args()
    if args.serve:
        recommendation_pipeline.serve(name="daily-recsys", cron="0 2 * * *")
    else:
        recommendation_pipeline()
