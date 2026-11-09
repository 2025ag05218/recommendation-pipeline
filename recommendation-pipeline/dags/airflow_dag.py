"""Optional Apache Airflow version of the same workflow (alternative to Prefect).

Place/symlink this file in your Airflow `dags/` folder and set REPO to the repository path.
Each task shells out to `run_pipeline.py --stage ...`, so Airflow and Prefect share identical code.
NOTE: written for Airflow 2.x; the Prefect flow in recommendation_flow.py is the tested orchestrator.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

REPO = os.environ.get("RECSYS_REPO", "/opt/airflow/recommendation-pipeline")
MODELS = ["popularity", "item_cf", "content_based", "hybrid"]

with DAG(
    dag_id="recommendation_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["recsys", "dm4ml"],
) as dag:
    def stage(name: str, model: str | None = None) -> BashOperator:
        cmd = f"cd {REPO} && python run_pipeline.py --stage {name}" + (f" --model {model}" if model else "")
        return BashOperator(task_id=name if not model else f"{name}_{model}", bash_command=cmd)

    ingest, validate, prepare = stage("ingestion"), stage("validation"), stage("preparation")
    transform, features = stage("transformation"), stage("features")
    train_tasks = [stage("training", m) for m in MODELS]
    evaluate = stage("evaluation")

    ingest >> validate >> prepare >> transform >> features >> train_tasks >> evaluate
