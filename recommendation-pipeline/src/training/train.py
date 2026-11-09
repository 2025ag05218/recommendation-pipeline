"""Stage 6 - Model training with MLflow experiment tracking.

For each model: fit on the training split, persist to models/<name>.joblib and log
parameters, dataset info and the model artifact to an MLflow run. The run id is stored in
models/<name>.run.json so the evaluation stage can attach its metrics to the same run.
"""
from __future__ import annotations

import time

import joblib
import mlflow
import numpy as np

from src.common import ROOT, get_logger, get_path, logged_stage, write_json
from src.training.models import MODEL_REGISTRY, TrainData

log = get_logger("training")

MODEL_NAMES = list(MODEL_REGISTRY)


def setup_mlflow(cfg: dict) -> None:
    uri = cfg["models"]["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(ROOT / uri.replace("sqlite:///", ""))      # anchor DB at repo root
    mlflow.set_tracking_uri(uri)
    exp = cfg["models"]["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(exp)


@logged_stage("training")
def run(cfg: dict, model_name: str) -> dict:
    np.random.seed(cfg["seed"])
    setup_mlflow(cfg)
    data = TrainData.load(cfg)
    params = dict(cfg["models"].get(model_name, {}) or {})
    model = MODEL_REGISTRY[model_name](**params)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tags({"stage": "training", "model_family": "baseline" if model_name == "popularity" else "personalised"})
        mlflow.log_params({"model": model_name, "seed": cfg["seed"], **model.params,
                           "split_strategy": cfg["preparation"]["split"]["strategy"],
                           "positive_threshold": cfg["preparation"]["positive_threshold"]})
        t0 = time.time()
        model.fit(data)
        fit_seconds = round(time.time() - t0, 3)
        mlflow.log_metrics({"train_seconds": fit_seconds, "train_interactions": int(data.R.nnz),
                            "n_users": data.R.shape[0], "n_items": data.R.shape[1]})
        model_path = get_path(cfg, "models_dir", f"{model_name}.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        run_id = run.info.run_id

    write_json({"run_id": run_id, "model": model_name, "fit_seconds": fit_seconds},
               get_path(cfg, "models_dir", f"{model_name}.run.json"))
    return {"model": model_name, "run_id": run_id, "fit_seconds": fit_seconds}
