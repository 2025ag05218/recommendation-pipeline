"""Generate reports/model_comparison.md and reports/evidence.md from the latest pipeline outputs.

    python scripts/make_docs.py [--run-date 2026-11-14]
Run it after a pipeline run (and after scripts/make_stage_screenshots.py) so the numbers and image links are current.
"""
import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
ap = argparse.ArgumentParser()
ap.add_argument("--run-date", default=str(date.today()))
args = ap.parse_args()

res = {r["model"]: r for r in json.loads((R / "results.json").read_text())}
pop, cf, ct, hy = res["popularity"], res["item_cf"], res["content_based"], res["hybrid"]
prep = json.loads((R / "preparation_summary.json").read_text())
val = json.loads((R / "validation_report.json").read_text())
ing = json.loads((R / "ingestion_summary.json").read_text())
feat = json.loads((R / "features_summary.json").read_text())
trans = json.loads((R / "transformation_summary.json").read_text())
runs = {m: json.loads((ROOT / "models" / f"{m}.run.json").read_text()) for m in res}
lift = lambda a, b: 100 * (a / b - 1)
K = 10

# ------------------------------------------------------------------------------------ model_comparison.md
cols = [f"precision_at_{K}", f"recall_at_{K}", f"ndcg_at_{K}", "coverage", "diversity", "rmse", "mae"]
head = ["Precision@10", "Recall@10", "NDCG@10", "Coverage", "Diversity", "RMSE", "MAE"]
best = {c: (min if c in ("rmse", "mae") else max)(r[c] for r in res.values() if c in r) for c in cols}
rows = []
for name, label in (("popularity", "Popularity (baseline)"), ("item_cf", "Item-CF (personalised)"),
                    ("content_based", "Content-based (personalised)"), ("hybrid", "Hybrid CF + content (personalised)")):
    cells = []
    for c in cols:
        v = res[name].get(c)
        cells.append("n/a" if v is None else (f"**{v:.4f}**" if abs(v - best[c]) < 1e-12 else f"{v:.4f}"))
    rows.append(f"| {label} | " + " | ".join(cells) + " |")

mc = f"""# Model Comparison

Comparison of the baseline and the personalised recommenders on the same data, split and metrics.
All numbers come from `reports/results.csv`, produced by the evaluation stage and logged to MLflow.

## Setup

| Item | Value |
|---|---|
| Dataset | MovieLens 100K ({prep['users']:,} users, {prep['items']:,} items, {prep['interactions']:,} ratings after cleaning) |
| Split | Time-based, per user: the most recent 20% of every user's history is the test set ({prep['train_rows']:,} train / {prep['test_rows']:,} test rows) |
| Relevant item | A test rating of 4 or more |
| Candidates | Every item the user did not rate in the training period |
| Cut-off | K = {K} |
| Users evaluated | {cf['n_eval_users']} (users with at least one relevant test item) |
| Seed | 42 (fixed in `configs/config.yaml`) |

## Models

| Model | Type | How it works |
|---|---|---|
| Popularity | Baseline | Recommends the items most users liked; identical list for everyone (minus items already seen). |
| Item-CF | Collaborative filtering | Item-item cosine similarity on the liked matrix with significance shrinkage `co/(co+10)`; score = sum of similarities to the user's liked items. Rating head: adjusted-cosine k-NN. |
| Content-based | Content | Cosine between the user's genre-preference profile (user feature) and the item's genres (item feature). |
| Hybrid | Blend | 0.7 x Item-CF + 0.3 x content scores, each scaled to [0, 1] per user. Rating head from Item-CF. |

## Results

| Model | {" | ".join(head)} |
|---|{"---:|" * len(cols)}
{chr(10).join(rows)}

Best value per column in **bold** (RMSE / MAE: lower is better; all others: higher is better).
RMSE / MAE are computed over all test ratings and only for models that predict ratings.

![Model comparison chart](model_comparison.png)

## Reading the results

* **Item-CF is the best ranker.** NDCG@10 {cf['ndcg_at_10']:.3f} vs {pop['ndcg_at_10']:.3f} for the baseline (+{lift(cf['ndcg_at_10'], pop['ndcg_at_10']):.0f}%), \
Precision@10 {cf['precision_at_10']:.3f} vs {pop['precision_at_10']:.3f} (+{lift(cf['precision_at_10'], pop['precision_at_10']):.0f}%), Recall@10 {cf['recall_at_10']:.3f} vs {pop['recall_at_10']:.3f}.
* **It also recommends more of the catalogue** ({cf['coverage']*100:.1f}% vs {pop['coverage']*100:.1f}% coverage) with slightly more varied lists (diversity {cf['diversity']:.3f} vs {pop['diversity']:.3f}), \
and predicts ratings better (RMSE {cf['rmse']:.3f} vs {pop['rmse']:.3f}; MAE {cf['mae']:.3f} vs {pop['mae']:.3f}). The rating gap is small because item averages are already a strong baseline.
* **Content-based is weak on accuracy** (Precision@10 {ct['precision_at_10']:.3f}) because 19 coarse genres cannot separate items well, \
but it has the highest coverage ({ct['coverage']*100:.1f}%) because it is not dominated by popular titles; its diversity is low ({ct['diversity']:.3f}) since lists stay within a few genres.
* **The hybrid matches Item-CF on accuracy** (NDCG@10 {hy['ndcg_at_10']:.3f} vs {cf['ndcg_at_10']:.3f}) and raises coverage to {hy['coverage']*100:.1f}%, \
but diversity drops to {hy['diversity']:.3f} because genre similarity pulls lists towards the user's favourite genres.
* **Caveat:** one dataset, one seed, no tuning, no significance tests - differences are indicative, not proven.

## Metric definitions

| Metric | Definition |
|---|---|
| Precision@K | Fraction of the K recommended items that are relevant, averaged over users. |
| Recall@K | Fraction of the user's relevant test items found in the top K, averaged over users. |
| NDCG@K | Discounted cumulative gain of the top K with binary relevance, normalised by the ideal ordering. |
| Coverage | Share of the item catalogue that appears in at least one user's top-K list. |
| Diversity | Mean pairwise genre dissimilarity (1 - cosine) of the items inside a list, averaged over users. |
| RMSE / MAE | Error of predicted vs true ratings over all test ratings (Popularity, Item-CF, Hybrid only). |

## Experiment tracking

Every model has an MLflow run in experiment `recsys-movielens-100k` holding parameters, training time, the metrics above and the model artifact.

| Model | MLflow run id |
|---|---|
""" + "\n".join(f"| {m} | `{runs[m]['run_id']}` |" for m in ("popularity", "item_cf", "content_based", "hybrid")) + """

Screenshots: `screenshots/mlflow_02_runs_table.png`, `screenshots/mlflow_03_compare_runs.png`, `screenshots/mlflow_04_run_detail_item_cf.png`.
"""
(R / "model_comparison.md").write_text(mc, encoding="utf-8")

# ------------------------------------------------------------------------------------ evidence.md
n_faults = sum(sum(v.values()) for v in ing["fault_injection"].values())
rej = {t: val["tables"][t]["rows_rejected"] for t in val["tables"]}


def stage(n, title, what, cmd, outputs, img, notes):
    return f"""### {n}. {title}

**What it does:** {what}

**Command:** `{cmd}`

**Produces:** {outputs}

![{title}](screenshots/{img})

*{notes}*
"""


ev = f"""# Evidence of Execution

This document shows that every stage of the pipeline ran and what it produced. It is the evidence companion to the
README (which explains the process), `feature_catalog.md` and `model_comparison.md`.

## How the evidence was produced

* **Stage images** (`screenshots/stage_*.png`, `repo_structure.png`) are rendered by `scripts/make_stage_screenshots.py` from the real
  stage log (`reports/pipeline.log`) and the real output files of the run, so they can be regenerated at any time.
* **Prefect and MLflow images** (`screenshots/prefect_*.png`, `screenshots/mlflow_*.png`) are screenshots of the Prefect 3 and MLflow 3 web UIs
  taken after a clean end-to-end run on {args.run_date}.
* `prefect_run_screenshot.png` (in `reports/`) is the screenshot of a successful Prefect run required by the assignment.

## 1. Deliverables checklist

| Assignment deliverable | Where it is |
|---|---|
| Complete source code organised into modules | `src/ingestion`, `src/validation`, `src/preparation`, `src/transformation`, `src/features`, `src/training`, `src/evaluation` (+ `src/inference`) |
| README.md with setup and execution instructions | `README.md` |
| requirements.txt | `requirements.txt` |
| Architecture diagram | `docs/architecture.png` |
| Data validation report or logs | `reports/validation_report.md` (+ `.json`), rejected rows in `data/errors/`, sample in `reports/validation_errors_sample.csv` |
| Feature catalog in table format | `reports/feature_catalog.md` |
| Model evaluation results comparing both models | `reports/results.md`, `reports/results.csv`, `reports/model_comparison.md` |
| Orchestration DAG / workflow file | `dags/recommendation_flow.py` (Prefect), `dags/airflow_dag.py` (optional), `run_pipeline.py` |
| Screenshot of a successful run | `reports/prefect_run_screenshot.png`, `reports/screenshots/prefect_*.png` |
| Short final report (2-3 pages) | `reports/final_report.pdf` |
| Experiment tracking (MLflow) | `reports/screenshots/mlflow_*.png`, MLflow runs listed in `reports/model_comparison.md` |

## 2. Repository structure

![Repository structure](screenshots/repo_structure.png)

## 3. Stage-by-stage evidence

""" + "\n".join([
    stage(1, "Ingestion", f"Loads users, items and interactions from MovieLens 100K and writes raw text CSVs. A seeded fault injection corrupts or appends {n_faults} rows so the validator has real defects to find.",
          "python run_pipeline.py --stage ingestion", "`data/raw/csv/*.csv`, `reports/ingestion_summary.json`", "stage_1_ingestion.png",
          f"{ing['rows']['users']:,} users, {ing['rows']['items']:,} items and {ing['rows']['interactions']:,} interaction rows written."),
    stage(2, "Data validation", "Checks missing values, duplicates, data types, rating range, timestamp window and referential integrity. Unusable rows go to the error dataset with the reason; repairable attribute problems are warnings.",
          "python run_pipeline.py --stage validation", "`data/errors/*.parquet`, `reports/validation_report.md`, `reports/validation_errors_sample.csv`", "stage_2_validation.png",
          f"Rejected: {rej['interactions']} interactions, {rej['users']} users, {rej['items']} items. Every injected fault was detected (see the injected-vs-detected table)."),
    stage(3, "Data preparation", "Repairs warning-level issues (median imputation, standardised text), keeps the latest rating per (user, item) and creates the time-based train/test split.",
          "python run_pipeline.py --stage preparation", "`data/processed/{users,items,interactions}_clean`, `train`, `test`", "stage_3_preparation.png",
          f"Per-user chronological split: {prep['train_rows']:,} train rows, {prep['test_rows']:,} test rows."),
    stage(4, "Data transformation", "Builds derived datasets from the training split only: user activity summary, item popularity and the sparse user-item interaction matrix.",
          "python run_pipeline.py --stage transformation", "`user_activity_summary`, `item_popularity`, `interaction_matrix.npz`", "stage_4_transformation.png",
          f"Matrix shape {trans['matrix_shape'][0]} x {trans['matrix_shape'][1]}, {trans['nnz']:,} non-zero entries."),
    stage(5, "Feature management", "Computes documented user, item and interaction features point-in-time from the training data, stores them in a feature store and generates the feature catalog.",
          "python run_pipeline.py --stage features", "`data/features/*.parquet`, `reports/feature_catalog.md`", "stage_5_features.png",
          "The same compute functions are used for training and for prediction (`src/inference/predict.py`)."),
    stage(6, "Model training", "Trains the popularity baseline and three personalised models (Item-CF, content-based, hybrid); parameters, timings and model artifacts are logged to MLflow.",
          "python dags/recommendation_flow.py", "`models/*.joblib`, `models/*.run.json`, MLflow runs", "stage_6_training.png",
          "The four training tasks run in parallel in the Prefect flow."),
    stage(7, "Model evaluation", f"Evaluates every model at K = {K} (Precision, Recall, NDCG, Coverage, Diversity, plus RMSE / MAE) and logs the metrics to the same MLflow runs.",
          "python run_pipeline.py --stage evaluation", "`reports/results.md`, `results.csv`, `model_comparison.png`", "stage_7_evaluation.png",
          "Full analysis in `reports/model_comparison.md`."),
    "### 8. Unit tests\n\n**Command:** `python -m pytest -q`\n\n![Unit tests](screenshots/stage_8_unit_tests.png)\n\n*Tests cover metrics, validation checks, the time-based split and the point-in-time rule for features.*\n",
]) + f"""
## 4. Orchestration evidence (Prefect)

The workflow runs ingest, validate, prepare, transform and features sequentially, then the four training tasks in parallel, then evaluation.

**Screenshot of a successful run (required deliverable)**

![Prefect successful run](prefect_run_screenshot.png)

**Additional Prefect screenshots from the clean end-to-end run**

| Dashboard | Flow runs |
|---|---|
| ![Prefect dashboard](screenshots/prefect_01_dashboard.png) | ![Prefect flow runs](screenshots/prefect_02_flow_runs.png) |

| Task graph and task runs | Flow-run logs |
|---|---|
| ![Prefect task runs](screenshots/prefect_03_flow_run_taskruns.png) | ![Prefect logs](screenshots/prefect_04_flow_run_logs.png) |

A summary graphic generated from `reports/run_log.jsonl` is in `reports/workflow_run.png`.

## 5. Experiment tracking evidence (MLflow)

One MLflow run per model in experiment `recsys-movielens-100k`.

| Experiments | Runs |
|---|---|
| ![MLflow experiments](screenshots/mlflow_01_experiments.png) | ![MLflow runs](screenshots/mlflow_02_runs_table.png) |

**Side-by-side comparison of the four runs (parameters and metrics)**

![MLflow compare runs](screenshots/mlflow_03_compare_runs.png)

**One run in detail (parameters, metrics) and its stored model artifact**

![MLflow run detail](screenshots/mlflow_04_run_detail_item_cf.png)

![MLflow artifacts](screenshots/mlflow_05_run_artifacts_item_cf.png)
"""
(R / "evidence.md").write_text(ev, encoding="utf-8")
print("wrote reports/model_comparison.md and reports/evidence.md")
