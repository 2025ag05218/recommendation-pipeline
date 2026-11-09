"""Draw docs/architecture.png (pipeline architecture)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 15.4)
ax.set_ylim(-0.3, 8.2)
ax.axis("off")

STAGE, STORE, OUT, ORCH = "#DCE9F7", "#FFF2CC", "#E2F0D9", "#EADCF4"


def box(x, y, w, h, text, color, fs=9, bold=False, ec="#444"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(p, q, style="-|>", color="#333", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


ax.text(7.5, 7.9, "End-to-End Data Management Pipeline for a Recommendation System", ha="center", fontsize=14, fontweight="bold")

# orchestration band
box(0.3, 6.75, 14.4, 0.7, "Orchestration  -  Prefect flow (dags/recommendation_flow.py)  |  optional Airflow DAG  |  run_pipeline.py   -   "
    "task dependencies, retries, daily schedule", ORCH, fs=9.5, bold=True)

# source
box(0.3, 4.55, 1.7, 1.2, "Source data\nMovieLens 100K\nusers | items |\nratings", STORE, fs=8.5, bold=True)

# main stages
xs = [2.5, 4.55, 6.6, 8.65, 10.7]
labels = ["1. Ingestion\nload -> raw CSV\n(+ fault injection)", "2. Validation\nschema, range, dupes,\nreferential integrity",
          "3. Preparation\nclean / impute /\ntime-based split", "4. Transformation\nuser activity, item\npopularity, matrix",
          "5. Feature mgmt\nuser / item /\ninteraction features"]
for x, l in zip(xs, labels):
    box(x, 4.55, 1.85, 1.2, l, STAGE, fs=8.3, bold=False)
arrow((2.0, 5.15), (2.5, 5.15))
for a, b in zip(xs[:-1], xs[1:]):
    arrow((a + 1.85, 5.15), (b, 5.15))

# training + evaluation
box(12.85, 5.0, 1.85, 0.75, "6a. Popularity\n(baseline)", STAGE, fs=8.3)
box(12.85, 4.1, 1.85, 0.75, "6b. Item-CF", STAGE, fs=8.3)
box(12.85, 3.2, 1.85, 0.75, "6c. Content-based", STAGE, fs=8.3)
box(12.85, 2.3, 1.85, 0.75, "6d. Hybrid", STAGE, fs=8.3)
for y in (5.375, 4.475, 3.575, 2.675):
    arrow((12.55, 5.15), (12.85, y), rad=0.0)
box(12.85, 0.75, 1.85, 1.1, "7. Evaluation\nP@K R@K NDCG@K\nCoverage Diversity\nRMSE MAE", OUT, fs=8, bold=True)
# bracket: every trained model feeds the evaluation stage
for y in (5.375, 4.475, 3.575, 2.675):
    ax.plot([14.7, 15.0], [y, y], color="#333", lw=1.3)
ax.plot([15.0, 15.0], [5.375, 1.3], color="#333", lw=1.3)
arrow((15.0, 1.3), (14.7, 1.3))

# storage layer
box(2.5, 2.55, 1.85, 1.1, "Error dataset\ndata/errors/*.parquet\n+ validation report", "#FDE2E2", fs=8)
box(4.55, 2.55, 1.85, 1.1, "Validated + clean\ndata/processed/\nuser/item/train/test", STORE, fs=8)
box(6.6, 2.55, 1.85, 1.1, "Derived datasets\nactivity, popularity,\nsparse matrix (.npz)", STORE, fs=8)
box(8.65, 2.55, 1.85, 1.1, "Feature store\ndata/features/*.parquet\n+ feature catalog", STORE, fs=8)
arrow((5.5, 4.55), (3.45, 3.65), ls="--", color="#B22222")
arrow((5.5, 4.55), (5.5, 3.65), ls="--", color="#777")
arrow((7.5, 4.55), (7.5, 3.65), ls="--", color="#777")
arrow((9.6, 4.55), (9.6, 3.65), ls="--", color="#777")
arrow((11.6, 4.55), (9.8, 3.65), ls="--", color="#777", rad=0.2)

# mlflow + outputs
box(10.6, 0.75, 1.95, 1.1, "MLflow tracking\nparams, metrics,\nmodel artifacts\n(sqlite + mlartifacts/)", OUT, fs=7.8, bold=True)
box(6.6, 0.75, 3.7, 0.95, "Reports\nresults.md | validation_report.md |\nfeature_catalog.md | final report PDF", OUT, fs=8.3)
box(2.5, 0.75, 3.6, 0.95, "Serving: src/inference/predict.py\nsame feature code as training\n-> top-N recommendations", OUT, fs=8.5)
arrow((12.85, 1.3), (12.55, 1.3), color="#2E7D32")
arrow((13.5, 0.75), (8.4, 0.75), ls="--", color="#2E7D32", rad=-0.12)
arrow((9.6, 2.55), (5.0, 1.7), ls="--", color="#777", rad=0.1)
arrow((13.6, 2.3), (11.6, 1.85), ls="--", color="#2E7D32", rad=-0.1)

ax.text(0.3, -0.2, "Solid arrows = data flow between stages   |   dashed = artifacts written / read   |   "
        "all features and derived data come from the TRAIN split only (no leakage)", fontsize=8.5, color="#333")
plt.savefig(ROOT / "docs" / "architecture.png", dpi=160, bbox_inches="tight", facecolor="white")
print("saved docs/architecture.png")
