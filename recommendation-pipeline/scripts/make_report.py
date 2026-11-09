"""Build reports/final_report.pdf (2-3 pages) from the latest pipeline outputs.

All numbers are read from reports/*.json, so re-run this after any pipeline run.
The narrative sentences describe the default configuration - re-read them if you change data/config.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
rd = lambda n: json.loads((R / n).read_text())
res = {r["model"]: r for r in rd("results.json")}
val, prep, trans = rd("validation_report.json"), rd("preparation_summary.json"), rd("transformation_summary.json")
ing, feat = rd("ingestion_summary.json"), rd("features_summary.json")
pop, cf, ct, hy = res["popularity"], res["item_cf"], res["content_based"], res["hybrid"]
lift = lambda a, b: 100 * (a / b - 1)

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.2, leading=11.8, alignment=TA_JUSTIFY, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1F3A5F"))
title = ParagraphStyle("t", parent=ss["Title"], fontSize=16, spaceAfter=2)
sub = ParagraphStyle("s", parent=ss["Normal"], fontSize=9, alignment=1, textColor=colors.HexColor("#444444"), spaceAfter=6)
cell = ParagraphStyle("c", parent=ss["Normal"], fontSize=7.8, leading=9.4)
bul = ParagraphStyle("b", parent=body, leftIndent=11, bulletIndent=2, spaceAfter=1.5)


def P(t, st=body):
    return Paragraph(t, st)


def table(rows, widths, header=True, fs=7.8):
    t = Table([[Paragraph(str(c), cell) if isinstance(c, str) else c for c in r] for r in rows], colWidths=widths, repeatRows=1)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE9F7"))]
    t.setStyle(TableStyle(style))
    return t


fm = lambda v: "n/a" if v is None else f"{v:.4f}"
story = [
    P("End-to-End Data Management Pipeline for a Recommendation System", title),
    P("Data Management for Machine Learning (AIMLZG529) - Assignment 2 &nbsp;|&nbsp; Rajavarman Mani &nbsp;|&nbsp; Roll No. 2025ag05218", sub),

    P("1. Objective and dataset", h1),
    P(f"The goal is a clean, modular and reproducible pipeline that takes raw user, item and interaction data to evaluated "
      f"recommenders: ingestion, validation, preparation, transformation, feature management, training, evaluation and orchestration. "
      f"I used <b>MovieLens 100K</b> ({prep['users']} users, {prep['items']:,} movies, {prep['interactions']:,} ratings after cleaning; "
      f"{prep['sparsity_pct']}% sparse). The task is top-10 movie recommendation; a rating of 4 or more counts as a positive signal. "
      f"The full source code, configuration and instructions are in the GitHub repository (see README)."),

    P("2. Architecture and tools", h1),
    Image(str(ROOT / "docs" / "architecture.png"), width=17.2 * cm, height=17.2 * cm * 0.55),
    P("<b>Tools:</b> Python/pandas/NumPy/SciPy for processing; Parquet for storage (typed, compressed, fast); a custom, file-based feature layer; "
      "MLflow (SQLite backend) for experiment tracking; Prefect for orchestration (an equivalent Airflow DAG is included); pytest for unit tests. "
      "Everything is driven by one <font face='Courier'>configs/config.yaml</font> and a fixed seed, so every run is reproducible."),

    P("3. Design decisions per stage", h1),
]
for b in [
    "<b>Ingestion.</b> Data is loaded as untyped text to mimic an external source. MovieLens is almost perfectly clean, so a <i>seeded, documented</i> "
    f"fault-injection step corrupts about {sum(sum(v.values()) for v in ing['fault_injection'].values()):,} raw rows (duplicates, invalid ratings/timestamps, missing values, orphan ids). "
    "This gives the validator real defects and allows an injected-vs-detected check; it can be disabled in the config.",
    "<b>Validation.</b> Checks cover missing values, data types, rating range (1-5), timestamp window, duplicates and referential integrity. "
    "Defects are split by severity: unusable records are <b>rejected</b> to an error dataset with the reason, while repairable attribute problems "
    "(e.g. age 250, missing title) are logged as warnings and fixed in preparation, so we do not throw away good interactions because of a bad profile field. "
    "A run fails if more than 20% of a table is rejected.",
    f"<b>Preparation and split.</b> Imputation (median), text/case standardisation, UTC timestamps and one rating per (user, item). A <b>time-based split</b> "
    f"places the most recent 20% of each user's history in the test set ({prep['train_rows']:,} train / {prep['test_rows']:,} test rows). "
    "Per-user chronological splitting keeps nearly every user evaluable, which a single global cut-off would not on a 7-month dataset (a global option is available).",
    f"<b>Transformation and features.</b> User activity summary, item popularity and a sparse {trans['matrix_shape'][0]} x {trans['matrix_shape'][1]} rating matrix "
    "are built from <b>train only</b>. Features are documented in a registry that generates the catalog, are computed point-in-time at the last training timestamp, "
    "and use one set of functions for both training and prediction (<font face='Courier'>src/inference/predict.py</font>), which avoids training/serving skew. "
    "I chose a custom feature layer over Feast because the requirement is reuse, which this provides without extra infrastructure.",
    "<b>Models.</b> (a) <i>Popularity</i> baseline: the most-liked items for everyone. (b) <i>Item-CF</i>: item-item cosine similarity on the liked matrix with significance "
    "shrinkage co/(co+10), plus an adjusted-cosine k-NN rating predictor. (c) <i>Content-based</i>: cosine between the user's genre-preference features and the item's genres. "
    "(d) <i>Hybrid</i>: 0.7 x CF + 0.3 x content scores, each scaled per user.",
    "<b>Orchestration.</b> A Prefect flow runs ingest, validate, prepare, transform and features sequentially, then the four training tasks in parallel, then evaluation. "
    "Dependencies are explicit, ingestion retries once, and a cron schedule can be attached with <font face='Courier'>--serve</font>.",
]:
    story.append(P(b, bul))
    story[-1].bulletText = "\u2022"

vt = [["Table", "Rows in", "Valid", "Rejected", "Main reject reasons"]]
for t in ("interactions", "users", "items"):
    s = val["tables"][t]
    top = sorted(val["reject_counts"][t].items(), key=lambda kv: -kv[1])
    top = ", ".join(f"{k} ({v})" for k, v in top if v > 0)
    vt.append([t, f"{s['rows_in']:,}", f"{s['rows_valid']:,}", f"{s['rows_rejected']:,} ({s['rejected_pct']:.2f}%)", top])
story += [P("4. Data validation outcome", h1),
          table(vt, [2.0 * cm, 1.6 * cm, 1.6 * cm, 2.4 * cm, 9.6 * cm]),
          Spacer(1, 3),
          P("All injected faults were detected (the report <font face='Courier'>reports/validation_report.md</font> lists injected vs detected per fault type); "
            "rejected rows are stored in <font face='Courier'>data/errors/</font>. The feature catalog (user, item and interaction features) is in "
            "<font face='Courier'>reports/feature_catalog.md</font>.")]

hdr = ["Model", "P@10", "R@10", "NDCG@10", "Coverage", "Diversity", "RMSE", "MAE"]
rows = [hdr]
for name, r in (("Popularity (baseline)", pop), ("Item-CF", cf), ("Content-based", ct), ("Hybrid (CF + content)", hy)):
    rows.append([name, fm(r["precision_at_10"]), fm(r["recall_at_10"]), fm(r["ndcg_at_10"]), fm(r["coverage"]), fm(r["diversity"]),
                 fm(r.get("rmse")), fm(r.get("mae"))])
story += [
    P("5. Results", h1),
    P(f"Evaluation uses K = 10 on {cf['n_eval_users']} users who have at least one liked item in their held-out period; candidates are all items the user has not rated in training. "
      "Coverage is the share of the catalogue that appears in any list; diversity is the mean pairwise genre dissimilarity inside a list.", body),
    table(rows, [4.0 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.7 * cm, 1.7 * cm]),
    Spacer(1, 4),
    Image(str(R / "model_comparison.png"), width=11.5 * cm, height=11.5 * cm * 4.2 / 9),
    P(f"<b>Findings.</b> The personalised Item-CF model clearly beats the popularity baseline: NDCG@10 {cf['ndcg_at_10']:.3f} vs {pop['ndcg_at_10']:.3f} "
      f"(+{lift(cf['ndcg_at_10'], pop['ndcg_at_10']):.0f}%), Precision@10 {cf['precision_at_10']:.3f} vs {pop['precision_at_10']:.3f} "
      f"(+{lift(cf['precision_at_10'], pop['precision_at_10']):.0f}%), and it also recommends a wider part of the catalogue "
      f"({cf['coverage']*100:.1f}% vs {pop['coverage']*100:.1f}%). Its rating predictions are better too (RMSE {cf['rmse']:.3f} vs {pop['rmse']:.3f}), "
      "although that gap is small, because item averages are already a strong rating baseline. "
      f"The hybrid matches Item-CF on accuracy (NDCG {hy['ndcg_at_10']:.3f}) but trades diversity ({hy['diversity']:.2f} vs {cf['diversity']:.2f}) for higher coverage "
      f"({hy['coverage']*100:.1f}%): blending in genre similarity pulls lists toward the user's favourite genres. The content-only model is weak on accuracy "
      f"(P@10 {ct['precision_at_10']:.3f}) because 19 coarse genres cannot separate items well, yet it has the highest coverage ({ct['coverage']*100:.1f}%) as it is not dominated by popular titles. "
      "With a single dataset and seed these differences are indicative, not statistically tested.", body),
    P("6. Limitations and future work", h1),
]
for b in [
    f"<b>Cold start:</b> users and items without training history get no personalised scores; item-CF cannot recommend the {trans['items_without_train_interactions']} items that never appear in training. Richer item metadata or an embedding model would help.",
    "<b>Evaluation:</b> offline top-K metrics on one time-based split and one seed; no confidence intervals or A/B test. The per-user split lets one user's test events precede other users' training events (mild cross-user temporal overlap); the global split removes this at the cost of evaluable users.",
    "<b>Scale:</b> dense user x item score matrices are fine for 943 x 1,682 but not for millions of items; production would need sparse retrieval or approximate nearest neighbours, and an online feature store with freshness guarantees.",
    "<b>Data realism:</b> faults are injected, so validation scores demonstrate the checks rather than real-world data quality; thresholds (e.g. rating >= 4 as positive) are assumptions. Tuning (neighbours, shrinkage, blend weights) was not performed.",
    "<b>Operations:</b> orchestration is local (Prefect ephemeral/server); drift monitoring, model registry promotion and incremental ingestion are natural next steps.",
]:
    story.append(P(b, bul))
    story[-1].bulletText = "\u2022"

doc = SimpleDocTemplate(str(R / "final_report.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.5 * cm, bottomMargin=1.4 * cm, title="Recommendation Pipeline - Final Report",
                        author="Rajavarman Mani")
doc.build(story)
print("saved reports/final_report.pdf")
