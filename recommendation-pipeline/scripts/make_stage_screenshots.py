"""Render one terminal-style evidence image per pipeline stage into reports/screenshots/.

Each image shows the stage's real log lines (reports/pipeline.log) and a peek at the files the stage produced,
so a reviewer can see what every stage did without re-running anything.  Run AFTER a pipeline run:

    python scripts/make_stage_screenshots.py
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"
BOLD_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono-Bold.ttf"
SIZE, LINE_H, PAD, MAX_CHARS = 14, 20, 22, 150
COLORS = {"prompt": "#7fdc7f", "out": "#d4d4d4", "dim": "#8a8f98", "head": "#f0c674", "ok": "#7fdc7f", "bad": "#ff7b72"}
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 28)


def fit(line: str) -> str:
    line = line.replace(str(ROOT), ".")                      # do not print machine-specific absolute paths
    return line if len(line) <= MAX_CHARS else line[: MAX_CHARS - 1] + "\u2026"


def render(name: str, title: str, lines: list[tuple[str, str]]) -> None:
    lines = [(k, fit(t)) for k, t in lines]
    font, bold = ImageFont.truetype(str(FONT_PATH), SIZE), ImageFont.truetype(str(BOLD_PATH), SIZE)
    char_w = font.getlength("M")
    width = int(char_w * MAX_CHARS + 2 * PAD)
    bar = 38
    height = bar + PAD + LINE_H * len(lines) + PAD
    im = Image.new("RGB", (width, height), "#1e1e1e")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, width, bar], fill="#323232")
    for i, c in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        d.ellipse([16 + i * 22, 12, 28 + i * 22, 24], fill=c)
    d.text((width // 2 - len(title) * char_w / 2, 10), title, fill="#bdbdbd", font=font)
    y = bar + PAD
    for kind, text in lines:
        d.text((PAD, y), text, fill=COLORS.get(kind, COLORS["out"]), font=bold if kind in ("prompt", "head") else font)
        y += LINE_H
    im.save(OUT / name)
    print("saved", name)


def log_lines(stage: str) -> list[tuple[str, str]]:
    path = ROOT / "reports" / "pipeline.log"
    out = []
    for l in path.read_text(encoding="utf-8").splitlines():
        if f"| {stage} |" in l and "| START " not in l:
            out.append(("ok" if "DONE" in l else "out", l))
    return out


def sh(cmd: str, text: str | list[str]) -> list[tuple[str, str]]:
    body = text.splitlines() if isinstance(text, str) else text
    return [("prompt", f"$ {cmd}")] + [("out", l) for l in body] + [("out", "")]


def df_text(df: pd.DataFrame) -> list[str]:
    return df.to_string(index=False).splitlines()


def js(path: str, keys=None) -> list[str]:
    obj = json.loads((ROOT / path).read_text())
    if keys:
        obj = {k: obj[k] for k in keys if k in obj}
    return json.dumps(obj, indent=2, default=str).splitlines()


def head(path: str, n: int) -> list[str]:
    return (ROOT / path).read_text(encoding="utf-8").splitlines()[:n]


def main() -> None:
    P = lambda p: ROOT / "data" / p
    # ---------------------------------------------------------------- 1 ingestion
    L = [("prompt", "$ python run_pipeline.py --stage ingestion")] + log_lines("ingestion") + [("out", "")]
    L += sh("head -4 data/raw/csv/interactions.csv", head("data/raw/csv/interactions.csv", 4))
    L += sh("head -4 data/raw/csv/users.csv", head("data/raw/csv/users.csv", 4))
    L += sh("head -3 data/raw/csv/items.csv", head("data/raw/csv/items.csv", 3))
    s = json.loads((ROOT / "reports/ingestion_summary.json").read_text())
    L += sh("cat reports/ingestion_summary.json   # rows written + injected faults", [f"rows: {s['rows']}", f"fault_injection: {s['fault_injection']}"])
    render("stage_1_ingestion.png", "Stage 1 - Ingestion", L)

    # ---------------------------------------------------------------- 2 validation
    L = [("prompt", "$ python run_pipeline.py --stage validation")] + log_lines("validation") + [("out", "")]
    rep = (ROOT / "reports/validation_report.md").read_text().splitlines()
    i0 = rep.index("## 1. Summary")
    L += sh("sed -n '1,/## 2/p' reports/validation_report.md   # summary table", [l for l in rep[i0:i0 + 8]])
    j = [i for i, l in enumerate(rep) if l.startswith("## 3.")]
    if j:
        L += sh("grep -A22 'Injected faults' reports/validation_report.md   # injected vs detected", rep[j[0]:j[0] + 24])
    err = head("reports/validation_errors_sample.csv", 6)
    L += sh("head -6 reports/validation_errors_sample.csv   # rejected rows + reason", err)
    render("stage_2_validation.png", "Stage 2 - Data validation", L)

    # ---------------------------------------------------------------- 3 preparation
    tr, te = pd.read_parquet(P("processed/train.parquet")), pd.read_parquet(P("processed/test.parquet"))
    L = [("prompt", "$ python run_pipeline.py --stage preparation")] + log_lines("preparation") + [("out", "")]
    L += sh("cat reports/preparation_summary.json", js("reports/preparation_summary.json"))
    L += sh("python -c \"print(train.head(4)); print(test.head(4))\"", df_text(tr.head(4)) + [""] + df_text(te.head(4)))
    render("stage_3_preparation.png", "Stage 3 - Data preparation (cleaning + time-based split)", L)

    # ---------------------------------------------------------------- 4 transformation
    ua, ip = pd.read_parquet(P("processed/user_activity_summary.parquet")), pd.read_parquet(P("processed/item_popularity.parquet"))
    L = [("prompt", "$ python run_pipeline.py --stage transformation")] + log_lines("transformation") + [("out", "")]
    L += sh("cat reports/transformation_summary.json   # sparse user-item matrix", js("reports/transformation_summary.json"))
    L += sh("python -c \"print(user_activity_summary.head(3))\"", df_text(ua.head(3)))
    L += sh("python -c \"print(item_popularity.head(4))\"", df_text(ip.head(4)))
    render("stage_4_transformation.png", "Stage 4 - Data transformation (derived datasets)", L)

    # ---------------------------------------------------------------- 5 features
    uf, itf, inf = (pd.read_parquet(P(f"features/{n}_features.parquet")) for n in ("user", "item", "interaction"))
    L = [("prompt", "$ python run_pipeline.py --stage features")] + log_lines("features") + [("out", "")]
    ucols = ["user_id", "user_activity_count", "user_avg_rating", "user_positive_ratio", "days_since_last_interaction", "preferred_genre"]
    icols = ["item_id", "item_popularity", "item_popularity_pct", "item_avg_rating", "item_bayes_rating", "item_age_years"]
    L += sh("python -c \"print(user_features[...].head(3))\"", df_text(uf[ucols].round(3).head(3)))
    L += sh("python -c \"print(item_features[...].head(3))\"", df_text(itf[icols].round(3).head(3)))
    L += sh("python -c \"print(interaction_features.head(3))\"", df_text(inf.round(3).head(3)))
    L += sh("ls -1 data/features/", sorted(p.name for p in (ROOT / "data/features").glob("*.parquet")))
    render("stage_5_features.png", "Stage 5 - Feature management (feature store tables)", L)

    # ---------------------------------------------------------------- 6 training
    L = [("prompt", "$ python dags/recommendation_flow.py   # four training tasks run in parallel")] + log_lines("training")[:12] + [("out", "")]
    L += sh("ls -1 models/", sorted(p.name for p in (ROOT / "models").iterdir() if p.name != ".gitkeep"))
    runs = [json.loads(p.read_text()) for p in sorted((ROOT / "models").glob("*.run.json"))]
    L += sh("cat models/*.run.json   # MLflow run ids", [f"{r['model']:14s} run_id={r['run_id']}  fit_seconds={r['fit_seconds']}" for r in runs])
    render("stage_6_training.png", "Stage 6 - Model training (MLflow-tracked)", L)

    # ---------------------------------------------------------------- 7 evaluation
    L = [("prompt", "$ python run_pipeline.py --stage evaluation")]
    for kind, l in log_lines("evaluation"):
        if "->" in l:
            model = l.split("evaluation | ")[1].split(" ->")[0]
            L.append(("out", f"evaluation | {model}"))
        else:
            L.append((kind, l))
    L.append(("out", ""))
    L += sh("cat reports/results.md", head("reports/results.md", 12)[0:12])
    render("stage_7_evaluation.png", "Stage 7 - Model evaluation (results table)", L)

    # ---------------------------------------------------------------- 8 tests
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    L = sh("python -m pytest -q", (r.stdout.strip().splitlines() or ["(no output)"])[-12:])
    render("stage_8_unit_tests.png", "Unit tests", L)

    # ---------------------------------------------------------------- 9 repo structure
    skip = {".git", "__pycache__", ".pytest_cache", "mlartifacts", ".venv", "venv", ".prefect", "screenshots"}
    lines = ["recommendation-pipeline/"]

    def walk(path: Path, prefix: str, depth: int) -> None:
        entries = sorted([e for e in path.iterdir() if e.name not in skip and e.name != ".gitkeep" and not e.name.endswith(".pyc")],
                         key=lambda e: (e.is_file(), e.name))
        if path.name in ("processed", "features", "errors", "csv", "validated", "ml-100k") and depth >= 2:
            entries = entries[:6]
        for i, e in enumerate(entries):
            last = i == len(entries) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{e.name}{'/' if e.is_dir() else ''}")
            if e.is_dir() and depth < 3:
                walk(e, prefix + ("    " if last else "│   "), depth + 1)

    walk(ROOT, "", 1)
    render("repo_structure.png", "Repository structure", [("prompt", "$ tree -L 3 recommendation-pipeline")] + [("out", l) for l in lines[:70]])


if __name__ == "__main__":
    main()
