"""Draw reports/workflow_run.png from reports/run_log.jsonl (the stage timings of the last run).

This is a generated summary of a successful run. For the submission also take a real screenshot of the
Prefect UI (or Airflow UI) - see README, section 'Orchestration screenshot'.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
log = [json.loads(l) for l in (ROOT / "reports" / "run_log.jsonl").read_text().splitlines() if l.strip()]
status = {r["stage"]: r for r in log}
EXPECTED = ["ingestion", "validation", "preparation", "transformation", "features", "training[popularity]",
            "training[item_cf]", "training[content_based]", "training[hybrid]", "evaluation"]
all_ok = all(n in status and status[n]["status"] == "success" for n in EXPECTED)

layout = {  # name -> (x, y)
    "ingestion": (0.2, 3), "validation": (2.4, 3), "preparation": (4.6, 3), "transformation": (6.8, 3), "features": (9.0, 3),
    "training[popularity]": (11.4, 5.1), "training[item_cf]": (11.4, 3.7), "training[content_based]": (11.4, 2.3),
    "training[hybrid]": (11.4, 0.9), "evaluation": (14.0, 3),
}
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16.6)
ax.set_ylim(0, 6.6)
ax.axis("off")
ax.text(0.2, 6.25, f"Flow run: recommendation-pipeline   -   {'COMPLETED' if all_ok else 'FAILED'}   "
        f"(total task time {sum(r['seconds'] for r in log):.1f}s)", fontsize=13, fontweight="bold",
        color="#1B5E20" if all_ok else "#B71C1C")
w, h = 2.0, 0.95
for name, (x, y) in layout.items():
    r = status.get(name)
    ok = r and r["status"] == "success"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#C8E6C9" if ok else "#FFCDD2", ec="#2E7D32" if ok else "#B71C1C", lw=1.5))
    short = name.replace("training", "train")
    ax.text(x + w / 2, y + 0.62, short, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(x + w / 2, y + 0.27, ("Completed  " if ok else ("Failed  " if r else "Not run")) + (f"{r['seconds']}s" if r else ""),
            ha="center", va="center", fontsize=8.5, color="#1B5E20" if ok else "#B71C1C")


def arr(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333"))


chain = ["ingestion", "validation", "preparation", "transformation", "features"]
for a, b in zip(chain[:-1], chain[1:]):
    arr((layout[a][0] + w, layout[a][1] + h / 2), (layout[b][0], layout[b][1] + h / 2))
for t in [k for k in layout if k.startswith("training")]:
    arr((layout["features"][0] + w, layout["features"][1] + h / 2), (layout[t][0], layout[t][1] + h / 2))
    arr((layout[t][0] + w, layout[t][1] + h / 2), (layout["evaluation"][0], layout["evaluation"][1] + h / 2))
plt.savefig(ROOT / "reports" / "workflow_run.png", dpi=150, bbox_inches="tight", facecolor="white")
print("saved reports/workflow_run.png")
