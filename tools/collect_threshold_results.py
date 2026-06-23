# -*- coding: utf-8 -*-
import csv
from pathlib import Path

base = Path("train_outputs/mm5_rgb1_t8_u8_anno_class_fixed_w4/threshold_sensitivity")

items = [
    ("0.40", "eval_thr040"),
    ("0.50", "eval_thr050"),
    ("0.60", "eval_thr060"),
    ("0.70", "eval_thr070"),
    ("0.80", "eval_thr080"),
]

rows = []

for th, folder in items:
    p = base / folder / "metrics_summary.csv"

    with open(p, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["method"] == "OursMaskFusion":
                rows.append({
                    "threshold": th,
                    "EN": row["EN"],
                    "SD": row["SD"],
                    "AG": row["AG"],
                    "SF": row["SF"],
                    "MI_SUM": row["MI_SUM"],
                    "MI_RGB": row["MI_RGB"],
                    "MI_Thermal": row["MI_Thermal"],
                    "MI_UV": row["MI_UV"],
                })

out = base / "threshold_summary.csv"

with open(out, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["threshold", "EN", "SD", "AG", "SF", "MI_SUM", "MI_RGB", "MI_Thermal", "MI_UV"]
    )
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print("saved:", out)

for r in rows:
    print(r)
