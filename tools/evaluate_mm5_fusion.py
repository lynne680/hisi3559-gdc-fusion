# -*- coding: utf-8 -*-
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("failed to read: %s" % str(path))
    return img.astype(np.uint8)


def norm(img):
    img = img.astype(np.float32)
    mn = np.percentile(img, 1)
    mx = np.percentile(img, 99)
    out = np.clip((img - mn) / (mx - mn + 1e-6), 0, 1)
    return out


def entropy(img):
    hist = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256])
    p = hist[:, 0] / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def avg_gradient(img):
    img = img.astype(np.float32)
    dx = img[:, 1:] - img[:, :-1]
    dy = img[1:, :] - img[:-1, :]
    dx = dx[:-1, :]
    dy = dy[:, :-1]
    ag = np.sqrt((dx * dx + dy * dy) / 2.0)
    return float(ag.mean())


def spatial_frequency(img):
    img = img.astype(np.float32)
    rf = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    cf = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
    return float(np.sqrt(rf * rf + cf * cf))


def mutual_information(a, b):
    a = a.astype(np.uint8).ravel()
    b = b.astype(np.uint8).ravel()

    hist_2d, _, _ = np.histogram2d(a, b, bins=256, range=[[0, 255], [0, 255]])
    pxy = hist_2d / (hist_2d.sum() + 1e-12)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    px_py = px[:, None] * py[None, :]
    nz = pxy > 0

    mi = np.sum(pxy[nz] * np.log2(pxy[nz] / (px_py[nz] + 1e-12)))
    return float(mi)


def metric_row(sample, method, img, rgb, thermal, uv):
    return {
        "sample": sample,
        "method": method,
        "EN": entropy(img),
        "SD": float(np.std(img)),
        "AG": avg_gradient(img),
        "SF": spatial_frequency(img),
        "MI_RGB": mutual_information(img, rgb),
        "MI_Thermal": mutual_information(img, thermal),
        "MI_UV": mutual_information(img, uv),
        "MI_SUM": mutual_information(img, rgb) + mutual_information(img, thermal) + mutual_information(img, uv),
    }


def to_u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def make_color(img):
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)


def make_preview(rgb, thermal, uv, avg_fusion, fixed_fusion, ours, out_path):
    h, w = rgb.shape

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    thermal_bgr = make_color(thermal)
    uv_bgr = make_color(uv)
    avg_bgr = cv2.cvtColor(avg_fusion, cv2.COLOR_GRAY2BGR)
    fixed_bgr = cv2.cvtColor(fixed_fusion, cv2.COLOR_GRAY2BGR)
    ours_bgr = cv2.cvtColor(ours, cv2.COLOR_GRAY2BGR)

    canvas = np.zeros((h, w * 6, 3), dtype=np.uint8)

    canvas[:, 0:w] = rgb_bgr
    canvas[:, w:2*w] = thermal_bgr
    canvas[:, 2*w:3*w] = uv_bgr
    canvas[:, 3*w:4*w] = avg_bgr
    canvas[:, 4*w:5*w] = fixed_bgr
    canvas[:, 5*w:6*w] = ours_bgr

    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--fused-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data)
    fused_dir = Path(args.fused_dir)
    out_dir = Path(args.out)

    preview_dir = out_dir / "comparison_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rgb_dir = data_dir / "rgb"
    thermal_dir = data_dir / "thermal"
    uv_dir = data_dir / "uv"

    rows = []
    rgb_files = sorted(rgb_dir.glob("*.png"))

    for idx, rgb_path in enumerate(rgb_files):
        name = rgb_path.name
        sample = Path(name).stem

        rgb = read_gray(rgb_dir / name)
        thermal = read_gray(thermal_dir / name)
        uv = read_gray(uv_dir / name)
        ours = read_gray(fused_dir / name)

        rgb_n = norm(rgb)
        t_n = norm(thermal)
        u_n = norm(uv)

        visible_only = rgb
        avg_fusion = to_u8((rgb_n + t_n + u_n) / 3.0)
        fixed_fusion = to_u8(0.60 * rgb_n + 0.25 * t_n + 0.15 * u_n)

        methods = [
            ("VisibleOnly", visible_only),
            ("AverageFusion", avg_fusion),
            ("FixedWeightedFusion", fixed_fusion),
            ("OursMaskFusion", ours),
        ]

        for method, img in methods:
            rows.append(metric_row(sample, method, img, rgb, thermal, uv))

        if idx < 60:
            make_preview(
                rgb,
                thermal,
                uv,
                avg_fusion,
                fixed_fusion,
                ours,
                preview_dir / ("comparison_%s.jpg" % sample)
            )

        if idx % 50 == 0:
            print("processed:", idx, name)

    metric_csv = out_dir / "metrics_detail.csv"
    summary_csv = out_dir / "metrics_summary.csv"

    keys = ["sample", "method", "EN", "SD", "AG", "SF", "MI_RGB", "MI_Thermal", "MI_UV", "MI_SUM"]

    with open(metric_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    methods = sorted(set(r["method"] for r in rows))

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "EN", "SD", "AG", "SF", "MI_RGB", "MI_Thermal", "MI_UV", "MI_SUM"])

        for m in methods:
            sub = [r for r in rows if r["method"] == m]
            writer.writerow([
                m,
                np.mean([r["EN"] for r in sub]),
                np.mean([r["SD"] for r in sub]),
                np.mean([r["AG"] for r in sub]),
                np.mean([r["SF"] for r in sub]),
                np.mean([r["MI_RGB"] for r in sub]),
                np.mean([r["MI_Thermal"] for r in sub]),
                np.mean([r["MI_UV"] for r in sub]),
                np.mean([r["MI_SUM"] for r in sub]),
            ])

    print("done")
    print("detail:", metric_csv)
    print("summary:", summary_csv)
    print("preview:", preview_dir)


if __name__ == "__main__":
    main()
