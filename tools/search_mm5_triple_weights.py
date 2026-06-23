# -*- coding: utf-8 -*-
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("failed to read: %s" % str(path))
    return img.astype(np.uint8)


def norm(img):
    img = img.astype(np.float32)
    mn = np.percentile(img, 1)
    mx = np.percentile(img, 99)
    return np.clip((img - mn) / (mx - mn + 1e-6), 0, 1)


def to_u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


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


def fuse(rgb_n, t_n, u_n, mask, base_t, base_u, target_t, target_u):
    base_r = 1.0 - base_t - base_u
    target_r = 1.0 - target_t - target_u

    base = base_r * rgb_n + base_t * t_n + base_u * u_n
    target = target_r * rgb_n + target_t * t_n + target_u * u_n

    m = mask.astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (11, 11), 0)

    out = base * (1.0 - m) + target * m
    return to_u8(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_dir = data_dir / "rgb"
    thermal_dir = data_dir / "thermal"
    uv_dir = data_dir / "uv"

    files = sorted(rgb_dir.glob("*.png"))

    if args.max_samples > 0:
        files = files[:args.max_samples]

    print("samples:", len(files))

    samples = []

    for p in files:
        name = p.name
        rgb = read_gray(rgb_dir / name)
        thermal = read_gray(thermal_dir / name)
        uv = read_gray(uv_dir / name)
        mask = read_gray(mask_dir / name)

        samples.append((
            norm(rgb),
            norm(thermal),
            norm(uv),
            rgb,
            thermal,
            uv,
            mask
        ))

    # 背景区域只允许轻微融合，目标区域融合更强
    base_t_values = [0.02, 0.04, 0.06, 0.08, 0.10]
    base_u_values = [0.02, 0.04, 0.06, 0.08, 0.10]

    target_t_values = [0.20, 0.25, 0.30, 0.35, 0.40]
    target_u_values = [0.10, 0.15, 0.20, 0.25, 0.30]

    rows = []

    total = 0

    for base_t in base_t_values:
        for base_u in base_u_values:
            if base_t + base_u >= 0.25:
                continue

            for target_t in target_t_values:
                for target_u in target_u_values:
                    if target_t + target_u >= 0.65:
                        continue

                    ENs, SDs, AGs, SFs = [], [], [], []
                    MIRs, MITs, MIUs, MISUMs = [], [], [], []

                    for rgb_n, t_n, u_n, rgb, thermal, uv, mask in samples:
                        out = fuse(
                            rgb_n,
                            t_n,
                            u_n,
                            mask,
                            base_t,
                            base_u,
                            target_t,
                            target_u
                        )

                        mi_r = mutual_information(out, rgb)
                        mi_t = mutual_information(out, thermal)
                        mi_u = mutual_information(out, uv)

                        ENs.append(entropy(out))
                        SDs.append(float(np.std(out)))
                        AGs.append(avg_gradient(out))
                        SFs.append(spatial_frequency(out))
                        MIRs.append(mi_r)
                        MITs.append(mi_t)
                        MIUs.append(mi_u)
                        MISUMs.append(mi_r + mi_t + mi_u)

                    row = {
                        "base_rgb": 1.0 - base_t - base_u,
                        "base_t": base_t,
                        "base_u": base_u,
                        "target_rgb": 1.0 - target_t - target_u,
                        "target_t": target_t,
                        "target_u": target_u,
                        "EN": float(np.mean(ENs)),
                        "SD": float(np.mean(SDs)),
                        "AG": float(np.mean(AGs)),
                        "SF": float(np.mean(SFs)),
                        "MI_RGB": float(np.mean(MIRs)),
                        "MI_Thermal": float(np.mean(MITs)),
                        "MI_UV": float(np.mean(MIUs)),
                        "MI_SUM": float(np.mean(MISUMs)),
                    }

                    # 综合排序分数：优先 MI_SUM，同时要求细节指标不能太差
                    row["score"] = row["MI_SUM"] + 0.03 * row["AG"] + 0.005 * row["SF"]

                    rows.append(row)

                    total += 1
                    if total % 20 == 0:
                        print("tested:", total)

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)

    csv_path = out_dir / "triple_weight_search.csv"

    keys = [
        "score",
        "base_rgb", "base_t", "base_u",
        "target_rgb", "target_t", "target_u",
        "EN", "SD", "AG", "SF",
        "MI_RGB", "MI_Thermal", "MI_UV", "MI_SUM"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("done")
    print("csv:", csv_path)
    print("\nTop 10:")

    for i, r in enumerate(rows[:10]):
        print(
            i,
            "score=%.4f" % r["score"],
            "MI_SUM=%.4f" % r["MI_SUM"],
            "EN=%.4f" % r["EN"],
            "SD=%.4f" % r["SD"],
            "AG=%.4f" % r["AG"],
            "SF=%.4f" % r["SF"],
            "base=(%.2f, %.2f, %.2f)" % (r["base_rgb"], r["base_t"], r["base_u"]),
            "target=(%.2f, %.2f, %.2f)" % (r["target_rgb"], r["target_t"], r["target_u"]),
        )


if __name__ == "__main__":
    main()
