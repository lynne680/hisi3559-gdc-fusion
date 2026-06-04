# -*- coding: utf-8 -*-
"""
prepare_fusion_dataset.py

用途：
1. 解压或读取 Hi3559 采集到的 Cam2/Cam3 NV21 YUV 文件；
2. 批量转换为 PNG；
3. 使用 OpenCV 单应性矩阵将 Cam3 配准到 Cam2，生成 Cam3_warp；
4. 生成 224x224 三通道训练输入：Cam2_Y、Cam3_warp_Y、abs(Cam2_Y-Cam3_warp_Y)；
5. 基于亮度差与边缘差自动生成第一版伪标签 mask。

mask 约定：
白色 255：适合融合区域；
黑色 0：不适合融合，保留 Cam2。
"""

import argparse
import csv
import os
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np


# 你当前标定得到的 OpenCV 正向矩阵：Cam3 -> Cam2
H_CAM3_TO_CAM2 = np.array([
    [0.9840656540,  -0.0872238974, -39.6684505],
    [0.0805662801,   1.0014957400, -24.3853648],
    [-0.0000047859,  0.0000004491,   1.0000000],
], dtype=np.float64)


CAM_RE = re.compile(r"cam([23])_capture_(\d+)_1920x1080_nv21\.yuv$")


def read_nv21_to_bgr(path: Path, width: int, height: int) -> np.ndarray:
    expected_size = width * height * 3 // 2
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size != expected_size:
        raise ValueError(f"{path} size mismatch: got {data.size}, expected {expected_size}")

    yuv = data.reshape((height * 3 // 2, width))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    return bgr


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unzip_if_needed(zip_path: Path, work_dir: Path) -> Path:
    ensure_dir(work_dir)
    if zip_path is None:
        return work_dir

    extract_dir = work_dir / "raw_yuv"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    ensure_dir(extract_dir)

    print(f"[INFO] unzip {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(extract_dir))
    return extract_dir


def collect_pairs(raw_root: Path):
    cam2 = {}
    cam3 = {}

    for p in raw_root.rglob("*.yuv"):
        m = CAM_RE.search(p.name)
        if not m:
            continue
        cam_id = int(m.group(1))
        idx = int(m.group(2))
        if cam_id == 2:
            cam2[idx] = p
        elif cam_id == 3:
            cam3[idx] = p

    common = sorted(set(cam2.keys()) & set(cam3.keys()))
    pairs = [(idx, cam2[idx], cam3[idx]) for idx in common]
    return pairs


def normalize_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) * 255.0 / (mx - mn)
    return np.clip(y, 0, 255).astype(np.uint8)


def make_input_and_mask(cam2_bgr: np.ndarray,
                        cam3_warp_bgr: np.ndarray,
                        size: int,
                        white_ratio: float,
                        center_prior: bool):
    cam2_y = cv2.cvtColor(cam2_bgr, cv2.COLOR_BGR2GRAY)
    cam3_y = cv2.cvtColor(cam3_warp_bgr, cv2.COLOR_BGR2GRAY)

    cam2_224 = cv2.resize(cam2_y, (size, size), interpolation=cv2.INTER_AREA)
    cam3_224 = cv2.resize(cam3_y, (size, size), interpolation=cv2.INTER_AREA)

    diff = cv2.absdiff(cam2_224, cam3_224)

    sobel2_x = cv2.Sobel(cam2_224, cv2.CV_32F, 1, 0, ksize=3)
    sobel2_y = cv2.Sobel(cam2_224, cv2.CV_32F, 0, 1, ksize=3)
    sobel3_x = cv2.Sobel(cam3_224, cv2.CV_32F, 1, 0, ksize=3)
    sobel3_y = cv2.Sobel(cam3_224, cv2.CV_32F, 0, 1, ksize=3)

    edge2 = cv2.magnitude(sobel2_x, sobel2_y)
    edge3 = cv2.magnitude(sobel3_x, sobel3_y)
    edge_diff = np.abs(edge2 - edge3)

    diff_norm = normalize_u8(diff).astype(np.float32)
    edge_norm = normalize_u8(edge_diff).astype(np.float32)

    # 分数越小，说明两图越接近，越适合融合。
    score = 0.65 * diff_norm + 0.35 * edge_norm

    if center_prior:
        yy, xx = np.mgrid[0:size, 0:size]
        cx = (size - 1) / 2.0
        cy = (size - 1) / 2.0
        rx = size * 0.45
        ry = size * 0.45
        dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        # 边缘区域提高分数，减少边缘融合。
        score = score + np.clip(dist - 0.6, 0, None) * 60.0

    white_ratio = max(0.05, min(0.80, white_ratio))
    threshold = np.percentile(score, white_ratio * 100.0)
    mask = (score <= threshold).astype(np.uint8) * 255

    # 去除碎点，填补小洞。
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # 删除太小的白色连通区域，减少块状噪声。
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    min_area = max(20, int(size * size * 0.002))
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == lab] = 255
    mask = cleaned

    input_224 = cv2.merge([cam2_224, cam3_224, diff])
    score_vis = normalize_u8(score)
    return input_224, mask, diff, score_vis


def make_preview(cam2_bgr, cam3_bgr, cam3_warp_bgr, input_224, mask_224, idx, out_path: Path):
    h, w = 240, 426
    cam2_s = cv2.resize(cam2_bgr, (w, h))
    cam3_s = cv2.resize(cam3_bgr, (w, h))
    warp_s = cv2.resize(cam3_warp_bgr, (w, h))

    mask_color = cv2.cvtColor(cv2.resize(mask_224, (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    cam2_y = input_224[:, :, 0]
    cam3_y = input_224[:, :, 1]
    diff = input_224[:, :, 2]
    diff_s = cv2.cvtColor(cv2.resize(diff, (w, h)), cv2.COLOR_GRAY2BGR)

    row1 = np.hstack([cam2_s, cam3_s, warp_s])
    row2 = np.hstack([diff_s, mask_color, cam2_s])
    canvas = np.vstack([row1, row2])
    cv2.putText(canvas, f"idx={idx:03d} | cam2 | cam3 | cam3_warp | diff | mask", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, default=None, help="image.zip path. If omitted, use --raw-dir")
    parser.add_argument("--raw-dir", type=str, default=None, help="directory containing yuv files")
    parser.add_argument("--out", type=str, default="fusion_dataset", help="output dataset directory")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--size", type=int, default=224, help="network input size")
    parser.add_argument("--white-ratio", type=float, default=0.35, help="rough ratio of white fusion area in pseudo mask")
    parser.add_argument("--no-center-prior", action="store_true", help="disable center prior")
    parser.add_argument("--limit", type=int, default=0, help="process first N pairs only; 0 means all")
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    if args.zip:
        raw_root = unzip_if_needed(Path(args.zip), out_dir)
    elif args.raw_dir:
        raw_root = Path(args.raw_dir)
    else:
        raise ValueError("Please provide --zip or --raw-dir")

    dirs = {
        "cam2_png": out_dir / "cam2_png",
        "cam3_png": out_dir / "cam3_png",
        "cam3_warp_png": out_dir / "cam3_warp_png",
        "input_224": out_dir / "input_224",
        "mask_224": out_dir / "mask_224",
        "diff_224": out_dir / "diff_224",
        "score_224": out_dir / "score_224",
        "preview": out_dir / "preview",
    }
    for d in dirs.values():
        ensure_dir(d)

    pairs = collect_pairs(raw_root)
    if args.limit > 0:
        pairs = pairs[:args.limit]

    print(f"[INFO] found pairs: {len(pairs)}")
    if not pairs:
        raise RuntimeError("No cam2/cam3 pairs found")

    csv_path = out_dir / "dataset_index.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["idx", "cam2_png", "cam3_png", "cam3_warp_png", "input_224", "mask_224"])

        for count, (idx, cam2_path, cam3_path) in enumerate(pairs, start=1):
            print(f"[INFO] processing {idx:03d} ({count}/{len(pairs)})")

            cam2_bgr = read_nv21_to_bgr(cam2_path, args.width, args.height)
            cam3_bgr = read_nv21_to_bgr(cam3_path, args.width, args.height)

            cam3_warp_bgr = cv2.warpPerspective(
                cam3_bgr,
                H_CAM3_TO_CAM2,
                (args.width, args.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            name = f"{idx:03d}.png"
            cam2_png = dirs["cam2_png"] / name
            cam3_png = dirs["cam3_png"] / name
            warp_png = dirs["cam3_warp_png"] / name
            input_png = dirs["input_224"] / name
            mask_png = dirs["mask_224"] / name
            diff_png = dirs["diff_224"] / name
            score_png = dirs["score_224"] / name

            cv2.imwrite(str(cam2_png), cam2_bgr)
            cv2.imwrite(str(cam3_png), cam3_bgr)
            cv2.imwrite(str(warp_png), cam3_warp_bgr)

            input_224, mask_224, diff_224, score_224 = make_input_and_mask(
                cam2_bgr,
                cam3_warp_bgr,
                args.size,
                args.white_ratio,
                center_prior=(not args.no_center_prior)
            )

            cv2.imwrite(str(input_png), input_224)
            cv2.imwrite(str(mask_png), mask_224)
            cv2.imwrite(str(diff_png), diff_224)
            cv2.imwrite(str(score_png), score_224)

            if count <= 20 or count % 10 == 0:
                make_preview(cam2_bgr, cam3_bgr, cam3_warp_bgr, input_224, mask_224, idx,
                             dirs["preview"] / f"preview_{idx:03d}.jpg")

            writer.writerow([idx, str(cam2_png), str(cam3_png), str(warp_png), str(input_png), str(mask_png)])

    print("[DONE] dataset prepared:", out_dir)
    print("[DONE] csv:", csv_path)
    print("[INFO] mask convention: white=use fusion, black=keep Cam2")


if __name__ == "__main__":
    main()
