# -*- coding: utf-8 -*-
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("failed to read image: %s" % str(path))

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        mn = img.min()
        mx = img.max()
        img = ((img - mn) / (mx - mn + 1e-6) * 255.0).astype(np.uint8)

    return img


def read_index(data_dir):
    index_path = Path(data_dir) / "dataset_index.csv"
    mapping = {}

    with open(index_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["id"]
            key = row["key"]
            mapping[sample_id] = key

    return mapping


def resize_nearest(img, size=224):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def make_mask_from_anno(anno, dilate_ksize=15, close_ksize=7, open_ksize=3):
    """
    ANNO_CLASS:
    0 = 背景
    >0 = 目标类别区域

    这里取 anno > 0，表示所有标注目标都作为融合区域。
    """
    mask = (anno > 0).astype(np.uint8) * 255

    if dilate_ksize > 1:
        kernel_dilate = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    if close_ksize > 1:
        kernel_close = np.ones((close_ksize, close_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    if open_ksize > 1:
        kernel_open = np.ones((open_ksize, open_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask


def make_fusion_preview(rgb, thermal, uv, mask):
    rgb_f = rgb.astype(np.float32)
    t_f = thermal.astype(np.float32)
    u_f = uv.astype(np.float32)

    aux = 0.60 * t_f + 0.40 * u_f
    fused = 0.70 * rgb_f + 0.30 * aux

    m = mask.astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (9, 9), 0)

    out = rgb_f * (1.0 - m) + fused * m
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def make_preview(rgb, thermal, uv, anno, mask, fused, out_path):
    h, w = 224, 224

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    thermal_color = cv2.applyColorMap(thermal, cv2.COLORMAP_JET)
    uv_color = cv2.applyColorMap(uv, cv2.COLORMAP_JET)

    anno_vis = anno.copy()
    if anno_vis.max() > 0:
        anno_vis = (anno_vis.astype(np.float32) / anno_vis.max() * 255.0).astype(np.uint8)
    anno_color = cv2.applyColorMap(anno_vis, cv2.COLORMAP_JET)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    fused_bgr = cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR)

    canvas = np.zeros((h, w * 6, 3), dtype=np.uint8)

    canvas[:, 0:w] = rgb_bgr
    canvas[:, w:2*w] = thermal_color
    canvas[:, 2*w:3*w] = uv_color
    canvas[:, 3*w:4*w] = anno_color
    canvas[:, 4*w:5*w] = mask_bgr
    canvas[:, 5*w:6*w] = fused_bgr

    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--anno-dir", required=True)
    parser.add_argument("--dilate", type=int, default=15)
    parser.add_argument("--close", type=int, default=7)
    parser.add_argument("--open", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data)
    anno_dir = Path(args.anno_dir)

    rgb_dir = data_dir / "rgb"
    thermal_dir = data_dir / "thermal"
    uv_dir = data_dir / "uv"

    mask_dir = data_dir / "mask_224"
    preview_dir = data_dir / "anno_mask_preview"
    fusion_dir = data_dir / "fusion_preview"

    for d in [mask_dir, preview_dir, fusion_dir]:
        d.mkdir(parents=True, exist_ok=True)

    index_map = read_index(data_dir)
    rgb_files = sorted(rgb_dir.glob("*.png"))

    print("data:", data_dir)
    print("anno:", anno_dir)
    print("samples:", len(rgb_files))
    print("dilate:", args.dilate, "close:", args.close, "open:", args.open)

    valid_count = 0

    for idx, rgb_path in enumerate(rgb_files):
        name = rgb_path.name
        sample_id = Path(name).stem

        if sample_id not in index_map:
            print("skip no index:", sample_id)
            continue

        key = index_map[sample_id]
        anno_path = anno_dir / ("%s.png" % key)

        if not anno_path.exists():
            print("skip missing anno:", anno_path)
            continue

        rgb = read_gray(rgb_dir / name)
        thermal = read_gray(thermal_dir / name)
        uv = read_gray(uv_dir / name)

        anno = read_gray(anno_path)
        anno = resize_nearest(anno, 224)

        mask = make_mask_from_anno(
            anno,
            dilate_ksize=args.dilate,
            close_ksize=args.close,
            open_ksize=args.open
        )

        fused = make_fusion_preview(rgb, thermal, uv, mask)

        Image.fromarray(mask).save(mask_dir / name)
        Image.fromarray(fused).save(fusion_dir / name)

        if idx < 100:
            make_preview(
                rgb,
                thermal,
                uv,
                anno,
                mask,
                fused,
                preview_dir / ("anno_mask_preview_%s.jpg" % sample_id)
            )

        valid_count += 1

        if idx % 50 == 0:
            white_ratio = float((mask > 0).sum()) / float(mask.shape[0] * mask.shape[1])
            print("processed:", idx, name, "key=", key, "white_ratio=%.4f" % white_ratio)

    print("done")
    print("valid:", valid_count)
    print("mask dir:", mask_dir)
    print("preview dir:", preview_dir)


if __name__ == "__main__":
    main()
