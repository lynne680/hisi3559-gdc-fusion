# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("failed to read image: %s" % str(path))
    return img.astype(np.uint8)


def norm(img):
    img = img.astype(np.float32)
    mn = np.percentile(img, 1)
    mx = np.percentile(img, 99)
    img = np.clip((img - mn) / (mx - mn + 1e-6), 0, 1)
    return img


def fuse_one(rgb, thermal, uv, mask):
    rgb_f = norm(rgb)
    t_f = norm(thermal)
    u_f = norm(uv)

    # 三路融合策略：
    # 非 mask 区域主要保留可见光；
    # mask 区域加入热红外和紫外增强信息。
    # aux = 0.60 * t_f + 0.40 * u_f
    # fused_region = 0.65 * rgb_f + 0.35 * aux

    # m = mask.astype(np.float32) / 255.0
    # m = cv2.GaussianBlur(m, (11, 11), 0)

    # fused = rgb_f * (1.0 - m) + fused_region * m 

    # 背景区域：以可见光为主，轻微加入热红外和紫外
    base = 0.90 * rgb_f + 0.07 * t_f + 0.03 * u_f

    # 目标区域：适当增强热红外和紫外信息
    target = 0.55 * rgb_f + 0.30 * t_f + 0.15 * u_f

    m = mask.astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (11, 11), 0)

    fused = base * (1.0 - m) + target * m


    fused = np.clip(fused * 255.0, 0, 255).astype(np.uint8)

    return fused


def make_preview(rgb, thermal, uv, mask, fused, out_path):
    h, w = 224, 224

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    thermal_color = cv2.applyColorMap(thermal, cv2.COLORMAP_JET)
    uv_color = cv2.applyColorMap(uv, cv2.COLORMAP_JET)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    fused_bgr = cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR)

    canvas = np.zeros((h, w * 5, 3), dtype=np.uint8)

    canvas[:, 0:w] = rgb_bgr
    canvas[:, w:2*w] = thermal_color
    canvas[:, 2*w:3*w] = uv_color
    canvas[:, 3*w:4*w] = mask_bgr
    canvas[:, 4*w:5*w] = fused_bgr

    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out)

    fused_dir = out_dir / "fused"
    preview_dir = out_dir / "preview"

    fused_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    rgb_dir = data_dir / "rgb"
    thermal_dir = data_dir / "thermal"
    uv_dir = data_dir / "uv"

    rgb_files = sorted(rgb_dir.glob("*.png"))

    print("samples:", len(rgb_files))
    print("mask:", mask_dir)
    print("out:", out_dir)

    for idx, rgb_path in enumerate(rgb_files):
        name = rgb_path.name

        rgb = read_gray(rgb_dir / name)
        thermal = read_gray(thermal_dir / name)
        uv = read_gray(uv_dir / name)
        mask = read_gray(mask_dir / name)

        fused = fuse_one(rgb, thermal, uv, mask)

        Image.fromarray(fused).save(fused_dir / name)

        if idx < 100:
            make_preview(
                rgb,
                thermal,
                uv,
                mask,
                fused,
                preview_dir / ("fusion_preview_%s.jpg" % Path(name).stem)
            )

        if idx % 50 == 0:
            print("processed:", idx, name)

    print("done")
    print("preview:", preview_dir)
    print("fused:", fused_dir)


if __name__ == "__main__":
    main()
