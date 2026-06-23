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


def normalize_01(img):
    img = img.astype(np.float32)
    mn = np.percentile(img, 1)
    mx = np.percentile(img, 99)
    out = (img - mn) / (mx - mn + 1e-6)
    out = np.clip(out, 0.0, 1.0)
    return out


def edge_strength(img):
    img_f = img.astype(np.uint8)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = normalize_01(mag)
    return mag


def make_pseudo_mask(rgb, thermal, uv, white_ratio=0.28):
    """
    根据三路图像自动生成融合区域伪标签。

    思路：
    1. 热红外和紫外中显著、边缘明显的区域更需要参与融合；
    2. 可见光本身纹理很强的区域少融合，避免破坏自然画面；
    3. 只保留分数最高的一部分区域，保证 mask 保守；
    4. 通过形态学去碎块、填小洞、平滑边界。
    """
    rgb_n = normalize_01(rgb)
    t_n = normalize_01(thermal)
    u_n = normalize_01(uv)

    e_rgb = edge_strength(rgb)
    e_t = edge_strength(thermal)
    e_u = edge_strength(uv)

    # 热红外/紫外显著性
    t_sal = np.abs(t_n - rgb_n)
    u_sal = np.abs(u_n - rgb_n)

    # 三路辅助信息分数
    # 热红外权重大一些，紫外次之
    score = (
        0.38 * t_sal +
        0.25 * u_sal +
        0.20 * e_t +
        0.12 * e_u -
        0.10 * e_rgb
    )

    score = normalize_01(score)

    # 保守取前 white_ratio 比例作为融合区域
    thr = np.percentile(score, 100.0 * (1.0 - white_ratio))
    mask = (score >= thr).astype(np.uint8) * 255

    # 去除太小碎块
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask, dtype=np.uint8)

    min_area = 180
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == label_id] = 255

    # 闭运算：连接区域、填小洞
    kernel_close = np.ones((7, 7), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close)

    # 开运算：去毛刺
    kernel_open = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_open)

    return clean, (score * 255.0).astype(np.uint8)


def make_fusion_preview(rgb, thermal, uv, mask):
    """
    生成一个简单融合预览图，不作为训练真值，只用于肉眼检查。
    mask白色区域：加入热红外和紫外信息
    mask黑色区域：主要保留可见光
    """
    rgb_f = rgb.astype(np.float32)
    t_f = thermal.astype(np.float32)
    u_f = uv.astype(np.float32)

    aux = 0.60 * t_f + 0.40 * u_f
    fused = 0.70 * rgb_f + 0.30 * aux

    m = (mask.astype(np.float32) / 255.0)
    m = cv2.GaussianBlur(m, (9, 9), 0)

    out = rgb_f * (1.0 - m) + fused * m
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def make_preview(rgb, thermal, uv, mask, score, fused, out_path):
    h, w = 224, 224

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    thermal_color = cv2.applyColorMap(thermal, cv2.COLORMAP_JET)
    uv_color = cv2.applyColorMap(uv, cv2.COLORMAP_JET)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    score_color = cv2.applyColorMap(score, cv2.COLORMAP_JET)
    fused_bgr = cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR)

    canvas = np.zeros((h, w * 6, 3), dtype=np.uint8)

    canvas[:, 0:w] = rgb_bgr
    canvas[:, w:2*w] = thermal_color
    canvas[:, 2*w:3*w] = uv_color
    canvas[:, 3*w:4*w] = score_color
    canvas[:, 4*w:5*w] = mask_bgr
    canvas[:, 5*w:6*w] = fused_bgr

    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="prepared MM5 dataset directory")
    parser.add_argument("--white-ratio", type=float, default=0.28)
    args = parser.parse_args()

    data_dir = Path(args.data)

    rgb_dir = data_dir / "rgb"
    thermal_dir = data_dir / "thermal"
    uv_dir = data_dir / "uv"

    mask_dir = data_dir / "mask_224"
    score_dir = data_dir / "score_224"
    fusion_dir = data_dir / "fusion_preview"
    preview_dir = data_dir / "mask_preview"

    for d in [mask_dir, score_dir, fusion_dir, preview_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(rgb_dir.glob("*.png"))

    print("data:", data_dir)
    print("samples:", len(rgb_files))
    print("white_ratio:", args.white_ratio)

    if len(rgb_files) == 0:
        raise RuntimeError("no rgb images found: %s" % str(rgb_dir))

    for idx, rgb_path in enumerate(rgb_files):
        name = rgb_path.name

        thermal_path = thermal_dir / name
        uv_path = uv_dir / name

        if not thermal_path.exists() or not uv_path.exists():
            print("skip missing pair:", name)
            continue

        rgb = read_gray(rgb_path)
        thermal = read_gray(thermal_path)
        uv = read_gray(uv_path)

        mask, score = make_pseudo_mask(
            rgb,
            thermal,
            uv,
            white_ratio=args.white_ratio
        )

        fused = make_fusion_preview(rgb, thermal, uv, mask)

        Image.fromarray(mask).save(mask_dir / name)
        Image.fromarray(score).save(score_dir / name)
        Image.fromarray(fused).save(fusion_dir / name)

        if idx < 100:
            make_preview(
                rgb,
                thermal,
                uv,
                mask,
                score,
                fused,
                preview_dir / ("mask_preview_%s.jpg" % Path(name).stem)
            )

        if idx % 50 == 0:
            white = int((mask > 0).sum())
            total = mask.shape[0] * mask.shape[1]
            print("processed:", idx, name, "white_ratio_actual=%.3f" % (white / float(total)))

    print("done")
    print("mask dir:", mask_dir)
    print("preview dir:", preview_dir)


if __name__ == "__main__":
    main()
