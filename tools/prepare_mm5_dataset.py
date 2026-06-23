# -*- coding: utf-8 -*-
import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def natural_key(s):
    s = str(s)
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def collect_images(root):
    root = Path(root)
    files = []
    for ext in IMG_EXTS:
        files.extend(root.rglob("*" + ext))
        files.extend(root.rglob("*" + ext.upper()))
    files = sorted(files, key=lambda p: natural_key(p.name))
    return files


def stem_key(path):
    return Path(path).stem


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("failed to read image: %s" % str(path))

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        mn = np.percentile(img, 1)
        mx = np.percentile(img, 99)
        img = np.clip((img - mn) / (mx - mn + 1e-6), 0, 1)
        img = (img * 255.0).astype(np.uint8)

    return img


def resize_gray(img, size=224):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def save_png(gray, path):
    Image.fromarray(gray.astype(np.uint8)).save(path)


def make_preview(rgb, thermal, uv, input_img, out_path):
    h, w = 224, 224

    rgb_s = resize_gray(rgb, 224)
    thermal_s = resize_gray(thermal, 224)
    uv_s = resize_gray(uv, 224)

    rgb_vis = cv2.cvtColor(rgb_s, cv2.COLOR_GRAY2BGR)
    thermal_vis = cv2.applyColorMap(thermal_s, cv2.COLORMAP_JET)
    uv_vis = cv2.applyColorMap(uv_s, cv2.COLORMAP_JET)
    input_vis = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    canvas = np.zeros((h, w * 4, 3), dtype=np.uint8)
    canvas[:, 0:w] = rgb_vis
    canvas[:, w:2*w] = thermal_vis
    canvas[:, 2*w:3*w] = uv_vis
    canvas[:, 3*w:4*w] = input_vis

    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--thermal-dir", required=True)
    parser.add_argument("--uv-dir", required=True)
    parser.add_argument("--out", default="mm5_fusion_dataset")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--max-num", type=int, default=0)
    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir)
    thermal_dir = Path(args.thermal_dir)
    uv_dir = Path(args.uv_dir)
    out_dir = Path(args.out)

    out_rgb = out_dir / "rgb"
    out_thermal = out_dir / "thermal"
    out_uv = out_dir / "uv"
    out_input = out_dir / "input_224"
    out_preview = out_dir / "preview"

    for d in [out_rgb, out_thermal, out_uv, out_input, out_preview]:
        d.mkdir(parents=True, exist_ok=True)

    rgb_files = collect_images(rgb_dir)
    thermal_files = collect_images(thermal_dir)
    uv_files = collect_images(uv_dir)

    print("rgb files:", len(rgb_files))
    print("thermal files:", len(thermal_files))
    print("uv files:", len(uv_files))

    rgb_map = {stem_key(p): p for p in rgb_files}
    thermal_map = {stem_key(p): p for p in thermal_files}
    uv_map = {stem_key(p): p for p in uv_files}

    common_keys = sorted(
        set(rgb_map.keys()) & set(thermal_map.keys()) & set(uv_map.keys()),
        key=natural_key
    )

    if args.max_num > 0:
        common_keys = common_keys[:args.max_num]

    print("paired samples:", len(common_keys))

    if len(common_keys) == 0:
        print("No paired samples found.")
        return

    index_path = out_dir / "dataset_index.csv"

    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "key", "rgb", "thermal", "uv", "input_224"])

        for idx, key in enumerate(common_keys):
            sample_id = "%06d" % idx

            rgb_path = rgb_map[key]
            thermal_path = thermal_map[key]
            uv_path = uv_map[key]

            rgb = read_gray(rgb_path)
            thermal = read_gray(thermal_path)
            uv = read_gray(uv_path)

            rgb_224 = resize_gray(rgb, args.size)
            thermal_224 = resize_gray(thermal, args.size)
            uv_224 = resize_gray(uv, args.size)

            input_224 = np.stack([rgb_224, thermal_224, uv_224], axis=2)

            save_png(rgb_224, out_rgb / ("%s.png" % sample_id))
            save_png(thermal_224, out_thermal / ("%s.png" % sample_id))
            save_png(uv_224, out_uv / ("%s.png" % sample_id))
            Image.fromarray(input_224.astype(np.uint8)).save(out_input / ("%s.png" % sample_id))

            if idx < 100:
                make_preview(
                    rgb,
                    thermal,
                    uv,
                    input_224,
                    out_preview / ("preview_%s.jpg" % sample_id)
                )

            writer.writerow([
                sample_id,
                key,
                str(rgb_path),
                str(thermal_path),
                str(uv_path),
                str(out_input / ("%s.png" % sample_id))
            ])

            if idx % 50 == 0:
                print("processed:", idx)

    print("done")
    print("output:", out_dir)
    print("index:", index_path)


if __name__ == "__main__":
    main()
