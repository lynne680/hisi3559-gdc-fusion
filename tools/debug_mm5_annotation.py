# -*- coding: utf-8 -*-
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("failed to read: %s" % str(path))

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def resize_nearest(img, size=224):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def normalize_vis(img):
    img = img.astype(np.float32)
    if img.max() <= img.min():
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return (out * 255).astype(np.uint8)


def load_key_from_index(data_dir, sample_id):
    index_path = Path(data_dir) / "dataset_index.csv"

    with open(index_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] == sample_id:
                return row["key"]

    raise RuntimeError("sample id not found in dataset_index.csv: %s" % sample_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--mm5-root", required=True)
    parser.add_argument("--sample-id", default="000000")
    parser.add_argument("--out", default="mm5_annotation_debug")
    args = parser.parse_args()

    data_dir = Path(args.data)
    mm5_root = Path(args.mm5_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_id = args.sample_id
    key = load_key_from_index(data_dir, sample_id)

    rgb_path = data_dir / "rgb" / ("%s.png" % sample_id)
    rgb = read_gray(rgb_path)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)

    anno_dirs = [
        "ANNO_CLASS",
        "ANNO_VIS_CLASS",
        "ANNO_INST",
        "ANNO_VIS_INST",
    ]

    print("sample_id:", sample_id)
    print("original key:", key)

    panels = [rgb_bgr]
    titles = ["rgb"]

    for anno_name in anno_dirs:
        anno_path = mm5_root / anno_name / ("%s.png" % key)

        if not anno_path.exists():
            print("missing:", anno_path)
            continue

        anno = read_gray(anno_path)
        anno = resize_nearest(anno, 224)

        vals, counts = np.unique(anno, return_counts=True)

        print("\n[%s]" % anno_name)
        for v, c in zip(vals, counts):
            ratio = c / float(anno.shape[0] * anno.shape[1])
            print("  value=%s count=%d ratio=%.4f" % (str(v), int(c), ratio))

        anno_vis = normalize_vis(anno)
        anno_color = cv2.applyColorMap(anno_vis, cv2.COLORMAP_JET)
        panels.append(anno_color)
        titles.append(anno_name)

        # 为每个非零 value 单独输出二值 mask，方便看哪个类是目标
        for v in vals:
            if int(v) == 0:
                continue

            mask = (anno == v).astype(np.uint8) * 255
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            overlay = rgb_bgr.copy()
            overlay[mask > 0] = (0, 0, 255)

            one = np.zeros((224, 224 * 3, 3), dtype=np.uint8)
            one[:, 0:224] = rgb_bgr
            one[:, 224:448] = mask_bgr
            one[:, 448:672] = overlay

            out_path = out_dir / ("%s_%s_value_%s.jpg" % (sample_id, anno_name, str(int(v))))
            cv2.imwrite(str(out_path), one)

    canvas = np.zeros((224, 224 * len(panels), 3), dtype=np.uint8)

    for i, p in enumerate(panels):
        canvas[:, i * 224:(i + 1) * 224] = p

    cv2.imwrite(str(out_dir / ("%s_all_annos.jpg" % sample_id)), canvas)

    print("\nsaved:", out_dir)
    print("open:")
    print("xdg-open %s" % str(out_dir / ("%s_all_annos.jpg" % sample_id)))


if __name__ == "__main__":
    main()
