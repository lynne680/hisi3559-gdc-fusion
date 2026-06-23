# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import cv2
import numpy as np


def read_mask(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("failed to read: %s" % str(path))
    return img > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True, help="ground truth mask directory")
    parser.add_argument("--pred-dir", required=True, help="predicted mask directory")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    if not gt_dir.exists():
        raise RuntimeError("gt-dir not found: %s" % str(gt_dir))

    if not pred_dir.exists():
        raise RuntimeError("pred-dir not found: %s" % str(pred_dir))

    ious = []
    dices = []
    precisions = []
    recalls = []
    pred_white_ratios = []
    gt_white_ratios = []

    gt_files = sorted(gt_dir.glob("*.png"))

    for gt_path in gt_files:
        name = gt_path.name
        pred_path = pred_dir / name

        if not pred_path.exists():
            print("skip missing pred:", pred_path)
            continue

        gt = read_mask(gt_path)
        pred = read_mask(pred_path)

        inter = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()

        pred_sum = pred.sum()
        gt_sum = gt.sum()

        iou = inter / (union + 1e-9)
        dice = 2.0 * inter / (pred_sum + gt_sum + 1e-9)
        precision = inter / (pred_sum + 1e-9)
        recall = inter / (gt_sum + 1e-9)

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        pred_white_ratios.append(pred.mean())
        gt_white_ratios.append(gt.mean())

    if len(ious) == 0:
        raise RuntimeError("no matched mask pairs found")

    print("samples:", len(ious))
    print("IoU:", np.mean(ious))
    print("Dice:", np.mean(dices))
    print("Precision:", np.mean(precisions))
    print("Recall:", np.mean(recalls))
    print("Pred white ratio:", np.mean(pred_white_ratios))
    print("GT white ratio:", np.mean(gt_white_ratios))


if __name__ == "__main__":
    main()
