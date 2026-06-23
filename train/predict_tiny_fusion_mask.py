# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn


class TinyFusionMaskNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.mid = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2)

        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)

        x = self.enc2(x)
        x = self.pool2(x)

        x = self.mid(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        x = self.out(x)
        return x


def postprocess_mask(mask, min_area=500, close_ksize=7, open_ksize=3):
    """
    mask: uint8, 0/255, size=224x224

    作用：
    1. 去除面积太小的白色碎块；
    2. 闭运算填补白色区域内部小黑洞；
    3. 开运算去除细碎边缘。
    """
    mask = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean[labels == label_id] = 255

    if close_ksize > 1:
        kernel_close = np.ones((close_ksize, close_ksize), np.uint8)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close)

    if open_ksize > 1:
        kernel_open = np.ones((open_ksize, open_ksize), np.uint8)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_open)

    return clean


def predict_one(model, img_path, device, threshold):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0

    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob_fusion = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

    # 保守预测：
    # 只有融合类概率 >= threshold 的像素才判为白色融合区域
    pred = (prob_fusion >= threshold).astype(np.uint8)

    mask_224 = (pred * 255).astype(np.uint8)

    # 后处理：去碎块、填小洞、平滑区域
    mask_224 = postprocess_mask(
        mask_224,
        min_area=500,
        close_ksize=7,
        open_ksize=3
    )

    return img, Image.fromarray(mask_224.astype(np.uint8))


def collect_input_names(input_dir):
    names = []
    names.extend([p.name for p in input_dir.glob("*.png")])
    names.extend([p.name for p in input_dir.glob("*.jpg")])
    names.extend([p.name for p in input_dir.glob("*.jpeg")])
    names = sorted(list(set(names)))
    return names


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="tools/fusion_dataset_train"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="train_outputs/tiny_fusion_mask_ratio25_epoch120/best_tiny_fusion_mask.pth"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="train_outputs/tiny_fusion_mask_ratio25_epoch120/pred_masks_thr080_post"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80
    )

    parser.add_argument(
        "--export-raw-index",
        type=str,
        default="000"
    )

    args = parser.parse_args()

    data_dir = Path(args.data)
    input_dir = data_dir / "input_224"
    gt_mask_dir = data_dir / "mask_224"

    out_dir = Path(args.out)
    out_224 = out_dir / "mask_224"
    out_1080 = out_dir / "mask_1920x1080"
    preview_dir = out_dir / "preview"

    out_224.mkdir(parents=True, exist_ok=True)
    out_1080.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise RuntimeError("input_224 directory not found: %s" % str(input_dir))

    device = torch.device("cpu")

    model = TinyFusionMaskNet().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    names = collect_input_names(input_dir)

    print("data:", data_dir)
    print("model:", args.model)
    print("threshold:", args.threshold)
    print("input samples:", len(names))

    if len(names) == 0:
        raise RuntimeError("No input images found in %s" % str(input_dir))

    for idx, name in enumerate(names):
        img_path = input_dir / name
        stem = Path(name).stem

        img, mask_224 = predict_one(
            model,
            img_path,
            device,
            args.threshold
        )

        mask_224.save(out_224 / ("%s.png" % stem))

        # 放大到 1920x1080。
        # 这里仍用 NEAREST，因为当前板端 C 代码使用的是二值 mask 快速区间复制。
        mask_1080 = mask_224.resize((1920, 1080), Image.NEAREST)
        mask_1080.save(out_1080 / ("%s_1920x1080.png" % stem))

        # 保存板端可直接读取的 raw mask 文件
        with open(out_1080 / ("%s_1920x1080.y" % stem), "wb") as f:
            f.write(mask_1080.tobytes())

        # 预览图：
        # 三列：输入 / 伪标签 / 预测
        if idx < 12:
            gt_path = gt_mask_dir / name

            if gt_path.exists():
                gt = Image.open(gt_path).convert("L")

                canvas = Image.new("RGB", (224 * 3, 224))
                canvas.paste(img.convert("RGB"), (0, 0))
                canvas.paste(gt.convert("RGB"), (224, 0))
                canvas.paste(mask_224.convert("RGB"), (448, 0))
            else:
                canvas = Image.new("RGB", (224 * 2, 224))
                canvas.paste(img.convert("RGB"), (0, 0))
                canvas.paste(mask_224.convert("RGB"), (224, 0))

            canvas.save(preview_dir / ("pred_preview_%s.png" % stem))

    print("saved:", out_dir)

    export_name = "%s_1920x1080.y" % args.export_raw_index
    export_path = out_1080 / export_name

    if export_path.exists():
        print("selected raw mask:", export_path)
        print("copy command:")
        print("cp %s /home/xia/nfs/fusion_mask_selected_578_1920x1080.y" % str(export_path))
    else:
        print("selected raw mask not found:", export_path)
        print("available examples:")

        for p in sorted(out_1080.glob("*_1920x1080.y"))[:10]:
            print(" ", p.name)


if __name__ == "__main__":
    main()
