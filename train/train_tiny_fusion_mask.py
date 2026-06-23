# -*- coding: utf-8 -*-
import os
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FusionMaskDataset(Dataset):
    def __init__(self, dataset_dir, split="train", val_ratio=0.2, seed=42):
        self.dataset_dir = Path(dataset_dir)
        self.input_dir = self.dataset_dir / "input_224"
        self.mask_dir = self.dataset_dir / "mask_224"

        names = sorted([p.name for p in self.input_dir.glob("*.png")])

        random.seed(seed)
        random.shuffle(names)

        val_count = int(len(names) * val_ratio)

        if split == "train":
            self.names = names[val_count:]
        else:
            self.names = names[:val_count]

        print(f"{split}: {len(self.names)} samples")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        input_path = self.input_dir / name
        mask_path = self.mask_dir / name

        img = Image.open(input_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_np = np.array(img).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.uint8)

        # input: HWC -> CHW
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # mask: 0/255 -> 0/1
        mask_label = (mask_np >= 128).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_label)

        return img_tensor, mask_tensor, name


class TinyFusionMaskNet(nn.Module):
    """
    输入：3 x 224 x 224
    输出：2 x 224 x 224

    0 类：不融合，保留 Cam2
    1 类：融合区域
    """
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


def calc_iou(pred, target):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    inter = np.logical_and(pred == 1, target == 1).sum()
    union = np.logical_or(pred == 1, target == 1).sum()

    if union == 0:
        return 1.0

    return inter / union


def save_prediction_preview(model, dataset, device, out_dir, max_num=8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for i in range(min(max_num, len(dataset))):
            img, mask, name = dataset[i]

            x = img.unsqueeze(0).to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask_np = (mask.numpy().astype(np.uint8) * 255)
            pred_np = pred * 255

            input_vis = Image.fromarray(img_np)
            gt_vis = Image.fromarray(mask_np)
            pred_vis = Image.fromarray(pred_np)

            canvas = Image.new("RGB", (224 * 3, 224))
            canvas.paste(input_vis.convert("RGB"), (0, 0))
            canvas.paste(gt_vis.convert("RGB"), (224, 0))
            canvas.paste(pred_vis.convert("RGB"), (448, 0))

            canvas.save(out_dir / f"preview_{i:03d}_{name}")

    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="fusion_dataset_train")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out", type=str, default="train_outputs/tiny_fusion_mask")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_set = FusionMaskDataset(args.data, split="train")
    val_set = FusionMaskDataset(args.data, split="val")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = TinyFusionMaskNet().to(device)

    # 类别不均衡时，给融合区域一点权重
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0], device=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_iou = -1.0

    log_path = out_dir / "train_log.txt"
    with open(log_path, "w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0

            for imgs, masks, _ in train_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = criterion(logits, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * imgs.size(0)

            avg_loss = total_loss / len(train_set)

            model.eval()
            val_ious = []

            with torch.no_grad():
                for imgs, masks, _ in val_loader:
                    imgs = imgs.to(device)
                    masks = masks.to(device)

                    logits = model(imgs)
                    pred = torch.argmax(logits, dim=1)

                    pred_np = pred.squeeze(0).cpu().numpy()
                    mask_np = masks.squeeze(0).cpu().numpy()

                    val_ious.append(calc_iou(pred_np, mask_np))

            mean_iou = float(np.mean(val_ious)) if val_ious else 0.0

            line = f"epoch={epoch:03d}, loss={avg_loss:.6f}, val_iou={mean_iou:.4f}"
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            if mean_iou > best_iou:
                best_iou = mean_iou
                torch.save(model.state_dict(), out_dir / "best_tiny_fusion_mask.pth")
                print("saved best model:", out_dir / "best_tiny_fusion_mask.pth")

            if epoch % 10 == 0 or epoch == 1:
                save_prediction_preview(
                    model,
                    val_set,
                    device,
                    out_dir / "preview",
                    max_num=8,
                )

    torch.save(model.state_dict(), out_dir / "last_tiny_fusion_mask.pth")
    print("training finished")
    print("best val_iou:", best_iou)


if __name__ == "__main__":
    main()
