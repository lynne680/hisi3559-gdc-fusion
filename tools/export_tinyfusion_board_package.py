import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.predict_tiny_fusion_mask import TinyFusionMaskNet

W, H = 224, 224
N = W * H

ORDER = [
    "enc1.0.weight", "enc1.0.bias",
    "enc1.2.weight", "enc1.2.bias",
    "enc2.0.weight", "enc2.0.bias",
    "enc2.2.weight", "enc2.2.bias",
    "mid.0.weight", "mid.0.bias",
    "mid.2.weight", "mid.2.bias",
    "up2.weight", "up2.bias",
    "dec2.0.weight", "dec2.0.bias",
    "up1.weight", "up1.bias",
    "dec1.0.weight", "dec1.0.bias",
    "out.weight", "out.bias",
]

def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt

def read_y(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size != N:
        raise ValueError(f"size error: {path}, got {arr.size}, expect {N}")
    return arr.reshape(H, W)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--board-dir", default="/home/xia/nfs/mm5_board_test")
    parser.add_argument("--sample", default="000000")
    parser.add_argument("--out", default="/home/xia/nfs/tinyfusion_board_test")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = get_state_dict(ckpt)

    model = TinyFusionMaskNet()
    model.load_state_dict(state, strict=True)
    model.eval()

    # export weights
    weights_path = out / "tinyfusion_weights.bin"
    meta_path = out / "tinyfusion_weights_meta.txt"

    offset = 0
    with open(weights_path, "wb") as fw, open(meta_path, "w", encoding="utf-8") as fm:
        for name in ORDER:
            t = model.state_dict()[name].detach().cpu().contiguous().numpy().astype(np.float32)
            t.tofile(fw)
            fm.write(f"{name} offset={offset} num={t.size} shape={tuple(t.shape)}\n")
            offset += t.size

    # prepare one sample input and pytorch reference
    board_dir = Path(args.board_dir)
    sid = args.sample

    rgb = read_y(board_dir / f"rgb_{sid}.y")
    thermal = read_y(board_dir / f"thermal_{sid}.y")
    uv = read_y(board_dir / f"uv_{sid}.y")

    x_np = np.stack([rgb, thermal, uv], axis=0).astype(np.float32) / 255.0
    x = torch.from_numpy(x_np).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
        mask = (prob >= args.threshold).astype(np.uint8) * 255

    x_np.astype(np.float32).tofile(str(out / f"input_{sid}_chw_float.bin"))
    prob.astype(np.float32).tofile(str(out / f"ref_prob_{sid}.bin"))
    mask.astype(np.uint8).tofile(str(out / f"ref_mask_{sid}.y"))

    Image.fromarray(mask).save(out / f"ref_mask_{sid}.png")

    print("export done")
    print("weights:", weights_path)
    print("meta:", meta_path)
    print("sample:", sid)
    print("input:", out / f"input_{sid}_chw_float.bin")
    print("ref_prob:", out / f"ref_prob_{sid}.bin")
    print("ref_mask:", out / f"ref_mask_{sid}.y")
    print("threshold:", args.threshold)

if __name__ == "__main__":
    main()
