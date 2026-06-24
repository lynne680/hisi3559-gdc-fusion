import argparse
from pathlib import Path
import numpy as np
from PIL import Image

W, H = 224, 224
N = W * H

def read_y(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size != N:
        raise ValueError(f"size error: {path}, got {arr.size}, expect {N}")
    return arr.reshape(H, W)

def read_prob(path):
    arr = np.fromfile(str(path), dtype=np.float32)
    if arr.size != N:
        raise ValueError(f"size error: {path}, got {arr.size}, expect {N}")
    return arr.reshape(H, W)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/home/xia/nfs/tinyfusion_board_test")
    parser.add_argument("--sample", default="000000")
    args = parser.parse_args()

    d = Path(args.dir)
    sid = args.sample

    ref_prob = read_prob(d / f"ref_prob_{sid}.bin")
    board_prob = read_prob(d / f"board_prob_{sid}.bin")
    ref_mask = read_y(d / f"ref_mask_{sid}.y")
    board_mask = read_y(d / f"board_mask_{sid}.y")

    prob_diff = np.abs(ref_prob - board_prob)
    mask_diff = np.abs(ref_mask.astype(np.int16) - board_mask.astype(np.int16))

    print("[PROB]")
    print("mae:", float(prob_diff.mean()))
    print("rmse:", float(np.sqrt((prob_diff ** 2).mean())))
    print("max_abs:", float(prob_diff.max()))

    print("[MASK]")
    print("diff_pixels:", int((mask_diff != 0).sum()))
    print("total_pixels:", N)
    print("exact_match:", bool((mask_diff != 0).sum() == 0))

    diff_vis = np.clip(mask_diff * 20, 0, 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(d / f"board_mask_diff_{sid}.png")
    print("saved:", d / f"board_mask_diff_{sid}.png")

if __name__ == "__main__":
    main()
