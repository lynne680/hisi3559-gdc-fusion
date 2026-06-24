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

def soft_mask_integral(mask):
    mask_u32 = mask.astype(np.uint32)
    integral = np.zeros((H + 1, W + 1), dtype=np.uint32)
    integral[1:, 1:] = mask_u32.cumsum(axis=0).cumsum(axis=1)

    soft = np.zeros((H, W), dtype=np.uint8)

    for y in range(H):
        y1 = max(0, y - 5)
        y2 = min(H - 1, y + 5)

        for x in range(W):
            x1 = max(0, x - 5)
            x2 = min(W - 1, x + 5)

            A = integral[y1, x1]
            B = integral[y1, x2 + 1]
            C = integral[y2 + 1, x1]
            D = integral[y2 + 1, x2 + 1]

            s = D - B - C + A
            count = (x2 - x1 + 1) * (y2 - y1 + 1)
            soft[y, x] = s // count

    return soft

def reference_fusion(rgb, thermal, uv, mask):
    rgb_i = rgb.astype(np.int32)
    t_i = thermal.astype(np.int32)
    u_i = uv.astype(np.int32)

    soft = soft_mask_integral(mask).astype(np.int32)

    base = (94 * rgb_i + 4 * t_i + 2 * u_i + 50) // 100
    target = (50 * rgb_i + 40 * t_i + 10 * u_i + 50) // 100

    out = (base * (255 - soft) + target * soft + 127) // 255
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, soft.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-dir", required=True)
    parser.add_argument("--samples", default="000000,000013,000020,000050")
    parser.add_argument("--out", default="docs/experiments/mm5_three_modal/tables/board_consistency_check.csv")
    parser.add_argument("--save-diff", action="store_true")
    args = parser.parse_args()

    board_dir = Path(args.board_dir)
    samples = [s.strip() for s in args.samples.split(",") if s.strip()]
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    diff_dir = out_csv.parent.parent / "images" / "board_consistency_diff"
    if args.save_diff:
        diff_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append("sample,mae,rmse,max_abs_diff,diff_pixels,total_pixels,exact_match")

    for sid in samples:
        rgb = read_y(board_dir / f"rgb_{sid}.y")
        thermal = read_y(board_dir / f"thermal_{sid}.y")
        uv = read_y(board_dir / f"uv_{sid}.y")
        mask = read_y(board_dir / f"mask_{sid}.y")
        board = read_y(board_dir / f"fused_{sid}.y")

        ref, soft = reference_fusion(rgb, thermal, uv, mask)

        diff = board.astype(np.int16) - ref.astype(np.int16)
        abs_diff = np.abs(diff)

        mae = float(abs_diff.mean())
        rmse = float(np.sqrt((diff.astype(np.float32) ** 2).mean()))
        max_abs = int(abs_diff.max())
        diff_pixels = int((abs_diff != 0).sum())
        exact_match = diff_pixels == 0

        rows.append(
            f"{sid},{mae:.6f},{rmse:.6f},{max_abs},{diff_pixels},{N},{exact_match}"
        )

        print(
            f"{sid}: MAE={mae:.6f}, RMSE={rmse:.6f}, "
            f"MAX={max_abs}, diff_pixels={diff_pixels}, exact={exact_match}"
        )

        if args.save_diff:
            Image.fromarray(ref).save(diff_dir / f"ref_{sid}.png")
            Image.fromarray(board).save(diff_dir / f"board_{sid}.png")
            Image.fromarray((abs_diff * 20).clip(0, 255).astype(np.uint8)).save(diff_dir / f"diff_{sid}.png")
            Image.fromarray(soft).save(diff_dir / f"soft_mask_{sid}.png")

    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("saved:", out_csv)

if __name__ == "__main__":
    main()
