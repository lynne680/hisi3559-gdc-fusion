# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_l(path):
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.uint8)

    if arr.shape != (224, 224):
        raise RuntimeError("image must be 224x224: %s, got %s" % (str(path), str(arr.shape)))

    return arr


def save_raw(arr, path):
    arr.astype(np.uint8).tofile(str(path))


def save_pgm(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", default="000000,000013,000020,000050")
    args = parser.parse_args()

    data_dir = Path(args.data)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [s.strip() for s in args.samples.split(",") if s.strip()]

    print("data:", data_dir)
    print("mask:", mask_dir)
    print("out:", out_dir)
    print("samples:", sample_ids)

    for sid in sample_ids:
        name = sid + ".png"

        rgb_path = data_dir / "rgb" / name
        thermal_path = data_dir / "thermal" / name
        uv_path = data_dir / "uv" / name
        mask_path = mask_dir / name

        if not rgb_path.exists():
            raise RuntimeError("missing rgb: %s" % str(rgb_path))
        if not thermal_path.exists():
            raise RuntimeError("missing thermal: %s" % str(thermal_path))
        if not uv_path.exists():
            raise RuntimeError("missing uv: %s" % str(uv_path))
        if not mask_path.exists():
            raise RuntimeError("missing mask: %s" % str(mask_path))

        rgb = read_l(rgb_path)
        thermal = read_l(thermal_path)
        uv = read_l(uv_path)
        mask = read_l(mask_path)

        save_raw(rgb, out_dir / ("rgb_%s.y" % sid))
        save_raw(thermal, out_dir / ("thermal_%s.y" % sid))
        save_raw(uv, out_dir / ("uv_%s.y" % sid))
        save_raw(mask, out_dir / ("mask_%s.y" % sid))

        save_pgm(rgb, out_dir / ("rgb_%s.pgm" % sid))
        save_pgm(thermal, out_dir / ("thermal_%s.pgm" % sid))
        save_pgm(uv, out_dir / ("uv_%s.pgm" % sid))
        save_pgm(mask, out_dir / ("mask_%s.pgm" % sid))

        print("exported:", sid)

    print("done")
    print("board path should be: /get/mm5_board_test")


if __name__ == "__main__":
    main()
