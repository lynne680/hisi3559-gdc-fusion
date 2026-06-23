# -*- coding: utf-8 -*-
import sys
import time
from pathlib import Path

import torch

# 把项目根目录加入 Python 搜索路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.predict_tiny_fusion_mask import TinyFusionMaskNet


def main():
    model = TinyFusionMaskNet()
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(1, 3, 224, 224)

    # 预热
    for _ in range(30):
        with torch.no_grad():
            _ = model(x)

    runs = 200
    t0 = time.time()

    for _ in range(runs):
        with torch.no_grad():
            _ = model(x)

    t1 = time.time()

    avg_ms = (t1 - t0) * 1000.0 / runs

    print("params:", params)
    print("trainable params:", trainable_params)
    print("avg inference time on CPU: %.3f ms" % avg_ms)
    print("fps on CPU: %.2f" % (1000.0 / avg_ms))


if __name__ == "__main__":
    main()
