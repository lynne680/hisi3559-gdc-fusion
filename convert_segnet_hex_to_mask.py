import numpy as np
from PIL import Image

hex_file = "seg0_layer38_output0_inst.linear.hex"

chn = 11
height = 224
width = 224

values = []

with open(hex_file, "r") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue

        v = int(s, 16)

        # 转成 signed int32
        if v >= 0x80000000:
            v -= 0x100000000

        values.append(v)

expected = chn * height * width
print("read values:", len(values))
print("expected:", expected)

if len(values) != expected:
    raise ValueError("hex values count mismatch")

# 官方 dump 顺序是：C -> H -> W
arr = np.array(values, dtype=np.int32).reshape(chn, height, width)

# 对每个像素取 11 个通道中分数最大的类别
label = np.argmax(arr, axis=0).astype(np.uint8)

# 为了方便看，把 0~10 类拉伸到 0~255
label_vis = (label * (255 // (chn - 1))).astype(np.uint8)

Image.fromarray(label_vis).save("segnet_label_vis.png")
Image.fromarray(label).save("segnet_label_raw.png")

print("saved: segnet_label_vis.png")
print("saved: segnet_label_raw.png")
print("unique labels:", np.unique(label))
unique, counts = np.unique(label, return_counts=True)
print("label histogram:")
for u, c in zip(unique, counts):
    print("class %d: %d pixels, %.2f%%" % (u, c, c * 100.0 / (224 * 224)))

for cls in range(11):
    mask = np.zeros_like(label, dtype=np.uint8)

fusion_classes = [8]

mask = np.zeros_like(label, dtype=np.uint8)
for cls in fusion_classes:
    mask[label == cls] = 255

Image.fromarray(mask).save("fusion_mask_selected_578.png")
print("saved: fusion_mask_selected_578.png")


seg_vis = Image.open("segnet_label_vis.png").convert("RGB")
mask_rgb = Image.fromarray(mask).convert("RGB")

overlay = Image.blend(seg_vis, mask_rgb, alpha=0.35)
overlay.save("fusion_mask_selected_578_overlay.png")
print("saved: fusion_mask_selected_578_overlay.png")
# 放大到 1920x1080，供后续板端融合使用
from PIL import ImageFilter

mask_1920 = Image.fromarray(mask).resize((1920, 1080), Image.BILINEAR)
mask_1920 = mask_1920.filter(ImageFilter.GaussianBlur(radius=6))
mask_1920.save("fusion_mask_selected_soft_1920x1080.png")
mask_1920.save("fusion_mask_selected_578_1920x1080.png")
print("saved: fusion_mask_selected_578_1920x1080.png")
