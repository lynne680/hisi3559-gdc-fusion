from PIL import Image

src = "fusion_mask_selected_578_1920x1080.png"
dst = "fusion_mask_selected_578_1920x1080.y"

img = Image.open(src).convert("L")
img = img.resize((1920, 1080), Image.NEAREST)

# 二值化，确保只有 0 和 255
img = img.point(lambda p: 255 if p >= 128 else 0)

with open(dst, "wb") as f:
    f.write(img.tobytes())

print("saved:", dst)
print("size should be:", 1920 * 1080)
