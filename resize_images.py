# resize_images.py
from pathlib import Path

import cv2
from tqdm import tqdm

SRC_DIR = Path("datasets/实拍_油茶花")
DST_DIR = Path("datasets/实拍_油茶花_resized")
TARGET_SIZE = 1280

DST_DIR.mkdir(parents=True, exist_ok=True)

img_paths = [p for p in SRC_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

for img_path in tqdm(img_paths, desc="处理图片"):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[跳过-读取失败] {img_path}")
        continue

    h, w = img.shape[:2]
    out_path = DST_DIR / img_path.name

    if max(h, w) <= TARGET_SIZE:
        cv2.imwrite(str(out_path), img)  # 已经不大于目标尺寸,原样保存
        continue

    scale = TARGET_SIZE / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), resized)

print("完成")