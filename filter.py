import os
import random
import shutil
from pathlib import Path

# =====================
# 配置
# =====================
dataset_root = "./"

train_images = os.path.join(dataset_root, "datasets/Corn_Fruit_Segment/train/images")
train_labels = os.path.join(dataset_root, "datasets/Corn_Fruit_Segment/train/labels")

val_images = os.path.join(dataset_root, "datasets/Corn_Fruit_Segment/val/images")
val_labels = os.path.join(dataset_root, "datasets/Corn_Fruit_Segment/val/labels")

val_ratio = 0.2
seed = 42

# =====================
# 创建目录
# =====================
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# =====================
# 获取图片列表
# =====================
image_files = []

for f in os.listdir(train_images):
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ):
        image_files.append(f)

print(f"发现图片: {len(image_files)}")

# =====================
# 随机划分
# =====================
random.seed(seed)
random.shuffle(image_files)

val_num = int(len(image_files) * val_ratio)

val_files = image_files[:val_num]

print(f"验证集数量: {val_num}")

# =====================
# 移动文件
# =====================
for img_name in val_files:

    stem = Path(img_name).stem

    img_src = os.path.join(train_images, img_name)
    img_dst = os.path.join(val_images, img_name)

    label_src = os.path.join(train_labels, stem + ".txt")
    label_dst = os.path.join(val_labels, stem + ".txt")

    shutil.move(img_src, img_dst)

    if os.path.exists(label_src):
        shutil.move(label_src, label_dst)
    else:
        print(f"警告: 标签不存在 -> {label_src}")

print("划分完成")