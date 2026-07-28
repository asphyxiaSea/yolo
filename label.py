from ultralytics import RTDETR
import os

# 加载你训练好的权重
model = RTDETR("runs/detect/检测_油茶嫩芽_RTDETR_1920/weights/best.pt")

# 未标注图片所在目录
source_dir = "datasets/油茶"

# 批量推理并保存 YOLO 格式标注(txt)
results = model.predict(
    source=source_dir,
    imgsz=1920,          # 要和训练时保持一致
    conf=0.25,            # 置信度阈值,低于这个的框不会被保留,可以先设低一点方便人工筛查
    save=True,            # 不需要保存可视化图片,只要标注文件的话设 False,想顺便看效果检查可以设 True
    # save_txt=True,         # 关键参数:保存 YOLO 格式的 txt 标注
    # save_conf=False,       # 是否在 txt 里附带置信度分数,不需要就设 False
    # project="datasets/labels",
    # name="camellia_sprouts_pseudo_label",
)

print(f"标注文件已保存到: datasets/labels/camellia_sprouts_pseudo_label/labels/")