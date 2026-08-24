from ultralytics import YOLO, RTDETR


# 继续训练
# model = YOLO("runs/detect/检测_农业综合虫害/weights/last.pt")
# model.train(resume=True)



#  YOLO 检测_模型训练
# model = YOLO("models/yolo26s.pt")
# model.train(
#     data="datasets/油茶嫩芽/data.yaml",
#     epochs=100,
#     batch=16,
#     imgsz=1920,
#     device=1,
#     workers=8,
#     name="检测_油茶嫩芽"
# )


# RTDETR 检测_模型训练
model = RTDETR("rtdetr-l.pt")  # 加载预训练权重做迁移学习

results = model.train(
    data="datasets/玉米苗_无人机视角_400/data.yaml",
    epochs=100,
    imgsz=400,
    batch=16,          # RT-DETR 显存占用比 YOLO 大不少,batch 别设太高
    device=1,          # 用第几张 GPU
    workers=8,
    patience=20,       # early stopping
    # project="runs/rtdetr",
    name="检测_玉米苗_无人机视角_RTDETR_400",
)


# RTDETR 分类_模型训练
# model = YOLO("yolov8s-cls.pt")  # n/s/m/l/x 几种规模可选

# model.train(
#     data="datasets/玉米_病虫害_分类_640",   # 指向包含train/val的根目录，不是某个yaml
#     epochs=100,
#     imgsz=640,                 # 分类任务默认224，比检测任务的640小很多
#     batch=64,                  # 分类模型显存占用小，batch可以给大一些
#     device=1,
#     patience=20,               # early stopping
# )

# 继续训练
# model = RTDETR("runs/detect/检测_玉米雄穗_无人机视角_RTDETR_640/weights/last.pt") 
# model.train(resume=True)
