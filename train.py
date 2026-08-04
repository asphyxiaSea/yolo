from ultralytics import YOLO, RTDETR


# 继续训练
# model = YOLO("runs/detect/检测_农业综合虫害/weights/last.pt")
# model.train(resume=True)



# 训练
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


# RTDETR 模型训练
model = RTDETR("rtdetr-l.pt")  # 加载预训练权重做迁移学习

results = model.train(
    data="datasets/玉米雄穗_无人机视角_640/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,          # RT-DETR 显存占用比 YOLO 大不少,batch 别设太高
    device=1,          # 用第几张 GPU,你 5090 应该单卡够用
    workers=8,
    patience=20,       # early stopping
    # project="runs/rtdetr",
    name="检测_玉米雄穗_无人机视角_RTDETR_640",
)

# 继续训练
# model = RTDETR("runs/detect/检测_玉米雄穗_无人机视角_RTDETR_640/weights/last.pt") 
# model.train(resume=True)
