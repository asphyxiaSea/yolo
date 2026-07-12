from ultralytics import YOLO


# 继续训练
# model = YOLO("runs/detect/检测_羊/weights/last.pt")
# model.train(resume=True)



# 训练
model = YOLO("models/yolo26n.pt")
model.train(
    data="datasets/stone_v2/data.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device=1,
    workers=8,
    name="检测_危岩"
)
