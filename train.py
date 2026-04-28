from ultralytics import YOLO


# 继续训练
# model = YOLO("runs/detect/train7/weights/last.pt")
# model.train(resume=True)



# 训练
model = YOLO("yolo26l.pt")
model.train(
            data="datasets/GarBage/data.yaml",
            epochs=100,
            batch=8,
            imgsz=1024,
            device=0,
            workers=8,
            )
