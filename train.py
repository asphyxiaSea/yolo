from ultralytics import YOLO


# 继续训练
model = YOLO("runs/detect/train5/weights/last.pt")
model.train(resume=True)



# 训练
# model = YOLO("yolo26l.pt")
model.train(
            data="datasets/VisDrone/VisDrone.yaml",
            epochs=100,
            batch=8,
            imgsz=1920,
            device=0,
            workers=8,
            )
