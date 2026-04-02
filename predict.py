from ultralytics import YOLO

model = YOLO("/home/user/models/yolov8/runs/detect/train/weights/best.pt")

# 方案一：只要速度最快（不保存视频）
results = model.predict(
    source="assets/record20260326152040.mp4",
    conf=0.4,
    device=1,
    half=True,        # FP16推理，RTX5090必开
    stream=True,      # 流式处理，不占内存
    save=True,       
    verbose=False,    # 关掉每帧打印，减少IO开销
)

for result in results:
    pass  # 或者在这里处理每帧结果