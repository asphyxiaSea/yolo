# config.py
from ultralytics import RTDETR, YOLO

MODEL_REGISTRY = {
    "seg_corn": {"weights": "models/分割_玉米果穗/weights/best.pt", "loader": YOLO},
    "检测_稻叶病害": {"weights": "models/检测_稻叶病害/weights/best.pt", "loader": YOLO},
    "worker_pose": {"weights": "models/检测_油茶嫩芽_RTDETR_1920/weights/best.pt", "loader": RTDETR},
}

DEVICE = 1
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.25

MAX_STREAM_DURATION = 1800  # 单路最长运行时长(秒),超时强制停止