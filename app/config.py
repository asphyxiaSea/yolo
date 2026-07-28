# config.py
from ultralytics import RTDETR, YOLO

MODEL_REGISTRY = {
    "camellia_det": {"weights": "models/rtdetr_camellia.pt", "loader": RTDETR},
    "field_seg": {"weights": "models/yolo_seg.pt", "loader": YOLO},
    "worker_pose": {"weights": "models/yolo_pose.pt", "loader": YOLO},
}

DEVICE = 1
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.25

MAX_STREAM_DURATION = 1800  # 单路最长运行时长(秒),超时强制停止