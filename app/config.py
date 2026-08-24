# config.py
from ultralytics import RTDETR, YOLO

MODEL_REGISTRY = {
    "detect_visdrone_yolo26n_1920": {"weights": "models/检测_visdrone_yolov26n_1920/weights/best.pt", "loader": YOLO},
    "detect_visdrone_rtdetr_1280": {"weights": "models/检测_VisDrone_RTDETR_1280/weights/best.pt", "loader": RTDETR},
    "detect_water_surface_floating_yolo26s_1024": {"weights": "models/检测_水面漂浮物_yolo26s_1024/weights/best.pt", "loader": YOLO},
    "segment_corn_ears_yolo_1024": {"weights": "models/分割_玉米果穗_yolo_1024/weights/best.pt", "loader": YOLO},
    "classify_corn_pests_yolov8s_640": {"weights": "models/分类_玉米病虫害_yolov8s_640/weights/best.pt", "loader": YOLO},
    "detect_rice_leaf_disease_yolo26s_1024": {"weights": "models/检测_稻叶病害_yolo26s_1024/weights/best.pt", "loader": YOLO},
    "detect_citrus_yolo26s_1024": {"weights": "models/检测_柑橘_yolo26s_1024/weights/best.pt", "loader": YOLO},
    "detect_forest_pests_yolo26s_640": {"weights": "models/检测_林业虫害_yolo26s_640/weights/best.pt", "loader": YOLO},
    "detect_cotton_leaf_disease_yolo26s_640": {"weights": "models/检测_棉花叶片病害_yolo26s_640/weights/best.pt", "loader": YOLO},
    "detect_agricultural_pests_yolo26s_640": {"weights": "models/检测_农业综合虫害_yolo26s_640/weights/best.pt", "loader": YOLO},
    "detect_rice_pest_lamp_yolo26s_640": {"weights": "models/检测_水稻虫情灯虫害_yolo26s_640/weights/best.pt", "loader": YOLO},
    "detect_rice_field_pests_yolo26s_640": {"weights": "models/检测_水稻田间虫害_yolo26s_640/weights/best.pt", "loader": YOLO},
    "detect_dangerous_rock_yolo26n_640": {"weights": "models/检测_危岩_yolo26n_640/weights/best.pt", "loader": YOLO},
    "detect_sheep_yolo26n_1024": {"weights": "models/检测_羊_yolo26n_1024/weights/best.pt", "loader": YOLO},
    "detect_oil_tea_fruit_flower_yolo26s_1024": {"weights": "models/检测_油茶果_油茶花_yolo26s_1024/weights/best.pt", "loader": YOLO},
    "detect_oil_tea_buds_yolo26s_1920": {"weights": "models/检测_油茶嫩芽_yolo26s_1920/weights/best.pt", "loader": YOLO},
    "detect_oil_tea_buds_rtdetr_1920": {"weights": "models/检测_油茶嫩芽_rtdetr_1920/weights/best.pt", "loader": RTDETR},
    "detect_corn_seedlings_drone_view_rtdetr_400": {"weights": "models/检测_玉米苗_无人机视角_rtdetr_400/weights/best.pt", "loader": RTDETR},
    "detect_corn_tassels_drone_view_rtdetr_640": {"weights": "models/检测_玉米雄穗_无人机视角_rtdetr_640/weights/best.pt", "loader": RTDETR},
    "detect_corn_leaf_disease_yolo26s_1024": {"weights": "models/检测_玉米叶片病害_yolo26s_1024/weights/best.pt", "loader": YOLO},
    "detect_comprehensive_grass_harm_yolo26s_640": {"weights": "models/检测_综合草害_yolo26s_640/weights/best.pt", "loader": YOLO},
}

DEVICE = 1
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.25

MAX_STREAM_DURATION = 1800  # 单路最长运行时长(秒),超时强制停止