# inference.py
import gc

import torch


def parse_detection(result) -> list[dict]:
    boxes = result.boxes
    if boxes is None:
        return []
    return [
        {"cls": int(b.cls), "conf": float(b.conf), "xyxy": b.xyxy[0].tolist()}
        for b in boxes
    ]


def parse_segmentation(result) -> list[dict]:
    masks = result.masks
    boxes = result.boxes
    if masks is None or boxes is None:
        return []
    return [
        {
            "cls": int(b.cls),
            "conf": float(b.conf),
            "xyxy": b.xyxy[0].tolist(),
            "polygon": m.xy[0].tolist(),
        }
        for b, m in zip(boxes, masks)
    ]


def parse_pose(result) -> list[dict]:
    keypoints = result.keypoints
    boxes = result.boxes
    if keypoints is None or boxes is None:
        return []
    return [
        {
            "cls": int(b.cls),
            "conf": float(b.conf),
            "xyxy": b.xyxy[0].tolist(),
            "keypoints": kp.xy[0].tolist(),
            "kp_conf": kp.conf[0].tolist() if kp.conf is not None else None,
        }
        for b, kp in zip(boxes, keypoints)
    ]


def parse_result(result) -> dict:
    if result.masks is not None:
        return {"task": "segmentation", "detections": parse_segmentation(result)}
    if result.keypoints is not None:
        return {"task": "pose", "detections": parse_pose(result)}
    if result.boxes is not None:
        return {"task": "detection", "detections": parse_detection(result)}
    return {"task": "unknown", "detections": []}


def release_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_single_inference(
    model_name: str,
    model_registry: dict,
    img,
    conf: float,
    imgsz: int,
    device: int,
) -> dict:
    config = model_registry[model_name]
    model = config["loader"](config["weights"])
    try:
        results = model.predict(
            img, imgsz=imgsz, conf=conf,
            device=device, verbose=False, half=True,
        )
        return parse_result(results[0])
    finally:
        release_model(model)


def build_stream_generator(model, source: str, conf: float, imgsz: int, device: int):
    return model.predict(
        source=source, imgsz=imgsz, conf=conf,
        device=device, half=True, stream=True, verbose=False,
    )