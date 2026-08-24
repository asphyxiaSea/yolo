# tiling.py
import numpy as np


def split_image(img: np.ndarray, grid: int, overlap_ratio: float = 0.15):
    """
    将图像按网格切分,并留出重叠区域,避免目标被硬切断。

    grid: 4 -> 2x2, 9 -> 3x3
    overlap_ratio: 相邻tile的重叠比例(基于tile尺寸),用于减少边界漏检

    返回: [(tile_img, x_offset, y_offset), ...]
    """
    rows = cols = int(grid ** 0.5)
    if rows * cols != grid:
        raise ValueError(f"grid 必须是完全平方数(4/9/16...),收到: {grid}")

    h, w = img.shape[:2]
    tile_h, tile_w = h // rows, w // cols
    overlap_h, overlap_w = int(tile_h * overlap_ratio), int(tile_w * overlap_ratio)

    tiles = []
    for r in range(rows):
        for c in range(cols):
            y1 = max(0, r * tile_h - overlap_h)
            y2 = min(h, (r + 1) * tile_h + overlap_h)
            x1 = max(0, c * tile_w - overlap_w)
            x2 = min(w, (c + 1) * tile_w + overlap_w)
            tiles.append((img[y1:y2, x1:x2], x1, y1))

    return tiles


def offset_detection(det: dict, x_off: int, y_off: int) -> dict:
    """
    将单条检测结果(det)从tile局部坐标系映射回原图坐标系。
    兼容 detection(xyxy) / segmentation(xyxy+polygon) / pose(xyxy+keypoints)。
    就地修改后返回,调用方不需要关心具体任务类型。
    """
    x1, y1, x2, y2 = det["xyxy"]
    det["xyxy"] = [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off]

    if "polygon" in det and det["polygon"] is not None:
        det["polygon"] = [[px + x_off, py + y_off] for px, py in det["polygon"]]

    if "keypoints" in det and det["keypoints"] is not None:
        det["keypoints"] = [[kx + x_off, ky + y_off] for kx, ky in det["keypoints"]]

    return det


def merge_detections(all_dets: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """
    all_dets: 已经映射回原图坐标系的检测结果列表,每条至少含 xyxy/conf/cls。
    这里只做跨tile的重复框去重(NMS),polygon/keypoints 等附加字段跟随 xyxy 一起保留。
    """
    if not all_dets:
        return []

    boxes = np.array([d["xyxy"] for d in all_dets])
    scores = np.array([d["conf"] for d in all_dets])
    classes = np.array([d["cls"] for d in all_dets])

    keep_indices = []
    # 按类别分别做NMS,避免不同类别互相抑制
    for cls_id in np.unique(classes):
        cls_mask = classes == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_orig_idx = np.where(cls_mask)[0]

        order = cls_scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ious = _compute_iou(cls_boxes[i], cls_boxes[order[1:]])
            order = order[1:][ious < iou_threshold]

        keep_indices.extend(cls_orig_idx[keep].tolist())

    return [all_dets[i] for i in keep_indices]


def _compute_iou(box, boxes):
    box = np.asarray(box)
    boxes = np.asarray(boxes)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area_box + area_boxes - inter + 1e-9)