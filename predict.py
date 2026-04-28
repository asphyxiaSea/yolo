from ultralytics import YOLO
import cv2
import numpy as np


def _safe_to_numpy(value):
	"""Convert torch.Tensor / numpy array / list to numpy array."""
	if value is None:
		return None
	if hasattr(value, "cpu"):
		return value.cpu().numpy()
	return np.asarray(value)


def _calc_bbox_area(xyxy):
	x1, y1, x2, y2 = xyxy
	w = max(0.0, float(x2 - x1))
	h = max(0.0, float(y2 - y1))
	return w * h


def _calc_lr_symmetry_score(keypoints_xy, keypoints_conf, conf_thres=0.3):
	"""Return left-right keypoint distance mean as a simple symmetry indicator (smaller is better)."""
	# COCO 17-keypoints left/right pairs
	lr_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

	valid_dist = []
	for li, ri in lr_pairs:
		lc = keypoints_conf[li]
		rc = keypoints_conf[ri]
		if lc >= conf_thres and rc >= conf_thres:
			lxy = keypoints_xy[li]
			rxy = keypoints_xy[ri]
			valid_dist.append(float(np.linalg.norm(lxy - rxy)))

	if not valid_dist:
		return None
	return float(np.mean(valid_dist))


def analyze_pose_results(results, conf_thres=0.3):
	"""Analyze YOLO-pose predictions and print per-instance and global statistics."""
	total_images = len(results)
	total_people = 0
	kp_conf_all = []
	bbox_area_all = []

	for image_idx, result in enumerate(results):
		boxes = result.boxes
		keypoints = result.keypoints

		if boxes is None or keypoints is None:
			print(f"[Image {image_idx}] no boxes or keypoints")
			continue

		xyxy = _safe_to_numpy(boxes.xyxy)
		box_conf = _safe_to_numpy(boxes.conf)
		kp_xy = _safe_to_numpy(keypoints.xy)
		kp_conf = _safe_to_numpy(keypoints.conf)

		if xyxy is None or kp_xy is None or kp_conf is None:
			print(f"[Image {image_idx}] empty prediction data")
			continue

		n_person = len(xyxy)
		total_people += n_person
		print(f"\n[Image {image_idx}] detected people: {n_person}")

		for person_idx in range(n_person):
			person_box = xyxy[person_idx]
			person_box_conf = float(box_conf[person_idx]) if box_conf is not None else None
			person_kp_xy = kp_xy[person_idx]
			person_kp_conf = kp_conf[person_idx]

			valid_mask = person_kp_conf >= conf_thres
			valid_count = int(valid_mask.sum())
			kp_mean = float(person_kp_conf.mean())
			kp_min = float(person_kp_conf.min())
			kp_max = float(person_kp_conf.max())
			bbox_area = _calc_bbox_area(person_box)
			symmetry = _calc_lr_symmetry_score(person_kp_xy, person_kp_conf, conf_thres=conf_thres)

			kp_conf_all.extend(person_kp_conf.tolist())
			bbox_area_all.append(bbox_area)

			print(
				f"  person={person_idx:02d} "
				f"box_conf={person_box_conf:.3f} "
				f"bbox_area={bbox_area:.1f} "
				f"valid_kp={valid_count}/17 "
				f"kp_conf(mean/min/max)={kp_mean:.3f}/{kp_min:.3f}/{kp_max:.3f} "
				f"lr_sym={symmetry if symmetry is not None else 'NA'}"
			)

	print("\n===== Global Pose Summary =====")
	print(f"images={total_images}, people={total_people}")

	if kp_conf_all:
		kp_conf_arr = np.array(kp_conf_all, dtype=np.float32)
		print(
			"kp_conf(global mean/p50/p90/min/max)="
			f"{kp_conf_arr.mean():.3f}/"
			f"{np.percentile(kp_conf_arr, 50):.3f}/"
			f"{np.percentile(kp_conf_arr, 90):.3f}/"
			f"{kp_conf_arr.min():.3f}/"
			f"{kp_conf_arr.max():.3f}"
		)

	if bbox_area_all:
		bbox_area_arr = np.array(bbox_area_all, dtype=np.float32)
		print(
			"bbox_area(global mean/p50/p90/min/max)="
			f"{bbox_area_arr.mean():.1f}/"
			f"{np.percentile(bbox_area_arr, 50):.1f}/"
			f"{np.percentile(bbox_area_arr, 90):.1f}/"
			f"{bbox_area_arr.min():.1f}/"
			f"{bbox_area_arr.max():.1f}"
		)

model = YOLO("yolo26l-pose.pt")

# detect
# results = model.predict(
#     source="assets/record20260326152040.mp4",
#     conf=0.4,
#     device=1,
#     half=True,        # FP16推理，RTX5090必开
#     stream=True,      # 流式处理，不占内存
#     save=True,       
#     verbose=False,    # 关掉每帧打印，减少IO开销
# )


# pose
results = model.predict("assets/pose.png", save=False)  # 自动保存到 runs/pose/predict/
analyze_pose_results(results, conf_thres=0.3)