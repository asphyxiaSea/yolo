import os
import argparse
import math

import fiftyone as fo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 候选 imgsz：letterbox 时用 scale = imgsz / max(img_w, img_h)
CANDIDATE_IMGSZ = [640, 896, 1024, 1280, 1536]

# COCO 惯例：面积 < 32^2 记为 small，边长 32px 作为"安全线"下限
SAFE_EFFECTIVE_SIDE = 32.0
# 判断某个 imgsz 是否"够用"的标准：至少 P_THRESHOLD 分位的目标 effective_side 要 >= SAFE_EFFECTIVE_SIDE
P_THRESHOLD = 0.10  # 用 P10 而不是 median，避免被大目标掩盖小目标问题


def effective_side_for_imgsz(raw_side, img_w, img_h, imgsz):
    """给定原图尺寸和目标候选 imgsz，模拟 letterbox 后目标的有效边长（sqrt(area) 口径）"""
    scale = imgsz / max(img_w, img_h)
    return raw_side * scale


def recommend_imgsz(df):
    """
    对每个候选 imgsz，计算 letterbox 后目标有效边长的 P10。
    选出满足 P10 >= SAFE_EFFECTIVE_SIDE 的最小 imgsz；
    如果所有候选都不满足，返回最大候选并给出警告。
    """
    results = []
    for imgsz in CANDIDATE_IMGSZ:
        eff_side = df["raw_side"] * (imgsz / df[["img_w", "img_h"]].max(axis=1))
        p10 = np.percentile(eff_side, P_THRESHOLD * 100)
        median = np.median(eff_side)
        results.append({"imgsz": imgsz, "p10_effective_side": p10, "median_effective_side": median})

    result_df = pd.DataFrame(results)

    # 用 numpy 数组取值，避免 pandas .iloc/.loc 返回的 Scalar 联合类型
    # 传给 int() 时被 Pyright 判定为不兼容（Scalar 包含 complex 等 int() 不接受的类型）
    imgsz_arr = result_df["imgsz"].to_numpy()
    p10_arr = result_df["p10_effective_side"].to_numpy(dtype=float)

    qualified_mask = p10_arr >= SAFE_EFFECTIVE_SIDE
    if qualified_mask.any():
        recommended = int(imgsz_arr[qualified_mask][0])
        warn = False
    else:
        # 没有候选满足，选 P10 最大的那个（通常是最大 imgsz），并提示仍然不够
        best_idx = int(np.argmax(p10_arr))
        recommended = int(imgsz_arr[best_idx])
        warn = True

    return recommended, warn, result_df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="FiftyOne Dataset Name"
    )

    parser.add_argument(
        "--label-field",
        default="ground_truth"
    )

    parser.add_argument(
        "--output",
        default="analysis"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    dataset = fo.load_dataset(args.dataset)

    # metadata 兜底：没算过的话先算一遍，否则 sample.metadata 可能是 None
    if dataset.first().metadata is None:
        print("[信息] metadata 未计算，正在执行 dataset.compute_metadata() ...")
        dataset.compute_metadata()

    rows = []
    skipped_no_label = 0
    skipped_no_metadata = 0

    for sample in dataset:

        if sample.metadata is None or sample.metadata.width is None:
            skipped_no_metadata += 1
            continue

        img_w = sample.metadata.width
        img_h = sample.metadata.height

        label_obj = sample[args.label_field]
        if label_obj is None or label_obj.detections is None:
            skipped_no_label += 1
            continue

        detections = label_obj.detections
        if not detections:
            continue

        for det in detections:

            _, _, w, h = det.bounding_box

            pw = w * img_w
            ph = h * img_h
            area = pw * ph
            raw_side = math.sqrt(area) if area > 0 else 0.0

            rows.append(
                {
                    "sample_id": sample.id,
                    "filepath": sample.filepath,
                    "class": det.label,
                    "img_w": img_w,
                    "img_h": img_h,
                    "width_px": pw,
                    "height_px": ph,
                    "area_px": area,
                    "raw_side": raw_side,
                }
            )

    if skipped_no_metadata:
        print(f"[警告] {skipped_no_metadata} 个 sample 缺少 metadata，已跳过")
    if skipped_no_label:
        print(f"[信息] {skipped_no_label} 个 sample 在字段 '{args.label_field}' 下无标注（可能是背景图），已跳过")

    if not rows:
        raise RuntimeError("没有解析到任何目标框，请检查 --label-field 是否正确，以及 metadata 是否已计算")

    df = pd.DataFrame(rows)

    widths = df["width_px"].to_numpy()
    heights = df["height_px"].to_numpy()
    areas = df["area_px"].to_numpy()

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"Images : {dataset.count()}")
    print(f"Objects: {len(df)}")
    print(f"Classes: {df['class'].nunique()}")

    print()

    print("Width (pixel, 原图)")
    print(f"Mean   : {widths.mean():.2f}")
    print(f"Median : {np.median(widths):.2f}")
    print(f"P90    : {np.percentile(widths, 90):.2f}")

    print()

    print("Height (pixel, 原图)")
    print(f"Mean   : {heights.mean():.2f}")
    print(f"Median : {np.median(heights):.2f}")
    print(f"P90    : {np.percentile(heights, 90):.2f}")

    print()

    print("Area (pixel^2, 原图)")
    print(f"Median : {np.median(areas):.2f}")

    # ---------------------------------------------------------------- #
    # 基于 letterbox 有效尺寸的 imgsz 推荐（替换原来脱离分辨率的版本）
    # ---------------------------------------------------------------- #
    recommended_imgsz, warn, imgsz_table = recommend_imgsz(df)

    print()
    print("=" * 60)
    print("候选 imgsz 下 letterbox 后目标有效边长(px) 对比")
    print("=" * 60)
    print(imgsz_table.to_string(index=False))
    print()
    if warn:
        print(f"[警告] 所有候选 imgsz 都无法让 P{int(P_THRESHOLD*100)} 有效边长达到安全线({SAFE_EFFECTIVE_SIDE}px)，")
        print(f"       即使用最大候选 imgsz={recommended_imgsz}，仍有较多目标偏小。")
        print("       建议：提高原始采集分辨率 / 切图训练 / 使用带 P2 检测头的模型结构。")
    print(f"推荐 imgsz : {recommended_imgsz}  "
          f"(依据: 该 imgsz 下 P{int(P_THRESHOLD*100)} 有效边长 >= {SAFE_EFFECTIVE_SIDE}px)")
    print("=" * 60)

    # ---------------------------------------------------------------- #
    # 按类别统计（数量 + 尺寸），而不只是数量
    # ---------------------------------------------------------------- #
    class_stats = (
        df.groupby("class")
        .agg(
            count=("class", "size"),
            median_raw_side=("raw_side", "median"),
            p10_raw_side=("raw_side", lambda s: np.percentile(s, 10)),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    # 补充：在推荐 imgsz 下，各类别的有效边长与 small 占比
    scale_col = recommended_imgsz / df[["img_w", "img_h"]].max(axis=1)
    df["effective_side_at_recommended"] = df["raw_side"] * scale_col
    df["is_small_at_recommended"] = df["effective_side_at_recommended"] < SAFE_EFFECTIVE_SIDE

    small_ratio = (
        df.groupby("class")["is_small_at_recommended"]
        .mean()
        .reset_index()
        .rename(columns={"is_small_at_recommended": "small_ratio_at_recommended_imgsz"})
    )
    class_stats = class_stats.merge(small_ratio, on="class", how="left")
    class_stats["small_ratio_at_recommended_imgsz"] = (
        class_stats["small_ratio_at_recommended_imgsz"] * 100
    ).round(1)

    class_stats.to_csv(
        os.path.join(args.output, "class_statistics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 逐目标明细落盘，供后续做切图模拟 / 其他分析复用
    df.to_csv(
        os.path.join(args.output, "objects_full.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    imgsz_table.to_csv(
        os.path.join(args.output, "imgsz_candidates.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------- #
    # 图表
    # ---------------------------------------------------------------- #
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # Width
    plt.figure(figsize=(8, 5))
    plt.hist(widths, bins=50)
    plt.xlabel("Width (pixel, 原图)")
    plt.ylabel("Count")
    plt.title("Width Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "width_distribution.png"))
    plt.close()

    # Height
    plt.figure(figsize=(8, 5))
    plt.hist(heights, bins=50)
    plt.xlabel("Height (pixel, 原图)")
    plt.ylabel("Count")
    plt.title("Height Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "height_distribution.png"))
    plt.close()

    # Area
    plt.figure(figsize=(8, 5))
    plt.hist(np.log10(areas + 1), bins=50)
    plt.xlabel("log10(Area), 原图")
    plt.ylabel("Count")
    plt.title("Area Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "area_distribution.png"))
    plt.close()

    # Class count
    plt.figure(figsize=(10, 6))
    sorted_stats = class_stats.sort_values("count", ascending=False)
    plt.bar(sorted_stats["class"], sorted_stats["count"])
    plt.xticks(rotation=90)
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "class_distribution.png"))
    plt.close()

    # imgsz 候选对比图：P10 有效边长 vs imgsz，直观看拐点
    plt.figure(figsize=(8, 5))
    plt.plot(imgsz_table["imgsz"], imgsz_table["p10_effective_side"], marker="o", label="P10 effective side")
    plt.plot(imgsz_table["imgsz"], imgsz_table["median_effective_side"], marker="o", label="Median effective side")
    plt.axhline(SAFE_EFFECTIVE_SIDE, color="red", ls="--", lw=1, label=f"安全线 {SAFE_EFFECTIVE_SIDE}px")
    plt.xlabel("候选 imgsz")
    plt.ylabel("letterbox 后有效边长 (px)")
    plt.title("不同 imgsz 下目标有效尺寸")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "imgsz_recommendation.png"))
    plt.close()

    print()
    print("Analysis finished.")
    print(f"Output: {args.output}")
    print(" - objects_full.csv        : 逐目标明细（含原图尺寸、有效尺寸）")
    print(" - class_statistics.csv    : 按类别的数量/尺寸/small占比统计")
    print(" - imgsz_candidates.csv    : 各候选 imgsz 下的有效尺寸统计")
    print(" - imgsz_recommendation.png: imgsz 推荐可视化")


if __name__ == "__main__":
    main()