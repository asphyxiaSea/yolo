"""
测试 /custom/infer_stream/{model_name} 这个SSE接口,接真实的HLS推流。
先做一次原始流打开耗时诊断,再跑正式的SSE推理测试,并保存标注后的视频/推流HLS。
用法: python test_stream.py
按 Ctrl+C 可随时优雅停止,会正常关闭连接、让服务端把视频文件保存完整。
"""
import json
import time

import cv2
import requests

BASE_URL = "http://127.0.0.1:8008"
MODEL_NAME = "detect_visdrone_rtdetr_1280"
SOURCE = "assets/test.mp4"
CONF = 0.65
IMGSZ = 1280

# 保存标注后视频用的参数,不需要保存就把 SAVE_VIDEO_PATH 设为 None
SAVE_VIDEO_PATH = "/home/user/models/yolo/assets/test_output.mp4"
SAVE_VIDEO_FPS = 1  # 按你实测的推理吞吐大致填,不需要跟源视频真实fps一致

# 固定采样密度抽帧,不需要限制就设为 None(每帧都推理)
TARGET_FPS = 3  # 例如设成 3 表示固定每秒推理1次,不管源视频真实帧率

# 推流HLS,供展示系统直接播放,画面与检测框天然对齐
PUSH_HLS = True
PUSH_HLS_FPS = 1
HLS_SEGMENT_TIME = 2   # 每个.ts分片时长(秒)
HLS_LIST_SIZE = 5      # index.m3u8保留的分片数量

# 目标跟踪: 给同一目标在连续帧间保持一致的track_id
ENABLE_TRACKING = True
TRACKER = "bytetrack.yaml"  # 或 "botsort.yaml"(更准但更慢)

# 检测截图: 只在出现新目标时才截图(需要ENABLE_TRACKING=True才有意义)
SAVE_SNAPSHOT = True
SNAPSHOT_NEW_TARGETS_ONLY = True

SSE_NEW_TARGETS_ONLY = True  # SSE消息里只返回新目标,不返回所有目标,减少网络带宽占用

def diagnose_raw_stream():
    """
    不经过推理服务,直接用cv2打开这个HLS流,看看单纯是打开连接+读第一帧
    要花多久,用来判断瓶颈是不是在网络/协议层面,而不是推理本身。
    """
    print("=" * 60)
    print("[诊断] 开始测试原始流打开耗时(不经过推理服务)")
    print("=" * 60)

    t0 = time.time()
    cap = cv2.VideoCapture(SOURCE)
    t1 = time.time()
    print(f"[诊断] VideoCapture构造耗时: {t1 - t0:.1f}s, isOpened={cap.isOpened()}")

    if not cap.isOpened():
        print("[诊断] 流打不开,后面的SSE测试大概率也会失败")
        return

    print("[诊断] 开始读第一帧...")
    ret, frame = cap.read()
    t2 = time.time()
    print(f"[诊断] 读第一帧耗时: {t2 - t1:.1f}s, ret={ret}")

    if ret:
        print(f"[诊断] 帧尺寸: {frame.shape}")

    cap.release()
    print("=" * 60)
    print()


def test_sse_stream():
    params = {
        "source": SOURCE,
        "conf": CONF,
        "imgsz": IMGSZ,
    }

    if SAVE_VIDEO_PATH:
        params["save_video_path"] = SAVE_VIDEO_PATH
        params["save_video_fps"] = SAVE_VIDEO_FPS

    if TARGET_FPS:
        params["target_fps"] = TARGET_FPS

    if PUSH_HLS:
        params["push_hls"] = True
        params["push_hls_fps"] = PUSH_HLS_FPS
        params["hls_segment_time"] = HLS_SEGMENT_TIME
        params["hls_list_size"] = HLS_LIST_SIZE

    if ENABLE_TRACKING:
        params["enable_tracking"] = True
        params["tracker"] = TRACKER

    if SAVE_SNAPSHOT:
        params["save_snapshot"] = True
        if SNAPSHOT_NEW_TARGETS_ONLY:
            params["snapshot_new_targets_only"] = True
    if SSE_NEW_TARGETS_ONLY:
        params["sse_new_targets_only"] = True

    url = f"{BASE_URL}/custom/infer_stream/{MODEL_NAME}"
    print(f"连接: {url}")
    print(f"参数: {params}")
    print("提示: 按 Ctrl+C 可随时优雅停止")
    if PUSH_HLS:
        print(f"推流地址(展示系统播放这个): {BASE_URL}/custom/hls/live/index.m3u8")

    start_time = time.time()
    frame_count = 0

    try:
        with requests.get(url, params=params, stream=True, timeout=(5, None)) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue  # SSE消息之间的空行

                if not raw_line.startswith("data:"):
                    continue

                payload = raw_line[len("data:"):].strip()

                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    print(f"[解析失败] 原始内容: {payload}")
                    continue

                if "error" in data:
                    print(f"[服务端错误/终止] {data['error']}")
                    break

                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                print(f"[帧 {frame_count}] 耗时{elapsed:.1f}s | {fps:.1f} fps")
                print(json.dumps(data, ensure_ascii=False, indent=2))

                # 如果这一帧触发了截图,额外标注一下新目标信息,方便肉眼确认逻辑生效
                if "snapshot" in data:
                    new_ids = data.get("new_track_ids", "N/A")
                    print(f"  📸 触发截图: {data['snapshot']} | 新目标track_id: {new_ids}")

                print("-" * 60)

    except KeyboardInterrupt:
        # 手动Ctrl+C时,退出with块会正常关闭这次HTTP连接(即requests底层socket),
        # 服务端的 request.is_disconnected() 检测到后会走 finally 清理资源、
        # 保存视频文件,不会因为强行杀掉进程而导致文件损坏。
        print("\n[手动停止] 收到 Ctrl+C,正在优雅关闭连接...")

    print(f"\n流结束,共处理 {frame_count} 帧,总耗时 {time.time() - start_time:.1f}s")
    if SAVE_VIDEO_PATH:
        print(f"标注后视频应已保存至服务端路径: {SAVE_VIDEO_PATH}")
    if PUSH_HLS:
        print(f"推流应已生成: {BASE_URL}/custom/hls/live/index.m3u8 (可用VLC或浏览器测试播放)")
    if SAVE_SNAPSHOT:
        print(f"检测截图(限时保留)地址前缀: {BASE_URL}/custom/snapshots/")


if __name__ == "__main__":
    diagnose_raw_stream()
    test_sse_stream()