"""
验证推流地址是否真的在持续更新(而不是像Chrome直接打开那样只拿到一次性快照)。
分两部分:
1. 反复拉取 m3u8 播放列表文本,观察里面列出的分片是否在变化
2. 用 cv2.VideoCapture 持续读帧,对比帧内容是否随时间变化(判断画面是否卡死)

用法: python check_push_stream.py
"""
import hashlib
import time

import cv2
import requests

HLS_URL = "http://127.0.0.1:8008/hls/live/index.m3u8"  # 换成实际地址


def check_playlist_updates(rounds: int = 10, interval: float = 1.0):
    """
    反复拉取m3u8文本本身,观察里面列出的.ts分片名是否在变化。
    如果分片名一直不变,说明推流没有持续产出新内容,是服务端问题;
    如果分片名在滚动更新,说明推流本身没问题,是播放方式的问题。
    """
    print("=" * 60)
    print("[检查1] 观察 m3u8 播放列表内容是否持续更新")
    print("=" * 60)

    last_content = None

    for i in range(rounds):
        try:
            resp = requests.get(HLS_URL, timeout=5)
            resp.raise_for_status()
            content = resp.text
        except Exception as e:
            print(f"[第{i+1}次] 请求失败: {e}")
            time.sleep(interval)
            continue

        segments = [line for line in content.splitlines() if line.endswith(".ts")]
        changed = "变化了" if content != last_content else "没变化"

        print(f"[第{i+1}次] 分片列表: {segments} | 相比上次: {changed}")
        last_content = content
        time.sleep(interval)

    print()


def check_frame_updates(rounds: int = 10, interval: float = 1.0):
    """
    用cv2持续读帧,对每一帧算个简单哈希,对比是否随时间变化。
    如果哈希一直不变,说明画面卡死在某一帧;
    如果哈希在变,说明画面确实在持续更新。

    注意: 每次重新 VideoCapture(HLS_URL) 是为了模拟"重新拉取播放列表并
    读取最新内容"的行为,更贴近实际播放器的持续拉流方式,而不是打开一次
    连接后指望它自己感知到新分片(这更接近你之前遇到的Chrome直接打开的
    情况——只拿到当时那一刻的快照)。
    """
    print("=" * 60)
    print("[检查2] 持续读帧,观察画面内容是否随时间变化")
    print("=" * 60)

    last_hash = None

    for i in range(rounds):
        cap = cv2.VideoCapture(HLS_URL)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"[第{i+1}次] 读取帧失败")
            time.sleep(interval)
            continue

        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:12]
        changed = "画面变化了" if frame_hash != last_hash else "画面没变化(可能卡住了)"

        print(f"[第{i+1}次] 帧哈希: {frame_hash} | {changed}")
        last_hash = frame_hash
        time.sleep(interval)

    print()


if __name__ == "__main__":
    check_playlist_updates(rounds=10, interval=1.0)
    check_frame_updates(rounds=10, interval=1.0)