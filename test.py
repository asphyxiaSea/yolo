# import subprocess
# import numpy as np
# import time
# from ultralytics import YOLO

# URL = "http://183.230.196.249:18000/hls/H-87b4bcfca291f8bb/H-87b4bcfca291f8bb_live.m3u8"
# W, H = 2880, 1620

# cmd = [
#     "ffmpeg",
#     "-fflags", "nobuffer",
#     "-flags", "low_delay",
#     "-i", URL,
#     "-pix_fmt", "bgr24",
#     "-vcodec", "rawvideo",
#     "-an",
#     "-f", "rawvideo",
#     "pipe:1",
# ]

# print("加载模型...")
# yolo_model = YOLO("runs/detect/train5/weights/best.pt")

# print("启动 ffmpeg...")
# t = time.time()
# proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# frame_size = W * H * 3
# print("开始推理...")

# for i in range(10):
#     raw = proc.stdout.read(frame_size)
#     if len(raw) < frame_size:
#         print(f"第 {i} 帧：流中断，读到 {len(raw)} 字节")
#         break

#     frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
#     t_infer = time.time()
#     results = yolo_model.predict(
#         source=frame,
#         imgsz=1920,
#         conf=0.5,
#         device=0,
#         half=True,
#         verbose=False,
#     )
#     boxes = results[0].boxes
#     print(f"第 {i} 帧，距启动: {time.time()-t:.2f}s，推理: {time.time()-t_infer:.3f}s，检测数: {len(boxes)}")

# proc.kill()
# proc.wait()