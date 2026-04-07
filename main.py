import asyncio
import subprocess
import time
import threading
from contextlib import asynccontextmanager
from threading import Thread

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ─── 全局状态 ────────────────────────────────────────────
model_cache: dict[str, YOLO] = {}
model_cache_lock = threading.Lock()

detect_thread: Thread | None = None
is_running = False

latest_result: dict | None = None
latest_result_lock = threading.Lock()
new_result_event: asyncio.Event | None = None
main_loop: asyncio.AbstractEventLoop | None = None


def resolve_model(model_id: str) -> YOLO:
    ALIAS: dict[str, str] = {
        "visdrone": "runs/detect/train3/weights/best.pt",
    }
    resolved = ALIAS.get(model_id, model_id)
    with model_cache_lock:
        if resolved not in model_cache:
            print(f"加载模型: {resolved}")
            model_cache[resolved] = YOLO(resolved)
        return model_cache[resolved]


def probe_resolution(url: str, fallback_w=2880, fallback_h=1620) -> tuple[int, int]:
    """用 cv2 探测流分辨率，失败则返回 fallback"""
    try:
        cap = cv2.VideoCapture(url)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if W > 0 and H > 0:
            return W, H
    except Exception:
        pass
    return fallback_w, fallback_h


@asynccontextmanager
async def lifespan(app: FastAPI):
    global new_result_event, main_loop
    new_result_event = asyncio.Event()
    main_loop = asyncio.get_event_loop()
    yield
    stop_detect()
    with model_cache_lock:
        model_cache.clear()
    print("模型缓存已清理")


app = FastAPI(lifespan=lifespan)


# ─── 结果推送 ────────────────────────────────────────────
def _push_result(data: dict):
    global latest_result
    with latest_result_lock:
        latest_result = data
    if main_loop and new_result_event:
        main_loop.call_soon_threadsafe(new_result_event.set)


# ─── 检测线程 ────────────────────────────────────────────
def detect_worker(url: str, model_id: str):
    global is_running

    try:
        yolo_model = resolve_model(model_id)
    except Exception as e:
        print(f"模型加载失败: {e}")
        is_running = False
        return

    is_running = True

    while is_running:
        proc = None
        try:
            W, H = probe_resolution(url)
            print(f"连接流: {url}，分辨率: {W}x{H}，模型: {model_id}")

            cmd = [
                "ffmpeg",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-i", url,
                "-pix_fmt", "bgr24",
                "-vcodec", "rawvideo",
                "-an",
                "-f", "rawvideo",
                "pipe:1",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if proc.stdout is None:
                raise RuntimeError("ffmpeg stdout pipe is not available")
            frame_size = W * H * 3

            while is_running:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    print("流中断或结束")
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
                results = yolo_model.predict(
                    source=frame,
                    imgsz=1920,
                    conf=0.5,
                    device=0,
                    half=True,
                    save=True,
                    verbose=False,
                )
                boxes = results[0].boxes
                data = (
                    {
                        "count": len(boxes),
                        "model": model_id,
                        "boxes": [
                            {
                                "xyxy": b.xyxy[0].tolist(),
                                "conf": float(b.conf[0]),
                                "cls": int(b.cls[0]),
                            }
                            for b in boxes
                        ],
                    }
                    if boxes is not None and len(boxes) > 0
                    else {"count": 0, "model": model_id, "boxes": []}
                )
                _push_result(data)

        except Exception as e:
            if not is_running:
                break
            print(f"异常: {e}，5秒后重连...")
        finally:
            if proc:
                proc.kill()
                proc.wait()

        if is_running:
            time.sleep(5)


def stop_detect():
    global is_running, detect_thread
    is_running = False
    detect_thread = None


# ─── 接口 ────────────────────────────────────────────────
@app.post("/yolo/start")
def start_stream(url: str, model: str = "visdrone"):
    global detect_thread
    if detect_thread and detect_thread.is_alive():
        return {"status": "already_running"}
    stop_detect()
    detect_thread = Thread(target=detect_worker, args=(url, model), daemon=True)
    detect_thread.start()
    return {"status": "started", "url": url, "model": model}


@app.post("/yolo/stop")
def stop_stream():
    stop_detect()
    return {"status": "stopped"}


@app.get("/yolo/models")
def list_models():
    with model_cache_lock:
        return {"loaded_models": list(model_cache.keys())}


@app.websocket("/yolo/ws/results")
async def websocket_results(websocket: WebSocket):
    await websocket.accept()
    event = new_result_event
    if event is None:
        await websocket.send_json({"type": "error", "message": "result event not initialized"})
        await websocket.close(code=1011)
        return

    try:
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
                continue

            event.clear()
            with latest_result_lock:
                data = latest_result

            if data:
                await websocket.send_json(data)

    except WebSocketDisconnect:
        print("WebSocket 客户端断开")
        stop_detect()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)