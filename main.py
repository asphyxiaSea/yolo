import asyncio
import time
import threading
from contextlib import asynccontextmanager
from threading import Thread

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

# ─── 全局状态 ────────────────────────────────────────────
model_cache: dict[str, YOLO] = {}          # 模型缓存，key 为 model_id
model_cache_lock = threading.Lock()

detect_thread: Thread | None = None
latest_result = None
result_lock = threading.Lock()
is_running = False

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


def resolve_model(model_id: str) -> YOLO:
    """
    支持两种格式：
    - 路径：  "runs/detect/train2/weights/best.pt"
    - 名称：  "yolov8n" / "yolov8s" 等 ultralytics 内置名，
              或自定义别名（可在下方 ALIAS 表中扩展）
    同一 model_id 只加载一次，后续从缓存取。
    """
    ALIAS: dict[str, str] = {
        # 业务别名 → 实际权重路径，按需扩展
        "visdrone":  "runs/detect/train5/weights/best.pt",
    }

    resolved = ALIAS.get(model_id, model_id)   # 别名 → 路径/官方名

    with model_cache_lock:
        if resolved not in model_cache:
            print(f"加载模型: {resolved}")
            model_cache[resolved] = YOLO(resolved)
        return model_cache[resolved]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 不再强制预热，按需加载即可
    # 如果希望预热某个默认模型，可在此调用 resolve_model("yolov8n")
    yield
    stop_detect()
    with model_cache_lock:
        model_cache.clear()
    print("模型缓存已清理")


app = FastAPI(lifespan=lifespan)


# ─── 检测线程 ────────────────────────────────────────────
def detect_worker(url: str, model_id: str):
    global is_running, latest_result

    try:
        yolo_model = resolve_model(model_id)
    except Exception as e:
        print(f"模型加载失败: {e}")
        is_running = False
        return

    is_running = True

    while is_running:
        try:
            print(f"连接直播流: {url}，使用模型: {model_id}")
            results = yolo_model.predict(
                source=url,
                imgsz=1920,
                conf=0.5,
                device=0,
                half=True,
                stream=True,
                verbose=False,
                save=True,
                vid_stride=1,
            )
            for result in results:
                if not is_running:
                    break
                boxes = result.boxes
                data = (
                    {
                        "count": len(boxes),
                        "model": model_id,
                        "boxes": [
                            {
                                "xyxy": b.xyxy[0].tolist(),
                                "conf": float(b.conf[0]),
                                "cls":  int(b.cls[0]),
                            }
                            for b in boxes
                        ],
                    }
                    if boxes is not None and len(boxes) > 0
                    else {"count": 0, "model": model_id, "boxes": []}
                )

                with result_lock:
                    latest_result = data

        except Exception as e:
            if not is_running:
                break
            print(f"流中断: {e}，5秒后重连...")
            time.sleep(5)


def stop_detect():
    global is_running, detect_thread, latest_result
    is_running = False
    detect_thread = None
    latest_result = None


# ─── 接口 ────────────────────────────────────────────────
@app.post("/yolo/start")
def start_stream(url: str, model: str = "visdrone"):
    """
    参数:
      url   - 视频流地址
      model - 模型标识，支持:
              • 别名:  "person" / "car" / "helmet"
              • 路径:  "runs/detect/train2/weights/best.pt"
              • 官方名: "yolov8n" / "yolov8s" 等
    """
    global detect_thread

    if detect_thread and detect_thread.is_alive():
        return {"status": "already_running"}

    stop_detect()
    detect_thread = Thread(
        target=detect_worker,
        args=(url, model),
        daemon=True,
    )
    detect_thread.start()
    return {"status": "started", "url": url, "model": model}


@app.post("/yolo/stop")
def stop_stream():
    stop_detect()
    return {"status": "stopped"}


@app.get("/yolo/models")
def list_models():
    """查看当前已缓存（已加载）的模型"""
    with model_cache_lock:
        return {"loaded_models": list(model_cache.keys())}


@app.websocket("/yolo/ws/results")
async def websocket_results(websocket: WebSocket):
    await websocket.accept()
    last_sent = None

    try:
        while True:
            with result_lock:
                current = latest_result

            if current is not None and current is not last_sent:
                await websocket.send_json(current)
                last_sent = current
            else:
                await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("WebSocket 客户端断开")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)