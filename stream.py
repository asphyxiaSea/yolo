import asyncio
import time
import threading
from contextlib import asynccontextmanager
from threading import Thread

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

# ─── 全局状态 ────────────────────────────────────────────
model: YOLO | None = None
detect_thread: Thread | None = None
latest_result = None        # 只保留最新一帧结果
result_lock = threading.Lock()
is_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = YOLO("runs/detect/train2/weights/best.pt")
    print("模型加载完成")
    yield
    stop_detect()


app = FastAPI(lifespan=lifespan)


# ─── 检测线程 ────────────────────────────────────────────
def detect_worker(url: str):
    global is_running, latest_result
    if model is None:
        print("模型未初始化，无法开始检测")
        return

    yolo_model = model
    is_running = True

    while is_running:
        try:
            print(f"连接直播流: {url}")
            results = yolo_model.predict(
                source=url,
                imgsz=1024,
                conf=0.5,
                device=1,
                half=True,
                stream=True,
                verbose=False,
                save=True,
                vid_stride=5,
            )
            for result in results:
                if not is_running:
                    break
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    data = {
                        "count": len(boxes),
                        "boxes": [
                            {
                                "xyxy": b.xyxy[0].tolist(),
                                "conf": float(b.conf[0]),
                                "cls": int(b.cls[0]),
                            }
                            for b in boxes
                        ],
                    }
                else:
                    data = {"count": 0, "boxes": []}

                # 直接覆盖，永远只保留最新一帧
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
def start_stream(url: str):
    global detect_thread

    if detect_thread and detect_thread.is_alive():
        return {"status": "already_running"}

    stop_detect()
    detect_thread = Thread(target=detect_worker, args=(url,), daemon=True)
    detect_thread.start()
    return {"status": "started", "url": url}


@app.post("/yolo/stop")
def stop_stream():
    stop_detect()
    return {"status": "stopped"}


@app.websocket("/yolo/ws/results")
async def websocket_results(websocket: WebSocket):
    await websocket.accept()
    last_sent = None

    try:
        while True:
            with result_lock:
                current = latest_result

            # 有新结果才推，避免重复推同一帧
            if current is not None and current is not last_sent:
                await websocket.send_json(current)
                last_sent = current
            else:
                await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("WebSocket 客户端断开")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)