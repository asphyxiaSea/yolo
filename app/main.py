# app.py
import asyncio
import json
import time

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from config import MODEL_REGISTRY, DEVICE, DEFAULT_IMGSZ, DEFAULT_CONF, MAX_STREAM_DURATION
from inference import run_single_inference, build_stream_generator, parse_result, release_model

app = FastAPI()

infer_semaphore = asyncio.Semaphore(1)  # 单图推理并发限制


# ------------------ 单图推理接口 ------------------

@app.post("/infer/{model_name}")
async def infer_image(
    model_name: str,
    file: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    imgsz: int = Form(DEFAULT_IMGSZ),
):
    if model_name not in MODEL_REGISTRY:
        return {"error": f"未知模型: {model_name}，可选: {list(MODEL_REGISTRY.keys())}"}
    if not (0.0 <= conf <= 1.0):
        return {"error": "conf 必须在 0~1 之间"}

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "图片解码失败，请检查文件格式"}

    async with infer_semaphore:
        result = await run_in_threadpool(
            run_single_inference, model_name, MODEL_REGISTRY, img, conf, imgsz, DEVICE
        )

    return {"model": model_name, "conf": conf, "imgsz": imgsz, **result}


# ------------------ 实时推理接口:全局只允许一路 ------------------

# app.py 顶部,替换掉原来的 current_stream_token / stream_lock 定义

class StreamState:
    """用对象属性持有状态,闭包直接引用对象本身"""
    def __init__(self):
        self.current_token: str | None = None


stream_state = StreamState()
stream_lock = asyncio.Lock()


@app.get("/infer_stream/{model_name}")
async def infer_stream(
    model_name: str,
    source: str,
    request: Request,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
):
    """
    全局限制:同一时间只允许一路实时推理。
    新请求进来会自动顶替旧的,不需要前端做任何清理逻辑。
    """
    if model_name not in MODEL_REGISTRY:
        return {"error": f"未知模型: {model_name}，可选: {list(MODEL_REGISTRY.keys())}"}
    if not (0.0 <= conf <= 1.0):
        return {"error": "conf 必须在 0~1 之间"}

    # 生成本次请求的唯一token,并立刻成为"当前活跃任务"
    my_token = f"{model_name}:{time.time_ns()}"
    async with stream_lock:
        stream_state.current_token = my_token

    async def event_generator():
        config = MODEL_REGISTRY[model_name]
        model = config["loader"](config["weights"])
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()
            gen = await loop.run_in_executor(
                None, build_stream_generator, model, source, conf, imgsz, DEVICE
            )

            for r in gen:
                # 三重检查:客户端断开 / 被新请求顶替 / 超时
                if await request.is_disconnected():
                    break
                if stream_state.current_token != my_token:
                    yield f"data: {json.dumps({'error': '已被新的推理请求顶替,自动停止'})}\n\n"
                    break
                if time.time() - start_time > MAX_STREAM_DURATION:
                    yield f"data: {json.dumps({'error': '已达最大运行时长,自动停止'})}\n\n"
                    break

                result = parse_result(r)
                yield f"data: {json.dumps(result)}\n\n"
                await asyncio.sleep(0)

        finally:
            release_model(model)
            # 只有"我还是当前token"时才清空,避免误清理掉更新任务的状态
            async with stream_lock:
                if stream_state.current_token == my_token:
                    stream_state.current_token = None

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok", "models_available": list(MODEL_REGISTRY.keys())}